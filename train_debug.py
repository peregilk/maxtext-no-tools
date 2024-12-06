"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import os
import sys
import functools
import time
from typing import Sequence, Optional
from absl import app
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import grain.python as grain
import jax
import numpy as np
import orbax.checkpoint
import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager

import checkpointing
import max_utils
import maxtext_utils
import max_logging
import optimizers
import profiler
import pyconfig
import pathwaysutils  # pylint: disable=unused-import

from vertex_tensorboard import VertexTensorboardManager

from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models

import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh
from jax.experimental import checkify

from cloud_tpu_diagnostics import diagnostic
from cloud_tpu_diagnostics.configuration import debug_configuration
from cloud_tpu_diagnostics.configuration import diagnostic_configuration
from cloud_tpu_diagnostics.configuration import stack_trace_configuration

from layers import quantizations
from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3

def validate_train_config(config):
  """Validates the configuration is set correctly for train.py"""
  assert config.run_name, "Erroring out, need a real run_name"
  if not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."
  if config.quantization == "fp8":
    assert (
        config.gradient_accumulation_steps == 1
    ), "fp8 can't be used with gradient_accumulation_steps right now."

def get_first_step(state):
  with jax.spmd_mode("allow_all"):
    return int(state.step)

def load_next_batch(train_iter, example_batch, config):
  """Loads the next batch."""
  batch = example_batch if (config.reuse_example_batch and example_batch is not None) else next(train_iter)
  return batch

def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr, per_device_tokens):
  metrics["scalar"].update({"perf/step_time_seconds": step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tflops": per_device_tflops})
  metrics["scalar"].update({"perf/per_device_tflops_per_sec": per_device_tflops / step_time_delta.total_seconds()})
  metrics["scalar"].update({"perf/per_device_tokens": per_device_tokens})
  metrics["scalar"].update({"perf/per_device_tokens_per_sec": per_device_tokens / step_time_delta.total_seconds()})
  metrics["scalar"].update({"learning/current_learning_rate": lr})

_buffered_step = None
_buffered_metrics = None

def write_metrics(writer, local_metrics_file, running_gcs_metrics, metrics, step, config, is_training=True):
  metrics_to_write, steps_to_write = None, None
  if is_training:
    global _buffered_step, _buffered_metrics
    if _buffered_metrics is not None:
      if _buffered_step is None:
        raise ValueError(f"When writing metrics, {_buffered_step=} was none")
      metrics_to_write = _buffered_metrics
      steps_to_write = _buffered_step
  else:
    metrics_to_write = metrics
    steps_to_write = step

  if metrics_to_write:
    write_metrics_to_tensorboard(writer, metrics_to_write, steps_to_write, config, is_training)
    if config.metrics_file:
      max_utils.write_metrics_locally(metrics_to_write, steps_to_write, config, local_metrics_file, is_training)
    if config.gcs_metrics and jax.process_index() == 0:
      running_gcs_metrics = max_utils.write_metrics_for_gcs(
          metrics_to_write, steps_to_write, config, running_gcs_metrics, is_training
      )

  if is_training:
    _buffered_step = step
    _buffered_metrics = metrics

def write_metrics_to_tensorboard(writer, metrics, step, config, is_training=True):
  with jax.spmd_mode("allow_all"):
    if jax.process_index() == 0:
      for metric_name in metrics.get("scalar", []):
        writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
      for metric_name in metrics.get("scalars", []):
        writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

    if is_training:
      full_log = step % config.log_period == 0
      max_logging.log(
          f"completed step: {step}, seconds: {metrics['scalar']['perf/step_time_seconds']:.3f}, "
          f"TFLOP/s/device: {metrics['scalar']['perf/per_device_tflops_per_sec']:.3f}, "
          f"Tokens/s/device: {metrics['scalar']['perf/per_device_tokens_per_sec']:.3f}, "
          f"total_weights: {metrics['scalar']['learning/total_weights']}, "
          f"loss: {metrics['scalar']['learning/loss']:.3f}"
      )
      if full_log and jax.process_index() == 0:
        max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")
        writer.flush()

def clear_buffered_metrics():
  global _buffered_step
  global _buffered_metrics
  _buffered_step = None
  _buffered_metrics = None

def save_checkpoint(
    checkpoint_manager,
    step,
    state,
    dataset_type="c4",
    data_iterator=None,
    config: Optional[pyconfig.config] = None,
) -> bool:
  if config and config.enable_checkpointing:
    if (step % config.checkpoint_period == 0) or (
        config.enable_emergency_checkpoint and step % config.local_checkpoint_period == 0
    ):
      blocking_until_ready_start = time.time()
      max_logging.log(f"Waiting for step {step} to finish before checkpoint...")
      jax.block_until_ready(state)
      max_logging.log(
          f"Waited {time.time() - blocking_until_ready_start} seconds for step "
          f"{step} to finish before starting checkpointing."
      )
  chunk_byte_size = _DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE
  if config:
    chunk_byte_size = config.checkpoint_storage_target_data_file_size_bytes
  save_args = jax.tree.map(lambda _: orbax.checkpoint.SaveArgs(chunk_byte_size=chunk_byte_size), state)

  if isinstance(checkpoint_manager, emergency_checkpoint_manager.CheckpointManager):
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.PyTreeSave(item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size),
    )

  if dataset_type == "grain":
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            ),
            iter=grain.PyGrainCheckpointSave(data_iterator.local_iterator),
        ),
    )
  else:
    return checkpoint_manager.save(
        step,
        args=orbax.checkpoint.args.Composite(
            items=orbax.checkpoint.args.PyTreeSave(
                item=state, save_args=save_args, ocdbt_target_data_file_size=chunk_byte_size
            )
        ),
    )


def _split_dpo_state(state):
  reference_params = state.params["reference_params"]
  new_state = state.replace(params={k: v for k, v in state.params.items() if k != "reference_params"})
  return new_state, reference_params

def _merge_dpo_state(state, reference_params):
  return state.replace(params=dict(state.params, reference_params=reference_params))

def dpo_loss_fn(*args, **kwargs):
  # We won't actually run steps now, just returning a dummy value because we exit early.
  # This function won't be called before exit in our debugging run.
  return 0.0, {"total_loss":0.0, "total_weights":0.0, "moe_lb_loss":0.0, "reward_accuracy":0.0, "intermediate_outputs":{}}

def loss_fn(*args, **kwargs):
  # Similarly, not running steps long enough to reach here before exit.
  return 0.0, {"total_loss":0.0, "total_weights":0.0, "moe_lb_loss":0.0, "intermediate_outputs":{}}

def train_step(*args, **kwargs):
  # Not used here since we exit before training.
  return args[3], {"scalar":{"learning/loss":0.0,"learning/total_weights":0.0}}

def eval_step(*args, **kwargs):
  return {"scalar":{"evaluation/loss":0.0,"evaluation/total_loss":0.0,"evaluation/total_weights":0.0,"evaluation/moe_lb_loss":0.0}}

def create_goodput_recorder(config):
  return None

def record_goodput(*args, **kwargs):
  pass

def check_example_batch(config, example_batch):
  pass

def setup_mesh_and_model(config):
  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
        config.local_checkpoint_directory,
        config.checkpoint_dir,
        mesh,
        abstract_state,
        config.local_checkpoint_period,
        config.checkpoint_period,
        logger,
    )
  else:
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )
  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx

def setup_train_loop(config):
  recorder = create_goodput_recorder(config)
  record_goodput(recorder, config, None)
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  record_goodput(recorder, config, None)

  max_logging.log(f"[DEBUG] setup_train_loop: use_dpo={config.use_dpo}, per_device_batch_size={config.per_device_batch_size}, dataset_type={config.dataset_type}")

  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
  max_logging.log("[DEBUG] Data iterator created.")

  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  if config.use_dpo:
    max_logging.log("[DEBUG] DPO mode enabled. Attempting to restore reference parameters...")
    abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    try:
      step0_restored, _ = checkpointing.load_state_if_possible(
          checkpoint_manager,
          data_iterator,
          load_parameters_from_path="",
          load_full_state_from_path="",
          abstract_unboxed_pre_state=abstract_state,
          enable_single_replica_ckpt_restoring=False,
          dataset_type=config.dataset_type,
      )
    except FileNotFoundError:
      step0_restored = None
    if step0_restored is not None:
      reference_params = step0_restored["items"].params["params"]
      state = _merge_dpo_state(state, reference_params)
    else:
      max_logging.log("[DEBUG] Could not restore reference parameters for DPO.")
  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )

def train_loop(config, state=None):
  (init_rng, writer, checkpoint_manager, state_mesh_shardings,
   model, mesh, learning_rate_schedule, data_iterator, eval_data_iterator, state) = setup_train_loop(config)

  start_step = get_first_step(state)
  example_batch = None

  # Load the first batch and print debug info, then exit
  example_batch = load_next_batch(data_iterator, example_batch, config)
  max_logging.log("[DEBUG] First batch loaded.")
  max_logging.log(f"[DEBUG] Keys in batch: {list(example_batch.keys())}")
  for k, v in example_batch.items():
    max_logging.log(f"[DEBUG] {k}: shape={v.shape}, dtype={v.dtype}")

  # Check if chosen_segmentation present
  if "chosen_segmentation" in example_batch:
    max_logging.log("[DEBUG] 'chosen_segmentation' found. Exiting now.")
  else:
    max_logging.log("[DEBUG] 'chosen_segmentation' NOT found. Exiting now.")

  # Exit immediately after checking the first batch
  sys.exit(1)

def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  max_utils.print_system_information()
  config = pyconfig.config
  validate_train_config(config)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path
  vertex_tensorboard_manager = VertexTensorboardManager()
  if config.use_vertex_tensorboard or os.environ.get("UPLOAD_DATA_TO_TENSORBOARD"):
    vertex_tensorboard_manager.configure_vertex_tensorboard(config)

  if config.monitor_goodput and jax.process_index() == 0:
    logger_name = f"goodput_{config.run_name}"
    goodput_monitor = monitoring.GoodputMonitor(
        job_name=config.run_name,
        logger_name=logger_name,
        tensorboard_dir=config.tensorboard_dir,
        upload_interval=config.goodput_upload_interval_seconds,
        monitoring_enabled=True,
        pathway_enabled=config.enable_pathways_goodput,
        include_badput_breakdown=True,
    )
    goodput_monitor.start_goodput_uploader()
    max_logging.log("Started Goodput upload to Tensorboard in the background!")

  debug_config = debug_configuration.DebugConfig(
      stack_trace_config=stack_trace_configuration.StackTraceConfig(
          collect_stack_trace=config.collect_stack_trace,
          stack_trace_to_cloud=config.stack_trace_to_cloud,
          stack_trace_interval_seconds=config.stack_trace_interval_seconds,
      )
  )
  diagnostic_config = diagnostic_configuration.DiagnosticConfig(debug_config)
  with diagnostic.diagnose(diagnostic_config):
    train_loop(config)

if __name__ == "__main__":
  app.run(main)
