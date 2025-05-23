"""
 Copyright 2023 Google LLC
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      https://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

r"""Convert weights from a MaxText model to a HuggingFace model.

Usage:
python -m MaxText.llama_mistral_mixtral_orbax_to_hf \
    MaxText/configs/base.yml \
    base_output_directory=path/to/saving/intermediate_MaxText_files \
    load_parameters_path=/path/to/MaxText/checkpoint \
    run_name=<your run name> \
    model_name=<llama2|mistral|mixtral|llama3.1-8b> \
    hf_model_path=/local/path/to/save/HF/model/to

Note: `hf_model_path` must be the LAST argument.
"""

from typing import Sequence

import torch

from tqdm import tqdm

from absl import app # For app.run

import numpy as np

from jax.sharding import Mesh # JAX is still needed for Mesh

from transformers import LlamaForCausalLM, MistralForCausalLM, AutoModelForCausalLM, AutoConfig

from MaxText import checkpointing
from MaxText import llama_or_mistral_ckpt
from MaxText import max_logging # Changed from print to max_logging
from MaxText import maxtext_utils
from MaxText import pyconfig
from MaxText.generate_param_only_checkpoint import _read_train_checkpoint
# This is the original import from Script 1. We will use this for non-Llama3 models.
from MaxText.max_utils import unpermute_from_match_maxtext_rope as original_maxtext_rope_permuter


# --- START: Modified RoPE Permutation Handling ---
def apply_rope_permutation_conditionally(arr, model_size):
  """
  Applies RoPE-related permutation based on the model type.
  - Llama3 family: Returns the array as-is (no permutation).
  - Other models (Llama2, Mistral, Mixtral): Applies the original MaxText permutation.
  """
  if model_size.startswith("llama3"):
    max_logging.log(f"RoPE: Skipping permutation for Llama3 family model: {model_size}")
    return arr
  else:
    max_logging.log(f"RoPE: Applying original MaxText permutation for model: {model_size}")
    return original_maxtext_rope_permuter(arr, model_size) # original_maxtext_rope_permuter might need model_size
# --- END: Modified RoPE Permutation Handling ---


def reverse_scale(arr, scale):
  """
  MaxText has the scaling factor included into the weights,
  we reverse it when writing out the HuggingFace checkpoint
  """
  return arr * np.sqrt(scale)

def load_hf_model(model_size, model_dtype=torch.bfloat16):
  """
  Load the model that we are interested in from HuggingFace
  """
  max_logging.log(f"Loading empty HF model for {model_size} with dtype {model_dtype}")
  if model_size == "llama2-7b":
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=model_dtype, token="")
  elif model_size == "mistral-7b":
    model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=model_dtype)
  elif model_size == "mixtral-8x7b":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", device_map="auto", torch_dtype=model_dtype)
  elif model_size.startswith("llama3"):
    hf_model_identifier = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "llama3.1-70b": "meta-llama/Llama-3.1-70B",
        "llama3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama3.2-3b": "meta-llama/Llama-3.2-3B"
    }.get(model_size)

    if not hf_model_identifier: # Fallback naming attempt or for unlisted Llama3 models
        if "llama3." in model_size: # e.g. llama3.1-XYZ
            parts = model_size.split('-')
            version_suffix = parts[0].replace("llama3.", "Llama-3.") # Llama-3.1
            size_suffix = parts[1] # 8B, 70B
            hf_model_identifier = f"meta-llama/{version_suffix}-{size_suffix}"
        elif "llama3-" in model_size: # e.g. llama3-8B
             hf_model_identifier = f"meta-llama/Meta-{model_size.replace('llama3', 'Llama-3')}" # Meta-Llama-3-8B
        else:
            raise NotImplementedError(f"Cannot determine HuggingFace identifier for Llama3 model: {model_size}")
    max_logging.log(f"Attempting to load config for HF model: {hf_model_identifier}")
    try:
        config = AutoConfig.from_pretrained(hf_model_identifier, token="")
    except Exception as e:
        max_logging.warning(f"Could not load config for {hf_model_identifier} directly: {e}. "
                            f"Falling back to Llama-3.1-8B config structure for {model_size} and hoping for compatibility.")
        config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B", token="")
    config.torch_dtype = model_dtype
    model = AutoModelForCausalLM.from_config(config)
  else:
    raise NotImplementedError(f"Model size {model_size} not implemented in load_hf_model.")
  return model


def load_model_state(config):
  """
  Loads the MaxText model's TrainState from the Orbax checkpoint
  """
  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
      config.checkpoint_dir,
      config.enable_checkpointing,
      config.async_checkpointing,
      config.checkpoint_period,
  )
  load_path = config.load_parameters_path or config.load_full_state_path
  if not load_path:
      raise ValueError("Neither load_parameters_path nor load_full_state_path is set in config.")
  max_logging.log(f"Read training checkpoint from effective path: {load_path}")
  training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
  return training_state


def convert_state_to_hf(training_state, model_size, hf_dtype=torch.bfloat16):
  """
  Port the parameters from the Orbax training_state into the hf_model
  """
  if model_size not in llama_or_mistral_ckpt.MODEL_PARAMS_DICT:
    raise NotImplementedError(f"Model parameters for {model_size} not found in MODEL_PARAMS_DICT.")

  model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
  base_num_decoder_layers = model_params["num_layers"]
  base_num_query_heads = model_params["num_heads"]
  head_dim = model_params["dims_per_head"]
  base_num_kv_heads = model_params["num_kv_heads"]
  num_experts = model_params.get("num_experts")

  hf_model_params = {}

  hf_model_params["model.embed_tokens.weight"] = torch.tensor(
      np.asarray(training_state.params["params"]["token_embedder"]["embedding"]), dtype=hf_dtype
  )

  for layer_int in tqdm(range(base_num_decoder_layers), desc="Porting parameters layerwise"):
    # max_logging.log(f"Converting weights for layer {layer_int}") # tqdm provides progress

    q_kernel_orig = training_state.params["params"]["decoder"]["layers"]["self_attention"]["query"]["kernel"][:, layer_int, :, :]
    q_kernel_scaled = reverse_scale(q_kernel_orig, head_dim)
    q_kernel_permuted = apply_rope_permutation_conditionally(q_kernel_scaled, model_size)
    hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"] = torch.tensor(
        np.asarray(q_kernel_permuted.reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim).T),
        dtype=hf_dtype,
    )

    k_kernel_orig = training_state.params["params"]["decoder"]["layers"]["self_attention"]["key"]["kernel"][:, layer_int, :, :]
    k_kernel_permuted = apply_rope_permutation_conditionally(k_kernel_orig, model_size)
    hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"] = torch.tensor(
        np.asarray(k_kernel_permuted.reshape(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T),
        dtype=hf_dtype,
    )

    v_kernel = training_state.params["params"]["decoder"]["layers"]["self_attention"]["value"]["kernel"][:, layer_int, :, :]
    hf_model_params[f"model.layers.{layer_int}.self_attn.v_proj.weight"] = torch.tensor(
        np.asarray(v_kernel.reshape(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T),
        dtype=hf_dtype,
    )

    o_kernel = training_state.params["params"]["decoder"]["layers"]["self_attention"]["out"]["kernel"][:, layer_int, :, :]
    hf_model_params[f"model.layers.{layer_int}.self_attn.o_proj.weight"] = torch.tensor(
        np.asarray(o_kernel.reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim).T),
        dtype=hf_dtype,
    )

    if num_experts is None:
      hf_model_params[f"model.layers.{layer_int}.mlp.gate_proj.weight"] = torch.tensor(
          np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp"]["wi_0"]["kernel"][:, layer_int, :].T),
          dtype=hf_dtype,
      )
      hf_model_params[f"model.layers.{layer_int}.mlp.up_proj.weight"] = torch.tensor(
          np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp"]["wi_1"]["kernel"][:, layer_int, :].T),
          dtype=hf_dtype,
      )
      hf_model_params[f"model.layers.{layer_int}.mlp.down_proj.weight"] = torch.tensor(
          np.asarray(training_state.params["params"]["decoder"]["layers"]["mlp"]["wo"]["kernel"][:, layer_int, :].T),
          dtype=hf_dtype,
      )
    else: # MoE
      hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.gate.weight"] = torch.tensor(
          np.asarray(
              training_state.params["params"]["decoder"]["layers"]["MoeBlock_0"]["gate"]["kernel"][:, layer_int, :].T
          ),
          dtype=hf_dtype,
      )
      for k in range(num_experts):
        hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w1.weight"] = torch.tensor(
            np.asarray(training_state.params["params"]["decoder"]["layers"]["MoeBlock_0"]["wi_0"][k, layer_int, :, :].T),
            dtype=hf_dtype,
        )
        hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w3.weight"] = torch.tensor( # Llama-like SwiGLU has w1 and w3
            np.asarray(training_state.params["params"]["decoder"]["layers"]["MoeBlock_0"]["wi_1"][k, layer_int, :, :].T),
            dtype=hf_dtype,
        )
        hf_model_params[f"model.layers.{layer_int}.block_sparse_moe.experts.{k}.w2.weight"] = torch.tensor( # and w2 is down_proj
            np.asarray(training_state.params["params"]["decoder"]["layers"]["MoeBlock_0"]["wo"][k, layer_int, :, :].T),
            dtype=hf_dtype,
        )

    effective_embed_dim = base_num_query_heads * head_dim
    hf_model_params[f"model.layers.{layer_int}.input_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["pre_self_attention_layer_norm"]["scale"][:, layer_int]
            .reshape(effective_embed_dim)
        ),
        dtype=hf_dtype,
    )
    hf_model_params[f"model.layers.{layer_int}.post_attention_layernorm.weight"] = torch.tensor(
        np.asarray(
            training_state.params["params"]["decoder"]["layers"]["post_self_attention_layer_norm"]["scale"][:, layer_int]
            .reshape(effective_embed_dim)
        ),
        dtype=hf_dtype,
    )

  if "logits_dense" in training_state.params["params"]["decoder"]:
    hf_model_params["lm_head.weight"] = torch.tensor(
        np.asarray(training_state.params["params"]["decoder"]["logits_dense"]["kernel"].T), dtype=hf_dtype
    )
  else:
    max_logging.warning("logits_dense not found in checkpoint, lm_head.weight will be missing.")

  effective_embed_dim = base_num_query_heads * head_dim
  hf_model_params["model.norm.weight"] = torch.tensor(
      np.asarray(
          training_state.params["params"]["decoder"]["decoder_norm"]["scale"].reshape(effective_embed_dim)
      ),
      dtype=hf_dtype,
  )
  return hf_model_params


def convert_orbax_hf(hf_model_path, config):
  """
  Landing function to convert MaxText model's checkpoint to HuggingFace format
  """
  if config.model_name.startswith("llama3"):
    preferred_dtype = torch.bfloat16
  else:
    preferred_dtype = torch.bfloat16 # Default to bfloat16 for others too, with fallback

  hf_dtype = preferred_dtype
  try:
    torch.zeros(1, dtype=hf_dtype)
    max_logging.log(f"Using {hf_dtype} for HuggingFace model weights.")
  except (TypeError, RuntimeError):
    max_logging.warning(f"{hf_dtype} not supported on this system/PyTorch version, falling back to torch.float16.")
    hf_dtype = torch.float16
    try:
        torch.zeros(1, dtype=hf_dtype)
        max_logging.log(f"Now using {hf_dtype} for HuggingFace model weights.")
    except Exception as e:
        max_logging.error(f"torch.float16 also not supported? This is unexpected. Error: {e}")
        raise

  hf_model = load_hf_model(config.model_name, model_dtype=hf_dtype)
  training_state = load_model_state(config)
  new_hf_model_params = convert_state_to_hf(training_state, config.model_name, hf_dtype=hf_dtype)

  missing_keys = [k for k in hf_model.state_dict().keys() if k not in new_hf_model_params]
  if missing_keys:
      max_logging.warning(f"The following Hugging Face model keys are MISSING from the converted parameters: {missing_keys}")

  extra_keys = [k for k in new_hf_model_params if k not in hf_model.state_dict().keys()]
  if extra_keys:
      max_logging.warning(f"The following converted parameters are EXTRA and not part of the Hugging Face model: {extra_keys}")

  max_logging.log(f"Saving HuggingFace model to path = {hf_model_path} with dtype {hf_dtype}")
  hf_model.save_pretrained(hf_model_path, state_dict=new_hf_model_params)


def main(argv: Sequence[str]):
  """
  Main function for the script.
  IMPORTANT: Expects `hf_model_path` to be the *last* argument.
  Example argv from `absl.app.run(main)` when called with
  `python -m MaxText.module config.yml run_name=foo hf_model_path=save/path`:
  argv = ['config.yml', 'run_name=foo', 'hf_model_path=save/path']
  """
  if not argv: # Should not happen if absl.app calls it with sys.argv[1:]
      max_logging.error("No arguments provided to main function.")
      return

  if len(argv) < 2: # Need at least config.yml and hf_model_path=...
      max_logging.error(
          "Insufficient arguments. Usage: python -m YourModule config.yml [key=value_overrides...] hf_model_path=<path>"
      )
      max_logging.error(f"Received argv: {argv}")
      return

  # pyconfig.initialize expects a list of arguments where the first one is the config YAML file,
  # followed by overrides. It does not want the script name if called this way.
  # argv[:-1] will give [config.yml, run_name=foo]
  # It assumes the first element it receives is the YAML config.
  pyconfig_args = argv[:-1]
  if not pyconfig_args:
      max_logging.error(f"No arguments for pyconfig after removing hf_model_path. Original argv: {argv}")
      return

  try:
      config = pyconfig.initialize(pyconfig_args)
  except Exception as e:
      max_logging.error(f"Error during pyconfig.initialize with args: {pyconfig_args}")
      max_logging.error(f"Original argv to main: {argv}")
      raise e

  # The last argument is hf_model_path
  hf_model_path_arg = argv[-1]
  if not hf_model_path_arg.startswith("hf_model_path="):
      max_logging.error(
          f"The last argument is expected to be 'hf_model_path=<value>', but got: {hf_model_path_arg}"
      )
      return
  final_hf_model_path = hf_model_path_arg.split("=", 1)[1]

  max_logging.log(f"Will save converted HuggingFace checkpoint to path = {final_hf_model_path}")
  convert_orbax_hf(final_hf_model_path, config)


if __name__ == "__main__":
  app.run(main)
