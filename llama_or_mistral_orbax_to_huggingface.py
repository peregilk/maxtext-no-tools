from typing import Sequence
import torch
from tqdm import tqdm
from absl import app
import numpy as np
import pyconfig
import max_utils
import jax
from jax.sharding import Mesh
import max_logging
import checkpointing
from generate_param_only_checkpoint import _read_train_checkpoint
import llama_or_mistral_ckpt
from transformers import LlamaForCausalLM, MistralForCausalLM, AutoModelForCausalLM

def unpermute_from_match_maxtext_rope(arr):
    """
    Function to get the RoPE values in correct ordering
    """
    split_size = arr.shape[-1] // 2  # Assuming half for evens, half for odds
    evens = arr[..., :split_size]
    odds = arr[..., split_size:]
    return jax.numpy.stack([evens, odds], axis=len(arr.shape)).reshape(arr.shape)

def reverse_scale(arr, scale):
    """
    MaxText has the scaling factor included into the weights, 
    we reverse it when writing out the HuggingFace checkpoint
    """
    return arr * np.sqrt(scale)

def load_hf_model(model_size):
    """
    Load the model that we are interested in from HuggingFace
    """
    if model_size == "llama2-7b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    elif model_size == "llama3-8b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    elif model_size == "llama3.1-8b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    elif model_size == "llama3.1-70b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-70B")
    elif model_size == "llama3.2-1b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    elif model_size == "llama3.2-3b":
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    elif model_size == "mistral-7b":
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    else:
        raise NotImplementedError

    return model

def load_model_state(config):
    """
    Loads the MaxText model's TrainState from the Orbax checkpoint
    """
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

    # Create a checkpoint manager to load decode checkpoint at config.checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
    )

    # Read training state from config.load_paramaters_path
    max_logging.log(f"Read training checkpoint from: {config.load_full_state_path}")
    training_state, _ = _read_train_checkpoint(config, checkpoint_manager, mesh)
    return training_state

def convert_state_to_hf(training_state, model_size):
    """
    Port the parameters from the Orbax training_state into the hf_model
    """

    if model_size not in llama_or_mistral_ckpt.MODEL_PARAMS_DICT:
        print(f"Model size: {model_size}")
        raise NotImplementedError
    # Load the model specific parameters
    model_params = llama_or_mistral_ckpt.MODEL_PARAMS_DICT[model_size]
    base_num_decoder_layers = model_params['num_layers']
    base_num_query_heads = model_params['num_heads']
    head_dim = model_params['dims_per_head']
    base_num_kv_heads = model_params['num_kv_heads']

    hf_model_params = {}

    # Port the embedding weights
    hf_model_params["model.embed_tokens.weight"] = torch.tensor(
        np.asarray(training_state.params['params']['token_embedder']['embedding']),
        dtype=torch.bfloat16
    )

    for layer_int in tqdm(range(base_num_decoder_layers), desc="Porting parameters layerwise"):

        # Attention layers
        hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"] = torch.tensor(np.asarray(
            unpermute_from_match_maxtext_rope(
                reverse_scale(
                    training_state.params['params']["decoder"]["layers"]["self_attention"]["query"]["kernel"][:, layer_int, :, :],
                    head_dim
                )
            )),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"] = hf_model_params[f"model.layers.{layer_int}.self_attn.q_proj.weight"].view(base_num_query_heads * head_dim, base_num_query_heads * head_dim).T.view(base_num_query_heads, head_dim // 2, 2, base_num_query_heads * head_dim).transpose(1, 2).reshape(-1, base_num_query_heads * head_dim)

        hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"] = torch.tensor(np.asarray(
            unpermute_from_match_maxtext_rope(
                training_state.params['params']["decoder"]["layers"]["self_attention"]["key"]["kernel"][:, layer_int, :, :]
            )),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"] = hf_model_params[f"model.layers.{layer_int}.self_attn.k_proj.weight"].view(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T.reshape(base_num_kv_heads, head_dim // 2, 2, base_num_query_heads * head_dim).transpose(1, 2).reshape(-1, base_num_query_heads * head_dim)

        hf_model_params[f"model.layers.{layer_int}.self_attn.v_proj.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["self_attention"]["value"]["kernel"][:, layer_int, :, :]
            .reshape(base_num_query_heads * head_dim, base_num_kv_heads * head_dim).T),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.self_attn.o_proj.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["self_attention"]["out"]["kernel"][:, layer_int, :, :]
            .reshape(base_num_query_heads * head_dim, base_num_query_heads * head_dim).T),
            dtype=torch.bfloat16
        )

        # MLP Layers
        hf_model_params[f"model.layers.{layer_int}.mlp.gate_proj.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["mlp"]["wi_0"]["kernel"][:, layer_int, :].T),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.mlp.up_proj.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["mlp"]["wi_1"]["kernel"][:, layer_int, :].T),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.mlp.down_proj.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["mlp"]["wo"]["kernel"][:, layer_int, :].T),
            dtype=torch.bfloat16
        )

        # Pre/post attention layer norm
        hf_model_params[f"model.layers.{layer_int}.input_layernorm.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["pre_self_attention_layer_norm"]["scale"][:, layer_int]
            .reshape(base_num_query_heads * head_dim)),
            dtype=torch.bfloat16
        )
        hf_model_params[f"model.layers.{layer_int}.post_attention_layernorm.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["layers"]["post_self_attention_layer_norm"]["scale"][:, layer_int]
            .reshape(base_num_query_heads * head_dim)),
            dtype=torch.bfloat16
        )

    # LM head and layernorm
    if "logits_dense" in training_state.params["params"]["decoder"]:
        hf_model_params["lm_head.weight"] = torch.tensor(np.asarray(
            training_state.params['params']["decoder"]["logits_dense"]["kernel"].T),
            dtype=torch.bfloat16
        )

    hf_model_params["model.norm.weight"] = torch.tensor(np.asarray(
        training_state.params['params']["decoder"]["decoder_norm"]["scale"].reshape(base_num_query_heads * head_dim)),
        dtype=torch.bfloat16
    )

    return hf_model_params

def convert_orbax_hf(hf_model_path, config):
    """
    Landing function to convert MaxText model's checkpoint to HuggingFace format
    """
    hf_model = load_hf_model(config.model_name)
    training_state = load_model_state(config)
    new_hf_model_params = convert_state_to_hf(training_state, config.model_name)
    print(f"Saving HuggingFace model to path = {hf_model_path}")
    hf_model.save_pretrained(hf_model_path, state_dict=new_hf_model_params)

def main(argv: Sequence[str]):
    pyconfig.initialize(argv[:-1])
    # Assuming the last argument is the path to save the converted checkpoint in HuggingFace format
    hf_model_path = argv[-1].split("=")[1]
    print(f"Will save converted HuggingFace checkpoint to path = {hf_model_path}")
    convert_orbax_hf(hf_model_path, pyconfig.config)

if __name__ == "__main__":
    app.run(main)
