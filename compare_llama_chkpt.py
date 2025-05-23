import torch
import numpy as np
from transformers import AutoModelForCausalLM

def load_state_dict(model_path):
    # This works for both directory (from_pretrained) and .bin file
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
        return model.state_dict()
    except Exception as e:
        print(f"Error loading as HuggingFace model: {e}, will try torch.load...")
        return torch.load(model_path, map_location="cpu")

def compare_state_dicts(sd1, sd2, atol=1e-5, rtol=1e-3):
    all_keys = set(sd1.keys()).union(sd2.keys())
    only_in_1 = all_keys - set(sd2.keys())
    only_in_2 = all_keys - set(sd1.keys())
    shared = set(sd1.keys()) & set(sd2.keys())

    if only_in_1:
        print(f"Keys only in first model: {sorted(list(only_in_1))}")
    if only_in_2:
        print(f"Keys only in second model: {sorted(list(only_in_2))}")

    diffs = []
    for key in sorted(shared):
        t1 = sd1[key].cpu().float()
        t2 = sd2[key].cpu().float()
        if t1.shape != t2.shape:
            print(f"{key}: shape mismatch {t1.shape} vs {t2.shape}")
            continue
        absdiff = (t1 - t2).abs()
        maxdiff = absdiff.max().item()
        meandiff = absdiff.mean().item()
        rmsdiff = (absdiff.pow(2).mean().sqrt()).item()
        if maxdiff > atol + rtol * abs(t1).max().item():
            print(f"{key}: maxdiff={maxdiff:.6e}, mean={meandiff:.6e}, rms={rmsdiff:.6e}, shape={t1.shape}")
        diffs.append((key, maxdiff, meandiff, rmsdiff))
    print("Comparison done. Summary:")
    print(f"Total compared: {len(shared)} tensors.")
    print(f"Max abs diff across all: {max(x[1] for x in diffs):.6e}")
    print(f"Mean of means: {np.mean([x[2] for x in diffs]):.6e}")
    print(f"Mean of RMS: {np.mean([x[3] for x in diffs]):.6e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python compare_llama_ckpts.py <orig_model_path> <converted_model_path>")
        sys.exit(1)
    orig, new = sys.argv[1:3]
    print("Loading original...")
    sd1 = load_state_dict(orig)
    print("Loading converted...")
    sd2 = load_state_dict(new)
    compare_state_dicts(sd1, sd2)
