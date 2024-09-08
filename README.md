# maxtext-no-tools

```bash
# Clone the repo at the root. Make sure maxtext is installed.
# pip install torch
# pip install pyconfig
cd
cp maxtext-no-tools/convert_pipeline.sh maxtext/
cp maxtext-no-tools/llama_or_mistral_orbax_to_huggingface.py maxtext/MaxText/
cd maxtext/

# Script defaults to converting llama3
./convert_pipeline.sh --checkpoints 80000 --checkpoint_names north_llama3_edu_above_1_lr1e5_8192

# Convert llama3.1
bash convert_pipeline.sh --checkpoints 0 --checkpoint_names north_llama31instruct_test_edu_above_1_add_instructions_lr1e5_8192 --model_name llama3.1-8b --tokenizer_dir north/llama3.1-8b-reference


```
