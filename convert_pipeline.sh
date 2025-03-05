#!/bin/bash
export GOOGLE_APPLICATION_CREDENTIALS="/home/perk/.config/gcloud/application_default_credentials.json"
export CLOUDSDK_CONFIG="$HOME/.config/gcloud"

# Function to display usage
usage() {
    echo "Usage: $0 --checkpoints CHECKPOINTS --checkpoint_names CHECKPOINT_NAMES [--organization ORGANIZATION] [--tokenizer_dir TOKENIZER_DIR] [--model_name MODEL_NAME]"
    echo "Example: $0 --checkpoints 10000,20000 --checkpoint_names north_above05,north_above1 --organization north --tokenizer_dir north/llama3-8b-reference --model_name llama3-8b"
    exit 1
}

# Set default values
ORGANIZATION="north"
TOKENIZER_DIR="north/llama3-8b-reference"
MODEL_NAME="llama3-8b"
HOME="/mnt/lv_ai_2_dante/perk"
BUCKET_NAME="maxlog-eu"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --checkpoints) CHECKPOINTS="$2"; shift ;;
        --checkpoint_names) CHECKPOINT_NAMES="$2"; shift ;;
        --organization) ORGANIZATION="$2"; shift ;;
        --tokenizer_dir) TOKENIZER_DIR="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --bucket_name) BUCKET_NAME="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Check if required parameters are provided
if [ -z "$CHECKPOINTS" ] || [ -z "$CHECKPOINT_NAMES" ]; then
    echo "Error: Missing required parameters."
	usage
    exit 1
fi

# Convert comma-separated strings to arrays
IFS=',' read -r -a CHECKPOINT_ARRAY <<< "$CHECKPOINTS"
IFS=',' read -r -a CHECKPOINT_NAME_ARRAY <<< "$CHECKPOINT_NAMES"

# Iterate through each combination of checkpoints and checkpoint names
for CHECKPOINT in "${CHECKPOINT_ARRAY[@]}"; do
    for CHECKPOINT_NAME in "${CHECKPOINT_NAME_ARRAY[@]}"; do
        LOAD_PARAMETERS_PATH="gs://${BUCKET_NAME}/${CHECKPOINT_NAME}/checkpoints/${CHECKPOINT}/items"
        #LOAD_PARAMETERS_PATH="gs://${BUCKET_NAME}/${CHECKPOINT_NAME}/scanned_chkpt/${CHECKPOINT}/items"
        #LOAD_PARAMETERS_PATH="gs://maxlog-eu/${CHECKPOINT_NAME}/checkpoints/${CHECKPOINT}/items"
        TARGET_REPO="${CHECKPOINT_NAME}_${CHECKPOINT}"

        # Check if the checkpoint exists
        if ! gsutil -q stat "${LOAD_PARAMETERS_PATH}/*"; then
             echo "Error: Checkpoint ${CHECKPOINT} for ${CHECKPOINT_NAME} does not exist at ${LOAD_PARAMETERS_PATH}."
             exit 1
        fi

        # Attempt to create the repository on Hugging Face
        # if ! yes | huggingface-cli repo create $TARGET_REPO --organization $ORGANIZATION; then
        #    echo "Error: Repository ${TARGET_REPO} already exists or could not be created."
        #     exit 1
        # fi

        # Clean up on the disk
        rm -rf "${HOME}/.cache/huggingface/hub/*"
        find "${HOME}/hfrepo" -mindepth 1 -delete

        BASE_OUTPUT_DIRECTORY="${HOME}/modeltemp"
        RUN_NAME="CheckpointExportHF"
        HF_MODEL_PATH="${HOME}/hfrepo"

        # Change directory and run the Python script with the constants/variables
        cd "${HOME}/maxtext"
        #python ./MaxText/llama_or_mistral_orbax_to_huggingface.py  MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIRECTORY load_parameters_path=$LOAD_PARAMETERS_PATH run_name=$RUN_NAME model_name=$MODEL_NAME hf_model_path=$HF_MODEL_PATH
        python ./MaxText/llama_mistral_mixtral_orbax_to_hf.py  MaxText/configs/base.yml base_output_directory=$BASE_OUTPUT_DIRECTORY load_parameters_path=$LOAD_PARAMETERS_PATH run_name=$RUN_NAME model_name=$MODEL_NAME hf_model_path=$HF_MODEL_PATH
        
        ## PUSH TO HUGGINGFACE

        # Change directory to the target repo
        cd "${HOME}/hfrepo"

        # Get extra files from the specified tokenizer directory
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/tokenizer.json
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/tokenizer_config.json
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/LICENSE
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/special_tokens_map.json

        # Upload files to Hugging Face repository
        # for file in *; do huggingface-cli upload "$ORGANIZATION/$TARGET_REPO" "$file"; done

    done
done
