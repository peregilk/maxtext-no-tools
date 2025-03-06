#!/bin/bash
set -euo pipefail

# Set required environment variables for gcloud
export GOOGLE_APPLICATION_CREDENTIALS="/home/perk/.config/gcloud/application_default_credentials.json"
export CLOUDSDK_CONFIG="$HOME/.config/gcloud"

# Disable TPU detection and TPU metadata fetching
export TPU_NAME=""
export XRT_TPU_CONFIG=""
export TPU_IP_ADDRESS=""
export TPU_METADATA_URL=""

# Force JAX to use CPU only by setting both JAX_PLATFORM_NAME and JAX_PLATFORMS,
# and limit the device count via XLA_FLAGS.
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

# Hide GPUs from JAX
export CUDA_VISIBLE_DEVICES=""

# Reduce TensorFlow logging verbosity
export TF_CPP_MIN_LOG_LEVEL=2

# Enable faster transfers via hf_transfer in huggingface_hub
export HF_HUB_ENABLE_HF_TRANSFER=1

# Function to display usage information
usage() {
    echo "Usage: $0 --checkpoints CHECKPOINTS --checkpoint_names CHECKPOINT_NAMES [--organization ORGANIZATION] [--tokenizer_dir TOKENIZER_DIR] [--model_name MODEL_NAME] [--bucket_name BUCKET_NAME] [--overwrite_repo]"
    echo "Example: $0 --checkpoints 1200 --checkpoint_names nb-llama-3.2-3B --bucket_name nb_llama3_x --model_name llama3.2-3b --tokenizer_dir north/llama3.1-8b-reference --organization pere"
    exit 1
}

# Set default values
ORGANIZATION="north"
TOKENIZER_DIR="north/llama3-8b-reference"
MODEL_NAME="llama3-8b"
BUCKET_NAME="maxlog-eu"
OVERWRITE_REPO="false"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --checkpoints)
            CHECKPOINTS="$2"
            shift ;;
        --checkpoint_names)
            CHECKPOINT_NAMES="$2"
            shift ;;
        --organization)
            ORGANIZATION="$2"
            shift ;;
        --tokenizer_dir)
            TOKENIZER_DIR="$2"
            shift ;;
        --model_name)
            MODEL_NAME="$2"
            shift ;;
        --bucket_name)
            BUCKET_NAME="$2"
            shift ;;
        --overwrite_repo)
            OVERWRITE_REPO="true" ;;
        *)
            echo "Unknown parameter passed: $1"
            usage ;;
    esac
    shift
done

# Check for required parameters
if [[ -z "${CHECKPOINTS:-}" ]] || [[ -z "${CHECKPOINT_NAMES:-}" ]]; then
    echo "Error: Missing required parameters."
    usage
fi

# Convert comma-separated strings to arrays
IFS=',' read -r -a CHECKPOINT_ARRAY <<< "$CHECKPOINTS"
IFS=',' read -r -a CHECKPOINT_NAME_ARRAY <<< "$CHECKPOINT_NAMES"

# Determine the script's directory and set WORK_DIR relative to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="${SCRIPT_DIR}/../"

# Pre-conversion: for each target repository, if it already exists and overwrite isn't requested, exit.
for CHECKPOINT in "${CHECKPOINT_ARRAY[@]}"; do
    for CHECKPOINT_NAME in "${CHECKPOINT_NAME_ARRAY[@]}"; do
        TARGET_REPO="${CHECKPOINT_NAME}_${CHECKPOINT}"
        if huggingface-cli repo view "$TARGET_REPO" --organization "$ORGANIZATION" > /dev/null 2>&1; then
            if [[ "$OVERWRITE_REPO" != "true" ]]; then
                echo "Repository $TARGET_REPO already exists under organization $ORGANIZATION. Use --overwrite_repo to overwrite."
                exit 1
            else
                echo "Repository $TARGET_REPO already exists; overwriting as requested."
            fi
        fi
    done
done

# Function to upload a file with retries
upload_file_with_retry() {
    local file="$1"
    local max_retries=3
    local attempt=0
    local success=0
    while [ $attempt -lt $max_retries ]; do
        if huggingface-cli upload "$ORGANIZATION/$TARGET_REPO" "$file"; then
            success=1
            break
        else
            attempt=$((attempt + 1))
            echo "Error uploading $file. Attempt $attempt of $max_retries. Retrying in 10 seconds..."
            sleep 10
        fi
    done
    if [ $success -ne 1 ]; then
        echo "Failed to upload $file after $max_retries attempts. Exiting."
        exit 1
    fi
}

# Iterate through each combination of checkpoints and checkpoint names
for CHECKPOINT in "${CHECKPOINT_ARRAY[@]}"; do
    for CHECKPOINT_NAME in "${CHECKPOINT_NAME_ARRAY[@]}"; do

        LOAD_PARAMETERS_PATH="gs://${BUCKET_NAME}/${CHECKPOINT_NAME}/checkpoints/${CHECKPOINT}/items"
        TARGET_REPO="${CHECKPOINT_NAME}_${CHECKPOINT}"

        # Check if the checkpoint exists on GCS
        if ! gsutil -q stat "${LOAD_PARAMETERS_PATH}/*"; then
            echo "Error: Checkpoint ${CHECKPOINT} for ${CHECKPOINT_NAME} does not exist at ${LOAD_PARAMETERS_PATH}."
            exit 1
        fi

        # Clean up local cache and repository directories (using WORK_DIR)
        rm -rf "${WORK_DIR}/.cache/huggingface/hub/"*
        find "${WORK_DIR}/hfrepo" -mindepth 1 -delete

        BASE_OUTPUT_DIRECTORY="${WORK_DIR}/modeltemp"
        RUN_NAME="CheckpointExportHF"
        HF_MODEL_PATH="${WORK_DIR}/hfrepo"

        # Run the Python conversion script (in CPU mode)
        cd "${WORK_DIR}/maxtext"
        python ./MaxText/llama_mistral_mixtral_orbax_to_hf.py \
            MaxText/configs/base.yml \
            base_output_directory="$BASE_OUTPUT_DIRECTORY" \
            load_parameters_path="$LOAD_PARAMETERS_PATH" \
            run_name="$RUN_NAME" \
            model_name="$MODEL_NAME" \
            hf_model_path="$HF_MODEL_PATH"

        ## PUSH TO HUGGINGFACE
        cd "${WORK_DIR}/hfrepo"

        # Check if logged in to Hugging Face CLI (token should be in $HOME/.cache/huggingface/token)
        if ! huggingface-cli whoami > /dev/null 2>&1; then
            echo "Not logged in to Hugging Face CLI. Please run 'huggingface-cli login' and try again."
            exit 1
        fi

        # At push time, check if the repository exists. If not, attempt to create it.
        if ! huggingface-cli repo view "$TARGET_REPO" --organization "$ORGANIZATION" > /dev/null 2>&1; then
            set +e
            yes | huggingface-cli repo create "$TARGET_REPO" --organization "$ORGANIZATION"
            RC=$?
            set -e
            if [ $RC -ne 0 ]; then
                echo "Repository creation command returned a nonzero exit code (likely a conflict)."
                echo "Assuming repository $TARGET_REPO already exists; proceeding with overwrite."
            fi
        else
            echo "Repository $TARGET_REPO already exists; proceeding with overwrite."
        fi

        # Get extra files from the specified tokenizer directory
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/tokenizer.json
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/tokenizer_config.json
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/LICENSE
        wget https://huggingface.co/${TOKENIZER_DIR}/raw/main/special_tokens_map.json

        # Upload each file in the current directory using huggingface-cli with retry.
        for file in *; do
            if [ -f "$file" ]; then
                upload_file_with_retry "$file"
            fi
        done

    done
done
