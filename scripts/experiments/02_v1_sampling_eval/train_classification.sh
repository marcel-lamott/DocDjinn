#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "layoutlmv3|--model-name microsoft/layoutlmv3-base --tokenizer-name microsoft/layoutlmv3-base --use-segment-level-bboxes --train-batch-size 16 --eval-batch-size 16"
)

declare -a dataset_names=(
    # alpha 1.0
    "rvlcdip"
    # alpha 0.75
    "rvlcdip"
    # alpha 0.5
    "rvlcdip"
)

declare -a synthetic_dataset_names=(
    # alpha 1
    "rvlcdip_alpha=1.0_v1"
    # alpha 0.75
    "rvlcdip_alpha=0.75_v1"
    # alpha 0.5
    "rvlcdip_alpha=0.5_v1"
)

declare -a monitored_metrics=(
    "validation/accuracy"
    "validation/accuracy"
    "validation/accuracy"
)

TASK_ARGUMENTS="--optimizer adam --use-preprocessed-dataset"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
