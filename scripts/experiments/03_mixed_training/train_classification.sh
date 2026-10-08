#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "bert-base-uncased|--model-name bert-base-uncased --tokenizer-name bert-base-uncased --train-batch-size 32 --eval-batch-size 32" # text only
    "lilt|--model-name SCUT-DLVCLab/lilt-roberta-en-base --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base --train-batch-size 32 --eval-batch-size 32" # text + layout
    "layoutlmv3|--model-name microsoft/layoutlmv3-base --tokenizer-name microsoft/layoutlmv3-base --use-segment-level-bboxes --train-batch-size 32 --eval-batch-size 32"
)

declare -a dataset_names=(
    "rvlcdip"
    "tobacco3482"
    "doclaynet_4k_cls"
)

declare -a synthetic_dataset_names=(
    "rvlcdip_alpha=1.0"
    "tobacco3482_alpha=1.0"
    "doclaynet4k_alpha=1.0_CLS"
)

declare -a monitored_metrics=(
    "validation/accuracy"
    "validation/accuracy"
    "validation/accuracy"
)

TASK_ARGUMENTS="--optimizer adam" # --use-preprocessed-dataset"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
