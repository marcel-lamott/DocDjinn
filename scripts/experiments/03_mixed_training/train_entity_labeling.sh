#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "bert-base-uncased|--model-name bert-base-uncased --tokenizer-name bert-base-uncased" # text only
    "lilt|--model-name SCUT-DLVCLab/lilt-roberta-en-base --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base" # text + layout
    "layoutlmv3|--model-name microsoft/layoutlmv3-base --tokenizer-name microsoft/layoutlmv3-base --use-segment-level-bboxes"
)

declare -a dataset_names=(
    "cord"
    "funsd"
    "sroie"
)

declare -a synthetic_dataset_names=(
    "cord_alpha=1.0"
    "funsd_alpha=1.0"
    "sroie_alpha=1.0"
)

declare -a monitored_metrics=(
    "validation/seqeval/f1_score"
    "validation/seqeval/f1_score"
    "validation/seqeval/f1_score"
)

TASK_ARGUMENTS="--optimizer adamw --lr-start 2.0e-5 --train-batch-size 16 --eval-batch-size 16 --num-epochs 100 --use-preprocessed-dataset"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
