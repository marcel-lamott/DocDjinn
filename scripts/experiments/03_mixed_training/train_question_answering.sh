#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "bert-base-uncased|--model-name bert-base-uncased --tokenizer-name bert-base-uncased --train-batch-size 32" # text only
    "lilt|--model-name SCUT-DLVCLab/lilt-roberta-en-base --tokenizer-name SCUT-DLVCLab/lilt-roberta-en-base --train-batch-size 32" # text + layout
    "layoutlmv3|--model-name microsoft/layoutlmv3-base --tokenizer-name microsoft/layoutlmv3-base --use-segment-level-bboxes --train-batch-size 16"
)

declare -a dataset_names=(
    "ex_docvqa"
    "ex_klc"
    "ex_wiki"
)

declare -a synthetic_dataset_names=(
    "docvqa_alpha=1.0"
    "kleister_alpha=1.0"
    "wtq_alpha=1.0"
)

declare -a monitored_metrics=(
    "validation/ex_due_eval/ANLS"
    "validation/ex_due_eval/F1"
    "validation/ex_due_eval/WTQ"
)

TASK_ARGUMENTS="--optimizer adamw --lr-start 5.0e-5 --ignore-samples-with-no-answer --lr-schedule-warmup-steps-frac-of-total 0.02 --eval-batch-size 1 --num-epochs 50  --use-preprocessed-dataset"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
