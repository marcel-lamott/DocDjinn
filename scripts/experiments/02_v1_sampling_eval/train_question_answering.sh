#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "layoutlmv3|--model-name microsoft/layoutlmv3-base --tokenizer-name microsoft/layoutlmv3-base --use-segment-level-bboxes"
)

declare -a dataset_names=(
    # alpha 1.0
    "ex_docvqa"
    # alpha 0.75
    "ex_docvqa"
    # alpha 0.5
    "ex_docvqa"
)

declare -a synthetic_dataset_names=(
    # alpha 1.0
    "docvqa_alpha=1.0_v1"
    # alpha 0.75
    "docvqa_alpha=0.75_v1"
    # alpha 0.5
    "docvqa_alpha=0.5_v1"
)

declare -a monitored_metrics=(
    "validation/ex_due_eval/ANLS"
    "validation/ex_due_eval/ANLS"
    "validation/ex_due_eval/ANLS"
)

TASK_ARGUMENTS="--validate-every-n-epochs 0.2 --lr-start 5.0e-5 --ignore-samples-with-no-answer --lr-schedule-warmup-steps-frac-of-total 0.02 --train-batch-size 8 --eval-batch-size 1 --num-epochs 50  --use-preprocessed-dataset"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
