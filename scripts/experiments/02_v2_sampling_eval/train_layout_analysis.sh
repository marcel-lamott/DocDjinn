#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "faster-rcnn|--model-name faster-rcnn_r50_fpn_1x_coco --with-amp --train-batch-size 16 --eval-batch-size 16 --optimizer sgd --lr-start 0.02 --momentum 0.9 --weight-decay 0.0001"
)

declare -a dataset_names=(
    # alpha 1.0
    "publaynet"
    # alpha 0.75
    "publaynet"
    # alpha 0.5
    "publaynet"
)

declare -a synthetic_dataset_names=(
    # alpha 1.0
    "publaynet_correct-sampling_alpha=1.0"
    # alpha 0.75
    "publaynet_correct-sampling_alpha=0.75"
    # alpha 0.5
    "publaynet_correct-sampling_alpha=0.5"
)

declare -a monitored_metrics=(
    "validation/coco_eval_AP"
    "validation/coco_eval_AP"
    "validation/coco_eval_AP"
)

TASK_ARGUMENTS="--num-epochs 40 --validate-every-n-epochs 1 --lr-schedule-warmup-steps-frac-of-total 0.05"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
