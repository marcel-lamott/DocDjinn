#!/bin/bash

# Resolve relative path to run_base.sh (works no matter where you run this from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/base.sh"

declare -a model_configs=(
    "faster-rcnn|--model-name faster-rcnn_r50_fpn_1x_coco --with-amp --train-batch-size 16 --eval-batch-size 16 --optimizer sgd --lr-start 0.02 --momentum 0.9 --weight-decay 0.0001"
    "faster-rcnn-doclaynet|--model-name faster-rcnn_r50_fpn_1x_coco_doclaynet --with-amp --train-batch-size 16 --eval-batch-size 16 --optimizer sgd --lr-start 0.02 --momentum 0.9 --weight-decay 0.0001  --use-fixed-size --no-use-flip"
    "cascade-rcnn|--model-name cascade-rcnn_r50_fpn_1x_coco --with-amp --train-batch-size 16 --eval-batch-size 16 --optimizer sgd --lr-start 0.02 --momentum 0.9 --weight-decay 0.0001"
    "cascade-rcnn-doclaynet|--model-name cascade-rcnn_r50_fpn_1x_coco_doclaynet --with-amp --train-batch-size 16 --eval-batch-size 16 --optimizer sgd --lr-start 0.02 --momentum 0.9 --weight-decay 0.0001  --use-fixed-size --no-use-flip"
)

declare -a dataset_names=(
    "publaynet"
    "icdar2019"
    "doclaynet_4k_dla"
)

declare -a synthetic_dataset_names=(
    "publaynet_correct-sampling_alpha=1.0"
    "icdar2019_alpha=1.0"
    "doclaynet4k_alpha=1.0_DLA"
)

declare -a monitored_metrics=(
    "validation/coco_eval_AP"
    "validation/coco_eval_AP"
    "validation/coco_eval_AP"
)

TASK_ARGUMENTS="--num-epochs 40 --validate-every-n-epochs 1 --lr-schedule-warmup-steps-frac-of-total 0.05"

# Call the shared base script
source "$BASE_SCRIPT" "$@"
