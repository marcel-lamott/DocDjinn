#!/bin/bash
# Base script for running model-dataset experiments.
# Called by another script that defines dataset and model configs.
# Usage (inside another script):
#   source run_base.sh [DATASET_NAME] [MODEL_NAME] [extra_args...]

set -e
EXPERIMENT_NAME="experiment_02_v1"
DATASET_NAME=${1:-all}
SYNTHETIC_DATASET_NAME=${2:-all}
MODEL_NAME=${3:-all}
SEED=${4:-42}

# Verify configs are defined by the calling script
if [ -z "${dataset_names[*]}" ] || [ -z "${model_configs[*]}" ] || [ -z "${monitored_metrics[*]}" ]; then
    echo "Error: dataset_names and model_configs must be defined before calling run_base.sh"
    exit 1
fi

for idx in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$idx]}"
    synthetic_dataset="${synthetic_dataset_names[$idx]}"
    monitored_metric="${monitored_metrics[$idx]}"

    if [ "$DATASET_NAME" != "all" ] && [ "$DATASET_NAME" != "$dataset" ]; then
        continue
    fi

    if [ "$SYNTHETIC_DATASET_NAME" != "all" ] && [ "$SYNTHETIC_DATASET_NAME" != "$synthetic_dataset" ]; then
        continue
    fi

    for config in "${model_configs[@]}"; do
        IFS='|' read -r model_name model_args <<< "$config"

        if [ "$MODEL_NAME" != "all" ] && [ "$MODEL_NAME" != "$model_name" ]; then
            continue
        fi

        run_name_with_seed="${EXPERIMENT_NAME}-${model_name}-${dataset}-0-${synthetic_dataset}-all-seed-${SEED}"

        echo "=========================================================="
        echo "Running experiment: ${run_name_with_seed}"
        echo "Dataset: ${dataset} | Synthetic: ${synthetic_dataset} | Model: ${model_name} | Seed: ${SEED}"
        echo "=========================================================="

        # set -x
        python docdjinn/evaluation/runners/mixed_runner.py \
            --dataset-name "${dataset}" \
            --synthetic-dataset-name "${synthetic_dataset}" \
            --num-real-samples 0 \
            --num-synthetic-samples -1 \
            --run-name "${run_name_with_seed}" \
            --monitored-metric ${monitored_metric} \
            --seed ${SEED} \
            ${model_args} \
            ${TASK_ARGUMENTS} \
            ${@:5}
    done
done
