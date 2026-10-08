#!/bin/bash
# use scheduler python scripts/experiments/schedule_jobs.py --config-file scripts/experiments/02_v2_sampling_eval/jobs.sh --gpu-ids 1,2,3,4,5,6,7
EXPERIMENT_NAME="experiment_02_v1"

declare -a model_configs=(
    "layoutlmv3"
)

declare -a layout_model_configs=(
    "faster-rcnn"
)

declare -a dataset_name_and_script=(
    # classification datasets
    "scripts/experiments/02_v1_sampling_eval/train_classification.sh rvlcdip rvlcdip_alpha=1.0_v1"
    "scripts/experiments/02_v1_sampling_eval/train_classification.sh rvlcdip rvlcdip_alpha=0.75_v1"
    "scripts/experiments/02_v1_sampling_eval/train_classification.sh rvlcdip rvlcdip_alpha=0.5_v1"
    # # # entity labeling datasets
    "scripts/experiments/02_v1_sampling_eval/train_entity_labeling.sh cord cord_alpha=1.0_v1"
    "scripts/experiments/02_v1_sampling_eval/train_entity_labeling.sh cord cord_alpha=0.75_v1"
    "scripts/experiments/02_v1_sampling_eval/train_entity_labeling.sh cord cord_alpha=0.5_v1"
    # # qa datasets
    "scripts/experiments/02_v1_sampling_eval/train_question_answering.sh ex_docvqa docvqa_alpha=1.0_v1"
    "scripts/experiments/02_v1_sampling_eval/train_question_answering.sh ex_docvqa docvqa_alpha=0.75_v1"
    "scripts/experiments/02_v1_sampling_eval/train_question_answering.sh ex_docvqa docvqa_alpha=0.5_v1"
    # # layout analysis datasets
    "scripts/experiments/02_v1_sampling_eval/train_layout_analysis.sh publaynet publaynet_correct-sampling_alpha=1.0_v1"
    "scripts/experiments/02_v1_sampling_eval/train_layout_analysis.sh publaynet publaynet_correct-sampling_alpha=0.75_v1"
    "scripts/experiments/02_v1_sampling_eval/train_layout_analysis.sh publaynet publaynet_correct-sampling_alpha=0.5_v1"
)

CONFIGS=()

# Generate configurations based on model_configs and dataset_names
for dataset_script in "${dataset_name_and_script[@]}"; do
    dataset_script_parts=($dataset_script)
    dataset_script=${dataset_script_parts[0]}
    dataset_name=${dataset_script_parts[1]}
    synthetic_dataset=${dataset_script_parts[2]}
    if [[ "$dataset_name" == "publaynet" || "$dataset_name" == "doclaynet" || "$dataset_name" == "icdar2019" ]]; then
        for model_config in "${layout_model_configs[@]}"; do
            for seed in 42 826 548 401 335; do # randomly chosen seeds
                config_entry="${EXPERIMENT_NAME}-${model_config}-${dataset_name}-0-${synthetic_dataset}-all-seed-${seed} ${dataset_script} ${dataset_name} ${synthetic_dataset} ${model_config} ${seed}"
                CONFIGS+=("$config_entry")
            done
        done
    else
        for model_config in "${model_configs[@]}"; do
            for seed in 42 826 548 401 335; do # randomly chosen seeds
                config_entry="${EXPERIMENT_NAME}-${model_config}-${dataset_name}-0-${synthetic_dataset}-all-seed-${seed} ${dataset_script} ${dataset_name} ${synthetic_dataset} ${model_config} ${seed}"
                CONFIGS+=("$config_entry")
            done
        done
    fi
done
