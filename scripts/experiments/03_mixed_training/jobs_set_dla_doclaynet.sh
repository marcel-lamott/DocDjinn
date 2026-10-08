#!/bin/bash
# use scheduler python scripts/experiments/schedule_jobs.py --config-file scripts/experiments/03_mixed_training/jobs_set_dla.sh --gpu-ids 1,2,3,4,5,6,7
declare -a model_configs=(
    "faster-rcnn-doclaynet"
    "cascade-rcnn-doclaynet"
)

declare -a dataset_name_and_script=(
    # layout analysis datasets
    "scripts/experiments/03_mixed_training/train_layout_analysis.sh doclaynet_4k_dla doclaynet4k_alpha=1.0_DLA"
)

SEEDS=(42 826 548) # randomly chosen seeds
EXPERIMENT_NAME="experiment_03"

declare -a real_syn_sizes=(
    "100 -1" # train on 100 real samples
    "1000 -1" # train on 1000 real samples
    "-1 -1" # train on all real samples
    "0 -1" # train on 0 real samples
    "-1 0"
    "100 0" # train on 100 real samples + 0 synthetic samples
    "1000 0" # train on 1000 real samples + 0 synthetic samples
)

declare -a per_config_args=(
    ""
    ""
    ""
    ""
    ""
    "--no-enable-early-stopping"
    "--no-enable-early-stopping"
)

CONFIGS=()
# Generate configurations based on model_configs and dataset_names
for seed in "${SEEDS[@]}"; do
    for dataset_script in "${dataset_name_and_script[@]}"; do
        dataset_script_parts=($dataset_script)
        dataset_script=${dataset_script_parts[0]}
        dataset_name=${dataset_script_parts[1]}
        synthetic_dataset=${dataset_script_parts[2]}
        for model_config in "${model_configs[@]}"; do
            for idx in "${!real_syn_sizes[@]}"; do
                real_syn_size="${real_syn_sizes[$idx]}"
                real_size=$(echo $real_syn_size | cut -d' ' -f1)
                synthetic_size=$(echo $real_syn_size | cut -d' ' -f2)
                per_config_args="${per_config_args[$idx]}"
                config_entry="${EXPERIMENT_NAME}-${model_config}-${dataset_name}-${real_size}-${synthetic_dataset}-${synthetic_size}-seed-${seed} ${dataset_script} ${dataset_name} ${synthetic_dataset} ${model_config} ${seed} ${real_size} ${synthetic_size} ${per_config_args}"
                CONFIGS+=("$config_entry")
            done
        done
    done
done

# use for slurm
# echo "Total configurations to run: ${#CONFIGS[@]}"

# GPUS_PER_TASK=1
# CPUS_PER_TASK=8
# MEMORY="40G"

# for config in "${CONFIGS[@]}"; do
#     config_parts=($config)
#     job_name=${config_parts[0]}
#     dataset_script=${config_parts[1]}

#     echo "Submitting job: ${job_name}"
#     echo "Command: bash $dataset_script ${config_parts[@]:2}"
#     # bash $dataset_script ${config_parts[@]:2}

#     # # # Submit each configuration as a separate Slurm job using srun
#     srun --job-name="${job_name}" \
#         -n 1 \
#         --gpus-per-task="${GPUS_PER_TASK}" \
#         --cpus-per-task="${CPUS_PER_TASK}" \
#         --mem="${MEMORY}" \
#         --output="data/cache/slurm_logs/${job_name}.out" \
#         --error="data/cache/slurm_logs/${job_name}.err" \
#         bash "$dataset_script" "${config_parts[@]:2}" &
# done

# wait
