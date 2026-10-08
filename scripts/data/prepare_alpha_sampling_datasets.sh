#!/bin/bash

declare datasets=(
    "rvlcdip_alpha=0.5_v1"
    "rvlcdip_alpha=0.75_v1"
    "rvlcdip_alpha=1.0_v1"
    "rvlcdip_alpha=0.5" # this is v2
    "rvlcdip_alpha=0.75" # this is v2
    "rvlcdip_alpha=1.0" # this is v2

    "docvqa_alpha=0.5_v1"
    "docvqa_alpha=0.75_v1"
    "docvqa_alpha=1.0_v1"
    "docvqa_alpha=0.5"  # this is v2
    "docvqa_alpha=0.75"  # this is v2
    "docvqa_alpha=1.0"  # this is v2

    "cord_alpha=0.5_v1"
    "cord_alpha=0.75_v1"
    "cord_alpha=1.0_v1"
    "cord_alpha=0.5" # this is v2
    "cord_alpha=0.75" # this is v2
    "cord_alpha=1.0" # this is v2

    "publaynet_correct-sampling_alpha=0.5_v1"
    "publaynet_correct-sampling_alpha=0.75_v1"
    "publaynet_correct-sampling_alpha=1.0_v1"
    "publaynet_correct-sampling_alpha=0.5" # this is v2
    "publaynet_correct-sampling_alpha=0.75" # this is v2
    "publaynet_correct-sampling_alpha=1.0" # this is v2
)

for dataset in "${datasets[@]}"; do
    echo "Preparing synthetic dataset: ${dataset}"
    python docdjinn/data/cmds/prepare_synth_datasets.py --dataset-name ${dataset} $@
done
