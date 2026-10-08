#!/bin/bash

declare datasets=(
    # classification
    "rvlcdip_alpha=1.0"
    "tobacco3482_alpha=1.0"
    "doclaynet4k_alpha=1.0_CLS"

    # docvqa
    "docvqa_alpha=1.0"
    "wtq_alpha=1.0"

    # kie
    "cord_alpha=1.0"
    "funsd_alpha=1.0"
    "kleister_alpha=1.0"
    "sroie_alpha=1.0"

    # # layout analysis
    "publaynet_correct-sampling_alpha=1.0"
    "icdar2019_alpha=1.0"
    "doclaynet4k_alpha=1.0_DLA --clip-bboxes-to-foreground"
)

for dataset in "${datasets[@]}"; do
    echo "Preparing synthetic dataset: ${dataset}"
    python docdjinn/data/cmds/prepare_synth_datasets.py --dataset-name ${dataset} $@
done
