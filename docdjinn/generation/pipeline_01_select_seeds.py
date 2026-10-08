"""
TODO: select seeds based on clusters
"""

import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from docdjinn.analyzation.clustering.cmds.generate_seeds import (
    GenerateSeedsConfig,
    generate_seeds_for_embedding_type,
)
from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
from docdjinn.data.interface import load_dataset
from docdjinn.generation.constants import SEED_IMAGE_MAX_WIDTH, SEED_IMAGE_QUALITY
from docdjinn.generation.models import PipelineParameters, SynDatasetDefinition
from docdjinn.generation.utils.image import (
    downscale_and_compress,
)
from docdjinn.generation.utils.log import log_pipeline_level


def prepare_seed_images(dsdef: SynDatasetDefinition, seeds_df: pd.DataFrame):
    dsfiles = dsdef.get_file_structure()

    dataset = load_dataset(dsdef.base_dataset_name, split="train")

    all_doc_ids = set(seeds_df.stack())
    for seed in tqdm(all_doc_ids, desc="Downscaling and compressing seed images"):
        outfile = dsfiles.preprocessed_seed_images_directory / f"{seed}.jpg"
        if not outfile.exists():
            img = dataset.train.get_by_id(seed).image.content  # type: ignore

            downscale_and_compress(
                img=img,
                save_to_path=outfile,
                max_width=SEED_IMAGE_MAX_WIDTH,
                quality=SEED_IMAGE_QUALITY,
            )


def visualize_selected_seed_labels(
    dsdef: SynDatasetDefinition, seeds_df: pd.DataFrame, save_to: Path
):
    import matplotlib.pyplot as plt

    dataset = load_dataset(dsdef.base_dataset_name, split="train")

    all_doc_ids = set(seeds_df.stack())
    label_counter: dict[str, int] = {}
    for seed in tqdm(all_doc_ids, desc="Extracting class labels for seed images"):
        doc = dataset.train.get_by_id(seed)  # type: ignore

        document_label = None
        for annotation in doc.annotations:  # type: ignore
            if annotation._type == "classification":
                document_label = annotation.label.name
                break

        if document_label is not None:
            label_counter[document_label] = label_counter.get(document_label, 0) + 1

    if len(label_counter) == 0:
        return

    print("Seed image class label distribution:")
    for label, count in label_counter.items():
        print(f"Label: {label}, Count: {count}")

    # visualize the seed label distribution as a bar chart
    fig = plt.figure(figsize=(10, 6))
    plt.bar(list(label_counter.keys()), list(label_counter.values()))
    plt.xlabel("Class Labels")
    plt.ylabel("Frequency")
    plt.title("Seed Image Class Label Distribution")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close(fig)


def visualize_selected_clusters(clusters_df, save_to):
    from collections import Counter

    import matplotlib.pyplot as plt

    # Flatten all values into a single list
    all_clusters = clusters_df.values.flatten()

    # Count occurrences per cluster
    cluster_counts = Counter(all_clusters)

    # Sort by cluster index for plotting
    clusters_sorted = sorted(cluster_counts.keys())
    counts_sorted = [cluster_counts[c] for c in clusters_sorted]

    # Plot histogram
    plt.bar(clusters_sorted, counts_sorted)
    plt.xlabel("Cluster")
    plt.ylabel("Frequency")
    plt.title("Histogram of Cluster Occurrences")
    plt.savefig(save_to)


def pipeline_select_seeds(params: PipelineParameters):
    log_pipeline_level()

    dsdef: SynDatasetDefinition = params.dsdef
    dsfiles = dsdef.get_file_structure()

    total_prompt_calls = int(
        math.ceil(dsdef.documents_count / dsdef.prompt_params.num_solutions)
    )
    # Add a bit of buffer because some documents will fail and we need to prompt more often
    # total_prompt_calls += 100

    cfg = GenerateSeedsConfig(
        dataset_name=dsdef.base_dataset_name,
        hdbscan_min_cluster_size=dsdef.hdbscan_min_cluster_size,
        output_dir=dsfiles.base_path,
        total_seed_runs=total_prompt_calls,
        total_seeds_per_run=dsdef.seed_images_count,
        visualize_seeds=True,
        alpha=dsdef.alpha,
        max_pool_size=dsdef.max_seed_pool,
        seed=42,
        seed_selection_strategy=dsdef.seed_selection_strategy,
    )
    embedding_type = EmbeddingType(dsdef.embedding_type)
    seeds_path: Path
    clusters_path: Path
    seeds_path, clusters_path = generate_seeds_for_embedding_type(
        cfg=cfg, embedding_type=embedding_type
    )

    # Rename seeds file
    new_file = seeds_path.with_name("seeds.csv")
    seeds_path.rename(new_file)
    seeds_path = new_file

    # Rename clusters file
    new_file = clusters_path.with_name("clusters.csv")
    clusters_path.rename(new_file)
    clusters_path = new_file

    seeds_df = pd.read_csv(seeds_path)

    # visualize document classes if possible
    visualize_selected_seed_labels(
        dsdef=dsdef,
        seeds_df=seeds_df,
        save_to=seeds_path.with_name("seed_label_distribution.png"),
    )

    # Prepare seed images
    prepare_seed_images(dsdef=dsdef, seeds_df=seeds_df)

    clusters_df = pd.read_csv(clusters_path)
    visualize_selected_clusters(
        clusters_df, save_to=clusters_path.with_name("clusters_hist.png")
    )
