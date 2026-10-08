from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pydantic.v1 as pydantic
import pydantic_argparse
import tqdm

from docdjinn import ENV
from docdjinn.analyzation.clustering.core._embeddings import (
    _load_sample_ids_from_embeddings,
)
from docdjinn.analyzation.clustering.core._utilities import (
    EmbeddingType,
    _get_clustering_output_path,
    _visualize_images_grid,
)
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    import numpy as np


logger = get_logger(__name__)


def alpha_cluster_sampling_create_pool(
    cluster_labels: np.ndarray,
    max_seed_pool: int = -1,
) -> np.ndarray:
    """
    Create a pool of candidate seed images for LLM prompt construction.

    The pool is sampled **proportional to cluster sizes**, ensuring that
    each cluster is represented at least once if possible. This prevents
    small clusters from being entirely excluded from the pool.

    Args:
        cluster_labels: np.ndarray of cluster labels for all samples.
        max_seed_pool: int, maximum number of seed images to select for the pool.
            - If -1 or larger than the dataset, the full dataset is used.

    Returns:
        np.ndarray: indices of samples included in the pool.
    """
    n_samples = len(cluster_labels)
    unique_labels = np.unique(cluster_labels)

    # Use full dataset if max_seed_pool is -1 or larger than dataset
    if max_seed_pool == -1 or max_seed_pool >= n_samples:
        return np.arange(n_samples)

    # Step 1: guarantee one sample per cluster
    guaranteed_indices = [
        np.random.choice(np.where(cluster_labels == label)[0])
        for label in unique_labels
    ]

    remaining = max_seed_pool - len(guaranteed_indices)
    if remaining <= 0:
        # pool is smaller than number of clusters: return guaranteed samples
        return np.array(guaranteed_indices)

    # Step 2: sample remaining indices proportional to cluster sizes
    cluster_sizes = {
        label: np.sum(cluster_labels == label).item() for label in unique_labels
    }
    cluster_prob = {
        label: size / sum(cluster_sizes.values())
        for label, size in cluster_sizes.items()
    }
    doc_prob = np.array([cluster_prob[cluster_labels[i]] for i in range(n_samples)])

    # Exclude guaranteed indices
    available_indices = np.setdiff1d(np.arange(n_samples), guaranteed_indices)
    available_prob = doc_prob[available_indices]
    available_prob = available_prob / available_prob.sum()

    sampled_remaining = np.random.choice(
        available_indices, size=remaining, replace=False, p=available_prob
    )

    pool_indices = np.concatenate([guaranteed_indices, sampled_remaining])
    return pool_indices


def alpha_cluster_sampling_pool(
    cluster_labels: np.ndarray,
    total_seeds: int,
    pool_indices: np.ndarray,
    alpha: float = 1.0,
    seed_selection_strategy: str = "v1",
) -> tuple[list[int], list[int]]:
    """
    Sample seeds from a pool using two-stage alpha-based cluster probabilities:
    1) Pick a cluster based on alpha weighting
    2) Pick a random sample from that cluster

    Args:
        cluster_labels: np.ndarray of cluster labels for all samples
        total_seeds: number of seeds to sample
        pool_indices: available sample indices
        alpha: exponent for cluster weighting
            - alpha=1 -> proportional to cluster size
            - alpha=0 -> uniform across clusters
            - alpha<0 -> inverse-proportional to cluster size

    Returns:
        Tuple of (sampled_indices, sampled_clusters)
    """
    pool_labels = cluster_labels[pool_indices]
    unique_labels = np.unique(pool_labels)

    # Compute cluster sizes in pool
    cluster_sizes = {
        label: np.sum(pool_labels == label).item() for label in unique_labels
    }

    # Compute alpha-weighted cluster probabilities
    cluster_probs = np.array([cluster_sizes[label] ** alpha for label in unique_labels])
    cluster_probs = cluster_probs / cluster_probs.sum()

    if seed_selection_strategy == "v1":
        sampled_indices = []
        sampled_clusters = []

        for _ in range(total_seeds):
            # 1) Pick a cluster according to alpha probabilities
            cluster = np.random.choice(unique_labels, p=cluster_probs)

            # 2) Pick a random sample from that cluster in the pool
            cluster_pool_indices = pool_indices[pool_labels == cluster]
            sample = np.random.choice(cluster_pool_indices)

            sampled_indices.append(int(sample))
            sampled_clusters.append(int(cluster))

        return sampled_indices, sampled_clusters

    elif seed_selection_strategy == "v2":
        sampled_indices = []
        sampled_clusters = []

        cluster = np.random.choice(unique_labels, p=cluster_probs)
        for _ in range(total_seeds):
            cluster_pool_indices = pool_indices[pool_labels == cluster]
            sample = np.random.choice(cluster_pool_indices)

            sampled_indices.append(int(sample))
            sampled_clusters.append(int(cluster))

        return sampled_indices, sampled_clusters
    else:
        raise ValueError(f"Unknown seed selection strategy: {seed_selection_strategy}")


def generate_seeds_for_embedding_type(
    cfg: GenerateSeedsConfig, embedding_type: EmbeddingType
) -> tuple[Path, Path]:
    import random

    import numpy as np
    import pandas as pd

    # set seed
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # get paths
    output_dir = Path(cfg.clusters_dir) / cfg.dataset_name / embedding_type.value
    embeddings_path = (
        Path(cfg.embeddings_dir) / cfg.dataset_name / f"{embedding_type.value}.h5"
    )
    cluster_sample_ids = _load_sample_ids_from_embeddings(embeddings_path)
    clusters_path = _get_clustering_output_path(
        output_dir=output_dir,
        intermediate_num_dims=cfg.intermediate_num_dims,
        hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
        hdbscan_metric=cfg.hdbscan_metric,
        k_nn_n_neighbors=cfg.k_nn_n_neighbors,
        do_knn=cfg.do_knn,
        method=cfg.method,
    )

    # load data
    # cluster_labels coresponds to the document indices in the dataset
    cluster_results = np.load(clusters_path, allow_pickle=True).item()
    cluster_labels = cluster_results["cluster_labels"]
    logger.info(f"Cluster labels shape: {cluster_labels.shape}")
    assert len(cluster_labels) == len(cluster_sample_ids), (
        "Mismatch in number of samples"
    )

    pool_indices = alpha_cluster_sampling_create_pool(
        cluster_labels=cluster_labels, max_seed_pool=cfg.max_pool_size
    )
    seed_samples = []
    seed_clusters = []
    for _ in tqdm.tqdm(range(cfg.total_seed_runs), desc="Sampling seeds"):
        sampled_seeds, sampled_clusters = alpha_cluster_sampling_pool(
            cluster_labels=cluster_labels,
            total_seeds=cfg.total_seeds_per_run,
            pool_indices=pool_indices,
            alpha=cfg.alpha,
            seed_selection_strategy=cfg.seed_selection_strategy,
        )
        sampled_seed_ids = [cluster_sample_ids[i] for i in sampled_seeds]
        seed_samples.append(sampled_seed_ids)
        seed_clusters.append(sampled_clusters)

    # save the sampled seeds
    seeds_output_path = Path(cfg.output_dir) / clusters_path.name.replace(
        ".npy",
        f"_alpha={cfg.alpha}_max-pool-size={cfg.max_pool_size}_strategy={cfg.seed_selection_strategy}_seeds.csv",
    )
    dataframe = pd.DataFrame(seed_samples)
    logger.info(f"Saving sampled seeds to {seeds_output_path}...")
    dataframe.to_csv(seeds_output_path, index=False)

    # save the sampled clusters
    clusters_output_path = Path(cfg.output_dir) / clusters_path.name.replace(
        ".npy",
        f"_alpha={cfg.alpha}_max-pool-size={cfg.max_pool_size}_strategy={cfg.seed_selection_strategy}_clusters.csv",
    )
    dataframe = pd.DataFrame(seed_clusters)
    logger.info(f"Saving sampled seeds to {clusters_output_path}...")
    dataframe.to_csv(clusters_output_path, index=False)

    # also visualize the random 20 seed documents as an image grid
    # load all seed documents into an image grid
    if cfg.visualize_seeds:
        from docdjinn.data import load_dataset

        dataset = load_dataset(cfg.dataset_name, split="train")
        seed_images = []
        for seed in sampled_seeds[: cfg.n_seeds_to_visualize]:
            seed_images.append(
                dataset.train.get_by_id(cluster_sample_ids[seed]).image.content
            )
        vis_fname = seeds_output_path.parent / seeds_output_path.name.replace(
            ".csv", ".png"
        )
        _visualize_images_grid(
            images=seed_images,
            save_path=vis_fname,
        )

    return seeds_output_path, clusters_output_path


class GenerateSeedsConfig(pydantic.BaseModel):
    """
    Configuration for generating clustering seeds.
    """

    # same as clustering config
    dataset_name: str
    seed: int = 42
    hdbscan_min_cluster_size: int = 10
    intermediate_num_dims: int = 100
    hdbscan_metric: str = "euclidean"
    do_knn: bool = True
    k_nn_n_neighbors: int = 5
    embeddings_dir: str | Path = ENV.EMBEDDINGS_DIR
    clusters_dir: str | Path = ENV.CLUSTERS_DIR
    output_dir: str | Path
    method: str = "hdbscan"  # or "kmeans"
    seed_selection_strategy: str = "v1"

    # specific to seed generation
    total_seed_runs: int = 10000
    total_seeds_per_run: int = 10
    visualize_seeds: bool = False
    n_seeds_to_visualize: int = 20

    # sampling strategy
    max_pool_size: int = -1  # if -1, seeds are selected from complete dataset, otherwise a pool is generated via proportional sampling, where it is ensured that each cluster is selected at least once
    """
    sampling exponent for clusters.
        - alpha=1 -> proportional
        - alpha=0 -> uniform
        - alpha<0 -> inverse-proportional
    """
    alpha: float = 0


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=GenerateSeedsConfig,
    )
    generate_seeds_for_embedding_type(parser.parse_typed_args(), EmbeddingType.combined)
