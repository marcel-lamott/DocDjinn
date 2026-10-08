from __future__ import annotations

from typing import TYPE_CHECKING

from docdjinn.analyzation.clustering.core._utilities import (
    EmbeddingType,
)
from docdjinn.analyzation.clustering.core._embeddings import (
    _load_embeddings,
)
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    import torch

logger = get_logger(__name__)


def _normalized_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def _reduce_embeddings_dims(
    embeddings: torch.Tensor,
    intermediate_num_dims: int = None,
    reduce_dim_metric: str = "euclidean",
    seed: int = None,
):
    import math

    import umap

    if intermediate_num_dims is None:
        intermediate_num_dims = math.floor(math.sqrt(embeddings.shape[1]))

    if intermediate_num_dims < embeddings.shape[1]:
        logger.info(
            f"Reducing embedding dimensions from {embeddings.shape[1]} to {intermediate_num_dims=} before clustering..."
        )
        umap_engine = umap.UMAP(
            n_components=intermediate_num_dims,
            metric=reduce_dim_metric,
            n_jobs=-1,
            verbose=False,
            random_state=seed,
        )
        return umap_engine.fit_transform(embeddings)
    return embeddings


def _run_hdbscan(
    embeddings: torch.Tensor,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_metric: str = "euclidean",
    seed: int = None,
):
    import hdbscan
    import numpy as np

    approx_min_span_tree = True
    if seed is not None:
        np.random.seed(seed)
        approx_min_span_tree = False  # otherwise not deterministic

    logger.info("Running HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric=hdbscan_metric,
        core_dist_n_jobs=-1,
        approx_min_span_tree=approx_min_span_tree,
        algorithm="best",
        prediction_data=True,
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    return soft_clusters, cluster_labels


def _run_knn(
    embeddings: torch.Tensor,
    cluster_labels: np.ndarray,
    k_nn_n_neighbors: int = 5,
):
    import copy

    from sklearn.neighbors import KNeighborsClassifier

    # train k-NN classifier
    noise_mask = cluster_labels == -1
    non_noise_mask = cluster_labels != -1
    X_non_noise = embeddings[non_noise_mask]
    y_non_noise = cluster_labels[non_noise_mask]
    knn = KNeighborsClassifier(n_neighbors=k_nn_n_neighbors, n_jobs=-1)
    knn.fit(X_non_noise, y_non_noise)

    X_noise = embeddings[noise_mask]
    predicted_labels = knn.predict(X_noise)

    # assign predicted labels back to noise points
    cluster_labels = copy.deepcopy(cluster_labels)
    cluster_labels[noise_mask] = predicted_labels

    return cluster_labels


def _get_cached_reduced_embeddings(
    embeddings: np.ndarray,
    intermediate_num_dims: int,
    reduce_dim_metric: str,
    seed: int,
    cache_dir: str = None,
) -> np.ndarray:
    """Get reduced embeddings from cache or compute and cache them."""
    import os
    import pickle

    if cache_dir is None:
        # Compute without caching
        return _reduce_embeddings_dims(
            embeddings=embeddings,
            intermediate_num_dims=intermediate_num_dims,
            reduce_dim_metric=reduce_dim_metric,
            seed=seed,
        )

    # Create cache key from parameters
    cache_key = f"{intermediate_num_dims}_{reduce_dim_metric}_{seed}"
    cache_file = os.path.join(cache_dir, f"reduced_embeddings_{cache_key}.pkl")

    # Try to load from cache
    if os.path.exists(cache_file):
        logger.info(f"Loading reduced embeddings from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # Compute and cache
    os.makedirs(cache_dir, exist_ok=True)
    reduced_embeddings = _reduce_embeddings_dims(
        embeddings=embeddings,
        intermediate_num_dims=intermediate_num_dims,
        reduce_dim_metric=reduce_dim_metric,
        seed=seed,
    )

    with open(cache_file, "wb") as f:
        pickle.dump(reduced_embeddings, f)
    logger.info(f"Cached reduced embeddings to: {cache_file}")

    return reduced_embeddings


# layoutlm CLS token, clip, text, combined
def _read_and_cluster_embeddings(
    embeddings_dir: str,
    dataset_name: str,
    embedding_type: EmbeddingType,
    intermediate_num_dims: int = None,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_metric: str = "euclidean",
    method: str = "hdbscan",
    n_kmeans_clusters: int = 150,
    k_nn_n_neighbors: int = 5,
    seed: int = 42,
    do_knn: bool = True,
    cache_dir: str = None,
) -> dict:
    """
    Read embeddings from H5PY file, reduce dimensions, and cluster them.

    This function first loads the embeddings from an H5PY file, normalizes them to unit length,
    and then reduces their dimensions using UMAP if specified (by default we always use umap).
    It then applies the chosen clustering algorithm (HDBSCAN or KMeans) to the reduced embeddings. Usually we only
    use HDBSCAN currently with KNN, and optionally apply k-NN to label noise points. Without KNN, HDBSCAN returns
    clusters with noise points associated a label of -1. KMeans is also supported as an alternative clustering method.
    The function returns a dictionary containing the cluster labels, noise mask, number of noise points,
    reduced embeddings, and soft cluster assignments.

    Args:
        embeddings_dir (str): Directory where the embeddings H5PY file is located.
        dataset_name (str): Name of the dataset (used to construct the file name).
        embedding_type (EmbeddingType): Type of embeddings (layout, clip, text).
        intermediate_num_dims (int, optional): Number of dimensions to reduce embeddings to before clustering.
            If None, no dimensionality reduction is applied. Defaults to None.
        hdbscan_min_cluster_size (int, optional): Minimum cluster size for HDBSCAN algorithm.
            Defaults to 10.
        hdbscan_metric (str, optional): Distance metric used by HDBSCAN algorithm.
            Defaults to "euclidean".
        method (str, optional): The clustering method to use ("hdbscan" or "kmeans"). Defaults to "hdbscan".
        n_kmeans_clusters (int, optional): Number of clusters for KMeans algorithm.
            Only used if method is "kmeans". Defaults to 150.
        k_nn_n_neighbors (int, optional): Number of neighbors for k-NN algorithm.
            Only used if method is "hdbscan" and do_knn is True. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        do_knn (bool, optional): Whether to apply k-nearest neighbors processing.
            Only used if method is "hdbscan". Defaults to True.
        cache_dir (str, optional): Directory to cache reduced embeddings.
            If None, no caching is done. Defaults to None.
    """
    import numpy as np
    import torch
    from pathlib import Path

    # read the embeddings
    embeddings, _ = _load_embeddings(
        file_path=Path(embeddings_dir) / dataset_name / f"{embedding_type.value}.h5"
    )
    embeddings = torch.from_numpy(embeddings)

    # normalize the embeddings
    embeddings = _normalized_embeddings(embeddings)

    # we also reduce embeddings to embeddings_2d for visualization
    # we only run it to cache the embeddings
    _get_cached_reduced_embeddings(
        embeddings=embeddings,
        intermediate_num_dims=2,
        reduce_dim_metric=hdbscan_metric,
        seed=seed,
        cache_dir=cache_dir,
    )

    # reduce embedding dimensions for clustering
    embeddings_reduced_dim = _get_cached_reduced_embeddings(
        embeddings=embeddings,
        intermediate_num_dims=intermediate_num_dims,
        reduce_dim_metric=hdbscan_metric,
        seed=seed,
        cache_dir=cache_dir,
    )

    # convert embeddings to double
    embeddings_reduced_dim = embeddings_reduced_dim.astype(np.double)

    # normalize reduced embeddings
    embeddings_reduced_dim = _normalized_embeddings(embeddings_reduced_dim)

    if method == "hdbscan":
        # step 1: run the clustering algorithm on the embeddings
        soft_clusters, cluster_labels = _run_hdbscan(
            embeddings=embeddings_reduced_dim,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_metric=hdbscan_metric,
            seed=seed,
        )

        # step 2: train k-NN on non-noise points
        # select points that are not labeled as noise
        num_noise = np.sum(cluster_labels == -1)
        noise_mask = cluster_labels == -1

        logger.info("Number of noise points: %d", num_noise)

        # return if not using k-NN to label noise points
        if do_knn and num_noise > 0:
            cluster_labels = _run_knn(
                embeddings=embeddings_reduced_dim,
                cluster_labels=cluster_labels,
                k_nn_n_neighbors=k_nn_n_neighbors,
            )

        return {
            "cluster_labels": cluster_labels,
            "noise_mask": noise_mask,
            "num_noise": num_noise,
            "embeddings_reduced_dim": embeddings_reduced_dim,
            "soft_clusters": soft_clusters,
        }
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=seed, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_reduced_dim)
        soft_clusters = np.zeros((len(cluster_labels), n_kmeans_clusters))
        soft_clusters[np.arange(len(cluster_labels)), cluster_labels] = 1.0
        return {
            "cluster_labels": cluster_labels,
            "num_noise": 0,
            "embeddings_reduced_dim": embeddings_reduced_dim,
            "soft_clusters": soft_clusters,
        }
    else:
        raise ValueError(f"Unknown clustering method: {method}")
