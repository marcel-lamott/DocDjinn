from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# Distance / Connectivity
def _normalized_connectivity(X, labels, n_neighbors=10):
    """
    Normalized connectivity metric: measures if each point's nearest neighbors
    are in the same cluster. 0 = perfect connectivity, 1 = worst.

    Parameters:
    - X: data points (n_samples x n_features)
    - labels: cluster labels
    - n_neighbors: number of neighbors to consider

    Returns:
    - normalized connectivity score (0-1)
    """
    from sklearn.neighbors import NearestNeighbors

    n_samples = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Exclude self from neighbors
    indices = indices[:, 1:]
    score = 0
    for i in range(n_samples):
        for j in indices[i]:
            if labels[i] != labels[j]:
                score += 1 / n_neighbors  # penalize different cluster

    # Maximum possible score is n_samples (each point has all neighbors in other clusters)
    max_score = n_samples
    normalized_score = score / max_score
    return normalized_score


# Compactness / Separation
def _cluster_compactness_scores(embeddings, labels):
    """
    Compute compactness scores for clusters using various metrics.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    return {
        "silhouette_score": silhouette_score(embeddings, labels),
        "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
        "davies_bouldin_score": davies_bouldin_score(embeddings, labels),
    }


# Balance / Size Equity
def _cluster_balance_scores(cluster_sizes):
    """
    Compute balance scores for clusters using various metrics.
    """
    import numpy as np
    import scipy

    sizes = np.array(cluster_sizes)
    entropy = scipy.stats.entropy(sizes)
    norm_entropy = entropy / np.log(len(sizes))

    # Coefficient of variation
    cv = sizes.std() / sizes.mean()
    mmr = sizes.min() / sizes.max()

    # Gini coefficient
    sorted_sizes = np.sort(sizes)
    n = len(sizes)
    gini = (
        2 * np.sum((np.arange(1, n + 1)) * sorted_sizes) / (n * sorted_sizes.sum())
    ) - (n + 1) / n

    return {
        "entropy": norm_entropy.item(),
        "coefficient_of_variation": cv.item(),
        "min-to-max-ratio": mmr.item(),
        "gini-coefficient": gini.item(),
    }


def evaluate_clusters_unsupervised(
    embeddings: np.ndarray, cluster_labels: np.ndarray
) -> tuple[dict[str, float], int]:
    """
    Evaluate clustering quality using unsupervised metrics.
    """
    import numpy as np
    import torch

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    unique_entries, counts = np.unique(cluster_labels, return_counts=True)
    result = dict()
    result["connectivity"] = {
        "normalized_connectivity": _normalized_connectivity(
            X=embeddings,
            labels=cluster_labels,
            n_neighbors=int(embeddings.shape[0] * 0.01),
        ),
    }
    result["compactness"] = _cluster_compactness_scores(
        embeddings=embeddings, labels=cluster_labels
    )
    result["balance"] = _cluster_balance_scores(counts)
    return result, len(unique_entries)


def calculate_cluster_statistics(
    embeddings: np.ndarray, cluster_labels: np.ndarray
) -> "pd.DataFrame":
    """
    Calculate statistics for each cluster, including size and variance.
    Variance is computed as the average pairwise cosine distance within the cluster.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    unique_clusters = set(cluster_labels)
    cluster_stats = []
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_size = len(cluster_embeddings)
        sim_matrix = cosine_similarity(cluster_embeddings)
        cosine_distances = 1 - sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        cosine_diversity = np.mean(cosine_distances)
        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "size": cluster_size,
                "variance": cosine_diversity,
            }
        )
    return pd.DataFrame(cluster_stats)
