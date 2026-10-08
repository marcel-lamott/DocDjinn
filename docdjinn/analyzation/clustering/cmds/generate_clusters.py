from __future__ import annotations

from pathlib import Path

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.analyzation.clustering.core._metrics import calculate_cluster_statistics
from docdjinn.analyzation.clustering.core._utilities import (
    EmbeddingType,
    _save_clustering_metrics,
)
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(cfg: ClusteringConfig):
    """
    Generate clusters for all embedding types and save results.
    """

    import numpy as np

    from docdjinn.analyzation.clustering.core._algorithms import (
        _read_and_cluster_embeddings,
    )
    from docdjinn.analyzation.clustering.core._metrics import (
        evaluate_clusters_unsupervised,
    )
    from docdjinn.analyzation.clustering.core._utilities import (
        _get_clustering_output_path,
    )

    logger.info(f"Clustering with config:\n{cfg}")

    for embedding_type in EmbeddingType.__members__.values():
        logger.info(f"Generating clusters for {embedding_type.value=}")

        # see if embeddings exist
        embeddings_path = (
            Path(cfg.embeddings_dir) / cfg.dataset_name / (f"{embedding_type.value}.h5")
        )
        if not embeddings_path.exists():
            logger.warning(
                f"Embeddings not found for {cfg.dataset_name} at {embeddings_path}, skipping..."
            )
            continue

        # save cluster labels
        output_dir = Path(cfg.output_dir) / cfg.dataset_name / embedding_type.value
        clusters_path = _get_clustering_output_path(
            output_dir=output_dir,
            intermediate_num_dims=cfg.intermediate_num_dims,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
            hdbscan_metric=cfg.hdbscan_metric,
            k_nn_n_neighbors=cfg.k_nn_n_neighbors,
            do_knn=cfg.do_knn,
            method=cfg.method,
        )

        if not Path(clusters_path).exists():
            outputs = _read_and_cluster_embeddings(
                embeddings_dir=cfg.embeddings_dir,
                dataset_name=cfg.dataset_name,
                embedding_type=embedding_type,
                intermediate_num_dims=cfg.intermediate_num_dims,
                hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
                hdbscan_metric=cfg.hdbscan_metric,
                k_nn_n_neighbors=cfg.k_nn_n_neighbors,
                seed=cfg.seed,
                do_knn=cfg.do_knn,
                cache_dir=output_dir,
                method=cfg.method,
            )

            logger.info(f"Saving clusters to {clusters_path}...")
            Path(clusters_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(
                clusters_path,
                outputs,
            )
            cluster_labels = outputs["cluster_labels"]
            num_noise = outputs.get("num_noise", 0)
            embeddings_reduced_dim = outputs["embeddings_reduced_dim"]
        else:
            logger.info(f"Loading existing clusters from {clusters_path}...")
            cluster_results = np.load(clusters_path, allow_pickle=True).item()
            cluster_labels = cluster_results["cluster_labels"]
            num_noise = cluster_results["num_noise"]
            embeddings_reduced_dim = cluster_results["embeddings_reduced_dim"]

        # compute cluster statistics
        cluster_stats = calculate_cluster_statistics(
            embeddings_reduced_dim, cluster_labels
        )
        cluster_stats.to_csv(
            clusters_path.parent / clusters_path.name.replace(".npy", "_stats.csv"),
            index=False,
        )

        # compute metrics
        cluster_metrics, num_clusters = evaluate_clusters_unsupervised(
            embeddings=embeddings_reduced_dim, cluster_labels=cluster_labels
        )

        # save metrics
        _save_clustering_metrics(
            output_dir=cfg.output_dir,
            dataset_name=cfg.dataset_name,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size,
            intermediate_num_dims=cfg.intermediate_num_dims,
            hdbscan_metric=cfg.hdbscan_metric,
            k_nn_n_neighbors=cfg.k_nn_n_neighbors,
            method=cfg.method,
            embedding_type=embedding_type,
            embeddings=embeddings_reduced_dim,
            cluster_metrics=cluster_metrics,
            num_clusters=num_clusters,
            num_noise=num_noise,
            seed=cfg.seed,
            do_knn=cfg.do_knn,
        )


class ClusteringConfig(pydantic.BaseModel):
    """
    Configuration for clustering operations.
    """

    dataset_name: str
    seed: int = 42
    hdbscan_min_cluster_size: int = 10
    intermediate_num_dims: int = 100
    hdbscan_metric: str = "euclidean"
    do_knn: bool = True
    k_nn_n_neighbors: int = 5
    embeddings_dir: str | Path = ENV.EMBEDDINGS_DIR
    output_dir: str | Path = ENV.CLUSTERS_DIR
    method: str = "hdbscan"  # or "kmeans"


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=ClusteringConfig,
    )
    main(parser.parse_typed_args())
