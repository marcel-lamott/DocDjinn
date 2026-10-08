import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from docdjinn import ENV
from docdjinn.analyzation.clustering.core._embeddings import (
    _load_sample_ids_from_embeddings,
)
from docdjinn.data import load_dataset
from docdjinn.analyzation.clustering.core._utilities import _get_clustering_output_path
from .config import settings
from docdjinn.logging import get_logger

logger = get_logger(__name__)

class DataManager:
    """Manages dataset and clustering data loading."""

    def __init__(self):
        self.dataset = None
        self.dataset_name = None
        self.metrics = None
        self.cluster_data_cache = {}

    def load_dataset(self, dataset_name: str):
        """Load dataset and update internal state."""
        if self.dataset_name != dataset_name:
            self.dataset = load_dataset(dataset_name=dataset_name, split="train")
            self.dataset_name = dataset_name
            self.metrics = pd.read_csv(settings.metrics_csv_path)
            self.metrics = self.metrics[self.metrics["dataset_name"] == dataset_name]

    def get_cluster_data(
        self,
        dataset_name: str,
        embedding_src: str,
        intermediate_dims: int,
        min_cluster_size: int,
        method: str,
    ) -> Dict:
        """Load clustering results with caching."""
        cache_key = (
            dataset_name,
            embedding_src,
            intermediate_dims,
            min_cluster_size,
            method,
        )

        if cache_key not in self.cluster_data_cache:
            output_dir = ENV.CLUSTERS_DIR / dataset_name / embedding_src
            sample_ids = _load_sample_ids_from_embeddings(
                file_path=ENV.EMBEDDINGS_DIR / dataset_name / f"{embedding_src}.h5"
            )
            logger.info("Loading clustering results from %s", output_dir)
            clusters_path = _get_clustering_output_path(
                output_dir=output_dir,
                intermediate_num_dims=intermediate_dims,
                hdbscan_min_cluster_size=1 if method == "kmeans" else min_cluster_size,
                hdbscan_metric=settings.metric,
                k_nn_n_neighbors=settings.k_nn_n_neighbors,
                method=method,
            )

            # Load cluster data
            cluster_data = np.load(clusters_path, allow_pickle=True).item()

            # Load cluster statistics
            stats_path = clusters_path.parent / clusters_path.name.replace(
                ".npy", "_stats.csv"
            )
            cluster_stats = pd.read_csv(stats_path)

            # Load 2D embeddings
            emb_2d_path = (
                output_dir
                / f"reduced_embeddings_2_{settings.metric}_{settings.seed}.pkl"
            )
            if not os.path.exists(emb_2d_path):
                raise ValueError(f"2D embeddings not found: {emb_2d_path}")

            with open(emb_2d_path, "rb") as f:
                emb_2d = pickle.load(f)

            self.cluster_data_cache[cache_key] = {
                "sample_ids": sample_ids,
                "cluster_data": cluster_data,
                "cluster_stats": cluster_stats,
                "emb_2d": emb_2d,
            }

        return self.cluster_data_cache[cache_key]

    def create_scatter_dataframe(
        self,
        labels: np.ndarray,
        emb_2d: np.ndarray,
        soft_clusters: np.ndarray,
        sample_ids: np.ndarray,
        noise_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Create DataFrame for scatter plot visualization."""
        if noise_mask is None:
            noise_mask = np.array([False] * len(labels))

        x, y = emb_2d[:, 0], emb_2d[:, 1]
        return pd.DataFrame(
            {
                "doc_id": sample_ids,
                "x": x,
                "y": y,
                "label": labels,
                "prob": np.max(soft_clusters, axis=1),
                "index": np.arange(len(labels)),
                "noise_mask": noise_mask,
            }
        )


# Global instance
data_manager = DataManager()
