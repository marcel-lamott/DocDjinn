import os
from typing import List
from pydantic_settings import BaseSettings
from docdjinn import ENV
from pathlib import Path


class AppSettings(BaseSettings):
    # App Config
    debug: bool = True
    port: int = 8055
    graphs_base_dir: Path = ENV.CLUSTER_PLOTS
    external_stylesheets: List[str] = [
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
    ]

    # Clustering Options
    embedding_sources: List[str] = [
        "paper_kernel=4",
        "layout",
        "image",
        "text",
        "combined",
    ]
    intermediate_options: List[int] = [100]
    min_cluster_size_options: List[int] = [5, 10]
    dataset_options: List[str] = sorted([d for d in os.listdir(ENV.CLUSTERS_DIR)])

    # Default Values
    default_dataset: str = "tobacco3482"
    default_embedding: str = "paper_kernel=4"
    default_intermediate: int = 100
    default_min_cluster_size: int = 5
    default_method: str = "hdbscan"

    # Clustering Params
    seed: int = 42
    metric: str = "euclidean"
    k_nn_n_neighbors: int = 5
    do_knn: bool = False

    # Grid Config
    max_images: int = 12
    thumb_width: int = 200
    thumb_height: int = 280
    spacing: int = 10
    max_cols: int = 4

    # Metric Descriptions and Optimization Direction
    @property
    def metrics_list(self) -> dict:
        return {
            "num_clusters": {
                "direction": "min",
                "description": "Total number of clusters formed (excluding noise).",
            },
            "noise_percent": {
                "direction": "min",
                "description": "Proportion of points labeled as noise by HDBSCAN.",
            },
            "connectivity__normalized_connectivity": {
                "direction": "max",
                "description": "How connected clusters are (higher = more connected).",
            },
            "compactness__silhouette_score": {
                "direction": "max",
                "description": "Silhouette score (higher = better cluster separation).",
            },
            "compactness__calinski_harabasz_score": {
                "direction": "max",
                "description": "Calinski-Harabasz index (higher = better defined clusters).",
            },
            "compactness__davies_bouldin_score": {
                "direction": "min",
                "description": "Davies-Bouldin index (lower = better clustering).",
            },
            "balance__entropy": {
                "direction": "max",
                "description": "Entropy of cluster size distribution (higher = more balanced).",
            },
            "balance__coefficient_of_variation": {
                "direction": "min",
                "description": "Coefficient of variation of cluster sizes (lower = more balanced).",
            },
            "balance__min-to-max-ratio": {
                "direction": "max",
                "description": "Ratio of smallest to largest cluster size (higher = more balanced).",
            },
            "balance__gini-coefficient": {
                "direction": "min",
                "description": "Gini coefficient of cluster sizes (lower = more balanced).",
            },
        }

    # Load metrics CSV
    @property
    def metrics_csv_path(self) -> str:
        return str(ENV.CLUSTERS_DIR / f"metrics-seed={self.seed}.csv")


# Initialize settings
settings = AppSettings()
