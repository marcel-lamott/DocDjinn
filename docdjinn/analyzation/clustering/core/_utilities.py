from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from docdjinn.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image

logger = get_logger(__name__)


if TYPE_CHECKING:
    import torch


class EmbeddingType(str, enum.Enum):
    """
    Enum for different types of embeddings used in DocDjinn.
    """

    layout = "layout"
    image = "image"
    text = "text"
    combined = "combined"
    paper = "paper_kernel=4"


def _glob_clustering_output_paths(output_dir: str | Path):
    """
    List all clustering output files in the specified directory.

    Args:
        output_dir (str | Path): The directory to search for clustering output files.
            This must point to the `output_directory/dataset_name/embedding_type` level.
    """
    output_path = Path(output_dir)
    return list(
        output_path.glob("method=*_clusters_ind=*_hmcs=*_hm=*_do_knn=*_knn=*.npy")
    )


def _get_clustering_output_path(
    output_dir: str | Path,
    intermediate_num_dims: int,
    hdbscan_min_cluster_size: int = 10,
    hdbscan_metric: str = "euclidean",
    do_knn: bool = True,
    k_nn_n_neighbors: int = 5,
    method: str = "hdbscan",
):
    """
    Generate a standardized file path for clustering output results.

    This function creates a descriptive filename that encodes all the clustering
    parameters used, allowing for easy identification and retrieval of clustering
    results based on the specific configuration.

    Args:
        output_dir (str | Path): The base directory where clustering results will be saved.
            This must point to the `output_directory/dataset_name/embedding_type` level.
        intermediate_num_dims (int): The number of dimensions used in intermediate processing.
        hdbscan_min_cluster_size (int, optional): Minimum cluster size for HDBSCAN algorithm.
            Defaults to 10.
        hdbscan_metric (str, optional): Distance metric used by HDBSCAN algorithm.
            Defaults to "euclidean".
        do_knn (bool, optional): Whether to apply k-nearest neighbors processing.
            Defaults to True.
        k_nn_n_neighbors (int, optional): Number of neighbors for k-NN algorithm.
            Defaults to 5.
        method (str, optional): The clustering method being used. Defaults to "hdbscan".

    Returns:
        Path: A Path object pointing to the clustering output file with encoded parameters
              in the filename format: method={method}_clusters_ind={intermediate_num_dims}_
              hmcs={hdbscan_min_cluster_size}_hm={hdbscan_metric}_do_knn={do_knn}_
              knn={k_nn_n_neighbors}.npy
    """
    return (
        output_dir
        / f"method={method}_clusters_ind={intermediate_num_dims}_hmcs={hdbscan_min_cluster_size}_hm={hdbscan_metric}_do_knn={do_knn}_knn={k_nn_n_neighbors}.npy"
    )


def _save_clustering_metrics(
    output_dir: str | Path,
    dataset_name: str,
    hdbscan_min_cluster_size: int,
    intermediate_num_dims: int,
    hdbscan_metric: str,
    k_nn_n_neighbors: int,
    method: str,
    embedding_type: "EmbeddingType",
    embeddings: "np.ndarray",
    cluster_metrics: dict,
    num_clusters: int,
    num_noise: int,
    seed: int,
    do_knn: bool = True,
) -> None:
    import hashlib

    import torch

    cnt = embeddings.shape[0]
    noise_percent = num_noise / float(cnt)
    noise_percent = (
        noise_percent.item()
        if isinstance(noise_percent, torch.Tensor)
        else noise_percent
    )
    metrics_row = {
        "dataset_name": dataset_name,
        "embedding_type": embedding_type.value,
        "min_cluster_size": hdbscan_min_cluster_size,
        "intermediate_dims": intermediate_num_dims,
        "hdbscan_metric": hdbscan_metric,
        "k_nn_n_neighbors": k_nn_n_neighbors,
        "num_clusters": num_clusters,
        "num_noise": num_noise,
        "noise_percent": noise_percent,
        "do_knn": do_knn,
        "method": method,
    }

    # Add cluster metrics to the row
    for cat, items in cluster_metrics.items():
        for k, v in items.items():
            metrics_row[f"{cat}__{k}"] = v

    # Generate unique hash based on configuration parameters only (excluding results)
    config_items = {
        "dataset_name": dataset_name,
        "embedding_type": embedding_type.value,
        "min_cluster_size": hdbscan_min_cluster_size,
        "intermediate_dims": intermediate_num_dims,
        "hdbscan_metric": hdbscan_metric,
        "k_nn_n_neighbors": k_nn_n_neighbors,
        "seed": seed,
        "do_knn": do_knn,
        "method": method,
    }
    row_hash = hashlib.md5(str(sorted(config_items.items())).encode()).hexdigest()
    metrics_row["row_hash"] = row_hash

    # Save metrics
    metrics_path = Path(output_dir) / f"metrics-seed={seed}.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df = df[df["row_hash"] != row_hash]
        df = pd.concat([df, pd.DataFrame([metrics_row])], ignore_index=True)
    else:
        df = pd.DataFrame([metrics_row])

    logger.info(f"Saving clustering metrics to {metrics_path}...")
    df.to_csv(metrics_path, index=False)


def _visualize_images_grid(
    images: list[np.ndarray | "Image"],
    save_path: str | Path,
    nrow: int = 8,
    title: str | None = None,
    figsize: tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> None:
    """
    Create and save an image grid using torchvision's make_grid utility.

    Args:
        images: List of numpy arrays or PIL images to arrange in grid
        save_path: Path where the grid image will be saved
        nrow: Number of images displayed in each row of the grid
        title: Optional title for the saved image
        figsize: Figure size for matplotlib
        dpi: DPI for saved image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import resize
    from torchvision.utils import make_grid

    # Convert inputs to tensors
    tensor_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Handle different numpy array formats
            if img.ndim == 2:  # Grayscale
                img = np.expand_dims(img, axis=0)  # Add channel dimension
            elif img.ndim == 3 and img.shape[2] == 3:  # RGB with channels last
                img = np.transpose(img, (2, 0, 1))  # Convert to channels first
            elif img.ndim == 3 and img.shape[0] in [1, 3]:  # Already channels first
                pass
            else:
                raise ValueError(f"Unsupported numpy array shape: {img.shape}")

            tensor = torch.from_numpy(img).float()
        else:  # PIL Image
            transform = transforms.ToTensor()
            tensor = transform(img)

        tensor = resize(tensor, size=(512, 512))  # Resize to fixed size
        tensor_images.append(tensor)

    # Stack all tensors
    batch_tensor = torch.stack(tensor_images)

    # Create grid
    grid = make_grid(
        batch_tensor,
        nrow=nrow,
    )

    # Convert to numpy for matplotlib (channels last)
    grid_np = grid.permute(1, 2, 0).numpy()

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid_np)
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=16, pad=20)

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close()

    logger.info(f"Image grid saved to {save_path}")


def _load_pdfs_to_pil_images(pdf_paths: list[str | Path]) -> list["Image"]:
    """
    Loads a list of PDF document paths to PIL Images by rendering the first page of each PDF as PNG.

    Args:
        pdf_paths: List of paths to PDF files

    Returns:
        List of PIL Image objects, one for each PDF's first page
    """
    from pdf2image import convert_from_path

    pil_images = []

    for pdf_path in pdf_paths:
        try:
            # Convert first page of PDF to PIL Image
            images = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=72)

            if images:
                pil_images.append(images[0])
            else:
                logger.warning(f"No images converted from PDF: {pdf_path}")

        except Exception as e:
            logger.error(f"Failed to convert PDF {pdf_path}: {e}")
            continue

    return pil_images
