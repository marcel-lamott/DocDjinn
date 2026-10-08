import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

from ..config import settings


def ensure_dir(path: Path):
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def get_graph_save_path(
    dataset_name: str,
    graph_type: str,
    embedding_src: str,
    min_cluster_size: int,
    ext: str = "png",
    nclusters: int | None = None,
) -> Path:
    """
    Construct structured path to save graph image.

    Example:
        graphs/tobacco3482/scatter/tobacco3482_scatter_paper_kernel=4_min5.png
    """
    base_dir = Path(settings.graphs_base_dir)
    dataset_dir = base_dir / dataset_name / graph_type
    ensure_dir(dataset_dir)

    filename = f"{dataset_name}_{graph_type}_{embedding_src=}_{min_cluster_size=}_{nclusters=}.{ext}"

    return dataset_dir / filename


def save_plotly_figure(
    fig: go.Figure, save_path: Path, fmt: str = "png", scale: int = 2
):
    """
    Save Plotly figure to disk.
    Requires `kaleido` to be installed.
    """
    ensure_dir(save_path.parent)
    try:
        pio.write_image(fig, str(save_path), format=fmt, scale=scale)
        return str(save_path)
    except Exception as e:
        raise RuntimeError(f"Error saving figure to {save_path}: {e}")
