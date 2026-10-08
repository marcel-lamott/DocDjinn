import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


def map_embedding_name_to_final_name(embedding_name: str):
    match embedding_name:
        case "layout":
            return "layoutlm"
        case "image":
            return "clip"
        case "text":
            return "sentence"
        case "paper_kernel=4":
            return "pooled"
        case "combined":
            return "combined"


def create_scatter_plot(
    df: pd.DataFrame,
    embedding_src: str,
    dataset_name: str,
    min_cluster_size: int,
    n_cluster: int,
) -> go.Figure:
    """Create interactive scatter plot of document embeddings."""
    embedding_src = map_embedding_name_to_final_name(embedding_src)

    # # Force categorical colors if labels are numeric
    # df = df.copy()  # Avoid modifying original
    # df["label"] = df["label"].astype(str)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        # labels={"label": ""},
        hover_data={"index": True, "label": True, "doc_id": True},
        # title=f"{dataset_name}: '{embedding_src}' Embeddings, κ={min_cluster_size}, {n_cluster} Clusters",
    )
    margin = 0

    fig.update_traces(marker=dict(size=7, showscale=False), customdata=df["index"])
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        # legend_title="Cluster",
        showlegend=False,
        coloraxis_showscale=False,
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


"""This function is used to display the analysis plots as subplot i.e. one figure containing all plots"""


def create_cluster_analysis_plot(
    cluster_df: pd.DataFrame,
    dataset_name: str,
    cluster_labels: np.ndarray,
) -> go.Figure:
    """Create comprehensive cluster analysis visualization with clickable clusters."""
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Cluster Sizes",
            "Cluster Variances",
            "Size vs Variance",
            "Distribution",
        ),
        specs=[
            [{"type": "bar"}],
            [{"type": "bar"}],
            [{"type": "scatter"}],
            [{"type": "histogram"}],
        ],
    )

    # Prepare cluster indices for click events
    cluster_indices = {}
    for cluster_id in cluster_df["cluster_id"]:
        indices = np.where(cluster_labels == cluster_id)[0].tolist()
        cluster_indices[cluster_id] = indices

    # Plot 1: Cluster sizes (clickable)
    fig.add_trace(
        go.Bar(
            x=cluster_df["cluster_id"],
            y=cluster_df["size"],
            name="Size",
            customdata=[cluster_indices[cid] for cid in cluster_df["cluster_id"]],
            hovertemplate="Cluster %{x}<br>Size: %{y}<br>Click to view images<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Plot 2: Cluster variances
    fig.add_trace(
        go.Bar(
            x=cluster_df["cluster_id"],
            y=cluster_df["variance"],
            customdata=[cluster_indices[cid] for cid in cluster_df["cluster_id"]],
            name="Variance",
        ),
        row=2,
        col=1,
    )

    # Plot 3: Size vs Variance scatter (clickable)
    fig.add_trace(
        go.Scatter(
            x=cluster_df["size"],
            y=cluster_df["variance"],
            mode="markers",
            text=cluster_df["cluster_id"],
            name="Clusters",
            customdata=[cluster_indices[cid] for cid in cluster_df["cluster_id"]],
            hovertemplate="Cluster %{text}<br>Size: %{x}<br>Variance: %{y}<br>Click to view images<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Plot 4: Size distribution
    fig.add_trace(
        go.Histogram(x=cluster_df["size"], name="Size Distribution"),
        row=4,
        col=1,
    )

    fig.update_layout(
        title_text=f"Cluster Analysis for {dataset_name}",
        showlegend=False,
        height=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
    )

    _update_subplot_axes(fig)
    return fig


def _update_subplot_axes(fig: go.Figure) -> None:
    """Update axes labels for all subplots."""
    fig.update_xaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Size", row=1, col=1)
    fig.update_xaxes(title_text="Cluster ID", row=2, col=1)
    fig.update_yaxes(title_text="Variance", row=2, col=1)
    fig.update_xaxes(title_text="Size", row=3, col=1)
    fig.update_yaxes(title_text="Variance", row=3, col=1)
    fig.update_xaxes(title_text="Size", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)


"""This function is used to save cluster analysis plots separately not as a single plot"""


def generate_individual_cluster_plots(cluster_df, dataset_name: str) -> dict:
    """
    Given cluster_df (DataFrame with columns 'cluster_id', 'size', 'variance'),
    return a dict of plot_name -> go.Figure, one per subplot:
      - cluster_sizes
      - cluster_variances
      - size_vs_variance
      - distribution

    Note: dataset_name is unused for plotting but kept for potential titles.
    """
    plots = {}

    # Ensure expected columns exist
    if not {"cluster_id", "size", "variance"}.issubset(cluster_df.columns):
        raise ValueError(
            "cluster_df must contain 'cluster_id', 'size', and 'variance' columns"
        )

    # Cluster Sizes (bar)
    fig_sizes = go.Figure()
    fig_sizes.add_trace(
        go.Bar(
            x=cluster_df["cluster_id"],
            y=cluster_df["size"],
            name="Size",
        )
    )
    fig_sizes.update_layout(
        title_text=f"Cluster Sizes{' — ' + dataset_name if dataset_name else ''}",
        xaxis_title="Cluster ID",
        yaxis_title="Size",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    plots["cluster_sizes"] = fig_sizes

    # Cluster Variances (bar)
    fig_var = go.Figure()
    fig_var.add_trace(
        go.Bar(
            x=cluster_df["cluster_id"],
            y=cluster_df["variance"],
            name="Variance",
        )
    )
    fig_var.update_layout(
        title_text=f"Cluster Variances{' — ' + dataset_name if dataset_name else ''}",
        xaxis_title="Cluster ID",
        yaxis_title="Variance",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    plots["cluster_variances"] = fig_var

    # Size vs Variance (scatter)
    fig_sv = go.Figure()
    fig_sv.add_trace(
        go.Scatter(
            x=cluster_df["size"],
            y=cluster_df["variance"],
            mode="markers",
            text=cluster_df["cluster_id"],
            name="Size vs Variance",
        )
    )
    fig_sv.update_layout(
        title_text=f"Size vs Variance{' — ' + dataset_name if dataset_name else ''}",
        xaxis_title="Size",
        yaxis_title="Variance",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    plots["size_vs_variance"] = fig_sv

    # Distribution (histogram)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=cluster_df["size"], name="Size Distribution"))
    fig_dist.update_layout(
        title_text=f"Distribution{' — ' + dataset_name if dataset_name else ''}",
        xaxis_title="Size",
        yaxis_title="Count",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    plots["distribution"] = fig_dist

    return plots
