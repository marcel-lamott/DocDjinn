import time
import numpy as np
from dash import Input, Output, html, callback_context, dcc
import dash_bootstrap_components as dbc
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go
from .utils.save_utils import get_graph_save_path, save_plotly_figure
from sklearn.preprocessing import MinMaxScaler
from .data_manager import data_manager
from .visualizations import (
    create_scatter_plot,
    create_cluster_analysis_plot,
    generate_individual_cluster_plots,
)
from .config import settings


def register_callbacks(app):
    """Register all Dash callbacks."""

    @app.callback(
        [Input("dataset-dropdown", "value")],
    )
    def update_dataset(dataset_name: str):
        """Update global dataset when dropdown changes."""
        data_manager.load_dataset(dataset_name)

    @app.callback(
        [
            Output("scatter", "figure"),
            Output("cluster-analysis", "figure"),
        ],
        [
            Input("dataset-dropdown", "value"),
            Input("intermediate-dropdown", "value"),
            Input("min-cluster-size-dropdown", "value"),
            Input("embedding-dropdown", "value"),
            Input("method-dropdown", "value"),
        ],
    )
    def update_visualizations(
        dataset_name: str,
        intermediate_dims: int,
        min_cluster_size: int,
        embedding_src: str,
        method: str,
    ):
        """Update scatter plot and cluster analysis when parameters change."""
        # Get cluster data
        cluster_data = data_manager.get_cluster_data(
            dataset_name, embedding_src, intermediate_dims, min_cluster_size, method
        )

        labels = cluster_data["cluster_data"]["cluster_labels"]
        soft_clusters = cluster_data["cluster_data"]["soft_clusters"]
        noise_mask = cluster_data["cluster_data"].get(
            "noise_mask", np.array([False] * len(labels))
        )
        cluster_stats = cluster_data["cluster_stats"]
        emb_2d = cluster_data["emb_2d"]
        sample_ids = cluster_data["sample_ids"]

        # Create scatter plot dataframe
        df = data_manager.create_scatter_dataframe(
            labels, emb_2d, soft_clusters, sample_ids, noise_mask
        )

        # Create visualizations
        scatter_fig = create_scatter_plot(
            df,
            embedding_src,
            dataset_name,
            min_cluster_size,
            len(set(cluster_data["cluster_data"]["cluster_labels"])),
        )
        cluster_analysis_fig = create_cluster_analysis_plot(
            cluster_stats, dataset_name, labels
        )

        return scatter_fig, cluster_analysis_fig

    @app.callback(
        [
            Output("pdf-viewer", "src"),
            Output("pdf-viewer", "hidden"),
            Output("doc-info", "children"),
        ],
        [
            Input("scatter", "clickData"),
            Input("cluster-analysis", "clickData"),
            Input("dataset-dropdown", "value"),
            Input("intermediate-dropdown", "value"),
            Input("min-cluster-size-dropdown", "value"),
            Input("embedding-dropdown", "value"),
            Input("method-dropdown", "value"),
        ],
        prevent_initial_call=False,
    )
    def display_documents(
        scatter_click: dict,
        cluster_click: dict,
        dataset_name: str,
        intermediate_dims: int,
        min_cluster_size: int,
        embedding_src: str,
        method: str,
    ):
        """Handle document display for both single and cluster clicks."""
        ctx = callback_context
        if not ctx.triggered:
            return "", True, "Click a point or cluster to view documents"

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        cluster_data = data_manager.get_cluster_data(
            dataset_name, embedding_src, intermediate_dims, min_cluster_size, method
        )
        labels = cluster_data["cluster_data"]["cluster_labels"]
        sample_ids = cluster_data["sample_ids"]
        # Handle cluster click - show grid of documents
        if trigger_id == "cluster-analysis" and cluster_click:
            return _handle_cluster_click(cluster_click, labels, sample_ids)

        # Handle single document click
        if trigger_id == "scatter" and scatter_click:
            return _handle_scatter_click(scatter_click, labels, sample_ids)

        return "", True, "Click a point or cluster to view documents"

    @app.callback(
        Output("direction-selectors", "children"), Input("metric-checklist", "value")
    )
    def update_direction_selectors(selected_metrics: list):
        """Show dropdowns for choosing min/max and a description for each selected metric."""
        controls = []
        for m in selected_metrics:
            description = settings.metrics_list[m]["description"]
            controls.append(
                dbc.Card(
                    [
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label(m, className="fw-bold"),
                                                html.Div(
                                                    description,
                                                    className="text-muted small mb-2",
                                                ),
                                            ],
                                            width=7,
                                        ),
                                        dbc.Col(
                                            dcc.Dropdown(
                                                id={
                                                    "type": "direction-dropdown",
                                                    "metric": m,
                                                },
                                                options=[
                                                    {
                                                        "label": "Maximize",
                                                        "value": "max",
                                                    },
                                                    {
                                                        "label": "Minimize",
                                                        "value": "min",
                                                    },
                                                ],
                                                value=settings.metrics_list[m][
                                                    "direction"
                                                ],
                                                clearable=False,
                                            ),
                                            width=5,
                                        ),
                                    ]
                                )
                            ]
                        )
                    ],
                    className="mb-2",
                )
            )
        return controls

    @app.callback(
        Output("results-table", "data"),
        Output("results-table", "columns"),
        Input("compute-btn", "n_clicks"),
        State("metric-checklist", "value"),
        State({"type": "direction-dropdown", "metric": dash.ALL}, "value"),
        State({"type": "direction-dropdown", "metric": dash.ALL}, "id"),
    )
    def compute_best_results(n_clicks, selected_metrics, directions, ids):
        if n_clicks == 0 or not selected_metrics:
            return [], []

        # Map metrics to directions
        metric_directions = {i["metric"]: d for i, d in zip(ids, directions)}

        # Copy for normalization but keep original df for output
        df = data_manager.metrics.copy()
        df_norm = df.copy()

        for col in selected_metrics:
            scaler = MinMaxScaler()
            values = df[[col]].values
            normed = scaler.fit_transform(values)
            if metric_directions[col] == "min":
                normed = 1 - normed  # flip so higher is better
            df_norm[col] = normed

        df_norm["final_score"] = df_norm[selected_metrics].mean(axis=1)

        # Select top rows
        best_idx = df_norm.sort_values("final_score", ascending=False).index
        best = df.loc[best_idx].copy()
        best["final_score"] = df_norm.loc[best_idx, "final_score"]

        # Convert to table
        columns = [{"name": c, "id": c} for c in best.columns]
        data = best.to_dict("records")
        return data, columns

    @app.callback(
        Output("embedding-overview-table", "data"),
        Output("embedding-overview-table", "columns"),
        Input("dataset-dropdown", "value"),
        Input("min-cluster-size-dropdown", "value"),
    )
    def update_embedding_overview(dataset_name, min_cluster_size):
        """Compute summary metrics (num clusters, silhouette, entropy) for all embeddings."""
        # Placeholder for results
        rows = []

        for embedding_src in settings.embedding_sources:
            try:
                # Load cluster data for each embedding
                cluster_data = data_manager.get_cluster_data(
                    dataset_name,
                    embedding_src,
                    settings.default_intermediate,
                    min_cluster_size,
                    settings.default_method,
                )

                labels = cluster_data["cluster_data"]["cluster_labels"]

                # Number of clusters (excluding noise if labeled as -1)
                valid_labels = labels[labels >= 0]
                n_clusters = len(np.unique(valid_labels))

                # Silhouette score (skip if only 1 cluster)
                if n_clusters > 1:
                    from sklearn.metrics import silhouette_score

                    emb = cluster_data["emb_2d"]  # or full embeddings if available
                    sil = silhouette_score(emb, labels)
                else:
                    sil = np.nan

                # Entropy of cluster distribution
                from scipy.stats import entropy

                cluster_sizes = np.bincount(valid_labels)
                probs = cluster_sizes / cluster_sizes.sum()
                ent = entropy(probs)

                rows.append(
                    dict(
                        embedding=embedding_src,
                        n_clusters=n_clusters,
                        silhouette=round(sil, 3) if not np.isnan(sil) else "—",
                        entropy=round(ent, 3),
                    )
                )

            except Exception as e:
                rows.append(
                    dict(
                        embedding=embedding_src,
                        n_clusters="Error",
                        silhouette="Error",
                        entropy=str(e),
                    )
                )

        df = pd.DataFrame(rows)
        columns = [{"name": c.replace("_", " ").title(), "id": c} for c in df.columns]
        return df.to_dict("records"), columns

    @app.callback(
        Output("save-feedback", "children"),
        Input("save-all-graphs-btn", "n_clicks"),
        [
            State("scatter", "figure"),
            State("cluster-analysis", "figure"),
            State("dataset-dropdown", "value"),
            State("embedding-dropdown", "value"),
            State("min-cluster-size-dropdown", "value"),
            State("intermediate-dropdown", "value"),
            State("method-dropdown", "value"),
        ],
        prevent_initial_call=True,
    )
    def save_all_graphs(
        n_clicks,
        scatter_fig,
        cluster_fig,
        dataset_name,
        embedding_src,
        min_cluster_size,
        intermediate_dims,
        method,
    ):
        """Save scatter plot and the cluster-analysis subplots separately."""
        if not dataset_name or not embedding_src:
            return dbc.Alert("Missing dataset or embedding source.", color="danger")

        saved_paths = []
        errors = []
        ext = "pdf"

        cluster_data = data_manager.get_cluster_data(
            dataset_name, embedding_src, intermediate_dims, min_cluster_size, method
        )

        if scatter_fig:
            try:
                nclusters = len(set(cluster_data["cluster_data"]["cluster_labels"]))
                fig = go.Figure(scatter_fig)
                path = get_graph_save_path(
                    dataset_name,
                    "scatter",
                    embedding_src,
                    min_cluster_size,
                    ext=ext,
                    nclusters=nclusters,
                )
                saved = save_plotly_figure(fig, path, fmt=ext)
                saved_paths.append(saved)
            except Exception as e:
                errors.append(f"Scatter: {e}")

        try:
            cluster_data = data_manager.get_cluster_data(
                dataset_name, embedding_src, intermediate_dims, min_cluster_size, method
            )
            nclusters = len(set(cluster_data["cluster_data"]["cluster_labels"]))
            cluster_stats = cluster_data["cluster_stats"]

            """This function is generating cluster analysis figurs sepratley
            because in the webapp there is on plot having further subplots
            if I save that which I tried it looks odd like text is overlapping
            so I created separate plots for it and save that separately."""
            per_plot_figs = generate_individual_cluster_plots(
                cluster_stats, dataset_name=dataset_name
            )

            for plot_name, fig in per_plot_figs.items():
                try:
                    path = get_graph_save_path(
                        dataset_name,
                        plot_name,
                        embedding_src,
                        min_cluster_size,
                        ext=ext,
                    )
                    saved = save_plotly_figure(fig, path, fmt=ext)
                    saved_paths.append(saved)
                except Exception as e_plot:
                    errors.append(f"{plot_name}: {e_plot}")
        except Exception as e:
            errors.append(f"Cluster subplots generation failed: {e}")

        if errors and saved_paths:
            msg = "<br>".join(errors)
            return dbc.Alert(
                f"Some graphs saved, some failed:<br>{msg}",
                color="warning",
                dismissable=True,
            )
        elif errors and not saved_paths:
            msg = "<br>".join(errors)
            return dbc.Alert(
                f"Failed to save graphs:<br>{msg}", color="danger", dismissable=True
            )
        else:
            return dbc.Alert(
                f"Saved {len(saved_paths)} files:<br>" + "<br>".join(saved_paths),
                color="success",
                dismissable=True,
            )


def _handle_cluster_click(cluster_click: dict, labels, sample_ids):
    """Handle cluster visualization clicks."""
    try:
        cluster_indices = cluster_click["points"][0]["customdata"]
        if not cluster_indices:
            return "", True, "No documents in this cluster"

        # Limit documents for performance
        display_indices = cluster_indices[:12]

        # Create grid URL
        doc_ids_str = ",".join(str(sample_ids[idx]) for idx in display_indices)
        timestamp = int(time.time() * 1000)
        cluster_id = labels[display_indices[0]]

        return (
            "",
            True,
            html.Div(
                [
                    html.H5(f"Cluster {cluster_id} ({len(cluster_indices)} documents)"),
                    html.Img(
                        src=f"/cluster_grid/{doc_ids_str}?t={timestamp}",
                        style={
                            "width": "100%",
                            "max-height": "600px",
                            "object-fit": "contain",
                            "border": "1px solid #ddd",
                            "border-radius": "4px",
                        },
                    ),
                ]
            ),
        )
    except Exception as e:
        print(f"Error displaying cluster: {e}")
        return "", True, f"Error displaying cluster: {e}"


def _handle_scatter_click(scatter_click, labels, sample_ids):
    """Handle scatter plot clicks."""
    try:
        idx = int(scatter_click["points"][0]["pointIndex"])
        sample_id = str(sample_ids[idx])
        return (
            f"/image/{sample_id}",
            False,
            html.Div(
                [
                    html.P(f"Index: {idx}"),
                    html.P(f"Cluster: {labels[idx]}"),
                    html.P(f"DocID: {idx}"),
                ]
            ),
        )
    except Exception as e:
        return "", True, f"Error selecting point: {e}"
