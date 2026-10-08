import dash_bootstrap_components as dbc
from dash import dcc, html

from .config import settings
from dash import dash_table, dcc, html


def create_control_panel():
    """Create the main control panel with all dropdowns."""
    return html.Div(
        [
            _create_dropdown_row(
                "Dataset",
                "Choose the dataset to analyze",
                "dataset-dropdown",
                [{"label": ds, "value": ds} for ds in settings.dataset_options],
                settings.default_dataset,
            ),
            _create_dropdown_row(
                "Embedding source",
                "Which embedding model to use",
                "embedding-dropdown",
                [{"label": src, "value": src} for src in settings.embedding_sources],
                settings.default_embedding,
            ),
            _create_dropdown_row(
                "Intermediate dimensions",
                "Projection size before clustering",
                "intermediate-dropdown",
                [{"label": str(d), "value": d} for d in settings.intermediate_options],
                settings.default_intermediate,
            ),
            _create_dropdown_row(
                "Minimum cluster size",
                "Smallest allowed cluster size",
                "min-cluster-size-dropdown",
                [
                    {"label": str(d), "value": d}
                    for d in settings.min_cluster_size_options
                ],
                settings.default_min_cluster_size,
            ),
            _create_dropdown_row(
                "Clustering method",
                "Which clustering algorithm to use",
                "method-dropdown",
                [
                    {"label": "HDBSCAN", "value": "hdbscan"},
                ],
                settings.default_method,
            ),
        ],
        style={"gap": "15px"},
    )


def _create_dropdown_row(label, description, dropdown_id, options, value):
    """Create a standardized dropdown row."""
    return dbc.Row(
        [
            dbc.Col(
                [
                    html.Label(label, className="fw-bold"),
                    html.Div(description, className="text-muted small mb-2"),
                ],
                width=7,
            ),
            dbc.Col(
                dcc.Dropdown(
                    id=dropdown_id,
                    options=options,
                    value=value,
                    clearable=False,
                ),
                width=5,
            ),
        ]
    )


def create_visualization_panel():
    """Create main visualization panel with Save Graphs button."""
    return html.Div(
        [
            # Top Row: Control Buttons
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Save Graphs",
                            id="save-all-graphs-btn",
                            color="primary",
                            className="me-2",
                            style={"width": "100%"},
                        ),
                        width=3,
                    ),
                    dbc.Col(html.Div(id="save-feedback", style={"marginTop": "5px"}), width=9),
                ],
                className="mb-3",
            ),

            # Graphs
            dcc.Graph(id="scatter", style={"height": "700px"}),
            dcc.Graph(id="cluster-analysis", style={"height": "800px"}),
        ],
        style={"width": "65%", "display": "inline-block", "verticalAlign": "top"},
    )



def create_document_viewer():
    """Create the document viewer panel."""
    return html.Div(
        [
            html.H4("Selected document"),
            html.Div(id="doc-info", children="Click a point to open its document"),
            html.Iframe(
                id="pdf-viewer",
                src="",
                style={"width": "100%", "height": "700px"},
                hidden=True,
            ),
        ],
        style={
            "width": "34%",
            "display": "inline-block",
            "paddingLeft": "10px",
            "verticalAlign": "top",
        },
    )


def create_metrics_viewer():
    """Create the metrics evaluation panel."""
    return html.Div(
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H2(
                                    "Clustering Evaluation Dashboard",
                                    className="text-center my-3",
                                )
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Alert(
                                        [
                                            html.H5(
                                                "How embeddings and clustering are created",
                                                className="fw-bold",
                                            ),
                                            html.Ol(
                                                [
                                                    html.Li(
                                                        [
                                                            "Embeddings are created akin to ",
                                                            html.A(
                                                                "Unsupervised Document and Template Clustering using Multimodal Embeddings",
                                                                href="https://arxiv.org/pdf/2506.12116",
                                                                target="_blank",
                                                            ),
                                                            ":",
                                                            html.Br(),
                                                            "Get mean of all text tokens, concatenate with image embedding. Image embedding is concatenation of all image patch tokens and then applying a kernel.",
                                                        ]
                                                    ),
                                                    html.Li(
                                                        "Embeddings are clustered in 2 stages: first HDBSCAN, the points labeled as noise (no cluster membership) are then assigned to identified clusters via k-NN"
                                                    ),
                                                ]
                                            ),
                                        ],
                                        color="light",
                                        className="shadow-sm mb-4",
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Metric Selection"),
                                            dbc.CardBody(
                                                [
                                                    html.Label(
                                                        "Choose metrics to evaluate:"
                                                    ),
                                                    dcc.Checklist(
                                                        id="metric-checklist",
                                                        options=[
                                                            {"label": m, "value": m}
                                                            for m in settings.metrics_list.keys()
                                                        ],
                                                        value=[],
                                                        className="mb-3",
                                                    ),
                                                    html.Div(id="direction-selectors"),
                                                    dbc.Button(
                                                        "Compute Best Results",
                                                        id="compute-btn",
                                                        color="primary",
                                                        className="mt-3",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-4",
                                    )
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Top Results"),
                                            dbc.CardBody(
                                                [
                                                    dash_table.DataTable(
                                                        id="results-table",
                                                        page_size=10,
                                                        style_table={
                                                            "overflowX": "auto"
                                                        },
                                                        style_cell={
                                                            "textAlign": "left",
                                                            "padding": "8px",
                                                            "font_family": "monospace",
                                                        },
                                                        style_header={
                                                            "fontWeight": "bold",
                                                            "backgroundColor": "#f8f9fa",
                                                        },
                                                        style_data_conditional=[
                                                            {
                                                                "if": {
                                                                    "state": "active"
                                                                },
                                                                "backgroundColor": "#e9ecef",
                                                                "border": "1px solid #adb5bd",
                                                            },
                                                        ],
                                                    )
                                                ]
                                            ),
                                        ]
                                    )
                                ],
                                width=8,
                            ),
                        ]
                    ),
                ],
                fluid=True,
            )
        ]
    )


def create_overview_panel():
    """Overview table: shows per-embedding metrics for selected dataset."""
    return html.Div(
        [
            html.H4("Embedding Overview", className="mt-4"),
            html.Div(
                [
                    html.P(
                        "Shows summary metrics for each embedding method on the selected dataset:",
                        className="text-muted small",
                    ),
                    dash_table.DataTable(
                        id="embedding-overview-table",
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "font_family": "monospace",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f8f9fa",
                        },
                    ),
                ]
            ),
        ],
        style={"marginTop": "30px"},
    )


def create_app_layout():
    """Create the complete app layout."""
    return html.Div(
        [
            create_control_panel(),
            create_overview_panel(),
            create_visualization_panel(),
            create_document_viewer(),
            create_metrics_viewer(),
        ]
    )
