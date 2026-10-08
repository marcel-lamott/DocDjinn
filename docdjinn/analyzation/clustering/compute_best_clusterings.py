#!/usr/bin/env python3
"""
Compute top N clustering configurations per dataset
from a single global metrics file.

Example:
    python compute_best_clusterings_all_in_one.py \
        --metrics compactness__silhouette_score balance__entropy \
        --directions max max \
        --top 5
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from docdjinn import ENV


# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
METRICS_FILE = ENV.CLUSTERS_DIR / "metrics-seed=42.csv"


# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------
valid_datasets = [
    "cord",
    "doclaynet_4k",
    "ex_docvqa",
    "ex_klc",
    "ex_wiki",
    "funsd",
    "icdar2019",
    "publaynet",
    "rvlcdip",
    "sroie",
    "tobacco3482",
]


def compute_best_per_dataset(df, metrics, directions, top_n=5, filter_datasets=False):
    """Compute top N configs per dataset for selected metrics."""
    results = []

    for dataset, group in df.groupby("dataset_name"):
        if filter_datasets and dataset not in valid_datasets:
            continue

        df_norm = group.copy()
        scaler = MinMaxScaler()

        # normalize + direction handling
        for metric, direction in zip(metrics, directions):
            if metric not in group.columns:
                raise ValueError(
                    f"Metric '{metric}' not found in columns: {list(group.columns)}"
                )

            # normed = scaler.fit_transform(group[[metric]].values)
            normed = group[[metric]].values
            if direction == "min":
                normed = 1 - normed  # flip so higher is better
            df_norm[metric] = normed

        df_norm["final_score"] = df_norm[metrics].mean(axis=1)
        top = df_norm.sort_values("final_score", ascending=False).head(top_n)
        top["dataset_name"] = dataset
        results.append(top)

    combined = pd.concat(results, ignore_index=True)
    return combined


# Compute final embedding ranking
def compute_embedding_ranking(top_df, top_n, filter_datasets):
    """Aggregate top N positions across datasets per embedding type."""
    ranking_list = []

    for dataset, group in top_df.groupby("dataset_name"):
        if filter_datasets and dataset not in valid_datasets:
            continue

        # Sort by final_score descending
        group_sorted = group.sort_values("final_score", ascending=False).reset_index()
        # Assign position-based score
        group_sorted["rank_score"] = (
            top_n - group_sorted.index
        )  # top row = top_n, next = top_n-1 ...
        ranking_list.append(
            group_sorted[["embedding_type", "min_cluster_size", "rank_score"]]
        )

    # Combine all datasets
    all_scores = pd.concat(ranking_list)
    # Sum scores per embedding type
    final_ranking = (
        all_scores.groupby(["embedding_type", "min_cluster_size"], as_index=False)[
            "rank_score"
        ]
        .sum()
        .reset_index()
    )
    final_ranking = final_ranking.sort_values("rank_score", ascending=False)
    final_ranking["final_rank"] = range(1, len(final_ranking) + 1)

    return final_ranking


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute top N clustering configurations per dataset."
    )
    parser.add_argument(
        "--metrics", nargs="+", required=True, help="Metrics to consider"
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        required=True,
        help="Directions for each metric (max/min)",
    )

    parser.add_argument(
        "--filter",
        action="store_true",
        help="If set, only take into account used datasets",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        help="Only consider rows with this min_cluster_size",
    )
    parser.add_argument("--top", type=int, default=5, help="Top N results per dataset")
    parser.add_argument(
        "--outfile", default="best_clusterings_summary.csv", help="Output CSV path"
    )
    args = parser.parse_args()

    if len(args.metrics) != len(args.directions):
        parser.error("Number of metrics and directions must match.")

    print(f"📂 Loading metrics from {METRICS_FILE}")
    df = pd.read_csv(METRICS_FILE)

    # Apply filter if specified
    if args.min_cluster_size is not None:
        df = df[df["min_cluster_size"] == args.min_cluster_size]
        if df.empty:
            print(f"⚠️ No rows found with min_cluster_size = {args.min_cluster_size}")
            return

    print(f"✅ Found {len(df)} rows across {df['dataset_name'].nunique()} datasets")
    combined = compute_best_per_dataset(
        df, args.metrics, args.directions, args.top, filter_datasets=args.filter
    )

    # Select main display columns
    cols_to_show = [
        "dataset_name",
        "embedding_type",
        "min_cluster_size",
        "intermediate_dims",
        "method",
        *args.metrics,
        "final_score",
    ]
    cols_to_show = [c for c in cols_to_show if c in combined.columns]

    print("\n=== Top results per dataset ===")
    for ds, g in combined.groupby("dataset_name"):
        print(f"\n--- {ds} ---")
        print(g[cols_to_show])

    # Save combined summary
    out_path = Path(args.outfile)
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Summary saved to {out_path.resolve()}")

    final_ranking = compute_embedding_ranking(
        combined, top_n=args.top, filter_datasets=args.filter
    )

    print("\n=== Final Ranking of Embedding Types ===")
    print(final_ranking)


if __name__ == "__main__":
    main()
