"""
Supplementary analysis utilities for KIE GT comparison
Additional metrics for CVPR paper
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

from docdjinn import ENV


def compute_jensen_shannon_divergence(
    real_stats: Dict, synth_stats: Dict
) -> Dict[str, float]:
    """
    Compute Jensen-Shannon divergence for entity distributions.
    Lower is better (0 = identical distributions, 1 = completely different).
    """
    all_classes = sorted(
        set(
            list(real_stats["entity_counts"].keys())
            + list(synth_stats["entity_counts"].keys())
        )
    )

    real_counts = np.array(
        [real_stats["entity_counts"].get(cls, 0) for cls in all_classes]
    )
    synth_counts = np.array(
        [synth_stats["entity_counts"].get(cls, 0) for cls in all_classes]
    )

    # Normalize to probabilities
    real_probs = (
        real_counts / real_counts.sum() if real_counts.sum() > 0 else real_counts
    )
    synth_probs = (
        synth_counts / synth_counts.sum() if synth_counts.sum() > 0 else synth_counts
    )

    # JS divergence
    js_div = jensenshannon(real_probs, synth_probs)

    return {
        "overall_js_divergence": js_div,
        "overall_kl_divergence": entropy(real_probs, synth_probs),
    }


def compute_spatial_coverage_metrics(real_stats: Dict, synth_stats: Dict) -> Dict:
    """
    Compute spatial coverage metrics comparing how well synthetic data
    covers the spatial distribution of real data.
    """
    results = {}

    for entity_class in real_stats["spatial_distributions"].keys():
        real_pos = real_stats["spatial_distributions"].get(entity_class, [])
        synth_pos = synth_stats["spatial_distributions"].get(entity_class, [])

        if len(real_pos) == 0 or len(synth_pos) == 0:
            continue

        real_x = np.array([p[0] for p in real_pos])
        real_y = np.array([p[1] for p in real_pos])
        synth_x = np.array([p[0] for p in synth_pos])
        synth_y = np.array([p[1] for p in synth_pos])

        # Mean absolute difference in centroids
        real_centroid = (real_x.mean(), real_y.mean())
        synth_centroid = (synth_x.mean(), synth_y.mean())
        centroid_distance = np.sqrt(
            (real_centroid[0] - synth_centroid[0]) ** 2
            + (real_centroid[1] - synth_centroid[1]) ** 2
        )

        # Standard deviation comparison
        real_std = (real_x.std(), real_y.std())
        synth_std = (synth_x.std(), synth_y.std())
        std_diff = (abs(real_std[0] - synth_std[0]), abs(real_std[1] - synth_std[1]))

        results[entity_class] = {
            "centroid_distance": centroid_distance,
            "std_x_diff": std_diff[0],
            "std_y_diff": std_diff[1],
            "real_centroid": real_centroid,
            "synth_centroid": synth_centroid,
            "real_std": real_std,
            "synth_std": synth_std,
        }

    return results


def plot_entity_co_occurrence_matrix(
    real_stats: Dict, synth_stats: Dict, samples_real, samples_synth, output_prefix: str
):
    """
    Plot co-occurrence matrices showing which entities appear together in documents.
    Useful for understanding document structure preservation.
    """

    # Build co-occurrence matrices
    def build_cooccurrence(samples):
        from itertools import combinations

        co_occur = defaultdict(int)
        all_classes = set()

        for sample in samples:
            annotation = sample.annotations[0]
            word_labels_names = annotation.word_labels.name

            # Get unique entity classes in this sample
            entities_in_sample = set()
            for label in word_labels_names:
                if label.startswith("B-") or label.startswith("I-"):
                    entity_class = label[2:]
                    entities_in_sample.add(entity_class)
                    all_classes.add(entity_class)

            # Count co-occurrences
            for pair in combinations(sorted(entities_in_sample), 2):
                co_occur[pair] += 1

        return co_occur, sorted(all_classes)

    real_cooccur, real_classes = build_cooccurrence(samples_real)
    synth_cooccur, synth_classes = build_cooccurrence(samples_synth)

    all_classes = sorted(set(real_classes + synth_classes))
    n = len(all_classes)

    # Build matrices
    real_matrix = np.zeros((n, n))
    synth_matrix = np.zeros((n, n))

    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    for (cls1, cls2), count in real_cooccur.items():
        i, j = class_to_idx[cls1], class_to_idx[cls2]
        real_matrix[i, j] = count
        real_matrix[j, i] = count

    for (cls1, cls2), count in synth_cooccur.items():
        i, j = class_to_idx[cls1], class_to_idx[cls2]
        synth_matrix[i, j] = count
        synth_matrix[j, i] = count

    # Normalize
    real_matrix = (
        real_matrix / real_matrix.sum() if real_matrix.sum() > 0 else real_matrix
    )
    synth_matrix = (
        synth_matrix / synth_matrix.sum() if synth_matrix.sum() > 0 else synth_matrix
    )

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Real
    sns.heatmap(
        real_matrix,
        annot=False,
        cmap="Blues",
        ax=axes[0],
        xticklabels=all_classes,
        yticklabels=all_classes,
        cbar_kws={"label": "Frequency"},
    )
    axes[0].set_title("Real Data Co-occurrence", fontsize=14, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    # Synthetic
    sns.heatmap(
        synth_matrix,
        annot=False,
        cmap="Oranges",
        ax=axes[1],
        xticklabels=all_classes,
        yticklabels=all_classes,
        cbar_kws={"label": "Frequency"},
    )
    axes[1].set_title("Synthetic Data Co-occurrence", fontsize=14, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

    # Difference
    diff_matrix = np.abs(real_matrix - synth_matrix)
    sns.heatmap(
        diff_matrix,
        annot=False,
        cmap="Reds",
        ax=axes[2],
        xticklabels=all_classes,
        yticklabels=all_classes,
        cbar_kws={"label": "Abs Difference"},
    )
    axes[2].set_title("Absolute Difference", fontsize=14, fontweight="bold")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_yticklabels(axes[2].get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(
        ENV.KIE_GT_ANALYZATION_DIR / f"{output_prefix}_cooccurrence_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved: {ENV.KIE_GT_ANALYZATION_DIR / output_prefix}_cooccurrence_matrix.png"
    )
    plt.close()


def plot_document_level_statistics(
    real_stats: Dict, synth_stats: Dict, output_prefix: str
):
    """
    Plot document-level statistics (entities per document, etc.).
    """
    # Compute entities per document
    real_entities_per_doc = []
    synth_entities_per_doc = []

    for entity_class in real_stats["entity_counts_per_sample"].keys():
        real_entities_per_doc.extend(
            real_stats["entity_counts_per_sample"][entity_class]
        )

    for entity_class in synth_stats["entity_counts_per_sample"].keys():
        synth_entities_per_doc.extend(
            synth_stats["entity_counts_per_sample"][entity_class]
        )

    # Aggregate by document
    n_docs_real = real_stats["total_samples"]
    n_docs_synth = synth_stats["total_samples"]

    # Reshape data properly - sum across entity types per document
    # We need to reorganize the per_sample data
    all_classes = sorted(
        set(
            list(real_stats["entity_counts_per_sample"].keys())
            + list(synth_stats["entity_counts_per_sample"].keys())
        )
    )

    # Get max document count
    max_docs = max(
        max(
            [
                len(real_stats["entity_counts_per_sample"].get(cls, []))
                for cls in all_classes
            ],
            default=0,
        ),
        max(
            [
                len(synth_stats["entity_counts_per_sample"].get(cls, []))
                for cls in all_classes
            ],
            default=0,
        ),
    )

    real_total_per_doc = np.zeros(n_docs_real)
    synth_total_per_doc = np.zeros(n_docs_synth)

    for entity_class in all_classes:
        real_counts = real_stats["entity_counts_per_sample"].get(entity_class, [])
        synth_counts = synth_stats["entity_counts_per_sample"].get(entity_class, [])

        real_total_per_doc[: len(real_counts)] += np.array(real_counts)
        synth_total_per_doc[: len(synth_counts)] += np.array(synth_counts)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(real_total_per_doc, bins=20, alpha=0.6, label="Real", density=True)
    axes[0].hist(
        synth_total_per_doc, bins=20, alpha=0.6, label="Synthetic", density=True
    )
    axes[0].axvline(
        real_total_per_doc.mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Real μ={real_total_per_doc.mean():.1f}",
    )
    axes[0].axvline(
        synth_total_per_doc.mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Synth μ={synth_total_per_doc.mean():.1f}",
    )
    axes[0].set_xlabel("Total Entities per Document", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title(
        "Distribution of Entities per Document", fontsize=14, fontweight="bold"
    )
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Cumulative distribution
    real_sorted = np.sort(real_total_per_doc)
    synth_sorted = np.sort(synth_total_per_doc)
    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    synth_cdf = np.arange(1, len(synth_sorted) + 1) / len(synth_sorted)

    axes[1].plot(real_sorted, real_cdf, label="Real", linewidth=2)
    axes[1].plot(synth_sorted, synth_cdf, label="Synthetic", linewidth=2)
    axes[1].set_xlabel("Total Entities per Document", fontsize=12)
    axes[1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[1].set_title("Cumulative Distribution", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        ENV.KIE_GT_ANALYZATION_DIR / f"{output_prefix}_document_statistics.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(
        f"Saved: {ENV.KIE_GT_ANALYZATION_DIR / output_prefix}_document_statistics.png"
    )
    plt.close()


def generate_latex_table(stats_df, divergence_metrics, output_prefix: str):
    """
    Generate LaTeX table for paper.
    """
    latex_str = "\\begin{table}[t]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Comparison of Entity Distributions between Real and Synthetic Datasets}\n"
    latex_str += "\\label{tab:entity_comparison}\n"
    latex_str += "\\begin{tabular}{lrrrrr}\n"
    latex_str += "\\toprule\n"
    latex_str += (
        "Entity Class & Real & Synth & Real (\\%) & Synth (\\%) & p-value \\\\\n"
    )
    latex_str += "\\midrule\n"

    total_real = stats_df["Real Count"].sum()
    total_synth = stats_df["Synth Count"].sum()

    for _, row in stats_df.iterrows():
        entity = row["Entity Class"]
        real_count = int(row["Real Count"])
        synth_count = int(row["Synth Count"])
        real_pct = (real_count / total_real * 100) if total_real > 0 else 0
        synth_pct = (synth_count / total_synth * 100) if total_synth > 0 else 0
        p_val = row["Mann-Whitney p-value"]

        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "---"
        if not np.isnan(p_val) and p_val < 0.001:
            p_str = "$<$0.001"

        latex_str += f"{entity} & {real_count} & {synth_count} & {real_pct:.1f} & {synth_pct:.1f} & {p_str} \\\\\n"

    latex_str += "\\midrule\n"
    latex_str += f"Total & {total_real} & {total_synth} & 100.0 & 100.0 & --- \\\\\n"
    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    # Add divergence metrics as separate note
    latex_str += "\n% Divergence Metrics:\n"
    latex_str += f"% JS Divergence: {divergence_metrics['overall_js_divergence']:.4f}\n"
    latex_str += f"% KL Divergence: {divergence_metrics['overall_kl_divergence']:.4f}\n"

    with open(ENV.KIE_GT_ANALYZATION_DIR / f"{output_prefix}_table.tex", "w") as f:
        f.write(latex_str)

    print(f"Saved: {ENV.KIE_GT_ANALYZATION_DIR / output_prefix}_table.tex")
    print("\nLaTeX Table Preview:")
    print(latex_str)


def comprehensive_analysis(
    synth_dataset_name: str,
    real_stats: Dict,
    synth_stats: Dict,
    real_samples,
    synth_samples,
    output_prefix: str,
):
    """
    Run all supplementary analyses.
    """
    print("\n" + "=" * 80)
    print("ADVANCED ANALYSIS")
    print("=" * 80)

    # JS divergence
    print("\nComputing distribution divergence metrics...")
    divergence = compute_jensen_shannon_divergence(real_stats, synth_stats)
    print(f"  Jensen-Shannon Divergence: {divergence['overall_js_divergence']:.4f}")
    print(f"  KL Divergence: {divergence['overall_kl_divergence']:.4f}")

    # Spatial coverage
    print("\nComputing spatial coverage metrics...")
    spatial_metrics = compute_spatial_coverage_metrics(real_stats, synth_stats)
    print("  Centroid distances:")
    for entity_class, metrics in spatial_metrics.items():
        print(f"    {entity_class}: {metrics['centroid_distance']:.4f}")

    # Generate additional plots
    print("\nGenerating additional visualizations...")
    plot_entity_co_occurrence_matrix(
        real_stats, synth_stats, real_samples, synth_samples, output_prefix
    )
    plot_document_level_statistics(real_stats, synth_stats, output_prefix)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    ...
