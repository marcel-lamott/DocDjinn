from __future__ import annotations

import pydantic.v1 as pydantic
import pydantic_argparse
import torch

from docdjinn.data._core._utilities import TaskType
from docdjinn.data._core._visualization_utilities import (
    _compute_qa_stats,
    _extract_annotations,
)
from docdjinn.data.interfaces.mixed_dataset import load_mixed_dataset
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: PrintMixedDatasetInfo,
):
    torch.manual_seed(42)

    dataset = load_mixed_dataset(
        dataset_name=cfg.dataset_name,
        synthetic_dataset_name=cfg.synthetic_dataset_name,
        num_real_samples=cfg.num_real_samples,
        num_synthetic_samples=cfg.num_synthetic_samples,
    )

    split_reader = dataset.train
    if split_reader is None:
        split_reader = dataset.validation
    if split_reader is None:
        split_reader = dataset.test
    if split_reader is None:
        raise ValueError("No valid split found in the dataset.")

    logger.info(dataset)
    # logger.info(
    #     "Sample ids of first 10 samples: %s",
    #     [split_reader[idx].sample_id for idx in range(min(10, len(dataset.train)))],
    # )
    # for sample in split_reader:
    #     logger.info("First sample: {}".format(sample))
    #     break
    avg_num_pages = 0
    has_ocr = False
    has_normalized_bboxes = False
    for sample in split_reader:
        if sample.pdf is not None:
            avg_num_pages += sample.pdf.num_pages
        else:
            avg_num_pages += 1  # if no pdf, assume single page

        if sample.content is not None:
            if sample.content.words is not None:
                has_ocr = True
            if sample.content.word_bboxes is not None:
                has_normalized_bboxes = sample.content.word_bboxes.normalized
    avg_num_pages /= len(split_reader)

    logger.info(
        f"Average number of pages in the {cfg.dataset_name} dataset (train split): {avg_num_pages}"
    )
    logger.info(f"Has OCR={has_ocr}")
    logger.info(f"Has Normalized BBoxes={has_normalized_bboxes}")

    # find label distribution
    if dataset.task_type == TaskType.sequence_classification:
        label_counts = {}
        total_labels = 0
        for sample in split_reader:
            annotations = _extract_annotations(sample)
            if annotations["label"].name not in label_counts:
                label_counts[annotations["label"].name] = 0
            total_labels += 1
            label_counts[annotations["label"].name] += 1
            # if annotations["label"] is not None:
            # label_counts[label[0]] = label_counts.get(label, 0) + 1
            # total_labels += 1

        # sort the dict by key
        label_counts = dict(sorted(label_counts.items()))
        logger.info("Total labels: {}".format(total_labels))
        logger.info(
            f"Label distribution in the {cfg.dataset_name} dataset (train split):"
        )
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count} ({(count / total_labels) * 100:.2f}%)")

        import matplotlib.pyplot as plt

        labels = list(label_counts.keys())
        counts = [label_counts[label] for label in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts)
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.title(
            f"Label Distribution in mixed-{cfg.dataset_name}-{cfg.num_real_samples}-{cfg.synthetic_dataset_name}-{cfg.num_synthetic_samples}-label_distribution.png"
        )
        plt.xticks(rotation=45)
        plt.savefig(
            f"mixed-{cfg.dataset_name}-{cfg.num_real_samples}-{cfg.synthetic_dataset_name}-{cfg.num_synthetic_samples}-label_distribution.png"
        )

    if dataset.task_type == TaskType.extractive_qa:
        # Compute QA stats for each split
        if dataset.train:
            _compute_qa_stats(dataset.train, "train")
        if dataset.validation:
            _compute_qa_stats(dataset.validation, "validation")
        if dataset.test:
            _compute_qa_stats(dataset.test, "test")  # noqa: F821


class PrintMixedDatasetInfo(pydantic.BaseModel):
    """
    Configuration for visualizing dataset samples.
    """

    dataset_name: str
    synthetic_dataset_name: str
    num_real_samples: int = -1
    num_synthetic_samples: int = -1
    n_samples: int = 16


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=PrintMixedDatasetInfo,
    )
    main(parser.parse_typed_args())
