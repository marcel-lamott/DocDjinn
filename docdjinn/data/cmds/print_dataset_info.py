from __future__ import annotations

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn.data._core._visualization_utilities import _compute_qa_stats
from docdjinn.data.interface import load_dataset
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: PrintDatasetInfo,
):
    dataset = load_dataset(
        dataset_name=cfg.dataset_name,
        is_synthetic=cfg.synthetic,
    )

    split_reader = dataset.train
    if split_reader is None:
        split_reader = dataset.validation
    if split_reader is None:
        split_reader = dataset.test
    if split_reader is None:
        raise ValueError("No valid split found in the dataset.")

    logger.info(dataset)
    logger.info(
        "Sample ids of first 10 samples: %s",
        [split_reader[idx].sample_id for idx in range(min(10, len(dataset.train)))],
    )
    for sample in split_reader:
        print(sample)
        break

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

    if cfg.dataset_name.startswith("ex_"):
        # Compute QA stats for each split
        if dataset.train:
            _compute_qa_stats(dataset.train, "train")
        if dataset.validation:
            _compute_qa_stats(dataset.validation, "validation")
        if dataset.test:
            _compute_qa_stats(dataset.test, "test")  # noqa: F821


class PrintDatasetInfo(pydantic.BaseModel):
    """
    Configuration for visualizing dataset samples.
    """

    dataset_name: str
    n_samples: int = 16
    synthetic: bool = False


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=PrintDatasetInfo,
    )
    main(parser.parse_typed_args())
