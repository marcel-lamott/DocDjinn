from __future__ import annotations

import numpy as np
import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.data._core._data_types import DocumentInstance
from docdjinn.data.interface import (
    load_synthetic_dataset,
    prepare_synthetic_dataset,
)
from docdjinn.generation.models._syndatadef import SynDatasetDefinition
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: PrepareSynthDatasets,
):
    # prepare dataset config
    dataset_name = cfg.dataset_name

    dsdef = SynDatasetDefinition.from_file(
        ENV.SYN_DATASETS_DIR / dataset_name / f"{dataset_name}.yaml"
    )

    # manually fix the mapping for doclaynet datasets
    if dsdef.name.startswith("doclaynet"):
        if dsdef.task == "CLASSIFICATION":
            dsdef.base_dataset_name = "doclaynet_4k_cls"
        elif dsdef.task == "DLA":
            dsdef.base_dataset_name = "doclaynet_4k_dla"

    # prepare synthetic dataset
    prepare_synthetic_dataset(
        dsdef=dsdef,
        force_overwrite=cfg.force_overwrite,
        resize_images=cfg.resize_images,
        clip_bboxes_to_foreground=cfg.clip_bboxes_to_foreground,
    )

    # load and visualize samples
    synthesized_dataset = load_synthetic_dataset(
        dataset_name=cfg.dataset_name, split="train"
    )

    # log prepared info
    logger.info("Prepared synthetic dataset: %s", synthesized_dataset)

    # visualize single sample from this dataset
    assert synthesized_dataset.train is not None, (
        "Train split is None in the synthesized dataset"
    )
    for idx, sample in enumerate(synthesized_dataset.train):
        if idx == 0:
            logger.info("First sample in the synthesized dataset:")
            logger.info(sample)

        sample: DocumentInstance
        if cfg.run_sanity_check:
            # make sure here that we alway shave
            assert sample.content is not None
            if sample.content.word_bboxes:
                assert sample.content.word_bboxes.normalized, (
                    "Word bounding boxes are not normalized"
                )
                assert (
                    np.array(sample.content.word_bboxes.value).min() >= -0.01
                    and np.array(sample.content.word_bboxes.value).max() <= 1.01
                ), (
                    "Word bounding boxes for sample={}, words={}, are not in [0, 1] range, Got: min={}, max={}".format(
                        sample.sample_id,
                        sample.content.words,
                        np.array(sample.content.word_bboxes.value).min(),
                        np.array(sample.content.word_bboxes.value).max(),
                    )
                )
            if sample.content.word_segment_level_bboxes:
                assert sample.content.word_segment_level_bboxes.normalized, (
                    "Segment level bounding boxes are not normalized"
                )
                assert (
                    np.array(sample.content.word_segment_level_bboxes.value).min()
                    >= -0.01
                    and np.array(sample.content.word_segment_level_bboxes.value).max()
                    <= 1.01
                ), (
                    "Segment level bounding boxes for sample={}, words={}, are not in [0, 1] range, Got: min={}, max={}".format(
                        sample.sample_id,
                        sample.content.words,
                        np.array(sample.content.word_segment_level_bboxes.value).min(),
                        np.array(sample.content.word_segment_level_bboxes.value).max(),
                    )
                )


class PrepareSynthDatasets(pydantic.BaseModel):
    """
    Configuration for visualizing dataset samples.
    """

    dataset_name: str
    n_samples: int = 16
    run_sanity_check: bool = True
    force_overwrite: bool = False
    resize_images: bool = False
    clip_bboxes_to_foreground: bool = False


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=PrepareSynthDatasets,
    )
    main(parser.parse_typed_args())
