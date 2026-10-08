from __future__ import annotations

from pathlib import Path

import yaml

from docdjinn import ENV
from docdjinn.data._core._msgpack_dataset_writer import MsgpackDatasetWriter
from docdjinn.data._core._synth import SynthesizedDataset
from docdjinn.data._core._utilities import TaskType, get_logger
from docdjinn.generation.models import (
    SynDatasetDefinition,
)

from .dataset import load_dataset

logger = get_logger(__name__)


def _get_output_dataset_path(dataset_name: str) -> Path:
    # this is important must stay same as the other datasets
    # load the output file with the output directory structure
    # is used by DatasetFactory to load datasets
    return (
        ENV.SYN_DATASETS_PREPARED_DIR
        / dataset_name
        / "storage"
        / "default"
        / "msgpack"
        / "train"
        / "dataset.msgpack"
    )


def _get_output_metadata_path(dataset_name: str) -> Path:
    return ENV.SYN_DATASETS_PREPARED_DIR / dataset_name / "storage" / "metadata.yaml"


def prepare_synthetic_dataset(
    dsdef: SynDatasetDefinition,
    force_overwrite: bool = False,
    resize_images: bool = False,
    clip_bboxes_to_foreground: bool = False,
):
    output_file = _get_output_dataset_path(dsdef.name)

    # first load original dataset to get labels and task type
    original_dataset = load_dataset(
        dataset_name=dsdef.base_dataset_name,
        split="train",
    )

    # get the labels based on the task
    dataset_labels = None
    if original_dataset.task_type == TaskType.sequence_classification:
        dataset_labels = original_dataset.metadata.dataset_labels.classification
    elif original_dataset.task_type == TaskType.token_classification:
        dataset_labels = original_dataset.metadata.dataset_labels.ser
    elif original_dataset.task_type == TaskType.layout_analysis:
        dataset_labels = original_dataset.metadata.dataset_labels.layout

    # now create synthesized dataset
    dataset = SynthesizedDataset(
        dsdef=dsdef,
        task_type=original_dataset.task_type,
        dataset_labels=dataset_labels,
        resize_images=resize_images,
        clip_bboxes_to_foreground=clip_bboxes_to_foreground,
    )

    # msgpack writer
    MsgpackDatasetWriter(dataset, output_file).write(force_overwrite=force_overwrite)

    # ensure parent dir exists
    metadata_file = _get_output_metadata_path(dsdef.name)
    with open(metadata_file, "w") as f:
        yaml_content = original_dataset.metadata.model_dump()
        yaml.dump(yaml_content, f)


def load_synthetic_dataset(
    dataset_name: str,
    task_type_override: TaskType | None = None,
    split: str | None = None,
):
    dataset = load_dataset(
        dataset_name=dataset_name,
        is_synthetic=True,
        task_type_override=task_type_override,
        split=split,
    )
    return dataset
