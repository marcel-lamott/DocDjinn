"""
Defines interface for docdjinn components to load datasets using DatasetFactory and log relevant information.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import ConcatDataset

from docdjinn.data._core._dataset import Dataset
from docdjinn.data._core._dataset_factory import DatasetFactory  # noqa
from docdjinn.data._core._msgpack_dataset_writer import MsgpackDatasetWriter
from docdjinn.data._core._standard_splitter import StandardSplitter
from docdjinn.data._core._utilities import TaskType, get_logger
from docdjinn.data._transforms.utilities import generate_transform_hash
from docdjinn.data.constants import (
    DATASET_CONFIGS,
)

from .transform import load_transform

logger = get_logger(__name__)


def get_dataset_config(dataset_name: str, is_synthetic: bool = False):
    available_dataset_configs = list(
        filter(
            lambda x: x.is_synthetic == is_synthetic,
            DATASET_CONFIGS,
        )
    )

    dataset_config = list(
        filter(
            lambda x: x.dataset_name == dataset_name,
            available_dataset_configs,
        )
    )

    if len(dataset_config) == 0:
        raise ValueError(
            f"Dataset {dataset_name} with is_synthetic={is_synthetic} not found in predefined config list. "
            f"Available datasets: {[cfg.dataset_name for cfg in available_dataset_configs]}"
        )

    return dataset_config[0]


def preprocess_dataset(
    dataset: Dataset,
    transform_type: str = "default",
    force_overwrite: bool = False,
    output_dir: Path | dict[str, Path] | None = None,
    **transforms_kwargs,
):
    for split, split_reader in dataset.split_iterators.items():
        # load transform for this split
        logger.info("Transforms kwargs passed for preprocessing: %s", transforms_kwargs)

        # generate unique hash for transform
        transform_hash = generate_transform_hash(transforms_kwargs)

        # load the required transform
        transform = load_transform(
            task_type=dataset.task_type,
            for_train=(split == "train"),
            transform_type=transform_type,
            dataset_labels=dataset.metadata.dataset_labels,
            **transforms_kwargs,
        )

        # output file path
        if output_dir is None:
            output_dir = split_reader.data_dir / "preprocessed_v2"

        if isinstance(output_dir, dict):
            split_output_dir = output_dir[split]
        else:
            split_output_dir = output_dir

        output_file = (
            split_output_dir
            / split.value
            / f"dataset-{dataset.task_type.value}-{transform_hash}.msgpack"
        )

        # set transform
        if isinstance(split_reader, ConcatDataset):
            for ds in split_reader.datasets:
                ds.set_transform(transform)
        else:
            split_reader.set_transform(transform)

        # preprocess and save
        writer = MsgpackDatasetWriter(
            dataset_reader=split_reader,
            output_file=output_file,
            data_model=transform.get_output_data_model(),
        )
        dataset.split_iterators[split] = writer.write(force_overwrite=force_overwrite)
    return dataset


def load_dataset(
    dataset_name: str,
    is_synthetic: bool = False,
    task_type_override: TaskType | None = None,
    split: str | None = None,
    create_train_val_splits: bool = True,
    split_ratio: float = 0.95,
) -> Dataset:
    dataset_load_config = get_dataset_config(dataset_name, is_synthetic)
    dataset = DatasetFactory.load_dataset(
        dataset_load_config=dataset_load_config,
        split=split,
    )
    if task_type_override is not None:
        dataset.task_type = task_type_override

    if is_synthetic:
        return dataset

    if dataset.validation is None and create_train_val_splits:
        assert dataset.train is not None, (
            "Dataset splitting enabled but no training dataset found."
        )
        dataset_splitter = StandardSplitter(split_ratio=split_ratio, shuffle=True)
        train, validation = dataset_splitter(dataset.train)
        dataset.train = train
        dataset.validation = validation

        assert dataset.train is not None, (
            "Training split is None in the loaded dataset"
        )  # for our experiments we always make sure we have validation split present
        assert dataset.validation is not None, (
            "Validation split is None in the loaded dataset"
        )  # for our experiments we always make sure we have validation split present

    return dataset


def load_preprocessed_dataset(
    dataset_name: str,
    is_synthetic: bool = False,
    task_type_override: TaskType | None = None,
    # preprocess related args
    transform_type: str = "default",
    split: str | None = None,
    **transforms_kwargs,
) -> Dataset:
    # load raw dataset
    dataset = load_dataset(
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        task_type_override=task_type_override,
        split=split,
    )

    # preprocess dataset
    dataset = preprocess_dataset(
        dataset=dataset,
        force_overwrite=False,
        transform_type=transform_type,
        **transforms_kwargs,
    )

    return dataset
