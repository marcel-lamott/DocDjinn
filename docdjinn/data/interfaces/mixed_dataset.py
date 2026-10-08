"""
Defines interface for docdjinn components to load datasets using DatasetFactory and log relevant information.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from docdjinn.data._core._data_types import DatasetSplitType
from docdjinn.data._core._dataset import Dataset
from docdjinn.data._core._dataset_factory import DatasetFactory  # noqa
from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader
from docdjinn.data._core._utilities import TaskType, get_logger

from .dataset import load_dataset, preprocess_dataset

logger = get_logger(__name__)


def prepare_random_dataset_subset(
    dataset: MsgpackDatasetReader,
    num_samples: int,
    dataset_name: str,
    seed: int = 42,
    cache_subset_indices: bool = False,
) -> MsgpackDatasetReader:
    # Select random subset of indices using torch for reproducibility
    # copy dataset to avoid modifying the original

    cache_file_path = (
        Path("data")
        / "cached_subsets"
        / f"{dataset_name}_subset_{num_samples}_indices.txt"
    )
    if cache_subset_indices:
        if cache_file_path.exists():
            logger.info(
                f"Loading cached subset indices from {cache_file_path} for dataset {dataset_name}."
            )
            with open(cache_file_path, "r") as f:
                subset_indices = [int(line.strip()) for line in f]
            dataset.set_subset_indices(subset_indices)
            assert len(dataset) == num_samples, (
                f"Expected dataset length {num_samples}, but got {len(dataset)}"
            )
            return dataset

    # close open file readers before deepcopy
    dataset.close()

    # deepcopy the dataset
    dataset = copy.deepcopy(dataset)

    # select random subset of indices
    num_samples = min(num_samples, len(dataset))
    random_subset_indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    )[:num_samples].tolist()

    if cache_subset_indices:
        cache_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file_path, "w") as f:
            for idx in random_subset_indices:
                f.write(f"{idx}\n")
        logger.info(
            f"Cached subset indices to {cache_file_path} for dataset {dataset_name}."
        )

    # set subset indices
    dataset.set_subset_indices(random_subset_indices)

    assert len(dataset) == num_samples, (
        f"Expected dataset length {num_samples}, but got {len(dataset)}"
    )
    return dataset


def load_mixed_dataset(
    dataset_name: str,
    synthetic_dataset_name: str,
    num_real_samples: int = -1,
    num_synthetic_samples: int = -1,
    task_type_override: TaskType | None = None,
    split: str | None = None,
    load_preprocessed: bool = False,
    transform_type: str = "default",
    **transforms_kwargs,
) -> Dataset:
    # load real dataset
    real_dataset = load_dataset(
        dataset_name,
        is_synthetic=False,
        task_type_override=task_type_override,
        split=split,
    )

    # first extract all splits from real dataset
    real_train, real_validation, real_test = (
        real_dataset.train,
        real_dataset.validation,
        real_dataset.test,
    )

    assert real_train is not None, "No training split found in real dataset."

    # load synthetic dataset
    synthetic_dataset = load_dataset(
        synthetic_dataset_name,
        is_synthetic=True,
        task_type_override=task_type_override,
        split=split,
    )

    synthetic_train = synthetic_dataset.train
    assert synthetic_train is not None, "No training split found in synthetic dataset."

    # make a mixed dataset name based on number of samples
    assert real_dataset.task_type == synthetic_dataset.task_type, (
        "Task types of real and synthetic datasets do not match."
    )

    # now prepare subsets if required
    if num_real_samples >= 0:
        assert real_train is not None, "Real dataset does not have a training split."
        logger.info(
            f"Preparing real dataset training subset with {num_real_samples} samples."
        )
        train_length_before = len(real_train)
        real_train = prepare_random_dataset_subset(
            real_train,
            num_real_samples,
            dataset_name=dataset_name,
            cache_subset_indices=True,
        )
        logger.info(
            f"Real dataset size before: {train_length_before}, after: {len(real_train)}"
        )

    if num_synthetic_samples >= 0:
        logger.info(
            f"Preparing synthetic dataset training subset with {num_synthetic_samples} samples."
        )
        synthetic_length_before = len(synthetic_train)
        synthetic_train = prepare_random_dataset_subset(
            synthetic_train,
            num_synthetic_samples,
            dataset_name=synthetic_dataset_name,
        )
        logger.info(
            f"Synthetic dataset size before: {synthetic_length_before}, after: {len(synthetic_train)}"
        )

    # now combine datasets
    combined_train = ConcatDataset([real_train, synthetic_train])  # type: ignore[arg-type]

    # log dataset sizes
    logger.info(
        f"Combined training dataset size: {len(combined_train)} "
        f"(Real: {len(real_train)}, Synthetic: {len(synthetic_train)})"
    )

    mixed_dataset = Dataset(
        name=f"mixed_{dataset_name}-{num_real_samples}_{synthetic_dataset_name}-{num_synthetic_samples}",
        split_iterators={
            DatasetSplitType.train: combined_train,  # type: ignore[arg-type]
            DatasetSplitType.validation: real_validation,
            DatasetSplitType.test: real_test,
        },
        metadata=real_dataset.metadata,
        task_type=real_dataset.task_type,
    )

    # preprocess dataset if required
    if load_preprocessed:
        # slighly hacky but to save space and time only store train set preprocessed separately
        output_dir = {
            DatasetSplitType.train: Path("data") / "mixed_datasets" / mixed_dataset.name / "preprocessed",
            DatasetSplitType.test: Path("data") / "mixed_datasets" / dataset_name / "preprocessed",
            DatasetSplitType.validation: Path("data") / "mixed_datasets" / dataset_name / "preprocessed",
        }
        mixed_dataset = preprocess_dataset(
            dataset=mixed_dataset,
            force_overwrite=False,
            transform_type=transform_type,
            output_dir=output_dir,
            **transforms_kwargs,
        )

    return mixed_dataset
