from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import yaml
from atria_core.types import DatasetMetadata
from atria_core.types.common import DatasetSplitType
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from atria_core.types.data_instance.document_instance import (
    DocumentInstance,
)

from docdjinn.data.constants import DatasetLoadConfig
from docdjinn.logging import get_logger

from ._dataset import Dataset
from ._msgpack_dataset_reader import (
    MsgpackDatasetReader,
)

logger = get_logger(__name__)


class DatasetFactory:
    """
    Factory class for creating and loading datasets from msgpack shard files.

    The DatasetFactory provides a centralized way to load datasets stored in a specific
    directory structure with msgpack format. It automatically discovers available datasets
    and configurations, validates paths, and creates appropriate data iterators for each split.

    Expected Directory Structure:
        root_datasets_dir/
        └── dataset_name/
            └── storage/
                └── dataset_config_name/
                    └── msgpack/
                        ├── train/
                        │   ├── shard_001.msgpack
                        │   └── shard_002.msgpack
                        ├── validation/
                        │   └── shard_001.msgpack
                        └── test/
                            └── shard_001.msgpack

    Usage:
        # Basic usage with default DocumentInstance data model
        dataset = DatasetFactory.load_dataset(
            root_datasets_dir="/path/to/datasets",
            dataset_name="my_dataset",
            dataset_config_name="default"

        # With custom data model and output transformation
        dataset = DatasetFactory.load_dataset(
            root_datasets_dir="/path/to/datasets",
            dataset_name="my_dataset",
            dataset_config_name="processed",
            data_model=CustomDataInstance,
            output_transform=lambda x: preprocess(x)

        # Access splits
        for sample in dataset.train:
            # Process training samples
            pass

    The factory handles:
    - Automatic discovery of dataset splits (train, validation, test, etc.)
    - Loading msgpack shard files for each split
    - Data model instantiation and transformation
    - Error handling with helpful messages about available datasets/configs
    """

    @classmethod
    def get_preprocess_transform(self, preprocess_image_size: int) -> Callable:
        def preprocess_transform(sample: BaseDataInstance) -> dict:
            resized_image = sample.image.resize(
                width=preprocess_image_size, height=preprocess_image_size
            )
            return sample.model_copy(update={"image": resized_image})

        return preprocess_transform

    @classmethod
    def prepare_paths(
        cls, root_datasets_dir: str | Path, dataset_name: str, dataset_config_name: str
    ):
        # construct paths
        data_dir = Path(root_datasets_dir) / dataset_name / "storage"
        metadata_file = data_dir / "metadata.yaml"
        msgpack_dir = data_dir / dataset_config_name / "msgpack"

        if not data_dir.exists():
            raise ValueError(
                f"Data directory {data_dir} does not exist. "
                f"Please check the dataset {dataset_name} is prepared with config name {dataset_config_name}. "
            )

        assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist. "
        assert msgpack_dir.exists(), f"Data directory {msgpack_dir} does not exist. "
        return metadata_file, msgpack_dir

    @classmethod
    def load_metadata(
        cls,
        metadata_file: str | Path,
    ) -> DatasetMetadata:
        # load metadata
        with open(metadata_file, "r") as f:
            metadata = yaml.safe_load(f)
        return DatasetMetadata(**metadata)

    @classmethod
    def get_available_splits(cls, msgpack_dir: Path):
        available_splits = [DatasetSplitType(x) for x in os.listdir(msgpack_dir)]
        assert len(available_splits) > 0, (
            f"No splits found in {msgpack_dir}. Found {available_splits}"
        )
        return available_splits

    @classmethod
    def load_split_from_disk(
        cls,
        msgpack_dir: Path,
        split: DatasetSplitType,
        data_model: type[BaseDataInstance],
    ) -> MsgpackDatasetReader:
        # load msgpack files for this split
        split_files = list((msgpack_dir / split.value).glob("*.msgpack"))
        return MsgpackDatasetReader(msgpack_files=split_files, data_model=data_model)

    @classmethod
    def load_dataset(
        cls,
        dataset_load_config: DatasetLoadConfig,
        data_model: type[BaseDataInstance] = DocumentInstance,
        split: str | None = None,
    ) -> Dataset:
        # get dataset name and config name
        dataset_name, dataset_config_name = (
            dataset_load_config.dataset_name,
            dataset_load_config.dataset_config_name,
        )

        # handle tuple config names
        if isinstance(dataset_config_name, tuple):
            dataset_name, dataset_config_name = dataset_config_name

        # construct paths
        metadata_file, msgpack_dir = cls.prepare_paths(
            dataset_load_config.root_datasets_dir, dataset_name, dataset_config_name
        )

        # load metadata
        metadata = cls.load_metadata(metadata_file)

        # load split files
        available_splits = cls.get_available_splits(msgpack_dir)

        # load split iterators
        split_iterators = {}
        for current_split in available_splits:
            if split is not None and current_split.value != split:
                continue
            split_iterators[current_split] = cls.load_split_from_disk(
                msgpack_dir, current_split, data_model
            )

        return Dataset(
            name=dataset_name,
            split_iterators=split_iterators,
            metadata=metadata,
            task_type=dataset_load_config.task_type,
        )
