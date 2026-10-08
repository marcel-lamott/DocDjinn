"""
Msgpack shard list dataset module taken from atria_datasets
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from atria_core.types import BaseDataInstance
from datadings.reader import MsgpackReader as MsgpackFileReader

from docdjinn.data._core._data_types import DocumentInstanceModelInput
from docdjinn.logging import get_logger

logger = get_logger(__name__)

T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)


class MsgpackDatasetReader(Sequence[Any]):
    """
    A dataset class for reading Msgpack-based shard files.

    This class provides functionality for loading and iterating over datasets stored
    in Msgpack-based shard files. It supports efficient indexing and cumulative size
    calculations for handling multiple shards.

    Attributes:
        _shard_files (list[str]): A list of Msgpack file path for each shard.
        _cumulative_sizes (list[int]): Cumulative sizes of the shards for efficient indexing.
        _total_size (int): The total number of samples across all shards.
    """

    def __init__(
        self,
        msgpack_files: list[str] | list[Path],
        data_model: type,
        transform: Callable | None = None,
    ) -> None:
        """
        Initializes the `MsgpackShardListDataset`.

        Args:
            shard_files (List[DatasetShardInfo]): A list of shard metadata containing file URLs.
        """
        logger.info(f"Loading dataset from files: {msgpack_files}")
        self._msgpack_files = sorted(msgpack_files)
        self._total_size: int = 0

        cumulative_sizes: list[int] = []
        for f in self._msgpack_files:
            with MsgpackFileReader(f) as reader:
                self._total_size += len(reader)
                cumulative_sizes.append(self._total_size)
        self._cumulative_sizes = np.array(cumulative_sizes)

        self._data_model = data_model
        self._transform = transform
        self._subset_indices = None
        self._msgpack_file_readers = [MsgpackFileReader(f) for f in self._msgpack_files]
        self._data_dir = Path(self._msgpack_files[0]).parent

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def set_subset_indices(self, indices: list[int]) -> None:
        """
        Sets the subset indices for the dataset.

        Args:
            indices (List[int]): A list of indices to subset the dataset.
        """
        self._subset_indices = indices

    def set_transform(self, transform: Callable) -> None:
        """
        Sets the transform function for the dataset.

        Args:
            transform (Callable): A function to transform each data instance.
        """
        self._transform = transform

    def _transform_input(self, input: Any) -> BaseDataInstance:
        if issubclass(self._data_model, BaseDataInstance):
            if "total_num_pages" in input:
                input.pop("total_num_pages")
            data_instance: BaseDataInstance = self._data_model.model_validate(input)

            # assert that the transformed instance is of the expected data model type
            assert isinstance(data_instance, self._data_model), (
                f"self._input_transform(sample) should return {self._data_model}, but got {type(data_instance)}"
            )

            # load the data instance from disk if not already loaded
            data_instance.load()

            # yield the transformed data instance if output transform is enabled
            if self._transform is not None:
                data_instance = self._transform(data_instance)
            return data_instance
        elif issubclass(self._data_model, DocumentInstanceModelInput):
            data_instance = self._data_model.from_dict(input)
            if self._transform is not None:
                data_instance = self._transform(data_instance)
            return data_instance
        else:
            raise ValueError(
                f"Unsupported data model type: {self._data_model}. Must be a subclass of BaseDataInstance or DocumentInstanceModelInput."
            )

    def get_by_id(self, sample_id: str) -> int:
        for reader in self._msgpack_file_readers:
            try:
                sample_id = str(sample_id)
                index = reader.find_index(sample_id.replace(".", "_"))
                sample = reader[index]
                sample.pop("key", None)
                sample = self._transform_input(sample)
                assert sample.sample_id == sample_id, (  # this should never happen
                    f"Sample ID mismatch: expected {sample_id} ({type(sample_id)}), got {sample.sample_id} ({type(sample.sample_id)})"
                )
                return sample
            except KeyError:
                continue
        raise ValueError(f"Sample ID {sample_id} not found in any shard.")

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        """
        Retrieves a sample from the dataset by index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Dict[str, Any]: The sample at the specified index.
        """
        if self._subset_indices is not None:
            index = self._subset_indices[index]

        shard_index = np.searchsorted(self._cumulative_sizes, index, side="right")
        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - self._cumulative_sizes[shard_index - 1]
        sample = self._msgpack_file_readers[shard_index][inner_index]
        sample.pop("key", None)
        return self._transform_input(sample)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        if self._subset_indices is not None:
            return len(self._subset_indices)
        return self._total_size

    def close(self) -> None:
        """
        Closes all shard file readers to release resources.
        """
        for reader in self._msgpack_file_readers:
            reader._close()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}, "
            f"total_size={self._total_size}, num_shards={len(self._msgpack_files)})"
        )
