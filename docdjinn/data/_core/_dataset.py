"""
A simple dataset class that holds multiple split iterators.
"""

from __future__ import annotations

from typing import TypeVar

from atria_core.types import DatasetMetadata
from atria_core.types.common import DatasetSplitType
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from atria_core.utilities.repr import RepresentationMixin

from docdjinn.logging import get_logger

from ._msgpack_dataset_reader import MsgpackDatasetReader
from ._utilities import TaskType

logger = get_logger(__name__)


T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)


class Dataset(RepresentationMixin):
    def __init__(
        self,
        name: str,
        split_iterators: dict,
        metadata: DatasetMetadata,
        task_type: TaskType,
    ) -> None:
        self._name = name
        self._split_iterators: dict[DatasetSplitType, MsgpackDatasetReader] = (
            split_iterators
        )
        self._metadata = metadata
        self._task_type = task_type

    @property
    def name(self) -> str:
        """Dataset name."""
        return self._name

    @property
    def task_type(self) -> TaskType:
        """Dataset task type."""
        return self._task_type

    @task_type.setter
    def task_type(self, value: TaskType) -> None:
        self._task_type = value

    @property
    def split_iterators(
        self,
    ) -> dict[DatasetSplitType, MsgpackDatasetReader]:
        """Dictionary of split iterators."""
        return self._split_iterators

    @property
    def train(self) -> MsgpackDatasetReader | None:
        """Training split iterator. Returns None if training split is not available."""
        return self._split_iterators.get(DatasetSplitType.train, None)

    @property
    def validation(self) -> MsgpackDatasetReader | None:
        """Validation split iterator. Returns None if validation split is not available."""
        return self._split_iterators.get(DatasetSplitType.validation, None)

    @property
    def test(self) -> MsgpackDatasetReader | None:
        """Test split iterator. Returns None if test split is not available."""
        return self._split_iterators.get(DatasetSplitType.test, None)

    @train.setter
    def train(self, value: MsgpackDatasetReader) -> None:
        self._split_iterators[DatasetSplitType.train] = value

    @validation.setter
    def validation(self, value: MsgpackDatasetReader) -> None:
        self._split_iterators[DatasetSplitType.validation] = value

    @test.setter
    def test(self, value: MsgpackDatasetReader) -> None:
        self._split_iterators[DatasetSplitType.test] = value

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset metadata."""
        return self._metadata

    @property
    def train_size(self) -> int:
        """Length of the training split. Returns 0 if training split is not available."""
        if self.train is None:
            return 0
        return len(self.train)

    @property
    def validation_size(self) -> int:
        """Length of the validation split. Returns 0 if validation split is not available."""
        if self.validation is None:
            return 0
        return len(self.validation)

    @property
    def test_size(self) -> int:
        """Length of the test split. Returns 0 if test split is not available."""
        if self.test is None:
            return 0
        return len(self.test)
