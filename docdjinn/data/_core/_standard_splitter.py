"""
Dataset Splitter Module

This module defines the `StandardSplitter` class, which provides utilities for splitting
datasets into training and validation subsets. It supports both sequential and random
splitting strategies, with configurable options for shuffle, and split ratio.

Classes:
    - StandardSplitter: A class for splitting datasets into training and validation subsets.

Dependencies:
    - copy: For deep copying datasets.
    - typing: For type annotations.
    - torch.utils.data: For dataset splitting utilities.
    - atria_core.logger.logger: For logging utilities.
    - atria_registry: For registering dataset splitters.
    - atria_datasets.core.datasets.atria_dataset: For the base dataset class.

Date: 2025-04-07
Version: 1.0.0
License: MIT
"""

from atria_core.utilities.repr import RepresentationMixin

from ._msgpack_dataset_reader import MsgpackDatasetReader


class StandardSplitter(RepresentationMixin):
    """
    A class for splitting datasets into training and validation subsets.

    This class provides methods for creating sequential and random splits of datasets.
    It supports configurable options for shuffle, and split ratio.

    Attributes:
        split_ratio (float): The ratio of the training split. Defaults to 0.8.
        shuffle (bool): Whether to shuffle the dataset before splitting. Defaults to True.
    """

    def __init__(self, split_ratio: float = 0.8, shuffle: bool = True):
        """
        Initializes the `StandardSplitter`.

        Args:
            split_ratio (float): The ratio of the training split. Defaults to 0.8.
            shuffle (bool): Whether to shuffle the dataset before splitting. Defaults to True.
        """
        self.split_ratio = split_ratio
        self.shuffle = shuffle

    def create_sequential_split(
        self, train: "MsgpackDatasetReader"
    ) -> tuple["MsgpackDatasetReader", "MsgpackDatasetReader"]:
        """
        Creates a sequential split of the dataset.

        The dataset is split into training and validation subsets based on the split ratio,
        without shuffling.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.
        """
        import copy

        dataset_size = len(train)
        split_point = int(dataset_size * round(self.split_ratio, 2))

        validation = copy.deepcopy(train)
        train.set_subset_indices(list(range(split_point)))
        validation.set_subset_indices(list(range(split_point)))
        return train, validation

    def create_random_split(
        self, train: "MsgpackDatasetReader"
    ) -> tuple["MsgpackDatasetReader", "MsgpackDatasetReader"]:
        """
        Creates a random split of the dataset.

        The dataset is split into training and validation subsets based on the split ratio,
        with shuffling.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.
        """
        import copy

        from sklearn.model_selection import train_test_split

        assert train is not None, (
            "The dataset must have a 'train' split defined for sequential splitting."
        )

        train_dataset_size = len(train)
        validation = copy.deepcopy(train)
        train_subset, validation_subset = train_test_split(
            list(range(train_dataset_size)),
            test_size=1 - self.split_ratio,
            random_state=42,
        )
        train.set_subset_indices(list(train_subset))
        validation.set_subset_indices(list(validation_subset))
        return train, validation

    def __call__(
        self, train_split: "MsgpackDatasetReader"
    ) -> tuple["MsgpackDatasetReader", "MsgpackDatasetReader"]:
        """
        Splits the dataset into training and validation subsets.

        The splitting strategy (sequential or random) is determined by the `shuffle` attribute.

        Args:
            train_dataset (AtriaDataset): The dataset to split.

        Returns:
            Tuple[AtriaDataset, AtriaDataset]: The training and validation subsets.

        Raises:
            AssertionError: If the dataset is not an instance of `AtriaDataset` or if the
                            dataset size is unknown (e.g., in iterable mode).
        """
        assert isinstance(train_split, MsgpackDatasetReader), (
            "The dataset must be a PyTorch or Hugging Face dataset."
        )
        assert len(train_split) != "unknown", (
            "The dataset size is unknown. This means that the dataset is set up "
            "in iterable mode and splitting is not supported."
        )
        if self.shuffle:
            return self.create_random_split(train_split)
        else:
            return self.create_sequential_split(train_split)
