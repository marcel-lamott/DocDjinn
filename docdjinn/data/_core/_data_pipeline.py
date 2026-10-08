"""
A data pipeline that wraps around a dataset and provides dataloaders for training, validation, and testing.
"""

from typing import TYPE_CHECKING, Callable

from atria_core.utilities.repr import RepresentationMixin

from docdjinn.data._core._data_types import MMDetInput
from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader
from docdjinn.logging import get_logger

from ._dataset import Dataset
from ._utilities import (
    auto_dataloader,
    default_collate,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


logger = get_logger(__name__)


def mmdet_pseudo_collate(batch: list["MMDetInput"]):
    """
    Default collate function for MMDetInput inputs.

    This function collates a batch of data instances into a single batch. It is used when
    the `collate_fn` argument is not provided to the DataLoader.

    Args:
        batch (List[MMDetInput]): A batch of data instances.

    Returns:
        Any: The collated batch.

    Raises:
        ValueError: If the batch is empty or not a list.
    """
    from mmengine.dataset.utils import pseudo_collate

    return MMDetInput(
        **pseudo_collate(
            [
                {
                    "inputs": sample.inputs,
                    "data_samples": sample.data_samples,
                }
                for sample in batch
            ]
        )
    )


class DataPipeline(RepresentationMixin):
    def __init__(
        self,
        dataset: "Dataset",
        # dataset split args
        dataset_splitting_enabled: bool = False,
        split_ratio: float = 0.9,
        # collate_fn
        collate_fn: str | None = "default_collate",
    ):
        self._dataset = dataset
        self._sharded_storage_kwargs = {}
        self._dataset_splitter = None

        # if dataset_splitting_enabled and self._dataset.validation is None: # just make sure to turn this off for now
        #     assert self._dataset.train is not None, (
        #         "Dataset splitting enabled but no training dataset found."
        #     )
        #         self._dataset_splitter = StandardSplitter(
        #             split_ratio=split_ratio, shuffle=True
        #         )
        #     self._dataset.train, self._dataset.validation = self._dataset_splitter(
        #         self._dataset.train
        #     )

        #     logger.info("Dataset splitting enabled.")
        #     logger.info(
        #         f"Train set size: {self._dataset.train_size}, Validation set size: {self._dataset.validation_size}"
        #     )

        if collate_fn == "default_collate":
            self._collate_fn = default_collate
        elif collate_fn == "mmdet_pseudo_collate":
            self._collate_fn = mmdet_pseudo_collate
        elif collate_fn == "identity":
            self._collate_fn = lambda x: x
        else:
            raise ValueError(f"Invalid collate_fn: {collate_fn}")

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataset_metadata(self):
        return self._dataset.metadata

    def set_transform(
        self, transform: Callable, for_train: bool = True, for_eval: bool = True
    ):
        from torch.utils.data import ConcatDataset

        if for_train and self._dataset.train is not None:
            if isinstance(self._dataset.train, ConcatDataset):
                for ds in self._dataset.train.datasets:
                    ds.set_transform(transform)
            else:
                self._dataset.train.set_transform(transform)

        if for_eval and self._dataset.validation is not None:
            self._dataset.validation.set_transform(transform)

        if for_eval and self._dataset.test is not None:
            self._dataset.test.set_transform(transform)

    def dataloader(
        self,
        split: str,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        shuffle: bool = True,
    ):
        if split == "train":
            return self.train_dataloader(
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        elif split == "validation":
            return self.validation_dataloader(
                batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
            )
        elif split == "test":
            return self.test_dataloader(
                batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers
            )
        else:
            raise ValueError(f"Invalid split name: {split}")

    def train_dataloader(
        self,
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
        shuffle: bool = True,
    ) -> "DataLoader | None":
        import ignite.distributed as idist
        from torch.utils.data import RandomSampler, SequentialSampler

        if self._dataset.train is None:
            return

        return auto_dataloader(
            dataset=self._dataset.train,
            collate_fn=self._collate_fn,
            sampler=RandomSampler(self._dataset.train)
            if shuffle
            else SequentialSampler(self._dataset.train),
            drop_last=idist.get_world_size() > 1,
            batch_size=batch_size * idist.get_world_size(),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def validation_dataloader(
        self, batch_size: int = 1, pin_memory: bool = True, num_workers: int = 4
    ) -> "DataLoader | None":
        dataset = self._dataset.validation or self._dataset.test
        if dataset is None:
            return

        if self._dataset.validation is None:
            logger.warning(
                "No validation dataset found, using test dataset for validation."
            )

        return self._build_evaluation_dataloader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def test_dataloader(
        self, batch_size: int = 1, pin_memory: bool = True, num_workers: int = 4
    ) -> "DataLoader | None":
        if self._dataset.test is None:
            return None
        return self._build_evaluation_dataloader(
            self._dataset.test,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )

    def _build_evaluation_dataloader(
        self,
        dataset: "MsgpackDatasetReader",
        batch_size: int = 1,
        pin_memory: bool = True,
        num_workers: int = 4,
    ) -> "DataLoader":
        if dataset is None:
            return None

        import ignite.distributed as idist  # type: ignore
        from torch.utils.data import SequentialSampler  # type: ignore

        if idist.get_world_size() > 1:
            if len(dataset) % idist.get_world_size() != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
        return auto_dataloader(
            dataset=dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            drop_last=False,
            sampler=SequentialSampler(dataset),
            batch_size=batch_size * idist.get_world_size(),
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
