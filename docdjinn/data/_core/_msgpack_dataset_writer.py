"""
Defines interface for docdjinn components to load datasets using DatasetFactory and log relevant information.
"""

from __future__ import annotations

from pathlib import Path

import tqdm
from torch.utils.data import Dataset

from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader
from docdjinn.logging import get_logger

from ._data_types import BaseDataInstance, DocumentInstance

logger = get_logger(__name__)


class MsgpackDatasetWriter:
    def __init__(
        self,
        dataset_reader: MsgpackDatasetReader | Dataset,
        output_file: Path,
        data_model: type | type[BaseDataInstance] = DocumentInstance,
    ):
        self._dataset_reader = dataset_reader
        self._output_file = output_file
        self._data_model = data_model

    def _get_dataloader(self):
        import torch

        # setup dataloader
        dataloader = torch.utils.data.DataLoader(
            self._dataset_reader,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x,
            drop_last=False,
        )
        return dataloader

    def write(self, force_overwrite: bool = False) -> MsgpackDatasetReader:
        if force_overwrite:
            logger.warning(
                f"Force overwrite is enabled. Existing file at {self._output_file} will be deleted if it exists."
            )
            self._output_file.unlink(missing_ok=True)

        if not self._output_file.exists():
            self._write()
        return self.read()

    def read(self):
        return MsgpackDatasetReader(
            msgpack_files=[str(self._output_file)],
            data_model=self._data_model,
        )

    def _write(self):
        from datadings.writer import FileWriter

        try:
            dataloader = self._get_dataloader()
            total_sample = len(self._dataset_reader)
            self._output_file.parent.mkdir(parents=True, exist_ok=True)
            with FileWriter(
                self._output_file,
                overwrite=True,
            ) as writer:
                for batch in tqdm.tqdm(
                    dataloader,
                    desc=f"Preprocessing dataset to {self._output_file} with total samples {total_sample}",
                ):
                    for sample_or_sample_list in batch:
                        sample_list = (
                            [sample_or_sample_list]
                            if not isinstance(sample_or_sample_list, list)
                            else sample_or_sample_list
                        )
                        for sample in sample_list:
                            try:
                                sample_dict = (
                                    sample.to_dict()
                                    if hasattr(sample, "to_dict")
                                    else sample.model_dump()
                                )
                                writer.write(
                                    {
                                        "key": sample.sample_id,
                                        **sample_dict,
                                    }
                                )
                            except ValueError as e:
                                logger.error(
                                    f"[WriteError] Failed to write sample '{getattr(sample, 'sample_id', 'unknown')}': {e}"
                                )
                                continue
        except Exception as e:
            logger.error(f"Error while writing preprocessed data: {e}")
            self._output_file.unlink(missing_ok=True)
            raise e
        except KeyboardInterrupt as e:
            logger.error("Preprocessing interrupted by user.")
            self._output_file.unlink(missing_ok=True)
            raise e
