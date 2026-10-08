from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic_argparse

from docdjinn.data._core._utilities import TaskType
from docdjinn.data.interface import get_dataset_config, load_mixed_data_pipeline
from docdjinn.evaluation.runners._config import MixedRunnerConfig
from docdjinn.evaluation.runners.runner_v2 import Runner
from docdjinn.evaluation.runners.utilities import (
    prepare_transform_kwargs,
)
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


class MixedRunner(Runner):
    def setup_data_pipeline(self):
        """Setup and return the data pipeline."""
        logger.info("=" * 50)
        logger.info("BUILDING DATA PIPELINE")
        logger.info("=" * 50)

        self.config: MixedRunnerConfig

        task_type = get_dataset_config(self.config.dataset_name).task_type
        if task_type == TaskType.extractive_qa:
            assert self.config.use_preprocessed_dataset, (
                "Extractive QA datasets require preprocessed datasets. Pass the --use-preprocessed-dataset flag"
            )

        transform_kwargs = prepare_transform_kwargs(self.config)
        data_pipeline = load_mixed_data_pipeline(
            dataset_name=self.config.dataset_name,
            synthetic_dataset_name=self.config.synthetic_dataset_name,
            num_real_samples=self.config.num_real_samples,
            num_synthetic_samples=self.config.num_synthetic_samples,
            load_preprocessed=self.config.use_preprocessed_dataset,
            dataset_splitting_enabled=False,  # for now we don't use train/val split in mixed datasets
            **transform_kwargs,
        )

        # Log detailed data pipeline information
        logger.info(f"Dataset: {data_pipeline.dataset}")
        logger.info(f"Dataset metadata: {data_pipeline.dataset_metadata}")
        logger.info(f"Dataset labels: {data_pipeline.dataset_metadata.dataset_labels}")

        return data_pipeline


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=MixedRunnerConfig,
    )
    MixedRunner(parser.parse_typed_args()).run()
