from __future__ import annotations

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: TestModelPipeline,
):
    from docdjinn.data import (
        load_data_pipeline,
        load_preprocessed_data_pipeline,
        TaskType,
        DATASET_TASK_MAP,
    )
    from docdjinn.evaluation.model_pipeline import load_model_pipeline

    # setup data pipeline and dataloaders
    logger.info("Saving samples from split [train]...")
    data_pipeline = load_data_pipeline(
        dataset_name=cfg.dataset_name,
    )

    train_dataloader, validation_dataloader, test_dataloader = (
        data_pipeline.train_dataloader(batch_size=cfg.n_samples),
        data_pipeline.validation_dataloader(batch_size=cfg.n_samples),
        data_pipeline.test_dataloader(batch_size=cfg.n_samples),
    )

    task_type = DATASET_TASK_MAP.get(cfg.dataset_name)
    model_pipeline = load_model_pipeline(
        task_type=task_type,
        model_name=cfg.model_name,
        model_cache_dir=cfg.model_cache_dir,
        dataset_metadata=data_pipeline.dataset_metadata,
    )

    logger.info(f"Loaded model pipeline:\n{model_pipeline}")

    for split, dataloader in zip(
        ["train", "validation", "test"],
        [train_dataloader, validation_dataloader, test_dataloader],
    ):
        logger.info(f"Performing a single step on split [{split}]...")
        if dataloader is not None:
            batch = next(iter(dataloader))
            if split == "train":
                outputs = model_pipeline.training_step(batch)
            else:
                outputs = model_pipeline.evaluation_step(batch)
            logger.info(f"Outputs:\n{outputs}")
        else:
            logger.warning(f"No dataloader found for split [{split}]!")


class TestModelPipeline(pydantic.BaseModel):
    dataset_name: str
    model_name: str = "microsoft/layoutlmv3-base"
    model_cache_dir: str = ENV.MODELS_DIR
    root_datasets_dir: str = ENV.BASE_DATASETS_DIR
    n_samples: int = 16


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=TestModelPipeline,
    )
    main(parser.parse_typed_args())
