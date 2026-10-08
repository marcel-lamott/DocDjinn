from __future__ import annotations

from pathlib import Path

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.analyzation.clustering.core._embeddings import (
    _load_sample_ids_from_embeddings,
    _save_embeddings,
    embedding_extraction_with_cache,
)
from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
from docdjinn.data._core._utilities import TaskType
from docdjinn.data.interface import load_data_pipeline, load_preprocessed_data_pipeline
from docdjinn.evaluation.utils import get_device
from docdjinn.logging import get_logger

logger = get_logger(__name__)


class GenerateEmbeddingsConfig(pydantic.BaseModel):
    """
    Configuration for generating embeddings.
    """

    dataset_name: str
    is_synth: bool = False
    output_dir: str = ENV.EMBEDDINGS_DIR
    kernel_size: int = 4
    split: str = "train"
    batch_size: int = 16
    dataloader_num_workers: int = 8
    use_preprocessed: bool = False
    verify_only: bool = False
    is_synthetic: bool = False


def main(cfg: GenerateEmbeddingsConfig):
    # setup data pipeline and dataloaders with preprocessing
    # this will save preprocessed msgpacks
    if cfg.use_preprocessed:
        data_pipeline = load_preprocessed_data_pipeline(
            dataset_name=cfg.dataset_name,
            is_synthetic=cfg.is_synth,
            task_type=TaskType.generate_embeddings,
            split=cfg.split,
            is_synthetic=cfg.is_synthetic,
        )
    else:
        data_pipeline = load_data_pipeline(
            dataset_name=cfg.dataset_name,
            is_synthetic=cfg.is_synth,
            task_type=TaskType.generate_embeddings,
            split=cfg.split,
            is_synthetic=cfg.is_synthetic,
        )

    if cfg.verify_only:
        output_dir = Path(cfg.output_dir) / cfg.dataset_name
        sample_ids_per_type = {}
        for embedding_type in list(EmbeddingType):
            cache_file = Path(output_dir) / f"{embedding_type.value}.h5"
            if not cache_file.exists():
                logger.warning(
                    f"Cache file {cache_file} does not exist. Please run the script "
                    "without --verify_only to generate embeddings."
                )
                continue
            sample_ids = _load_sample_ids_from_embeddings(cache_file)
            logger.info(
                f"Cache file {cache_file} exists with {len(sample_ids)} samples."
            )
            sample_ids_per_type[embedding_type.value] = sample_ids

        # make sure sample ids are the same across all types
        sample_ids = sample_ids_per_type[
            sample_ids_per_type.keys().__iter__().__next__()
        ]
        for embedding_type, ids in sample_ids_per_type.items():
            assert ids == sample_ids, (
                f"Sample IDs for {embedding_type} do not match those for "
                f"{EmbeddingType.layout.value}"
            )

        logger.info(f"All cache files exist for dataset {cfg.dataset_name}.")
        return

    # print dataset info
    logger.info(data_pipeline.dataset)

    # setup dataloader
    dataloader = data_pipeline.dataloader(
        split=cfg.split,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader_num_workers,
    )

    # check whether batch in the dataset has ocr content
    batch = next(iter(dataloader))
    has_ocr_content = batch.words is not None

    output_dir = Path(cfg.output_dir) / cfg.dataset_name
    embeddings_per_type = {}
    sample_ids_per_type = {}
    for embedding_type in list(EmbeddingType):
        if embedding_type == EmbeddingType.combined:
            continue
        if (
            embedding_type
            in [EmbeddingType.layout, EmbeddingType.text, EmbeddingType.paper]
            and not has_ocr_content
        ):
            logger.warning(
                f"Skipping {embedding_type.value} embeddings for dataset {cfg.dataset_name} "
                "as it does not have OCR content."
            )
            continue
        embeddings, sample_ids = embedding_extraction_with_cache(
            dataloader=dataloader,
            output_dir=output_dir,
            embedding_type=embedding_type,
            device=get_device(),
        )
        embeddings_per_type[embedding_type.value] = embeddings
        sample_ids_per_type[embedding_type.value] = sample_ids
        logger.info(
            f"Generated {embedding_type.value} embeddings for {len(sample_ids)} samples."
        )

    # make sure sample ids are the same across all types
    sample_ids = sample_ids_per_type[sample_ids_per_type.keys().__iter__().__next__()]
    print("Sample ids of first 10 samples: ", sample_ids[:10])
    for embedding_type, ids in sample_ids_per_type.items():
        assert ids == sample_ids, (
            f"Sample IDs for {embedding_type} do not match those for "
            f"{EmbeddingType.layout.value}"
        )

    if not has_ocr_content:
        logger.warning(
            f"Skipping {EmbeddingType.combined.value} embeddings for dataset {cfg.dataset_name} "
            "as it does not have OCR content."
        )
        return
    cache_file = Path(output_dir) / f"{EmbeddingType.combined.value}.h5"
    if not cache_file.exists():
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        embeddings_per_type = {
            k: StandardScaler().fit_transform(v) for k, v in embeddings_per_type.items()
        }

        combined_embeddings = np.hstack(
            [
                v
                for k, v in embeddings_per_type.items()
                if k
                in [
                    EmbeddingType.layout.value,
                    EmbeddingType.text.value,
                    EmbeddingType.image.value,
                ]
            ]
        )

        logger.info(
            f"Generated {EmbeddingType.combined.value} embeddings for {len(sample_ids)} samples."
        )
        _save_embeddings(
            embeddings=combined_embeddings,
            sample_ids=sample_ids,
            file_path=Path(output_dir) / f"{EmbeddingType.combined.value}.h5",
        )


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=GenerateEmbeddingsConfig,
    )
    main(parser.parse_typed_args())
