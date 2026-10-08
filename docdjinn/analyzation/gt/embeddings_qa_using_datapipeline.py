"""
TODO: include answers in QA GT embeddings?
"""

from __future__ import annotations
import h5py
import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar
import numpy as np
import tqdm
from docdjinn import ENV
from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
from docdjinn.data._core._data_types import DocumentInstanceModelInput
from docdjinn.logging import get_logger
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
from docdjinn.data.interfaces.synthetic_data import (
    prepare_synthetic_dataset,
)
from docdjinn.data.interfaces.data_pipeline import (
    load_preprocessed_data_pipeline,
)
from typing import Literal
from docdjinn.data._core._utilities import TaskType
from docdjinn.data.interface import load_transform
from docdjinn.generation.models import (
    SyntheticDatasetFileStructure,
    SynDatasetDefinition,
)
from docdjinn.data._core._dataset import Dataset
from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader

T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)
if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _iterate_dataset(
    model_fn: Callable,
    embedding_fn: Callable,
    dataloader: "DataLoader",
    device: str = "cuda",
):
    """Inner function that actually generates the embeddings."""
    import torch

    model = model_fn()
    model.to(device)
    model.eval()
    print("Model is on:", next(model.parameters()).device)

    sample_ids = []
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting embeddings"):
            batch_dict = batch.to_dict()
            batch: DocumentInstanceModelInput
            batch = batch.select_first_overflow_samples()
            batch = batch.to(device)

            token_bboxes = batch.token_bboxes
            if token_bboxes is not None:
                if token_bboxes.min() >= 0 and token_bboxes.max() <= 1.0:
                    # if bboxes are normalized to [0, 1], convert to [0, 1000] as expected by layoutlmv3
                    token_bboxes = (token_bboxes * 1000).long()
                else:
                    logger.warning(
                        f"Token bboxes must be in the range [0, 1], but got min {token_bboxes.min()} and max {token_bboxes.max()}"
                    )
                    token_bboxes = (token_bboxes.clip(0, 1.0) * 1000).long()

                # assert check
                assert token_bboxes.min() >= 0 and token_bboxes.max() <= 1000, (
                    f"Token bboxes must be in the range [0, 1000], but got min {token_bboxes.min()} and max {token_bboxes.max()}"
                )

            # make sure if image is normlized 0-1 as in layoutlm we renormalize using clip stats
            assert batch.image.min() >= -1.1 and batch.image.max() <= 1.1, (
                f"Image pixel values must be in the range [0, 1], but got min {batch.image.min()} and max {batch.image.max()}"
            )

            # make inputs
            inputs = dict(
                qa_answers=batch.qa_answers,
                qa_question=batch.qa_question,
                sample_ids=batch.sample_id,
            )

            embeddings.append(embedding_fn(model, inputs))

            # in our preprocessed dataset indices are always unqiue
            # but sample_ids may not be always unique in some rare cases
            sample_ids.extend(batch.sample_id)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.cpu().numpy(), sample_ids


def _extract_text_embeddings(
    dataloader: "DataLoader",
    device: str = "cuda",
):
    """Inner function that actually generates the embeddings."""

    def model_fn():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        model.to(device)
        model.eval()
        return model

    print("Extracting embeddings only for Questions.................")

    def embedding_fn(model, inputs):
        sentences = [qa_question for qa_question in inputs["qa_question"]]
        return model.encode(sentences, convert_to_tensor=True)

    question_embeddings, question_sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    print("Extracting embeddings for both Questions and Answers...............")

    def qa_embedding_fn(model, inputs):
        """I asked gpt and It said this type of approach is common in SBERT/Text-encoders"""
        sentences = [
            f"Question: {q} Answer: {a}"
            for q, a in zip(inputs["qa_question"], inputs["qa_answers"])
        ]
        return model.encode(sentences, convert_to_tensor=True)

    qa_embeddings, qa_sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=qa_embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return dict(
        question_embeddings=question_embeddings,
        question_sample_ids=question_sample_ids,
        qa_embeddings=qa_embeddings,
        qa_sample_ids=qa_sample_ids,
    )


def embedding_extraction_with_cache(  # MsgpackDatasetReader[T_BaseDataInstance] | None
    dataloader: "DataLoader",
    output_dir: str | Path,
    embedding_type: EmbeddingType,
    device: str = "cuda",
    cache_outputs: bool = True,
    load_embeddings: Literal[
        "question_only", "QA"
    ] = "question_only",  # used to load embeddings from chache
):
    """By default it returns question only embeddings"""
    """Generic cacher function that handles caching logic for any embedding type."""
    if load_embeddings == "QA":
        cache_file = Path(output_dir) / f"QA_{embedding_type.value}.h5"
    elif load_embeddings == "question_only":
        cache_file = Path(output_dir) / f"Q_{embedding_type.value}.h5"

    if cache_outputs and cache_file.exists():
        logger.info(
            f"Loading cached {load_embeddings}_{embedding_type.value} embeddings from {cache_file}"
        )
        return _load_embeddings(cache_file)

    extraction_func = _extract_text_embeddings
    all_embeddings = extraction_func(dataloader, device)

    # Question only embeddings
    question_embeddings = all_embeddings["question_embeddings"]
    question_sample_ids = all_embeddings["question_sample_ids"]

    # Question + Answer embeddings
    qa_embeddings = all_embeddings["qa_embeddings"]
    qa_sample_ids = all_embeddings["qa_sample_ids"]

    if cache_outputs:
        """Checking that embeddings and sample_ids have same length"""
        assert len(question_sample_ids) == question_embeddings.shape[0], logger.warning(
            f"[Error in Questuion only Embedding] Number of sample IDs ({len(question_sample_ids)}) must match number of embeddings ({question_embeddings.shape[0]})"
        )

        assert len(qa_sample_ids) == qa_embeddings.shape[0], logger.warning(
            f"[Error in QA Embedding] Number of sample IDs ({len(qa_sample_ids)}) must match number of embeddings ({qa_embeddings.shape[0]})"
        )
        """Checking that sample_ids are unique"""
        assert len(set(question_sample_ids)) == len(question_sample_ids), (
            logger.warning(
                "[ERROR in Question only Embedding] Sample IDs must be unique"
            )
        )
        assert len(set(qa_sample_ids)) == len(qa_sample_ids), logger.warning(
            "[ERROR in QA Embedding] Sample IDs must be unique"
        )
        """Saving question only embeddings"""
        _save_embeddings(
            embeddings=question_embeddings,
            sample_ids=question_sample_ids,
            file_path=Path(output_dir) / f"Q_{embedding_type.value}.h5",
        )
        """Saving QA only embeddings"""
        _save_embeddings(
            embeddings=qa_embeddings,
            sample_ids=qa_sample_ids,
            file_path=Path(output_dir) / f"QA_{embedding_type.value}.h5",
        )
        return _load_embeddings(cache_file)

    return question_embeddings, question_sample_ids


def _save_embeddings(embeddings: "np.ndarray", sample_ids: list[str], file_path: Path):
    import h5py

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("sample_ids", data=sample_ids)


def _load_embeddings(file_path: Path):
    import h5py

    print(f"Loading embeddings from {file_path}")

    with h5py.File(file_path, "r") as f:
        sample_ids = f["sample_ids"][:]
        embeddings = f["embeddings"][:]
    return embeddings, [
        s.decode("utf-8") if isinstance(s, bytes) else s for s in sample_ids
    ]


def main(dataset_name: str, is_synth: bool):
    if is_synth:
        ymal_file = ENV.SYN_DATA_DEFINITIONS_DIR / f"{dataset_name}.yaml"
        dsdef: SynDatasetDefinition = SynDatasetDefinition.from_file(
            yaml_path=ymal_file
        )
        prepare_synthetic_dataset(dsdef=dsdef)

    data_pipeline = load_preprocessed_data_pipeline(
        dataset_name=dataset_name,
        # task_type=TaskType.generate_embeddings,
        is_synthetic=is_synth,
    )

    train_dataloader = data_pipeline.train_dataloader(batch_size=512, num_workers=2)

    output_dir = ENV.GT_EMBEDDINGS_DIR / dataset_name
    embedding, sample_ids = embedding_extraction_with_cache(
        dataloader=train_dataloader,
        output_dir=output_dir,
        embedding_type=EmbeddingType.text,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GT embeddings")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., docvqa, mysynthetic, pubtabnet)",
    )

    parser.add_argument(
        "--is_synth",
        action="store_true",
        help="If set, determines that the dataset is a synthetic dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset_name=args.dataset, is_synth=args.is_synth)
