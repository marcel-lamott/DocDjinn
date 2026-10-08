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
from docdjinn.data.interfaces.dataset import load_dataset
from docdjinn.logging import get_logger
from atria_core.types.data_instance.base import (
    BaseDataInstance,
)
from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
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

import numpy as np
from torch.utils.data import DataLoader

T_BaseDataInstance = TypeVar("T_BaseDataInstance", bound=BaseDataInstance)


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
    doc_ids = []
    questions = []
    answers = []
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting embeddings"):
            embeddings.append(embedding_fn(model, batch))
            sample_ids.extend(batch["sample_ids"])
            doc_ids.extend(batch["doc_ids"])
            questions.extend(batch["questions"])
            answers.extend(batch["answers"])

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.cpu().numpy(), questions, answers, sample_ids, doc_ids


def _extract_text_embeddings(dataloader: "DataLoader", device: str = "cuda"):
    """Inner function that actually generates the embeddings."""

    def model_fn():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        model.to(device)
        model.eval()
        return model

    def embedding_fn(model, inputs):
        sentences = [qa_question for qa_question in inputs["questions"]]
        return model.encode(sentences, convert_to_tensor=True)

    question_embeddings, questions, answers, sample_ids, doc_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    def qa_embedding_fn(model, inputs):
        sentences = [
            f"Question: {q} Answer: {a}"
            for q, a in zip(inputs["questions"], inputs["answers"])
        ]
        return model.encode(sentences, convert_to_tensor=True)

    qa_embeddings, questions, answers, sample_ids, doc_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=qa_embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return question_embeddings, qa_embeddings, questions, answers, sample_ids, doc_ids


def extract_embeddings(  # MsgpackDatasetReader[T_BaseDataInstance] | None
    dataloader: "DataLoader",
    output_dir: Path,
    device: str = "cuda",
):
    q_path = output_dir / f"Q.h5"
    qa_path = output_dir / f"QA.h5"

    if q_path.exists() and qa_path.exists():
        print(f"Found existing QA embeddings at {q_path} and {qa_path} - SKIPPING")
    else:
        extraction_func = _extract_text_embeddings
        question_embeddings, qa_embeddings, questions, answers, sample_ids, doc_ids = (
            extraction_func(dataloader, device)
        )

        _save_embeddings(
            embeddings=question_embeddings,
            questions=questions,
            answers=answers,
            sample_ids=sample_ids,
            document_ids=doc_ids,
            file_path=q_path,
        )

        _save_embeddings(
            embeddings=qa_embeddings,
            questions=questions,
            answers=answers,
            sample_ids=sample_ids,
            document_ids=doc_ids,
            file_path=Path(output_dir) / f"QA.h5",
        )


def _save_embeddings(
    embeddings: "np.ndarray",
    questions: list[str],
    answers: list[str],
    sample_ids: list[str],
    document_ids: list[str],
    file_path: Path,
):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("questions", data=questions)
        f.create_dataset("answers", data=answers)
        f.create_dataset("sample_ids", data=sample_ids)
        f.create_dataset("document_ids", data=document_ids)


def load_qa_embeddings(dataset_name: str, embedding_type: Literal["Q", "QA"]):
    file_path: Path = ENV.GT_EMBEDDINGS_DIR / dataset_name / f"{embedding_type}.h5"
    print(f"Loading embeddings from {file_path}")

    def decode_str_collection(col):
        return [s.decode("utf-8") if isinstance(s, bytes) else s for s in col]

    with h5py.File(file_path, "r") as f:
        embeddings = f["embeddings"][:]
        questions = f["questions"][:]
        answers = f["answers"][:]
        sample_ids = f["sample_ids"][:]
        doc_ids = f["document_ids"][:]

    return (
        embeddings,
        decode_str_collection(questions),
        decode_str_collection(answers),
        decode_str_collection(sample_ids),
        decode_str_collection(doc_ids),
    )


def collate_fn_extract_questions(batch):
    all_questions = []
    all_answers = []
    all_doc_ids = []
    all_sample_ids = []

    for doc in batch:
        for a in doc.annotations:
            for qa in a.qa_pairs:
                all_questions.append(qa.question_text)
                all_answers.append(qa.answer_text[0])
                all_doc_ids.append(doc.sample_id)
                all_sample_ids.append(f"{doc.sample_id}_{qa.id}")

    return {
        "questions": all_questions,
        "answers": all_answers,
        "sample_ids": all_sample_ids,
        "doc_ids": all_doc_ids,
    }


def main(dataset_name: str, is_synth: bool):
    if is_synth:
        ymal_file = ENV.SYN_DATA_DEFINITIONS_DIR / f"{dataset_name}.yaml"
        dsdef: SynDatasetDefinition = SynDatasetDefinition.from_file(
            yaml_path=ymal_file
        )
        # prepare_synthetic_dataset(dsdef=dsdef)

    # data_pipeline = load_preprocessed_data_pipeline(
    #     dataset_name=dataset_name,
    #     # task_type=TaskType.generate_embeddings,
    #     is_synthetic=is_synth,
    # )
    # train_dataloader = data_pipeline.train_dataloader(batch_size=512, num_workers=2)

    dataset = load_dataset(dataset_name=dataset_name, is_synthetic=is_synth)
    train_dataloader = DataLoader(
        dataset=dataset.train,
        batch_size=512,
        num_workers=0,
        collate_fn=collate_fn_extract_questions,
    )

    output_dir = ENV.GT_EMBEDDINGS_DIR / dataset_name
    extract_embeddings(
        dataloader=train_dataloader,
        output_dir=output_dir,
    )

    # embeddings, questions, answers, sample_ids, doc_ids = load_qa_embeddings(
    #     dataset_name=dataset_name, embedding_type="Q"
    # )
    # for e, q, a, s, d in zip(embeddings, questions, answers, sample_ids, doc_ids):
    #     print(e, q, a, s, d)
    #     input()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GT embeddings")
    parser.add_argument(
        "dataset",
        type=str,
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
