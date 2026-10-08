from __future__ import annotations

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn import ENV
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: PreprocessDataset,
):
    import torch

    from docdjinn.data import load_data_pipeline, load_preprocessed_data_pipeline

    # setup data pipeline and dataloaders with preprocessing
    # this will save preprocessed msgpacks
    preprocessed_dataset = load_preprocessed_data_pipeline(
        dataset_name=cfg.dataset_name,
        collate_fn=None,
    )

    if cfg.run_sanity_check:
        logger.info(
            "Running sanity check between preprocessed and unpreprocessed data..."
        )
        # for sanity check we make sure before and after preprocessing
        # we get the same results in the first unshuffled batch
        unprocessed_dataset = load_data_pipeline(
            dataset_name=cfg.dataset_name,
            collate_fn=None,
        )

        # get preprocessed batches
        batches = {"train": {}, "validation": {}, "test": {}}
        for split, dataloader in zip(
            ["train", "validation", "test"],
            [
                preprocessed_dataset.train_dataloader(batch_size=16, shuffle=False),
                preprocessed_dataset.validation_dataloader(batch_size=16),
                preprocessed_dataset.test_dataloader(batch_size=16),
            ],
        ):
            if dataloader is None:
                continue
            logger.info(f"Loading preprocessed batches from {split}...")
            batches[split]["preprocessed"] = next(iter(dataloader))

        for split, dataloader in zip(
            ["train", "validation", "test"],
            [
                unprocessed_dataset.train_dataloader(batch_size=16, shuffle=False),
                unprocessed_dataset.validation_dataloader(batch_size=16),
                unprocessed_dataset.test_dataloader(batch_size=16),
            ],
        ):
            if dataloader is None:
                continue
            logger.info(f"Loading unprocessed batches from {split}...")
            batches[split]["unprocessed"] = next(iter(dataloader))

        for split, data in batches.items():
            logger.info(f"Running sanity check for {split} split...")
            if "preprocessed" not in data or "unprocessed" not in data:
                logger.warning(
                    f"Skipping sanity check for {split} split as one of the dataloaders is missing."
                )
                continue
            for s1, s2 in zip(data["preprocessed"], data["unprocessed"]):

                def compare_values(val1, val2, name):
                    if val1 is None and val2 is None:
                        return
                    if torch.is_tensor(val1) and torch.is_tensor(val2):
                        assert torch.allclose(val1, val2), (
                            f"{name} do not match after preprocessing."
                        )
                    else:
                        assert val1 == val2, f"{name} do not match after preprocessing."

                compare_values(s1.token_ids, s2.token_ids, "Token IDs")
                compare_values(s1.token_bboxes, s2.token_bboxes, "Token bounding boxes")
                compare_values(s1.token_type_ids, s2.token_type_ids, "Token type IDs")
                compare_values(s1.token_labels, s2.token_labels, "Token labels")
                compare_values(s1.attention_mask, s2.attention_mask, "Attention masks")
                compare_values(s1.word_ids, s2.word_ids, "Word IDs")
                compare_values(s1.sequence_ids, s2.sequence_ids, "Sequence IDs")
                compare_values(
                    s1.overflow_to_sample_mapping,
                    s2.overflow_to_sample_mapping,
                    "Overflow to sample mappings",
                )
                compare_values(s1.index, s2.index, "Indices")
                compare_values(s1.sample_id, s2.sample_id, "Sample IDs")
                compare_values(s1.image, s2.image, "Images")
                compare_values(s1.label, s2.label, "Labels")
                compare_values(s1.words, s2.words, "Words")
                compare_values(s1.question_id, s2.question_id, "Question IDs")
                compare_values(s1.qa_question, s2.qa_question, "QA questions")
                compare_values(s1.qa_answers, s2.qa_answers, "QA answers")
                compare_values(
                    s1.token_answer_start, s2.token_answer_start, "Token answer starts"
                )


class PreprocessDataset(pydantic.BaseModel):
    """
    Configuration for visualizing dataset samples.
    """

    dataset_name: str
    root_datasets_dir: str = ENV.BASE_DATASETS_DIR
    n_samples: int = 16
    run_sanity_check: bool = True


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=PreprocessDataset,
    )
    main(parser.parse_typed_args())
