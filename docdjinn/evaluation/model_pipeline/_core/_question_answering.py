from __future__ import annotations

from typing import Any

import torch
from atria_core.types import (
    DatasetMetadata,
)
from torch.nn.modules import Module

from docdjinn.data._core._data_types import (
    DocumentInstanceModelInput,
    OverflowStrategy,
)
from docdjinn.evaluation.model_pipeline._core._data_types import (
    QAModelOutput,
    QAPair,
)
from docdjinn.logging import get_logger

from ._base import ModelPipeline

logger = get_logger(__name__)


class QuestionAnsweringPipeline(ModelPipeline):
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        dataset_metadata: DatasetMetadata,
        pretrained: bool = True,
        use_bbox: bool = True,
        use_image: bool = True,
        training_overflow_strategy: OverflowStrategy = OverflowStrategy.select_random,
        evaluation_overflow_strategy: OverflowStrategy = OverflowStrategy.select_all,
        input_stride: int = 0,
    ):
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            dataset_metadata=dataset_metadata,
            model_cache_dir=model_cache_dir,
        )
        self._use_bbox = use_bbox
        self._use_image = use_image
        self._training_overflow_strategy = training_overflow_strategy
        self._evaluation_overflow_strategy = evaluation_overflow_strategy
        self._input_stride = input_stride

    def _build_model(
        self, model_name: str, model_cache_dir: str, pretrained: bool = True, **kwargs
    ) -> Module:
        from rich.pretty import pretty_repr
        from transformers import AutoConfig, AutoModelForQuestionAnswering

        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir
            )

            logger.info(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                config=hf_config,
                cache_dir=model_cache_dir,
            )
        else:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir
            )
            return AutoModelForQuestionAnswering.from_config(
                config=hf_config,
            )

    def _verify_and_filter_inputs(
        self,
        batch: DocumentInstanceModelInput,
    ) -> dict[str, torch.Tensor | None]:
        # assume token bboxes are in [0, 1] range
        token_bboxes = (
            (batch.token_bboxes * 1000.0).clip(0, 1000).long()
            if batch.token_bboxes is not None and self._use_bbox
            else None
        )
        pixel_values = (
            batch.image if batch.image is not None and self._use_image else None
        )
        inputs = {
            "input_ids": batch.token_ids,
            "token_type_ids": batch.token_type_ids,
            "attention_mask": batch.attention_mask,
            "bbox": token_bboxes,
            "pixel_values": pixel_values,
            "start_positions": batch.token_answer_start,
            "end_positions": batch.token_answer_end,
            "segment_index": batch.segment_index,
            "segment_inner_token_rank": batch.segment_inner_token_rank,
            "first_token_idxes": batch.first_token_idxes,
            "first_token_idxes_mask": batch.first_token_idxes_mask,
        }

        # assert that we always get the arguments
        assert inputs["input_ids"] is not None, "Attention mask cannot be None"
        assert inputs["attention_mask"] is not None, "Attention mask cannot be None"
        assert inputs["pixel_values"] is not None, "Pixel values cannot be None"
        assert inputs["bbox"] is not None, "Labels cannot be None"
        assert inputs["start_positions"] is not None, "Start positions cannot be None"
        assert inputs["end_positions"] is not None, "End positions cannot be None"

        filtered_inputs = {}
        for key in list(inputs.keys()):
            if key not in self._possible_args:
                continue
            filtered_inputs[key] = inputs[key]
        return filtered_inputs

    def _model_forward(self, batch: DocumentInstanceModelInput) -> Any:
        inputs = self._verify_and_filter_inputs(batch)
        return self._model(**inputs)

    def training_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> QAModelOutput:
        batch = batch.resolve_sample_overflow(
            overflow_strategy=self._training_overflow_strategy
        )
        output = self._model_forward(batch)
        return QAModelOutput(
            loss=output.loss,
        )

    def evaluation_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> QAModelOutput:
        from .utilities import _postprocess_qa_predictions

        questions = batch.qa_question
        batch = batch.resolve_sample_overflow(
            overflow_strategy=self._evaluation_overflow_strategy
        )
        output = self._model_forward(batch)
        pred_answers_per_question_id = _postprocess_qa_predictions(
            words=batch.words,
            word_ids=batch.word_ids.detach().cpu(),
            sequence_ids=batch.sequence_ids.detach().cpu(),
            question_ids=batch.sample_id,  # each sample has a single question index for uniqueness
            start_logits=output.start_logits.detach().cpu(),
            end_logits=output.end_logits.detach().cpu(),
        )

        qa_outputs = []
        for (qid, preds), question in zip(
            pred_answers_per_question_id.items(), questions
        ):
            if "_page_" in qid:
                sample_id = qid.split("_page_")[0]  # get the original sample id
            else:
                sample_id = qid.split("_subsample_")[0]  # get the original sample id
            answer = preds[0]["text"]  # taking the top prediction

            # we ignore samples with no answer during evaluation
            qa_outputs.append(
                QAPair(sample_id=sample_id, question=question, answer=answer)
            )

        return QAModelOutput(loss=output.loss, qa_pairs=qa_outputs)
