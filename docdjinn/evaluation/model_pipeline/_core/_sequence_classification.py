from __future__ import annotations

from typing import Any

import torch
from atria_core.types import (
    DatasetMetadata,
)
from torch.nn.modules import Module

from docdjinn.data._core._data_types import (
    DocumentInstanceModelInput,
)
from docdjinn.evaluation.model_pipeline._core._data_types import (
    ClassificationModelOutput,
)
from docdjinn.evaluation.model_pipeline._core.utilities import _get_logits_from_output
from docdjinn.logging import get_logger

from ._base import ModelPipeline

logger = get_logger(__name__)


class SequenceClassificationPipeline(ModelPipeline):
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        dataset_metadata: DatasetMetadata,
        pretrained: bool = True,
        use_bbox: bool = True,
        use_image: bool = True,
        input_stride: int = 0,
    ):
        import torch

        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            dataset_metadata=dataset_metadata,
            model_cache_dir=model_cache_dir,
        )
        self._use_bbox = use_bbox
        self._use_image = use_image
        self._input_stride = input_stride
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _build_model(
        self,
        model_name: str,
        model_cache_dir: str,
        pretrained: bool = True,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> Module:
        from rich.pretty import pretty_repr
        from transformers import AutoConfig, AutoModelForSequenceClassification

        assert (
            dataset_metadata is not None
            and dataset_metadata.dataset_labels.classification is not None
        ), (
            "Dataset metadata with classification labels must be provided to build the model."
        )
        self._labels = dataset_metadata.dataset_labels.classification
        num_labels = len(self._labels)
        logger.info("Using %d labels for classification.", num_labels)
        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir, num_labels=num_labels
            )

            logger.debug(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=hf_config,
                cache_dir=model_cache_dir,
            )
        else:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir, num_labels=num_labels
            )
            return AutoModelForSequenceClassification.from_config(
                config=hf_config,
            )

    def _output_transform(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        batch: DocumentInstanceModelInput,
    ) -> ClassificationModelOutput:
        assert batch.label is not None, "Labels cannot be None"
        predicted_labels = logits.argmax(dim=-1)
        return ClassificationModelOutput(
            loss=loss,
            logits=logits,
            prediction_probs=logits.softmax(dim=-1),
            gt_label_value=batch.label,
            gt_label_name=[self._labels[i] for i in batch.label.tolist()],
            predicted_label_value=predicted_labels,
            predicted_label_name=[self._labels[i] for i in predicted_labels.tolist()],
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
            "labels": batch.label,
            "segment_index": batch.segment_index,
            "segment_inner_token_rank": batch.segment_inner_token_rank,
            "first_token_idxes": batch.first_token_idxes,
            "first_token_idxes_mask": batch.first_token_idxes_mask,
        }

        # assert that we always get the arguments
        assert inputs["input_ids"] is not None, "Attention mask cannot be None"
        assert inputs["attention_mask"] is not None, "Attention mask cannot be None"
        assert inputs["pixel_values"] is not None, "Pixel values cannot be None"
        assert inputs["labels"] is not None, "Labels cannot be None"
        assert inputs["bbox"] is not None, "Labels cannot be None"

        filtered_inputs = {}
        for key in list(inputs.keys()):
            if key not in self._possible_args:
                continue
            filtered_inputs[key] = inputs[key]
        return filtered_inputs

    def _model_forward(self, batch: DocumentInstanceModelInput) -> Any:
        inputs = self._verify_and_filter_inputs(batch)
        logits = _get_logits_from_output(self._model(**inputs))
        loss = self._loss_fn(logits, batch.label)
        return self._output_transform(loss, logits, batch)

    def training_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> ClassificationModelOutput:
        batch = batch.select_first_overflow_samples()
        return self._model_forward(batch)

    def evaluation_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> ClassificationModelOutput:
        batch = batch.select_first_overflow_samples()
        return self._model_forward(batch)
