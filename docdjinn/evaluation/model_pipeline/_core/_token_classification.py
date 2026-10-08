from __future__ import annotations

from typing import Any

import torch
from torch.nn.modules import Module

from docdjinn.data._core._data_types import (
    DatasetMetadata,
    DocumentInstanceModelInput,
    OverflowStrategy,
)
from docdjinn.evaluation.model_pipeline._core._data_types import (
    TokenClassificationModelOutput,
)
from docdjinn.logging import get_logger

from ._base import ModelPipeline

logger = get_logger(__name__)


class TokenClassificationPipeline(ModelPipeline):
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
        import torch

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
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def _build_model(
        self,
        model_name: str,
        model_cache_dir: str,
        pretrained: bool = True,
        dataset_metadata: DatasetMetadata | None = None,
    ) -> Module:
        from rich.pretty import pretty_repr
        from transformers import AutoConfig, AutoModelForTokenClassification

        assert (
            dataset_metadata is not None
            and dataset_metadata.dataset_labels.ser is not None
        ), "Dataset labels must be provided in dataset metadata."
        self._labels = dataset_metadata.dataset_labels.ser
        num_labels = len(self._labels)
        logger.info("Using %d labels for classification.", num_labels)
        if pretrained:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir, num_labels=num_labels
            )

            logger.info(
                f"Initializing the model with the following config:\n {pretty_repr(hf_config, expand_all=True)}"
            )
            return AutoModelForTokenClassification.from_pretrained(
                model_name,
                config=hf_config,
                cache_dir=model_cache_dir,
            )
        else:
            hf_config = AutoConfig.from_pretrained(
                model_name, cache_dir=model_cache_dir, num_labels=num_labels
            )
            return AutoModelForTokenClassification.from_config(
                config=hf_config,
            )

    def _gather_target_labels(self, batch: DocumentInstanceModelInput):
        assert batch.token_labels is not None, "Token labels cannot be None"
        target_label_names = []
        target_label_values = []
        for target in batch.token_labels:
            curr_target_label_names = [self._labels[i] for i in target[target != -100]]
            target_label_names.append(curr_target_label_names)

        return target_label_names, target_label_values

    def _output_transform(
        self,
        batch: DocumentInstanceModelInput,
        model_output: TokenClassificationModelOutput,
    ) -> TokenClassificationModelOutput:
        assert batch.token_labels is not None, "Token labels cannot be None"
        target_label_names = []
        predicted_label_names = []
        predictions = model_output.logits.argmax(-1)
        for prediction, target in zip(predictions, batch.token_labels, strict=True):
            curr_target_label_names = [self._labels[i] for i in target[target != -100]]
            curr_predicted_label_names = [
                self._labels[i] for i in prediction[target != -100]
            ]
            target_label_names.append(curr_target_label_names)
            predicted_label_names.append(curr_predicted_label_names)

        return TokenClassificationModelOutput(
            loss=model_output.loss,
            logits=model_output.logits,
            predicted_label_names=predicted_label_names,
            target_label_names=target_label_names,
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
            "labels": batch.token_labels,
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
        model_output = self._model(**inputs)
        return self._output_transform(batch, model_output)

    def training_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> TokenClassificationModelOutput:
        batch = batch.resolve_sample_overflow(self._training_overflow_strategy)
        return self._model_forward(batch)

    def evaluation_step(  # type: ignore[override]
        self, batch: DocumentInstanceModelInput, **kwargs
    ) -> TokenClassificationModelOutput:
        batch = batch.resolve_sample_overflow(self._evaluation_overflow_strategy)
        return self._model_forward(batch)
