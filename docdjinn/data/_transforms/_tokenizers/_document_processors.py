from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from pydantic import ConfigDict, Field

from docdjinn.data._core._data_types import (
    DocumentInstance,
    DocumentInstanceModelInput,
)
from docdjinn.data._transforms._generics._base import BaseTransform
from docdjinn.data._transforms._generics._hf_processor import HuggingfaceProcessor
from docdjinn.data._transforms._generics._image_processor import ImageProcessor
from docdjinn.logging import get_logger

from ._utilities import (
    _document_instance_to_hf_processor_inputs,
    _extract_annotations,
    _generate_qa_token_ids,
    _post_process_tokenizer_outputs,
)

logger = get_logger(__name__)


class BaseDocumentProcessor(BaseTransform[DocumentInstanceModelInput]):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    # tokenizer args
    tokenizer_name: str = "microsoft/layoutlmv3-base"
    init_kwargs: dict = Field(default_factory=dict)
    call_kwargs: dict = Field(default_factory=dict)
    overflow_sampling: str = "return_all"
    max_overflow_samples: int = 10
    use_segment_level_bboxes: bool = False
    cache_dir: str = "./cache"

    # image processor args
    do_normalize: bool = True  # Normalize the image to ImageNet mean and std
    do_resize: bool = True  # Resize the image to 224x224
    use_imagenet_mean_std: bool = False
    resize_height: int = 224
    resize_width: int = 224
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    # segment-level-rank info args
    add_segment_level_info: bool = False
    max_segment_num: int = 150

    def model_post_init(self, context) -> None:
        self._hf_processor = HuggingfaceProcessor(
            tokenizer_name=self.tokenizer_name,
            init_kwargs=self.init_kwargs,
            call_kwargs=self.call_kwargs,
            overflow_sampling=self.overflow_sampling,
            cache_dir=self.cache_dir,
        )
        self._image_transform = ImageProcessor(
            do_normalize=self.do_normalize,
            do_resize=self.do_resize,
            use_imagenet_mean_std=self.use_imagenet_mean_std,
            resize_height=self.resize_height,
            resize_width=self.resize_width,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )

    def get_output_data_model(self):
        return DocumentInstanceModelInput

    def __call__(
        self, document_instance: DocumentInstance
    ) -> DocumentInstanceModelInput | list[DocumentInstanceModelInput]:
        hf_processor_inputs = _document_instance_to_hf_processor_inputs(
            document_instance,
            use_segment_level_bboxes=self.use_segment_level_bboxes,
            image_transform=self._image_transform,
        )
        tokenization_data = self._hf_processor(**hf_processor_inputs)
        processed_outputs = _post_process_tokenizer_outputs(
            tokenization_data=tokenization_data,
            input_word_boxes=hf_processor_inputs.get("boxes", None),
            input_word_labels=hf_processor_inputs.get("word_labels", None),
            input_image=hf_processor_inputs.get("images", None),
            add_segment_level_info=self.add_segment_level_info,
            all_special_ids=self._hf_processor.tokenizer.all_special_ids,
            max_segment_num=self.max_segment_num,
        )
        return DocumentInstanceModelInput(
            index=torch.tensor(document_instance.index)
            if document_instance.index is not None
            else None,
            sample_id=document_instance.sample_id,
            words=hf_processor_inputs.pop("text", None),
            tokenizer_config=self._hf_processor.get_config(),
            **processed_outputs,
        )


class SequenceClassificationDocumentProcessor(BaseDocumentProcessor):
    def __call__(
        self, document_instance: DocumentInstance
    ) -> DocumentInstanceModelInput | list[DocumentInstanceModelInput]:
        instance = super().__call__(document_instance)
        annotations = _extract_annotations(document_instance)
        assert annotations.label is not None, "No label found in the document instance."
        if isinstance(instance, list):
            return [
                replace(
                    inst,
                    label=torch.tensor(annotations.label.value),
                )
                for inst in instance
            ]
        return replace(
            instance,
            label=torch.tensor(annotations.label.value),
        )


class TokenClassificationDocumentProcessor(BaseDocumentProcessor):
    pass


class QuestionAnsweringDocumentProcessor(BaseDocumentProcessor):
    ignore_samples_with_no_answer: bool = False
    is_training: bool = False

    def model_post_init(self, context) -> None:
        # update call kwargs
        self.call_kwargs["truncation"] = "only_second"

        super().model_post_init(context)

    def _is_no_answer_sample(
        self, token_answer_start, token_answer_end, tokenization_data
    ):
        total_answers = len(token_answer_start)
        for key, value in tokenization_data.items():
            if value is None:
                continue
            if key == "image":
                continue
            assert len(value) == total_answers, (
                f"Length mismatch in tokenization data for key {key}. "
                f"Expected length: {total_answers}, Actual length: {len(value)}"
            )

        valid_indices = []
        for idx, (s, e) in enumerate(zip(token_answer_start, token_answer_end)):
            if s != -1 and e != -1:
                valid_indices.append(idx)

        if len(valid_indices) == 0:
            return True  # skip this sample entirely

        if len(valid_indices) < total_answers:
            tokenization_data = {
                k: v[valid_indices] if v is not None and k not in ["image"] else v
                for k, v in tokenization_data.items()
            }
            token_answer_start = token_answer_start[valid_indices]
            token_answer_end = token_answer_end[valid_indices]

        assert (np.array(token_answer_end) != -1).all(), (
            f"Some end answer indices are -1 in document {token_answer_end}"
        )
        assert (np.array(token_answer_start) != -1).all(), (
            f"Some start answer indices are -1 in document {token_answer_start}"
        )
        total_answers = len(token_answer_start)
        for key, value in tokenization_data.items():
            if value is None:
                continue
            if key == "image":
                continue
            assert len(value) == total_answers, (
                f"Length mismatch in tokenization data for key {key}. "
                f"Expected length: {total_answers}, Actual length: {len(value)}"
            )
        return False

    def __call__(
        self, document_instance: DocumentInstance
    ) -> DocumentInstanceModelInput | list[DocumentInstanceModelInput]:
        qa_pairs = _extract_annotations(document_instance).qa_pairs
        assert qa_pairs is not None, "No QA pairs found in the document instance."
        assert len(qa_pairs) > 0, "No QA pairs found in the document instance."

        transformed_instances = []
        for qa_pair_index in range(len(qa_pairs)):
            # prepare model input
            hf_processor_inputs = _document_instance_to_hf_processor_inputs(
                document_instance,
                use_segment_level_bboxes=self.use_segment_level_bboxes,
                image_transform=self._image_transform,
                context=qa_pairs[qa_pair_index].question_text,
            )

            text_pair = hf_processor_inputs.get("text_pair", None)
            boxes = hf_processor_inputs.get("boxes", None)
            assert len(text_pair) == len(boxes), (
                f"Length mismatch between text_pair and boxes for sample {document_instance.sample_id}. "
                f"Length of text_pair: {len(text_pair)}, Length of boxes: {len(boxes)}"
            )

            tokenization_data = self._hf_processor(**hf_processor_inputs)
            processed_outputs = _post_process_tokenizer_outputs(
                tokenization_data=tokenization_data,
                input_word_boxes=hf_processor_inputs.get("boxes", None),
                input_word_labels=hf_processor_inputs.get("word_labels", None),
                input_image=hf_processor_inputs.get("images", None),
                add_segment_level_info=self.add_segment_level_info,
                all_special_ids=self._hf_processor.tokenizer.all_special_ids,
                max_segment_num=self.max_segment_num,
            )

            token_answer_start, token_answer_end = _generate_qa_token_ids(
                qa_pair=qa_pairs[qa_pair_index],
                word_ids=processed_outputs["word_ids"],
                sequence_ids=processed_outputs["sequence_ids"],
                sequence_length=processed_outputs["token_ids"].shape[-1],
            )

            # if all token_answer_start and token_answer_end are 0, it means we could not find the answer in the context
            # therefore using this sample as a training sample will not help the model learn anything
            if self.is_training and self.ignore_samples_with_no_answer:
                total_answers = len(token_answer_start)
                for key, value in processed_outputs.items():
                    if value is None:
                        continue
                    if key == "image":
                        continue
                    assert len(value) == total_answers, (
                        f"Length mismatch in tokenization data for key {key}. "
                        f"Expected length: {total_answers}, Actual length: {len(value)}"
                    )

                valid_indices = []
                for idx, (s, e) in enumerate(zip(token_answer_start, token_answer_end)):
                    if s != -1 and e != -1:
                        valid_indices.append(idx)

                if len(valid_indices) == 0:
                    continue  # skip this sample entirely

                if len(valid_indices) < total_answers:
                    processed_outputs = {
                        k: v[valid_indices]
                        if v is not None and k not in ["image"]
                        else v
                        for k, v in processed_outputs.items()
                    }
                    token_answer_start = token_answer_start[valid_indices]
                    token_answer_end = token_answer_end[valid_indices]

                assert (np.array(token_answer_end) != -1).all(), (
                    f"Some end answer indices are -1 in document {token_answer_end}"
                )
                assert (np.array(token_answer_start) != -1).all(), (
                    f"Some start answer indices are -1 in document {token_answer_start}"
                )
                total_answers = len(token_answer_start)
                for key, value in processed_outputs.items():
                    if value is None:
                        continue
                    if key == "image":
                        continue
                    assert len(value) == total_answers, (
                        f"Length mismatch in tokenization data for key {key}. "
                        f"Expected length: {total_answers}, Actual length: {len(value)}"
                    )

            # make sure afterwards we always have one length for all processed outputs
            sample_id = document_instance.sample_id + "_subsample_" + str(qa_pair_index)
            transformed_instance = DocumentInstanceModelInput(
                index=torch.tensor(document_instance.index)
                if document_instance.index is not None
                else None,
                sample_id=sample_id,
                words=hf_processor_inputs.pop("text_pair", None),
                question_id=qa_pair_index,
                qa_question=qa_pairs[qa_pair_index].question_text,
                qa_answers=qa_pairs[qa_pair_index].answer_text,
                token_answer_start=token_answer_start,
                token_answer_end=token_answer_end,
                tokenizer_config=self._hf_processor.get_config(),
                **processed_outputs,
            )
            transformed_instances.append(transformed_instance)
        return transformed_instances
