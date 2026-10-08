from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from transformers import BatchEncoding

from docdjinn.logging import get_logger

from ..._core._data_types import (
    AnnotatedObjectList,
    ClassificationAnnotation,
    DocumentInstance,
    EntityLabelingAnnotation,
    ExtractiveQAAnnotation,
    ExtractiveQAPair,
    Label,
    LabelList,
    LayoutAnalysisAnnotation,
)

logger = get_logger(__name__)


@dataclass
class Annotations:
    label: Label | None = None
    word_labels: LabelList | None = None
    qa_pairs: list[ExtractiveQAPair] | None = None
    annotated_objects: AnnotatedObjectList | None = None


def _document_instance_to_hf_processor_inputs(
    document_instance: DocumentInstance,
    use_segment_level_bboxes: bool = False,
    image_transform: Callable | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    if document_instance.content is None:
        return {}

    inputs = {}

    if context is None:
        if document_instance.content.words is not None:
            inputs["text"] = document_instance.content.words
    else:
        qa_pairs = _extract_annotations(document_instance).qa_pairs
        assert qa_pairs is not None and len(qa_pairs) > 0, (
            "No QA pairs found in the document instance for extractive QA task."
        )
        inputs["text"] = context
        inputs["text_pair"] = document_instance.content.words

    if document_instance.content.word_bboxes is not None:
        inputs["boxes"] = (
            document_instance.content.word_segment_level_bboxes.value
            if use_segment_level_bboxes
            and document_instance.content.word_segment_level_bboxes is not None
            else document_instance.content.word_bboxes.value
        )

    if document_instance.image is not None:
        inputs["images"] = (
            image_transform(document_instance.image.content)
            if image_transform is not None
            else document_instance.image.content
        )

        # extract annotations
        annotations = _extract_annotations(document_instance)
        if annotations.label is not None:
            inputs["label"] = annotations.label.value
        if annotations.word_labels is not None:
            inputs["word_labels"] = annotations.word_labels.value

    return inputs


def _extract_sequence_and_word_ids(
    tokenization_data: BatchEncoding,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence_ids = []
    word_ids = []
    input_ids = tokenization_data["input_ids"]
    num_overflow_samples = len(input_ids)  # type: ignore
    for i in range(num_overflow_samples):
        sequence_ids_per_overflow = tokenization_data.sequence_ids(i)
        word_ids_per_overflow = tokenization_data.word_ids(i)

        # filter sequence_ids
        sequence_ids_per_overflow = [
            -100 if x is None else x for x in sequence_ids_per_overflow
        ]
        word_ids_per_overflow = [
            -100 if x is None else x for x in word_ids_per_overflow
        ]
        if max(sequence_ids_per_overflow) > 0:
            word_ids_per_overflow = [
                -100 if sequence_id == 0 else word_id
                for word_id, sequence_id in zip(
                    word_ids_per_overflow, sequence_ids_per_overflow
                )
            ]
        sequence_ids.append(sequence_ids_per_overflow)
        word_ids.append(word_ids_per_overflow)

    sequence_ids = torch.tensor(sequence_ids)
    word_ids = torch.tensor(word_ids)
    return sequence_ids, word_ids


def _extract_token_bboxes_from_word_bboxes(
    word_bboxes: list[list[float]], word_ids: torch.Tensor
) -> torch.Tensor:
    token_bboxes = []
    for word_ids_per_sample in word_ids:
        token_bboxes_per_sample = [
            [0, 0, 0, 0] if word_id == -100 else word_bboxes[word_id]
            for word_id in word_ids_per_sample.tolist()
        ]
        token_bboxes.append(token_bboxes_per_sample)
    return torch.tensor(token_bboxes)


def _extract_token_labels_from_word_labels(
    word_labels: list[int], word_ids: Any
) -> torch.Tensor:
    token_labels = []
    for word_ids_per_sample in word_ids:
        token_labels_per_sample = []
        last_word_id = None
        for word_id in word_ids_per_sample.tolist():
            if word_id == -100 or word_id == last_word_id:
                token_labels_per_sample.append(-100)  # padding label
            else:
                token_labels_per_sample.append(word_labels[word_id])
            last_word_id = word_id
        token_labels.append(token_labels_per_sample)
    return torch.tensor(token_labels)


def _extract_segment_level_data(
    token_ids: torch.Tensor,
    token_bboxes: torch.Tensor,
    all_special_ids: set[int],
    max_segment_num: int = 150,
) -> Mapping[str, Any]:
    segment_index = _generate_segment_level_bbox_ranks(
        token_ids=token_ids,
        segment_level_bboxes=token_bboxes,
        all_special_ids=all_special_ids,
    )
    segment_inner_token_rank = _generate_segment_level_inner_ranks(
        line_rank_id=segment_index
    )
    first_token_idxes, first_token_idxes_mask = _generate_first_token_idxes(
        line_rank_id=segment_index, max_segment_num=max_segment_num
    )
    return {
        "segment_index": segment_index,
        "segment_inner_token_rank": segment_inner_token_rank,
        "first_token_idxes": first_token_idxes,
        "first_token_idxes_mask": first_token_idxes_mask,
    }


def _post_process_tokenizer_outputs(
    tokenization_data: BatchEncoding,
    input_word_boxes: list[list[float]] | None,
    input_word_labels: list[int] | None,
    input_image: Any | None,
    add_segment_level_info: bool = False,
    all_special_ids: set[int] = set(),
    max_segment_num: int = 150,
) -> Mapping[str, Any]:
    sequence_ids, word_ids = _extract_sequence_and_word_ids(tokenization_data)
    token_bboxes = tokenization_data.get("bbox", None)
    if token_bboxes is None and input_word_boxes is not None:
        token_bboxes = _extract_token_bboxes_from_word_bboxes(
            input_word_boxes, word_ids
        )
    token_labels = tokenization_data.get("labels", None)
    if token_labels is None and input_word_labels is not None:
        token_labels = _extract_token_labels_from_word_labels(
            input_word_labels, word_ids
        )
    image = tokenization_data.get("pixel_values", None)
    if image is not None:
        image = image[0]
    if image is None and input_image is not None:
        image = input_image

    outputs = {
        "token_ids": tokenization_data.get("input_ids"),
        "attention_mask": tokenization_data.get("attention_mask"),
        "token_bboxes": token_bboxes,
        "token_type_ids": tokenization_data.get("token_type_ids", None),
        "token_labels": token_labels,
        "sequence_ids": sequence_ids,
        "word_ids": word_ids,
        "image": image,
    }

    if add_segment_level_info:
        segment_level_data = _extract_segment_level_data(
            token_ids=outputs["token_ids"],
            token_bboxes=outputs["token_bboxes"],
            all_special_ids=all_special_ids,
            max_segment_num=max_segment_num,
        )
        outputs.update(segment_level_data)

    # assert that we have all the keys
    assert outputs["token_ids"] is not None, (
        "token_ids is None in the tokenizer outputs."
    )
    assert outputs["attention_mask"] is not None, (
        "attention_mask is None in the tokenizer outputs."
    )
    assert outputs["token_bboxes"] is not None, (
        "token_bboxes is None in the tokenizer outputs."
    )
    assert outputs["sequence_ids"] is not None, (
        "sequence_ids is None in the tokenizer outputs."
    )
    assert outputs["word_ids"] is not None, "word_ids is None in the tokenizer outputs."
    assert outputs["image"] is not None, "image is None in the tokenizer outputs."
    if input_word_labels is not None:
        assert outputs["token_labels"] is not None, (
            "token_labels is None in the tokenizer outputs."
        )

    return outputs


def _get_subword_start_end(word_start, word_end, word_ids, sequence_ids):
    start_of_context = -1
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_context = i
            break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(word_ids)):
        if word_start == word_ids[i] and subword_start == -1:
            subword_start = i
        if word_end == word_ids[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens


def _generate_qa_token_ids(
    qa_pair: ExtractiveQAPair,
    word_ids: torch.Tensor,
    sequence_ids: torch.Tensor,
    sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    # since we can have multiple answers per question, we need to handle that here and just take one which is not
    # -1
    word_ans_start, word_ans_end = -1, -1
    for ans_start, ans_end in zip(qa_pair.answer_start, qa_pair.answer_end):
        if ans_start != -1 and ans_end != -1:
            word_ans_start = ans_start
            word_ans_end = ans_end
            break

    # now we have one answer, with start and end indices in the word level
    # we need to convert them to token level

    token_answer_starts, token_answer_ends = [], []
    for word_ids_per_overflow, sequence_ids_per_overflow in zip(
        word_ids, sequence_ids, strict=True
    ):
        token_answer_start, token_answer_end = None, None
        if word_ans_start == -1:
            token_answer_start = -1
            token_answer_end = -1
        else:
            (token_answer_start, token_answer_end, _) = _get_subword_start_end(
                word_ans_start,
                word_ans_end,
                word_ids_per_overflow,
                sequence_ids_per_overflow,
            )
            if token_answer_start == -1:
                token_answer_start = -1
                token_answer_end = -1
            if token_answer_end == -1:
                token_answer_end = sequence_length - 1
            assert token_answer_end >= token_answer_start, (
                "End token index is less than start token index. "
                "Something is wrong in the conversion from answer word indices to answer token indices."
            )
        token_answer_starts.append(token_answer_start)
        token_answer_ends.append(token_answer_end)
    token_answer_start = torch.tensor(
        token_answer_starts, dtype=torch.long, device=word_ids.device
    )
    token_answer_end = torch.tensor(
        token_answer_ends, dtype=torch.long, device=word_ids.device
    )
    return token_answer_start, token_answer_end


def _extract_annotations(sample: DocumentInstance) -> Annotations:
    """Extract annotations from sample."""
    annotations = Annotations()

    if sample.annotations is not None:
        for ann in sample.annotations:
            if isinstance(ann, ClassificationAnnotation):
                annotations.label = ann.label
            elif isinstance(ann, ExtractiveQAAnnotation):
                annotations.qa_pairs = ann.qa_pairs
            elif isinstance(ann, EntityLabelingAnnotation):
                annotations.word_labels = ann.word_labels
            elif isinstance(ann, LayoutAnalysisAnnotation):
                annotations.annotated_objects = ann.annotated_objects

    return annotations


def _generate_segment_level_bbox_ranks(
    token_ids: torch.Tensor,
    segment_level_bboxes: torch.Tensor,
    all_special_ids: set[int],
):
    import torch

    line_rank_ids = []
    assert len(token_ids) == len(segment_level_bboxes), (
        f"Token ids and segment level bboxes must have the same batch size, Got {len(token_ids)} and {len(segment_level_bboxes)}"
    )
    for token_ids_per_sample, bboxes_per_sample in zip(
        token_ids, segment_level_bboxes
    ):  # this is a shape of [batch_size, seq_len, 4] in xyxy format and normalized 0-1000
        assert len(token_ids_per_sample) == len(bboxes_per_sample), (
            "Token ids and segment level bboxes must have the same sequence length"
        )
        line_rank_ids_per_sample = []
        line_rank = 0
        last_b = None
        for token_id, b in zip(token_ids_per_sample, bboxes_per_sample):
            if last_b is not None and not torch.equal(b, last_b):
                line_rank += 1
            if token_id in all_special_ids:
                line_rank_ids_per_sample.append(0)
            else:
                line_rank_ids_per_sample.append(line_rank)
            last_b = b
        line_rank_ids.append(line_rank_ids_per_sample)

    return torch.tensor(line_rank_ids, device=segment_level_bboxes.device)


def _generate_segment_level_inner_ranks(line_rank_id: torch.Tensor):
    # line_inner_rank_id is the inner rank as follows 1 means start 2 for all middle tokens 3 for end token ... for each token in the line/segment.
    # if there is no middle token, start token will be 1 and end token will be 3.
    inner_ranks = []
    for line_ranks_per_sample in line_rank_id:
        inner_ranks_per_sample = torch.zeros_like(
            line_ranks_per_sample, device=line_ranks_per_sample.device
        )

        line_segment_spans = []
        start_idx = 0
        last_lr = None
        for curr_idx, lr in enumerate(line_ranks_per_sample):
            if last_lr is not None and lr != last_lr:
                line_segment_spans.append((start_idx, curr_idx - 1))
                start_idx = curr_idx
            last_lr = lr
        line_segment_spans.append(
            (start_idx, start_idx)
        )  # add the last segment for sep token

        for span in line_segment_spans:
            span_start, span_end = span
            span_length = span_end - span_start
            if span_length == 0:
                inner_ranks_per_sample[span_start] = 1  # only one token in the line
            elif span_length == 1:
                inner_ranks_per_sample[span_start] = 1  # start
                inner_ranks_per_sample[span_end] = 3  # end
            else:
                inner_ranks_per_sample[span_start] = 1  # start
                inner_ranks_per_sample[span_start + 1 : span_end] = 2
                inner_ranks_per_sample[span_end] = 3  # end
        inner_ranks.append(inner_ranks_per_sample)
    return torch.stack(inner_ranks)


def _generate_first_token_idxes(line_rank_id: torch.Tensor, max_segment_num: int = 150):
    first_token_idxes = []
    first_token_idxes_mask = []
    for line_ranks_per_sample in line_rank_id:
        first_token_idxes_per_sample = []
        first_token_idxes_mask_per_sample = []
        last_lr = None
        for curr_idx, lr in enumerate(line_ranks_per_sample):
            if last_lr is not None and lr != last_lr and lr != 0:
                first_token_idxes_per_sample.append(curr_idx)
            last_lr = lr

        # make mask
        if len(first_token_idxes_per_sample) > max_segment_num:
            first_token_idxes_per_sample = first_token_idxes_per_sample[
                :max_segment_num
            ]

        first_token_idxes_mask_per_sample = [1] * len(first_token_idxes_per_sample) + [
            0
        ] * (max_segment_num - len(first_token_idxes_per_sample))
        first_token_idxes_per_sample = first_token_idxes_per_sample + [0] * (
            max_segment_num - len(first_token_idxes_per_sample)
        )
        first_token_idxes_mask.append(first_token_idxes_mask_per_sample)
        first_token_idxes.append(first_token_idxes_per_sample)

    first_token_idxes = torch.tensor(first_token_idxes, device=line_rank_id.device)
    first_token_idxes_mask = torch.tensor(
        first_token_idxes_mask, device=line_rank_id.device, dtype=torch.float32
    )
    return first_token_idxes, first_token_idxes_mask
