from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

from atria_core.types import *
from mmdet.structures import DetDataSample
from pydantic import ConfigDict

if TYPE_CHECKING:
    from typing import Any

    import torch


class OverflowStrategy(str, enum.Enum):
    select_first = "select_first"
    select_all = "select_all"
    select_random = "select_random"


if TYPE_CHECKING:
    import torch

T = TypeVar("T", bound="BaseModelInput")


@dataclass(frozen=True)
class MMDetInput:
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    inputs: list[Any] | Any
    data_samples: DetDataSample

    def to(self, device: torch.device | str) -> "MMDetInput":
        inputs = [tensor.to(device) for tensor in self.inputs]
        return MMDetInput(
            inputs=inputs,
            data_samples=self.data_samples,
        )


@dataclass(frozen=True)
class BaseModelInput:
    """
    Base class for model input dataclasses.
    - Frozen (immutable)
    - Prevents nested BaseModelInput instances
    - Provides transform utilities (like .to(device))
    """

    _is_batched: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self):
        # Disallow nested BaseModelInput instances
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, BaseModelInput):
                raise TypeError(
                    f"Field '{f.name}' cannot be another BaseModelInput "
                    f"({type(value).__name__}). Nesting is not allowed."
                )

    def _map_tensors(self, fn: callable):
        """
        Internal helper: apply a function to all torch.Tensor fields.
        Returns a new instance with transformed fields.
        """
        import torch

        updates = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                updates[f.name] = fn(val)
            elif isinstance(val, list):
                # If it's a list of tensors, map them too
                updates[f.name] = [
                    fn(v) if isinstance(v, torch.Tensor) else v for v in val
                ]
            else:
                updates[f.name] = val
        return replace(self, **updates)

    def to(self, device: torch.device | str):
        """Move all tensor fields to a given device."""
        return self._map_tensors(lambda t: t.to(device))

    def cpu(self):
        """Move all tensor fields to CPU."""
        return self._map_tensors(lambda t: t.cpu())

    def cuda(self):
        """Move all tensor fields to CUDA."""
        return self._map_tensors(lambda t: t.cuda())

    def numpy(self):
        """Convert all tensor fields to numpy arrays."""
        return self._map_tensors(
            lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        )

    @classmethod
    def batch(cls: Type[T], instances: list[T]) -> T:
        """
        Batch a list of BaseModelInput instances into a single instance.
        - Tensor fields are stacked along dim=0.
        - Non-tensor fields become lists.
        """
        import torch

        if not instances:
            raise ValueError("Cannot batch an empty list of inputs.")
        if not all(isinstance(x, cls) for x in instances):
            raise TypeError(f"All elements must be instances of {cls.__name__}.")

        field_values = {}
        for f in fields(instances[0]):
            if f.name.startswith("_"):
                field_values[f.name] = getattr(instances[0], f.name)
                continue

            vals = [getattr(x, f.name) for x in instances]
            if vals[0] is None:  # we assume if any value is None, all are None
                field_values[f.name] = None
                continue

            if all(isinstance(v, torch.Tensor) for v in vals):
                field_values[f.name] = torch.stack(vals, dim=0)
            else:
                field_values[f.name] = vals

        return cls(**field_values)

    def __repr__(self) -> str:
        """
        Generates a developer-friendly string representation of the object.

        Returns:
            str: A developer-friendly string representation of the object.
        """

        import torch
        from rich.pretty import pretty_repr

        torch.set_printoptions(edgeitems=2, threshold=100)

        return pretty_repr(self, max_length=4, max_string=128, max_depth=3)

    def __str__(self) -> str:
        """
        Generates a human-readable string representation of the object.

        Returns:
            str: A human-readable string representation of the object.
        """

        import torch
        from rich.pretty import pretty_repr

        torch.set_printoptions(edgeitems=2, threshold=100)

        return pretty_repr(self, max_length=4, max_string=128, max_depth=3)


@dataclass(frozen=True)
class DocumentInstanceModelInput(BaseModelInput):
    tokenizer_config: dict | None = None

    # token level fields
    token_ids: "torch.Tensor" = None
    token_bboxes: Optional["torch.Tensor"] = None
    token_type_ids: Optional["torch.Tensor"] = None
    token_labels: Optional["torch.Tensor"] = None
    attention_mask: "torch.Tensor" = None
    word_ids: "torch.Tensor" = None
    sequence_ids: "torch.Tensor" = None
    overflow_to_sample_mapping: "torch.Tensor" = None

    # segment level fields
    segment_index: "torch.Tensor" = None
    segment_inner_token_rank: "torch.Tensor" = None
    first_token_idxes: "torch.Tensor" = None
    first_token_idxes_mask: "torch.Tensor" = None

    # sample level fields
    index: Optional["torch.Tensor"] = (
        None  # index is used to uniquely identify a sample in a batch
    )
    sample_id: str = None
    image: Optional["torch.Tensor"] = None
    label: Optional["torch.Tensor"] = None
    words: list[str] = None

    # extractive QA specific fields
    question_id: int | None = None
    qa_question: str | None = None
    qa_answers: list[str] | None = None
    token_answer_start: Optional["torch.Tensor"] = None
    token_answer_end: Optional["torch.Tensor"] = None

    def select_overflow_samples_by_id(self, is_random: bool = False):
        import torch

        assert self._is_batched, (
            "select_all_overflow_samples can only be called on batched inputs."
        )

        def _gather_idx_from_sequence_list(
            samples_batch: list[torch.Tensor],
        ) -> torch.Tensor | None:
            if samples_batch is None:
                return None

            resolved_samples_batch = []
            for sample_data in samples_batch:
                if len(sample_data) == 1:
                    resolved_samples_batch.append(sample_data[0])
                else:
                    idx = (
                        0
                        if not is_random
                        else torch.randint(0, sample_data.shape[0], (1,)).item()
                    )
                    resolved_samples_batch.append(sample_data[idx])
            return torch.stack(resolved_samples_batch)

        token_ids = _gather_idx_from_sequence_list(self.token_ids)
        token_type_ids = _gather_idx_from_sequence_list(self.token_type_ids)
        token_bboxes = _gather_idx_from_sequence_list(self.token_bboxes)
        token_labels = _gather_idx_from_sequence_list(self.token_labels)
        attention_mask = _gather_idx_from_sequence_list(self.attention_mask)
        word_ids = _gather_idx_from_sequence_list(self.word_ids)
        sequence_ids = _gather_idx_from_sequence_list(self.sequence_ids)
        overflow_to_sample_mapping = _gather_idx_from_sequence_list(
            self.overflow_to_sample_mapping
        )

        # segment level fields
        segment_index = _gather_idx_from_sequence_list(self.segment_index)
        segment_inner_token_rank = _gather_idx_from_sequence_list(
            self.segment_inner_token_rank
        )
        first_token_idxes = _gather_idx_from_sequence_list(self.first_token_idxes)
        first_token_idxes_mask = _gather_idx_from_sequence_list(
            self.first_token_idxes_mask
        )

        # sample level fields remain unchanged
        token_answer_start, token_answer_end = None, None
        if self.token_answer_start is not None:
            token_answer_start = _gather_idx_from_sequence_list(self.token_answer_start)
            token_answer_end = _gather_idx_from_sequence_list(self.token_answer_end)

        return replace(
            self,
            token_ids=token_ids,
            token_type_ids=token_type_ids,
            token_bboxes=token_bboxes,
            token_labels=token_labels,
            attention_mask=attention_mask,
            word_ids=word_ids,
            sequence_ids=sequence_ids,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
            token_answer_start=token_answer_start,
            token_answer_end=token_answer_end,
            # segment level fields
            segment_index=segment_index,
            segment_inner_token_rank=segment_inner_token_rank,
            first_token_idxes=first_token_idxes,
            first_token_idxes_mask=first_token_idxes_mask,
            # stack tensors
            image=self.image if self.image is None else torch.stack(self.image),
            label=self.label if self.label is None else torch.stack(self.label),
            # index=self.index if self.index is None else torch.tensor(self.index),
        )

    def resolve_sample_overflow(
        self, overflow_strategy: OverflowStrategy = OverflowStrategy.select_all
    ) -> DocumentInstanceModelInput:
        if not isinstance(self.token_ids, list):
            # already resolved
            return self

        if overflow_strategy == OverflowStrategy.select_all:
            return self.select_all_overflow_samples()
        elif overflow_strategy == OverflowStrategy.select_first:
            return self.select_first_overflow_samples()
        elif overflow_strategy == OverflowStrategy.select_random:
            return self.select_random_overflow_samples()
        else:
            raise ValueError(f"Unknown overflow strategy: {overflow_strategy}")

    def select_first_overflow_samples(self):
        return self.select_overflow_samples_by_id(is_random=False)

    def select_random_overflow_samples(self):
        return self.select_overflow_samples_by_id(is_random=True)

    def select_all_overflow_samples(self) -> tuple[bool, list[int], list[str]]:
        import torch

        assert self._is_batched, (
            "select_all_overflow_samples can only be called on batched inputs."
        )
        repeat_indices = [sample.shape[0] for sample in self.token_ids]

        # we concatenate all lists of overflowed samples into a single tensor
        def _cat_tensor_fields(samples_list: list[torch.Tensor]) -> torch.Tensor | None:
            if samples_list is not None:
                return torch.cat(samples_list, dim=0)
            return None

        # these are all fields that are already in overflowed format
        token_ids = _cat_tensor_fields(self.token_ids)
        token_bboxes = _cat_tensor_fields(self.token_bboxes)
        token_type_ids = _cat_tensor_fields(self.token_type_ids)
        token_labels = _cat_tensor_fields(self.token_labels)
        attention_mask = _cat_tensor_fields(self.attention_mask)
        word_ids = _cat_tensor_fields(self.word_ids)
        sequence_ids = _cat_tensor_fields(self.sequence_ids)
        overflow_to_sample_mapping = _cat_tensor_fields(self.overflow_to_sample_mapping)

        # segment level fields
        segment_index = _cat_tensor_fields(self.segment_index)
        segment_inner_token_rank = _cat_tensor_fields(self.segment_inner_token_rank)
        first_token_idxes = _cat_tensor_fields(self.first_token_idxes)
        first_token_idxes_mask = _cat_tensor_fields(self.first_token_idxes_mask)

        token_answer_start, token_answer_end = None, None
        if self.token_answer_start is not None and self.token_answer_end is not None:
            token_answer_start = _cat_tensor_fields(self.token_answer_start)
            token_answer_end = _cat_tensor_fields(self.token_answer_end)

        # these are fields that are at sample level and need to be repeated in case of overflow
        # sample level fields
        index = self._repeat_field(self.index, repeat_indices)
        sample_id = self._repeat_field(self.sample_id, repeat_indices)
        image = self._repeat_field(self.image, repeat_indices)
        label = self._repeat_field(self.label, repeat_indices)
        words = self._repeat_field(self.words, repeat_indices)

        # extractive QA specific fields
        question_id = self._repeat_field(self.question_id, repeat_indices)
        qa_question = self._repeat_field(self.qa_question, repeat_indices)
        qa_answers = self._repeat_field(self.qa_answers, repeat_indices)

        repeated_instance = replace(
            self,
            token_ids=token_ids,
            token_bboxes=token_bboxes,
            token_type_ids=token_type_ids,
            token_labels=token_labels,
            attention_mask=attention_mask,
            word_ids=word_ids,
            sequence_ids=sequence_ids,
            overflow_to_sample_mapping=overflow_to_sample_mapping,
            # segment level fields
            segment_index=segment_index,
            segment_inner_token_rank=segment_inner_token_rank,
            first_token_idxes=first_token_idxes,
            first_token_idxes_mask=first_token_idxes_mask,
            # sample level fields
            index=index,
            sample_id=sample_id,
            image=image,
            label=label,
            words=words,
            question_id=question_id,
            qa_question=qa_question,
            qa_answers=qa_answers,
            token_answer_start=token_answer_start,
            token_answer_end=token_answer_end,
        )

        for key, value in repeated_instance.to_dict().items():
            if isinstance(value, list) and len(value) != sum(repeat_indices):
                raise ValueError(
                    f"Field '{key}' length {len(value)} does not match expected {sum(repeat_indices)}"
                )
            if isinstance(value, torch.Tensor) and value.size(0) != sum(repeat_indices):
                raise ValueError(
                    f"Field '{key}' size {value.size(0)} does not match expected {sum(repeat_indices)}"
                )
        return repeated_instance

    def _repeat_field(self, field_value: Any, repeat_indices: list[int]) -> Any:
        import torch

        if isinstance(field_value, list):
            if len(field_value) == 0:
                return field_value
            if len(field_value) != len(repeat_indices):
                raise ValueError(
                    f"List length ({len(field_value)}) doesn't match repeat_indices length ({len(repeat_indices)})"
                )
            repeated_list = [
                item
                for item, count in zip(field_value, repeat_indices, strict=True)
                for _ in range(count)
            ]

            if isinstance(field_value[0], torch.Tensor):
                return torch.stack(repeated_list, dim=0)
            return repeated_list

        elif isinstance(field_value, torch.Tensor):
            if field_value.size(0) != len(repeat_indices):
                raise ValueError(
                    f"Tensor batch size ({field_value.size(0)}) doesn't match repeat_indices length ({len(repeat_indices)})"
                )
            return field_value.repeat_interleave(
                torch.tensor(repeat_indices, device=field_value.device), dim=0
            )

        return field_value

    @classmethod
    def batch(cls: DocumentInstanceModelInput, instances: list[T]) -> T:
        if not instances:
            raise ValueError("Cannot batch an empty list of inputs.")
        if not all(isinstance(x, cls) for x in instances):
            raise TypeError(f"All elements must be instances of {cls.__name__}.")

        field_values = {}
        for f in fields(instances[0]):
            if f.name == "_is_batched":
                field_values[f.name] = True
                continue
            if f.name == "tokenizer_config":
                # For tokenizer_config, we take from the first instance
                field_values[f.name] = getattr(instances[0], f.name)
                continue
            if f.name.startswith("_"):
                field_values[f.name] = getattr(instances[0], f.name)
                continue

            vals = [getattr(x, f.name) for x in instances]
            if vals[0] is None:  # we assume if any value is None, all are None
                field_values[f.name] = None
                continue

            # we simply put all fields in a list and batch them later
            # for example we can have sequences like following due to overflow mapping
            # seq 1 -> token ids of size (2, 512)
            # seq 2 -> token ids of size (1, 512)
            # seq 3 -> token ids of size (4, 512)
            field_values[f.name] = vals

        return cls(**field_values)

    def print_info(self):
        import torch

        print("DocumentInstanceModelInput:")
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                print(f"  {f.name}: Tensor shape {val.shape}, dtype {val.dtype}")
            elif isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], torch.Tensor):
                    shapes = [v.shape for v in val]
                    print(f"  {f.name}: List of Tensors with shapes {shapes}")
                else:
                    print(f"  {f.name}: List of length {len(val)}")
            else:
                print(f"  {f.name}: {type(val).__name__} value: {val}")

    def to_dict(self):
        import torch

        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                result[f.name] = val.detach().cpu().numpy()
            elif isinstance(val, list):
                if len(val) > 0 and isinstance(val[0], torch.Tensor):
                    result[f.name] = [v.detach().cpu().numpy() for v in val]
                else:
                    result[f.name] = val
            else:
                result[f.name] = val

        return result

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        import numpy as np
        import torch

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = torch.tensor(value)
            elif (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], np.ndarray)
            ):
                data[key] = [torch.tensor(v) for v in value]
            else:
                data[key] = value

        return cls(**data)


@dataclass(frozen=True)
class ConditionalGenerationModelInput(BaseModelInput):
    index: Optional["torch.Tensor"] = None
    sample_id: Optional[str] = None
    input_ids: Optional["torch.Tensor"] = None
    bbox: Optional["torch.Tensor"] = None
    attention_mask: Optional["torch.Tensor"] = None
    pixel_values: Optional["torch.Tensor"] = None
    question_text: Optional[str] = None
    target_text: Optional[str] = None
    target_token_ids: Optional["torch.Tensor"] = None
    words: Optional[list[str]] = None
    word_labels: Optional[list[str]] = None
    label: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class VLMModelInput(BaseModelInput):
    index: Optional["torch.Tensor"] = None
    sample_id: Optional[str] = None
    input_ids: Optional["torch.Tensor"] = None
    bbox: Optional["torch.Tensor"] = None
    attention_mask: Optional["torch.Tensor"] = None
    pixel_values: Optional["torch.Tensor"] = None
    image_grid_thw: Optional["torch.Tensor"] = None
    question_text: Optional[str] = None
    target_text: Optional[str] = None
    target_token_ids: Optional["torch.Tensor"] = None
    words: Optional[list[str]] = None
    word_labels: Optional[list[str]] = None
    label: Optional["torch.Tensor"] = None
