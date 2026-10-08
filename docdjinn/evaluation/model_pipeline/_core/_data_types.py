from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ModelOutput:
    loss: torch.Tensor | None = None


@dataclass(frozen=True)
class ClassificationModelOutput(ModelOutput):
    logits: torch.Tensor | None = None
    prediction_probs: torch.Tensor | None = None
    gt_label_value: torch.Tensor | None = None
    gt_label_name: list[str] | None = None
    predicted_label_value: torch.Tensor | None = None
    predicted_label_name: list[str] | None = None


@dataclass(frozen=True)
class TokenClassificationModelOutput(ModelOutput):
    logits: torch.Tensor | None = None
    predicted_label_names: list[list[str]] | None = None
    target_label_names: list[list[str]] | None = None


@dataclass(frozen=True)
class LayoutTokenClassificationModelOutput(TokenClassificationModelOutput):
    token_labels: torch.Tensor | None = None
    token_bboxes: torch.Tensor | None = None


@dataclass(frozen=True)
class QAPair:
    sample_id: str
    question: str
    answer: str


@dataclass(frozen=True)
class QAModelOutput(ModelOutput):
    qa_pairs: list[QAPair] | None = None


@dataclass(frozen=True)
class MMDetEvaluationOutput(ModelOutput):
    loss_dict: dict | None = None
    det_data_samples: list[Any] | None = None
    class_labels: list[str] | None = None
