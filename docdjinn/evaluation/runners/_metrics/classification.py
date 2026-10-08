from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from docdjinn.evaluation.model_pipeline import ClassificationModelOutput
from docdjinn.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ignite.metrics import Metric


def load_classification_metrics(
    device: str, num_classes: int | None = None
) -> dict[str, Metric]:
    from ignite.metrics import Accuracy, ConfusionMatrix, Precision, Recall

    def _output_transform(output: ClassificationModelOutput):
        assert isinstance(output, ClassificationModelOutput), (
            f"Expected {ClassificationModelOutput}, got {type(output)}"
        )
        if output.logits is None:
            assert (
                output.predicted_label_value is not None
                and output.gt_label_value is not None
            ), (
                "Both predicted_label_value and gt_label_value must be provided when logits are None"
            )
            return (
                output.predicted_label_value,
                output.gt_label_value,
            )

        assert output.gt_label_value is not None, (
            "gt_label_value must be provided when logits are present"
        )
        return (
            output.logits,
            output.gt_label_value,
        )  # pred logits, gt_labels

    def _f1_score(output_transform: Callable, device: str = "cpu") -> Metric:
        from ignite.metrics import Precision, Recall

        precision = Precision(
            average=False, output_transform=output_transform, device=device
        )
        recall = Recall(average=False, output_transform=output_transform, device=device)
        return (precision * recall * 2 / (precision + recall)).mean()

    return {
        "accuracy": Accuracy(
            is_multilabel=False, device=device, output_transform=_output_transform
        ),
        "precision": Precision(
            average=True, device=device, output_transform=_output_transform
        ),
        "recall": Recall(
            average=True, device=device, output_transform=_output_transform
        ),
        "confusion_matrix": ConfusionMatrix(
            average="recall",
            device=device,
            num_classes=num_classes,
            output_transform=_output_transform,
        ),
        "f1": _f1_score(device=device, output_transform=_output_transform),
    }
