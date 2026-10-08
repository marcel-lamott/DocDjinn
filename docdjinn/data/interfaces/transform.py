from __future__ import annotations

from docdjinn.data._core._data_types import DatasetLabels
from docdjinn.data._core._utilities import TaskType, get_logger
from docdjinn.data._transforms._generics._base import BaseTransform

logger = get_logger(__name__)


def _get_default_transform(
    task_type: TaskType,
    for_train: bool = False,
    **kwargs,
) -> BaseTransform:
    """Get default transform for a given task type."""
    from docdjinn.data._transforms._tokenizers._document_processors import (
        BaseDocumentProcessor,
        QuestionAnsweringDocumentProcessor,
        SequenceClassificationDocumentProcessor,
        TokenClassificationDocumentProcessor,
    )

    if task_type == TaskType.generate_embeddings:
        return BaseDocumentProcessor(**kwargs)
    if task_type == TaskType.sequence_classification:
        return SequenceClassificationDocumentProcessor(**kwargs)
    elif task_type == TaskType.token_classification:
        return TokenClassificationDocumentProcessor(**kwargs)
    elif task_type == TaskType.extractive_qa:
        return QuestionAnsweringDocumentProcessor(is_training=for_train, **kwargs)
    elif task_type in [TaskType.layout_analysis, TaskType.table_extraction]:
        from docdjinn.data._transforms.mmdet import DocumentInstanceMMDetTransform

        return DocumentInstanceMMDetTransform(is_training=for_train, **kwargs)
    else:
        raise ValueError(f"Unsupported task type for transform: {task_type}")


def _get_conditional_generation_transform(
    task_type: TaskType,
    tokenizer_name: str,
    is_training: bool = False,
    dataset_labels: DatasetLabels | None = None,
    **kwargs,
) -> BaseTransform:
    """Get conditional generation transform."""
    from docdjinn.data._transforms._tokenizers._conditional_generation import (
        ConditionalGenerationTokenizer,
    )

    return ConditionalGenerationTokenizer(
        tokenizer_name=tokenizer_name,
        task_type=task_type,
        is_training=is_training,
        dataset_labels=dataset_labels,
        **kwargs,
    )


def load_transform(
    task_type: TaskType,
    for_train: bool = False,
    transform_type: str = "default",
    dataset_labels: DatasetLabels | None = None,
    **kwargs,
) -> BaseTransform:
    """Load transform based on type and task."""
    if transform_type == "default":
        return _get_default_transform(
            task_type=task_type, for_train=for_train, **kwargs
        )
    elif transform_type == "conditional_generation":
        return _get_conditional_generation_transform(
            task_type=task_type,
            is_training=for_train,
            dataset_labels=dataset_labels,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported transform type: {transform_type}")
