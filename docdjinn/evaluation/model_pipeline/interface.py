"""
Defines interface for docdjinn components to load datasets using DatasetFactory and log relevant information.
"""

from __future__ import annotations

from docdjinn.data._core._data_types import DatasetMetadata
from docdjinn.data._core._utilities import TaskType
from docdjinn.evaluation.model_pipeline._core._conditional_generation import (
    GenerativeLayoutAnalysisPipeline,
    GenerativeQuestionAnsweringPipeline,
    GenerativeSequenceClassificationPipeline,
    GenerativeTokenClassificationPipeline,
)
from docdjinn.evaluation.model_pipeline._core._detection import (
    DetectionPipeline,
)
from docdjinn.evaluation.model_pipeline._core._question_answering import (
    QuestionAnsweringPipeline,
)
from docdjinn.evaluation.model_pipeline._core._sequence_classification import (
    SequenceClassificationPipeline,
)
from docdjinn.evaluation.model_pipeline._core._token_classification import (
    TokenClassificationPipeline,
)
from docdjinn.logging import get_logger

from ._core._base import ModelPipeline  # noqa
from ._core._data_types import *  # noqa

logger = get_logger(__name__)


def load_model_pipeline(
    task_type: TaskType,
    model_name: str,
    model_cache_dir: str,
    dataset_metadata: DatasetMetadata,
    **kwargs,
) -> ModelPipeline:
    if model_name in [
        "microsoft/udop-large",
        "google-t5/t5-base",
        "google-t5/t5-large",
    ]:
        if task_type == TaskType.sequence_classification:
            return GenerativeSequenceClassificationPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
            )
        elif task_type == TaskType.token_classification:
            return GenerativeTokenClassificationPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
            )
        elif task_type == TaskType.extractive_qa:
            return GenerativeQuestionAnsweringPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
            )
        elif task_type == TaskType.layout_analysis:
            return GenerativeLayoutAnalysisPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    else:
        if task_type == TaskType.sequence_classification:
            return SequenceClassificationPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
                **kwargs,
            )
        elif task_type == TaskType.token_classification:
            return TokenClassificationPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
                **kwargs,
            )
        elif task_type == TaskType.extractive_qa:
            return QuestionAnsweringPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
                **kwargs,
            )
        elif task_type in [TaskType.layout_analysis, TaskType.table_extraction]:
            return DetectionPipeline(
                model_name=model_name,
                model_cache_dir=model_cache_dir,
                dataset_metadata=dataset_metadata,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
