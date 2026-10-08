from __future__ import annotations

from docdjinn.data._core._data_pipeline import DataPipeline
from docdjinn.data._core._utilities import TaskType, get_logger
from docdjinn.data.interfaces.mixed_dataset import load_mixed_dataset

from .dataset import load_dataset, load_preprocessed_dataset
from .transform import load_transform

logger = get_logger(__name__)


def get_collate_fn(task_type: TaskType | None) -> str:
    collate_fn = "default_collate"
    if task_type in [TaskType.layout_analysis, TaskType.table_extraction]:
        collate_fn = (
            "mmdet_pseudo_collate"  # for layout analysis + table detection / extraction
        )
    return collate_fn


def load_data_pipeline(
    dataset_name: str,
    dataset_splitting_enabled: bool = False,
    split_ratio: float = 0.9,
    use_collate_fn: bool = True,
    task_type: TaskType | None = None,
    transform_type: str = "default",
    set_task_transforms: bool = True,
    is_synthetic: bool = False,
    split: str | None = None,
    **transforms_kwargs,
) -> DataPipeline:
    # since each datasethas specific task, we infer task-level transforms from dataset name
    # e.g. FUNSD -> token classification, SROIE -> sequence classification

    # setup dataset
    dataset = load_dataset(
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        task_type_override=task_type,
        split=split,
    )

    # setup data pipeline and dataloaders
    data_pipeline = DataPipeline(
        dataset=dataset,
        dataset_splitting_enabled=dataset_splitting_enabled,
        split_ratio=split_ratio,
        collate_fn=get_collate_fn(dataset.task_type) if use_collate_fn else "identity",
    )

    # attach task-specific transforms if not preparing preprocessed dataset
    if set_task_transforms:
        # load transform for train
        transform = load_transform(
            task_type=dataset.task_type,
            for_train=True,
            transform_type=transform_type,
            dataset_labels=dataset.metadata.dataset_labels,
            **transforms_kwargs,
        )

        # set transform to data pipeline
        data_pipeline.set_transform(transform, for_train=True, for_eval=False)

        # load transform for eval
        transform = load_transform(
            task_type=dataset.task_type,
            for_train=False,
            transform_type=transform_type,
            dataset_labels=dataset.metadata.dataset_labels,
            **transforms_kwargs,
        )

        # set transform to data pipeline
        data_pipeline.set_transform(transform, for_train=False, for_eval=True)

    return data_pipeline


def load_preprocessed_data_pipeline(
    dataset_name: str,
    dataset_splitting_enabled: bool = False,
    split_ratio: float = 0.9,
    use_collate_fn: bool = True,
    task_type: TaskType | None = None,
    is_synthetic: bool = False,
    transform_type: str = "default",
    split: str | None = None,
    **transforms_kwargs,
) -> DataPipeline:
    # setup dataset with preprocessing
    dataset = load_preprocessed_dataset(
        dataset_name=dataset_name,
        is_synthetic=is_synthetic,
        task_type_override=task_type,
        transform_type=transform_type,
        split=split,
        **transforms_kwargs,
    )

    return DataPipeline(
        dataset=dataset,
        dataset_splitting_enabled=dataset_splitting_enabled,
        split_ratio=split_ratio,
        collate_fn=get_collate_fn(dataset.task_type) if use_collate_fn else "identity",
    )


def load_mixed_data_pipeline(
    dataset_name: str,
    synthetic_dataset_name: str,
    num_real_samples: int = -1,
    num_synthetic_samples: int = -1,
    load_preprocessed: bool = False,
    dataset_splitting_enabled: bool = False,
    split_ratio: float = 0.9,
    use_collate_fn: bool = True,
    task_type: TaskType | None = None,
    transform_type: str = "default",
    set_task_transforms: bool = True,
    split: str | None = None,
    **transforms_kwargs,
) -> DataPipeline:
    # since each datasethas specific task, we infer task-level transforms from dataset name
    # e.g. FUNSD -> token classification, SROIE -> sequence classification

    # setup dataset
    dataset = load_mixed_dataset(
        dataset_name=dataset_name,
        synthetic_dataset_name=synthetic_dataset_name,
        num_real_samples=num_real_samples,
        num_synthetic_samples=num_synthetic_samples,
        task_type_override=task_type,
        split=split,
        load_preprocessed=load_preprocessed,
        transform_type=transform_type,
        **transforms_kwargs,
    )

    # setup data pipeline and dataloaders
    data_pipeline = DataPipeline(
        dataset=dataset,
        dataset_splitting_enabled=dataset_splitting_enabled,
        split_ratio=split_ratio,
        collate_fn=get_collate_fn(dataset.task_type) if use_collate_fn else "identity",
    )

    # attach task-specific transforms if not preparing preprocessed dataset
    if not load_preprocessed and set_task_transforms:
        # load transform for train
        transform = load_transform(
            task_type=dataset.task_type,
            for_train=True,
            transform_type=transform_type,
            dataset_labels=dataset.metadata.dataset_labels,
            **transforms_kwargs,
        )

        # set transform to data pipeline
        data_pipeline.set_transform(transform, for_train=True, for_eval=False)

        # load transform for eval
        transform = load_transform(
            task_type=dataset.task_type,
            for_train=False,
            transform_type=transform_type,
            dataset_labels=dataset.metadata.dataset_labels,
            **transforms_kwargs,
        )

        # set transform to data pipeline
        data_pipeline.set_transform(transform, for_train=False, for_eval=True)

    return data_pipeline
