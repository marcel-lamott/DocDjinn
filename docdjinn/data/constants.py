from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docdjinn import ENV

from ._core._utilities import TaskType


@dataclass
class DatasetLoadConfig:
    dataset_name: str
    dataset_config_name: str | tuple[str, str]
    task_type: TaskType
    root_datasets_dir: Path | str = ENV.BASE_DATASETS_DIR
    is_synthetic: bool = False


@dataclass
class SyntheticDatasetLoadConfig:
    dataset_name: str
    task_type: TaskType
    dataset_config_name: str = "default"
    root_datasets_dir: Path | str = ENV.SYN_DATASETS_PREPARED_DIR
    is_synthetic: bool = True


DATASET_CONFIGS = [
    DatasetLoadConfig(
        dataset_name="tobacco3482",
        dataset_config_name="image_with_ocr",
        task_type=TaskType.sequence_classification,
    ),
    DatasetLoadConfig(
        dataset_name="rvlcdip",
        dataset_config_name="image_with_ocr_4k",
        task_type=TaskType.sequence_classification,
    ),
    DatasetLoadConfig(
        dataset_name="cord",
        dataset_config_name="default-bc8ca3a9",
        task_type=TaskType.token_classification,
    ),
    DatasetLoadConfig(
        dataset_name="funsd",
        dataset_config_name="default-d5de28ff",
        task_type=TaskType.token_classification,
    ),
    DatasetLoadConfig(
        dataset_name="sroie",
        dataset_config_name="default-c9d392fa",
        task_type=TaskType.token_classification,
    ),
    DatasetLoadConfig(
        dataset_name="wild_receipts",
        dataset_config_name="default-0efb7676",
        task_type=TaskType.token_classification,
    ),
    DatasetLoadConfig(
        dataset_name="docile",
        dataset_config_name="default-2b4fec84",
        task_type=TaskType.token_classification,
    ),
    DatasetLoadConfig(
        dataset_name="ex_docvqa",
        dataset_config_name=("due_benchmark", "ExDocVQA"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_deepform",
        dataset_config_name=("due_benchmark", "ExDeepForm"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_tabfact",
        dataset_config_name=("due_benchmark", "ExTabFact"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_wiki",
        dataset_config_name=("due_benchmark", "ExWikiTableQuestions"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_infographics",
        dataset_config_name=("due_benchmark", "ExInfographicsVQA"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_klc",
        dataset_config_name=("due_benchmark", "ExKleisterCharity"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="ex_pwc",
        dataset_config_name=("due_benchmark", "ExPWC"),
        task_type=TaskType.extractive_qa,
    ),
    DatasetLoadConfig(
        dataset_name="publaynet",
        dataset_config_name="4k",
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="doclaynet",  # backward compatibility
        dataset_config_name=("doclaynet", "1k"),
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="doclaynet_4k_cls",
        dataset_config_name=("doclaynet", "4k"),
        task_type=TaskType.sequence_classification,
    ),
    DatasetLoadConfig(
        dataset_name="doclaynet_4k_dla",
        dataset_config_name=("doclaynet", "4k"),
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="icdar2019",
        dataset_config_name="trackA_modern",
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="pubtables1m",
        dataset_config_name="structure_4k",
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="fintabnet",
        dataset_config_name="1k",
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="fintabnet_4k",
        dataset_config_name=("fintabnet", "4k"),
        task_type=TaskType.layout_analysis,
    ),
    DatasetLoadConfig(
        dataset_name="icdar2013",
        dataset_config_name="default-d32da75b",
        task_type=TaskType.layout_analysis,
    ),
    ###
    # ADD ALL SYNTHETIC DATASETS HERE
    ###
    ###
    # ALL SYNTHETIC DATASETS FOR V1 SAMPLING STRAT
    ###
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"docvqa_alpha={alpha}_v1",
            task_type=TaskType.extractive_qa,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"rvlcdip_alpha={alpha}_v1",
            task_type=TaskType.sequence_classification,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"cord_alpha={alpha}_v1",
            task_type=TaskType.token_classification,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"publaynet_alpha={alpha}",
            task_type=TaskType.layout_analysis,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"publaynet_correct-sampling_alpha={alpha}_v1",
            task_type=TaskType.layout_analysis,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    ###
    # ALL SYNTHETIC DATASETS FOR V2 SAMPLING STRAT
    ###
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"docvqa_alpha={alpha}",
            task_type=TaskType.extractive_qa,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"rvlcdip_alpha={alpha}",
            task_type=TaskType.sequence_classification,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"cord_alpha={alpha}",
            task_type=TaskType.token_classification,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"publaynet_correct-sampling_alpha={alpha}",
            task_type=TaskType.layout_analysis,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
    ###
    # ADDITIOANAL SYNTHETIC DATASETS HERE
    ###
    SyntheticDatasetLoadConfig(
        dataset_name="tobacco3482_alpha=1.0",
        task_type=TaskType.sequence_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_CLS",
        task_type=TaskType.sequence_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_CLS",
        task_type=TaskType.sequence_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="funsd_alpha=1.0",
        task_type=TaskType.token_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="sroie_alpha=1.0",
        task_type=TaskType.token_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="wtq_alpha=1.0",
        task_type=TaskType.extractive_qa,
    ),
    # SyntheticDatasetLoadConfig(
    #     dataset_name="ex_infographics",
    #     task_type=TaskType.extractive_qa,
    # ),
    SyntheticDatasetLoadConfig(
        dataset_name="wtq_alpha=1.0",
        task_type=TaskType.extractive_qa,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="kleister_alpha=1.0",
        task_type=TaskType.extractive_qa,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_DLA",
        task_type=TaskType.layout_analysis,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_DLA",
        task_type=TaskType.layout_analysis,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="icdar2019_alpha=1.0",
        task_type=TaskType.layout_analysis,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_CLS",
        task_type=TaskType.sequence_classification,
    ),
    SyntheticDatasetLoadConfig(
        dataset_name="doclaynet4k_alpha=1.0_DLA",
        task_type=TaskType.layout_analysis,
    ),
    *[
        SyntheticDatasetLoadConfig(
            dataset_name=f"publaynet_correct-sampling_alpha={alpha}",
            task_type=TaskType.layout_analysis,
        )
        for alpha in [0.5, 0.75, 1.0]
    ],
]
