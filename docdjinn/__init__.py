from __future__ import annotations

from enum import Enum
from pathlib import Path

_root_path = Path(__file__).parent.parent.resolve()


# Project paths
class ENV:
    # General
    ROOT_DIR: Path = _root_path
    DATA_DIR: Path = ROOT_DIR / "data"

    DATASETS_DIR: Path = ROOT_DIR / "data" / "datasets"
    BASE_DATASETS_DIR: Path = DATASETS_DIR / "base_v2"
    SYN_DATASETS_PREPARED_DIR: Path = DATASETS_DIR / "synthesized_prepared"
    SYN_DATASETS_DIR: Path = DATASETS_DIR / "synthesized_datasets"

    VISUAL_ELEMENT_PREFABS_DIR: Path = DATA_DIR / "visual_element_prefabs"

    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    GT_EMBEDDINGS_DIR: Path = DATA_DIR / "gt_embeddings"
    CLUSTERS_DIR: Path = DATA_DIR / "clusters"
    CLUSTER_PLOTS: Path = DATA_DIR / "cluster_plots"
    SYN_DATASET_STAT_PLOTS: Path = DATA_DIR / "syn_dataste_statistics_plots"

    ANALYZATION_DIR: Path = DATA_DIR / "analyzation"
    GT_ANALYZATION_DIR: Path = ANALYZATION_DIR / "gt"
    KIE_GT_ANALYZATION_DIR: Path = GT_ANALYZATION_DIR / "kie"
    CLS_GT_ANALYZATION_DIR: Path = GT_ANALYZATION_DIR / "cls"
    QA_GT_ANALYZATION_DIR: Path = GT_ANALYZATION_DIR / "qa"
    DLA_GT_ANALYZATION_DIR: Path = GT_ANALYZATION_DIR / "dla"

    WEBAPP_CACHE_DIR: Path = DATA_DIR / "webapp_cache"
    QA_GT_WEBAPP_CACHE_DIR: Path = WEBAPP_CACHE_DIR / "qa_gt"

    TEMP_DIR: Path = DATA_DIR / "temp"

    MODELS_DIR: Path = DATA_DIR / "models"
    RUNS_DIR: Path = DATA_DIR / "runs"

    EXPORTS_DIR: Path = DATA_DIR / "exports"

    # Contains combined datasets (original and synthetic)
    PREPARED_DATASETS_DIR: Path = DATASETS_DIR / "prepared"

    SYN_DATA_DEFINITIONS_DIR: Path = DATA_DIR / "syn_dataset_definitions"
    PROMPT_TEMPLATES_DIR: Path = DATA_DIR / "prompt_templates"
    SEED_IMAGES_DIR: Path = DATA_DIR / "seed-images"


ENV.BASE_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ENV.SYN_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ENV.SYN_DATASETS_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
ENV.VISUAL_ELEMENT_PREFABS_DIR.mkdir(parents=True, exist_ok=True)
ENV.PREPARED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ENV.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
ENV.CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
ENV.TEMP_DIR.mkdir(parents=True, exist_ok=True)
ENV.MODELS_DIR.mkdir(parents=True, exist_ok=True)
ENV.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
ENV.CLUSTER_PLOTS.mkdir(parents=True, exist_ok=True)
ENV.SYN_DATASET_STAT_PLOTS.mkdir(parents=True, exist_ok=True)
ENV.GT_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
ENV.KIE_GT_ANALYZATION_DIR.mkdir(parents=True, exist_ok=True)
ENV.CLS_GT_ANALYZATION_DIR.mkdir(parents=True, exist_ok=True)
ENV.DLA_GT_ANALYZATION_DIR.mkdir(parents=True, exist_ok=True)
ENV.QA_GT_ANALYZATION_DIR.mkdir(parents=True, exist_ok=True)
ENV.QA_GT_WEBAPP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LLM:
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5-20251001"
    TINYLLM_CLAUDE_SONNET_4 = "anthropic/claude-sonnet-4-20250514"


# Default values for generation
class GENERATION:
    LLM = LLM.CLAUDE_SONNET_4_5
    MAX_TOKENS = 16384
    HANDWRITING_MODEL_CHECKPOINT = ENV.MODELS_DIR / "handwriting" / "latest.pt"
