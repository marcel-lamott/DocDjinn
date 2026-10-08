from enum import Enum


class LLMType(Enum):
    CLAUDE = "claude"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"


class DatasetTask(Enum):
    KIE = "KIE"
    QA = "QA"
    DLA = "DLA"
    CLASSIFICATION = "CLASSIFICATION"
