from dataclasses import dataclass

from docdjinn.generation.models._consts import DatasetTask, LLMType
from docdjinn.generation.models._syndatadef import SynDatasetDefinition


@dataclass
class PipelineParameters:
    dsdef: SynDatasetDefinition
    llmtype: LLMType
    message_custom_id: str | None
    seedsonly: bool
    handwriting_batch_size: int
    debug: bool
    api_key_env_variable_name: str
    generate_handwriting: bool = True
    blur_handwriting_images: bool = True
