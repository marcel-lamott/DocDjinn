from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Iterable

import yaml

from docdjinn import ENV
from docdjinn.generation.models._file import SyntheticDatasetFileStructure
from docdjinn.generation.models._log import SynDocumentLog
from docdjinn.generation.utils.serialization import from_dict


@dataclass
class PromptParameters:
    num_solutions: int
    doc_type: str
    language: str
    gt_type: str  # eg "keys and their values"
    gt_format: str  # eg {"company": "company value", "date": "date value"...}


@dataclass
class SynDatasetDefinition:
    # General
    name: str
    task: str
    dataloader_model_task_as: (
        str | None
    )  # For Kleister which the data loading pipeline handles as QA
    base_dataset_name: str
    documents_count: int
    valid_labels: list[str]  # For DLA, KIE and Classification
    label_mapping: (
        dict[str, str] | None
    )  # For CORD because original labels have dots in them
    valid_secondary_labels: list[str]  # For groupings like in CORD or FUNSD

    # Prompt
    prompt_template: str
    prompt_task: str
    prompt_params: PromptParameters

    # Seed Documents
    hdbscan_min_cluster_size: int
    embedding_type: str
    seed_images_count: int
    alpha: float
    max_seed_pool: int
    seed_selection_strategy: str = "v1"

    def get_document_logs(self) -> Iterable[SynDocumentLog]:
        dsfiles = self.get_file_structure()
        # TODO: dont read files but read from dataset log
        for logfile in dsfiles.document_logs_directory.iterdir():
            docid = logfile.stem
            yield SynDocumentLog(
                document_id=docid, logdir=dsfiles.document_logs_directory
            )

    def write_to_document_log(self, document_id: str, vals: dict):
        dsfiles = self.get_file_structure()
        log_path = dsfiles.document_logs_directory / f"{document_id}.json"

        log = {}
        if log_path.exists():
            log = json.loads(log_path.read_text("utf-8"))

        log.update(vals)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    def reset_data_except_prompt_and_seeds(self):
        import shutil

        dsfiles = self.get_file_structure()

        dirs_to_delete = [
            dsfiles._annotations_directory,
            dsfiles._bbox_directory,
            dsfiles._debug_directory,
            dsfiles._handwriting_directory,
            dsfiles._html_directory,
            dsfiles.layout_element_definitions_directory,
            dsfiles.geometries_directory,
            dsfiles._pdf_directory,
            dsfiles._visual_elements_directory,
            dsfiles.img_directory,
            dsfiles.document_logs_directory,
            dsfiles.message_processing_logs_directory,
        ]
        for dir_path in dirs_to_delete:
            shutil.rmtree(dir_path)  # remove entire directory

        # Clear cache
        del self._file_structure
        # Recreate directory structure
        self.get_file_structure()

    def get_file_structure(self) -> SyntheticDatasetFileStructure:
        if hasattr(self, "_file_structure"):
            return self._file_structure
        else:
            self._file_structure = SyntheticDatasetFileStructure(ds_name=self.name)

        return self._file_structure

    def get_prompt_template(self) -> str:
        taskname = f"-{self.prompt_task}" if self.prompt_task else ""
        return (
            ENV.PROMPT_TEMPLATES_DIR
            / self.prompt_template
            / f"seed-based{taskname}.txt"
        ).read_text()

    def get_prompt(self) -> str:
        if hasattr(self, "_prompt"):
            return self._prompt
        else:
            prompt = self.get_prompt_template()
            prompt = prompt.replace(
                "{num_solutions}", f"{self.prompt_params.num_solutions}"
            )
            prompt = prompt.replace("{doc_type}", f"{self.prompt_params.doc_type}")
            prompt = prompt.replace("{language}", f"{self.prompt_params.language}")
            prompt = prompt.replace("{gt_type}", f"{self.prompt_params.gt_type}")
            prompt = prompt.replace("{gt_format}", f"{self.prompt_params.gt_format}")
            self._prompt = prompt

        return self._prompt

    @staticmethod
    def from_file(yaml_path: str | pathlib.Path) -> SynDatasetDefinition:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return from_dict(SynDatasetDefinition, data)
