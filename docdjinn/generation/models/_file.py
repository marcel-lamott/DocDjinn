import pathlib
from typing import Literal

from docdjinn import ENV


class SyntheticDatasetFileStructure:
    def __init__(self, ds_name: str):
        self.ds_name = ds_name

        self.prompt_batches_directory.mkdir(parents=True, exist_ok=True)
        self.message_results_directory.mkdir(parents=True, exist_ok=True)
        self.preprocessed_seed_images_directory.mkdir(parents=True, exist_ok=True)
        self.message_processing_logs_directory.mkdir(parents=True, exist_ok=True)
        self.raw_html_directory.mkdir(parents=True, exist_ok=True)
        self.render_html_directory.mkdir(parents=True, exist_ok=True)
        self.render_html_second_pass_directory.mkdir(parents=True, exist_ok=True)
        self.geometries_directory.mkdir(parents=True, exist_ok=True)
        self.raw_annotations_directory.mkdir(parents=True, exist_ok=True)
        self.pdf_initial_directory.mkdir(parents=True, exist_ok=True)
        self.pdf_with_handwriting_directory.mkdir(parents=True, exist_ok=True)
        self.pdf_without_handwriting_placeholder_directory.mkdir(
            parents=True, exist_ok=True
        )
        self.final_pdf_directory.mkdir(parents=True, exist_ok=True)
        self.bboxes_pdf_directory.mkdir(parents=True, exist_ok=True)
        self.bboxes_final_directory.mkdir(parents=True, exist_ok=True)
        self.bboxes_final_normalized_directory.mkdir(parents=True, exist_ok=True)
        self.ocr_results_directory.mkdir(parents=True, exist_ok=True)
        self.img_directory.mkdir(parents=True, exist_ok=True)
        self.gt_directory.mkdir(parents=True, exist_ok=True)
        self.document_logs_directory.mkdir(parents=True, exist_ok=True)
        self.handwritten_bboxes_directory.mkdir(parents=True, exist_ok=True)
        self.visual_element_definitions_directory.mkdir(parents=True, exist_ok=True)
        self.visual_elements_directory.mkdir(parents=True, exist_ok=True)
        self.layout_element_definitions_directory.mkdir(parents=True, exist_ok=True)
        # Directories for handwritten text images
        self.handwritten_text_images_directory.mkdir(parents=True, exist_ok=True)

        self.debug_pdf_visual_elements_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_handwriting_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_layout_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_geometries_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_bboxes_final_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_bboxes_directory.mkdir(parents=True, exist_ok=True)
        self.debug_pdf_bboxes_and_geos_directory.mkdir(parents=True, exist_ok=True)
        self.debug_ocr_bboxes_and_geos_directory.mkdir(parents=True, exist_ok=True)
        self.debug_html_raw_directory.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> pathlib.Path:
        return ENV.SYN_DATASETS_DIR / self.ds_name

    @property
    def ds_log_path(self) -> pathlib.Path:
        return self.base_path / "dataset_log.json"
    
    @property
    def ds_csv_log_path(self)->pathlib.Path:
        return self.base_path / "dataset_log.csv"

    # Keep on reset
    @property
    def prompt_batches_directory(self) -> pathlib.Path:
        return self.base_path / "logs" / "prompt_batches"

    # Keep on reset
    @property
    def message_results_directory(self) -> pathlib.Path:
        return self.base_path / "logs" / "prompt_message_results"

    # Keep on reset
    @property
    def preprocessed_seed_images_directory(self) -> pathlib.Path:
        return self.base_path / "preprocessed_seed_images"

    @property
    def message_processing_logs_directory(self) -> pathlib.Path:
        return self.base_path / "logs" / "message_processing_logs"

    @property
    def _html_directory(self) -> pathlib.Path:
        return self.base_path / "html"

    @property
    def raw_html_directory(self) -> pathlib.Path:
        return self._html_directory / "raw_html"

    @property
    def render_html_directory(self) -> pathlib.Path:
        return self._html_directory / "render_html_pass1"

    @property
    def render_html_second_pass_directory(self) -> pathlib.Path:
        return self._html_directory / "render_html_pass2"

    @property
    def geometries_directory(self) -> pathlib.Path:
        return self.base_path / "geometries"

    @property
    def _pdf_directory(self) -> pathlib.Path:
        return self.base_path / "pdf"

    @property
    def pdf_initial_directory(self) -> pathlib.Path:
        """Contains PDFs with handwriting-html-text visible"""
        return self._pdf_directory / "pdf_initial"

    @property
    def pdf_without_handwriting_placeholder_directory(self) -> pathlib.Path:
        """Contains PDFs with handwriting-html-text and visual element placeholders invisible"""
        return self._pdf_directory / "pdf_without_handwriting_placeholder"

    @property
    def pdf_with_handwriting_directory(self) -> pathlib.Path:
        """Contains PDFs where Handwriting and Visual Elements are invisible
        (need two render passes because transparent text is not included in PDF)"""
        return self._pdf_directory / "pdf_with_handwriting"

    @property
    def final_pdf_directory(self) -> pathlib.Path:
        """Contains final PDFs with handwriting and visual elements"""
        return self._pdf_directory / "pdf_final"

    @property
    def _bbox_directory(self) -> pathlib.Path:
        return self.base_path / "bbox"

    @property
    def bboxes_pdf_directory(self) -> pathlib.Path:
        """Contains the bounding boxes which were extracted from the PDF."""
        return self._bbox_directory / "bbox_pdf"

    @property
    def bboxes_final_directory(self) -> pathlib.Path:
        """For documents which contain handwriting or visual elements, this contains bounding boxes retrieved via OCR.
        Otherwise contains the bounding boxes which were extracted from the PDF."""
        return self._bbox_directory / "bbox_final"

    @property
    def bboxes_final_normalized_directory(self) -> pathlib.Path:
        """Contains the final bboxes but normalized to image size."""
        return self._bbox_directory / "bbox_final_normalized"

    @property
    def ocr_results_directory(self) -> pathlib.Path:
        """Contains OCR results for documents which contain handwriting or visual elements"""
        return self.base_path / "ocr_results"

    @property
    def img_directory(self) -> pathlib.Path:
        return self.base_path / "img"

    @property
    def _annotations_directory(self) -> pathlib.Path:
        return self.base_path / "annotations"

    @property
    def gt_directory(self) -> pathlib.Path:
        return self._annotations_directory / "gt"

    @property
    def raw_annotations_directory(self) -> pathlib.Path:
        return self._annotations_directory / "raw_annotations"

    @property
    def document_logs_directory(self) -> pathlib.Path:
        return self.base_path / "logs" / "document_logs"

    @property
    def _handwriting_directory(self) -> pathlib.Path:
        return self.base_path / "handwriting"

    @property
    def handwritten_bboxes_directory(self) -> pathlib.Path:
        return self._handwriting_directory / "handwriting_bbox"

    # Directories for handwritten text images
    @property
    def handwritten_text_images_directory(self) -> pathlib.Path:
        return self._handwriting_directory / "handwriting_raw_tokens"

    @property
    def _visual_elements_directory(self) -> pathlib.Path:
        return self.base_path / "visual_elements"

    @property
    def visual_element_definitions_directory(self) -> pathlib.Path:
        return self._visual_elements_directory / "visual_element_definitions"

    @property
    def visual_elements_directory(self) -> pathlib.Path:
        return self._visual_elements_directory / "visual_elements_images"

    @property
    def layout_element_definitions_directory(self) -> pathlib.Path:
        return self.base_path / "layout_element_definitions"

    @property
    def _debug_directory(self) -> pathlib.Path:
        return self.base_path / "debug"

    @property
    def debug_pdf_visual_elements_directory(self) -> pathlib.Path:
        return self._debug_directory / "visual_elements"

    @property
    def debug_pdf_handwriting_directory(self) -> pathlib.Path:
        return self._debug_directory / "handwriting"

    @property
    def debug_pdf_layout_directory(self) -> pathlib.Path:
        return self._debug_directory / "layout"

    @property
    def debug_pdf_geometries_directory(self) -> pathlib.Path:
        return self._debug_directory / "geometries"

    @property
    def debug_pdf_bboxes_final_directory(self) -> pathlib.Path:
        return self._debug_directory / "bboxes_final"

    @property
    def debug_pdf_bboxes_directory(self) -> pathlib.Path:
        return self._debug_directory / "bboxes"

    @property
    def debug_pdf_bboxes_and_geos_directory(self) -> pathlib.Path:
        return self._debug_directory / "bboxes_and_geos"

    @property
    def debug_ocr_bboxes_and_geos_directory(self) -> pathlib.Path:
        return self._debug_directory / "ocr_bboxes_and_geos"

    @property
    def debug_html_raw_directory(self) -> pathlib.Path:
        return self._debug_directory / "html_raw"

    def get_pdf_bbox_path(self, level: Literal["word", "char"], doc_id: str):
        return self.bboxes_pdf_directory / level / f"{doc_id}.txt"

    def get_final_bbox_path(self, level: Literal["word", "segment"], doc_id: str):
        return self.bboxes_final_directory / level / f"{doc_id}.txt"

    def get_final_normalized_bbox_path(
        self, level: Literal["word", "segment"], doc_id: str
    ):
        return self.bboxes_final_normalized_directory / level / f"{doc_id}.txt"
