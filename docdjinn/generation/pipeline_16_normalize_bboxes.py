from dataclasses import asdict
import json
from PIL import Image
import fitz
from docdjinn.generation.constants import IMAGE_RENDER_EXT, PDF_DPI
from docdjinn.generation.models import OCRBox, PipelineParameters, SynDatasetDefinition
from docdjinn.generation.models._bbox import LayoutBox
from docdjinn.generation.models._file import SyntheticDatasetFileStructure
from docdjinn.generation.models._log import SynDocumentLog
from docdjinn.generation.utils.bboxes import (
    read_syn_dataset_bboxes,
    save_bboxes,
)
from docdjinn.generation.utils.documentsize import get_document_size_for_bbox_unnormalization
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def normalize_ocrbox(bbox: OCRBox, width_px, height_px):
    """
    Convert a bounding box from PDF points to normalized image coordinates.
    """
    # Convert PDF points to pixels
    x_min_px = bbox.x0
    y_min_px = bbox.y0
    x_max_px = bbox.x2
    y_max_px = bbox.y2

    # Get image size in pixels
    img_w_px = width_px
    img_h_px = height_px

    # Normalize bounding box
    x_min_norm = x_min_px / img_w_px
    y_min_norm = y_min_px / img_h_px
    x_max_norm = x_max_px / img_w_px
    y_max_norm = y_max_px / img_h_px

    return OCRBox(
        x0=x_min_norm,
        y0=y_min_norm,
        x2=x_max_norm,
        y2=y_max_norm,
        text=bbox.text,
        block_no=bbox.block_no,
        line_no=bbox.line_no,
        word_no=bbox.word_no,
    )


def normalize_and_save_word_and_segment_bboxes(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()

    width_px, height_px = get_document_size_for_bbox_unnormalization(docid=docid, dsfiles=dsfiles)

    # word
    bbox_word_path = dsfiles.get_final_bbox_path(level="word", doc_id=docid)
    bbox_word = read_syn_dataset_bboxes(bbox_word_path)
    bbox_word_normalized = [
        normalize_ocrbox(bbox=b, width_px=width_px, height_px=height_px)
        for b in bbox_word
    ]
    bbox_word_normalized_path = dsfiles.get_final_normalized_bbox_path(
        level="word", doc_id=docid
    )
    save_bboxes(bboxes=bbox_word_normalized, bbox_path=bbox_word_normalized_path)

    # segment
    bbox_segment_path = dsfiles.get_final_bbox_path(level="segment", doc_id=docid)
    bbox_segment = read_syn_dataset_bboxes(bbox_segment_path)
    bbox_segment_normalized = [
        normalize_ocrbox(bbox=b, width_px=width_px, height_px=height_px)
        for b in bbox_segment
    ]
    bbox_segment_normalized_path = dsfiles.get_final_normalized_bbox_path(
        level="segment", doc_id=docid
    )
    save_bboxes(bboxes=bbox_segment_normalized, bbox_path=bbox_segment_normalized_path)


def normalize_layout_bboxes(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()

    pdf_path = dsfiles.final_pdf_directory / f"{docid}.pdf"
    doc = fitz.open(pdf_path)
    page = doc[0]
    width_pt, height_pt = page.rect.width, page.rect.height

    raw_annotations_path = dsfiles.raw_annotations_directory / f"{docid}.json"
    data = json.loads(raw_annotations_path.read_text(encoding="utf-8"))
    layout_bboxes: list[LayoutBox] = [LayoutBox(**d) for d in data]

    layout_bboxes_normalized = [
        LayoutBox.normalize_to_pdf(
            b, width_pt=width_pt, height_pt=height_pt, dpi=PDF_DPI
        )
        for b in layout_bboxes
    ]

    boxes_dicts = [asdict(b) for b in layout_bboxes_normalized]
    gt_path = dsfiles.gt_directory / f"{docid}.json"
    gt_path.write_text(json.dumps(boxes_dicts, indent=4), encoding="utf-8")

    doc.close()


def pipeline_normalize_bboxes(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef

    # Get documents valid for bbox normalization
    valid_documents = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1 and doclog.ocr_found:
            valid_documents.append(doclog.document_id)

    print(f"Found {len(valid_documents)} documents valid for BBox normalization.")

    with get_progress_bar() as progress:
        task = progress.add_task(
            "[white]Normalizing BBoxes...", total=len(valid_documents)
        )

        for docid in valid_documents:
            normalize_and_save_word_and_segment_bboxes(dsdef=dsdef, docid=docid)
            progress.update(task, advance=1)

    # We also normalize the DLA GT here as they are layout BBoxes
    # Get documents valid for layout bbox normalization
    valid_documents = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1 and doclog.layout_elements_num_elements > 0:
            valid_documents.append(doclog.document_id)

    print(
        f"Found {len(valid_documents)} documents valid for Layout BBox normalization."
    )

    with get_progress_bar() as progress:
        task = progress.add_task(
            "[white]Normalizing Layout BBoxes...", total=len(valid_documents)
        )

        for docid in valid_documents:
            normalize_layout_bboxes(dsdef=dsdef, docid=docid)
            progress.update(task, advance=1)
