import json
import pathlib
import shutil

import fitz
from docdjinn.generation.constants import IMAGE_RENDER_EXT, PDF_DPI
from docdjinn.generation.models import (
    OCRBox,
    PipelineParameters,
    SynDatasetDefinition,
    SynDocumentLog,
)
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from PIL import Image

from docdjinn.generation.utils.bboxes import (
    draw_bboxes_on_image,
    draw_bboxes_on_pdf,
    read_syn_dataset_bboxes,
    save_bboxes,
)
from docdjinn.generation.utils.geos import rect_to_ocrbox
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def mm_to_px(mm: int | float):
    return mm * 72 / 25.4


def draw_visual_elements_debug(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()
    bboxes = []
    data_path = dsfiles.visual_element_definitions_directory / f"{docid}.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    for d in data:
        if d["error"] is None:
            b = rect_to_ocrbox(d["rect"])
            bboxes.append(b)

    draw_bboxes_on_pdf(
        dsfiles.final_pdf_directory / f"{docid}.pdf",
        dsfiles.debug_pdf_visual_elements_directory / f"{docid}.pdf",
        bboxes,
        color=(0, 0, 1),  # visual elements blue
    )


def unnormalize_bboxes(bboxes: list[OCRBox], width: float, height: float):
    for b in bboxes:
        yield OCRBox(
            x0=b.x0 * width,
            y0=b.y0 * height,
            x2=b.x2 * width,
            y2=b.y2 * height,
            text=b.text,
            block_no=b.block_no,
            line_no=b.line_no,
            word_no=b.word_no,
        )


def draw_bbox_final_debug(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()

    bbox_norm_path = dsfiles.get_final_normalized_bbox_path(
        level="segment", doc_id=docid
    )
    bbox_norm = read_syn_dataset_bboxes(bbox_norm_path)

    img_path = dsfiles.img_directory / f"{docid}.{IMAGE_RENDER_EXT}"
    img = Image.open(img_path)
    width, height = img.size
    bbox_unnorm = list(unnormalize_bboxes(bboxes=bbox_norm, width=width, height=height))

    try:
        img_altered = draw_bboxes_on_image(img, bbox_unnorm, show_text=True)
        img_altered.save(
            dsfiles.debug_pdf_bboxes_final_directory / f"{docid}.{IMAGE_RENDER_EXT}"
        )
    except Exception as err:
        print(f"[ERROR]: Skipping debug PDF: {str(err)}")


def draw_bbox_debug(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()

    bbox_norm_path = dsfiles.get_pdf_bbox_path(level="word", doc_id=docid)
    bbox_unnorm = read_syn_dataset_bboxes(bbox_norm_path)

    pdf_path = dsfiles.pdf_initial_directory / f"{docid}.pdf"
    outpath = dsfiles.debug_pdf_bboxes_directory / f"{docid}.pdf"

    try:
        draw_bboxes_on_pdf(pdf_path=pdf_path, outpath=outpath, bboxes=bbox_unnorm)
    except Exception as err:
        print(f"[ERROR]: Skipping debug PDF: {str(err)}")


def pipeline_create_debug_data(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid documents
    valid_documents = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1

        if doclog.pdf_num_pages == 1:
            valid_documents.append(doclog)

    print(f"Found {len(valid_documents)} documents valid for debug PDF/Img drawing.")

    with get_progress_bar() as progress:
        task = progress.add_task(
            "[white]Draw Debug PDF/Images...", total=len(valid_documents)
        )

        doclog: SynDocumentLog
        for doclog in valid_documents:
            docid = doclog.document_id

            # Copy raw HTML to debug directory
            src = dsfiles.raw_html_directory / f"{docid}.html"
            tgt = dsfiles.debug_html_raw_directory / f"{docid}.html"
            shutil.copy(src, tgt)

            if doclog.visual_elements_num_elements > 0:
                draw_visual_elements_debug(dsdef=dsdef, docid=docid)

            # Handwriting debug is created when handwriting is inserted

            if doclog.ocr_found:
                draw_bbox_final_debug(dsdef=dsdef, docid=docid)

            progress.update(task, advance=1)

        # Copy debug script into debug html directory
        debug_script_fname = "debug.js"
        src_dir = pathlib.Path(__file__).parent
        src_path = src_dir / debug_script_fname
        dst_path = dsfiles.debug_html_raw_directory / debug_script_fname
        shutil.copy(src_path, dst_path)
