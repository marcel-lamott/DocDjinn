import pathlib
import shutil
from docdjinn.generation.models import (
    DocLogKey,
    PipelineParameters,
    SyntheticDatasetFileStructure,
    SynDocumentLog,
    OCRBox,
)
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from docdjinn.generation.constants import PIPELINE_04_3_SCALE_UP_FACTOR
import fitz
from fitz import Page
from PIL import Image
from io import BytesIO

import json
from typing import Union

from docdjinn.generation.utils.geos import rect_to_ocrbox
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar

__SCALE_UP__ = PIPELINE_04_3_SCALE_UP_FACTOR


def resize_to_bbox_highres(img, bbox_width, bbox_height, scale_up=3):
    """Resize with preserved aspect ratio, pad to bbox, upscale for sharpness."""

    """I am not directly resizing image to bbox coords,
    First calculate a scale factor that avoids overfllow
    in horizontal and vertical direction(that's why min)"""
    """Because scale is used for both width and height,
    aspect ratio = display_w/display_h = iw/ih (unchanged) ratio will remain same."""
    bbox_width = round(bbox_width)
    bbox_height = round(bbox_height)

    # -----------Aspect Ratio---------------
    iw, ih = img.size
    scale = min(bbox_width / iw, bbox_height / ih)

    new_w = int(iw * scale * scale_up)
    new_h = int(ih * scale * scale_up)
    # -----------Aspect Ratio---------------
    # ------------Resolution-----------------
    """f you embed an image whose pixel dimensions are exactly (display_w, display_h),
    those are the only pixels available to draw the strokes — often too few for a crisp
    rendering, especially if display_w or display_h is small.
    If we X with scale_up we have more pixels to draw image."""
    # ------------Resolution-----------------

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGBA")

    # Create high-res white background
    final_img = Image.new(
        "RGBA", (bbox_width * scale_up, bbox_height * scale_up), (255, 255, 255, 0)
    )

    # Paste resized image centered
    offset_x = (bbox_width * scale_up - new_w) // 2
    offset_y = (bbox_height * scale_up - new_h) // 2
    final_img.paste(img_resized, (offset_x, offset_y), mask=img_resized)

    return final_img


def mm_to_px(mm: Union[int, float]):
    return mm * 72 / 25.4


def insert_visual_elements(
    veds: list[dict],
    docid: str,
    dsfiles: SyntheticDatasetFileStructure,
):
    input_path = dsfiles.pdf_with_handwriting_directory / f"{docid}.pdf"
    output_pdf_path = dsfiles.final_pdf_directory / f"{docid}.pdf"

    ve_dir = dsfiles.visual_elements_directory / f"{docid}"
    ve_generated = ve_dir.exists()
    missing_ves = []

    doc = fitz.open(input_path)
    for d in veds:
        ve_id = d.get("id", None)

        if not ve_generated:
            print(
                f"[Warning] Visual elements directory does not exist for {docid}. Skipping"
            )
            if ve_id not in missing_ves:
                missing_ves.append(ve_id)
            continue
        img_path = ve_dir / f"{ve_id}.png"

        if not img_path.exists():
            print(
                f"[Warning] Visual element with id {ve_id} do not exist for {docid}. Skipping"
            )
            if ve_id not in missing_ves:
                missing_ves.append(ve_id)
            continue

        # computing bbox as in gitlab ticket
        # width_pt = mm_to_px(d["width_mm"])
        # height_pt = mm_to_px(d["height_mm"])
        # off_x, off_y = width_pt / 2.0, height_pt / 2.0
        # b = OCRBox(
        #     x0=d["center_x"] - off_x,
        #     x2=d["center_x"] - off_x + width_pt,
        #     y0=d["center_y"] - off_y,
        #     y2=d["center_y"] - off_y + height_pt,
        #     text="",
        #     block_no=-1,
        #     line_no=-1,
        #     word_no=-1,
        # )
        rect = d["rect"]
        b = rect_to_ocrbox(rect)
        bbox_w, bbox_h = b.width, b.height

        img = Image.open(img_path)
        img_resized = resize_to_bbox_highres(img, bbox_w, bbox_h, scale_up=__SCALE_UP__)

        img_bytes = BytesIO()
        img_resized.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        rect = fitz.Rect(b.x0, b.y0, b.x2, b.y2)
        assert len(doc) == 1, (
            f"Multipage: {dsfiles.pdf_initial_directory / f'{docid}.pdf'}, {dsfiles.pdf_with_handwriting_directory / f'{docid}.pdf'}"
        )
        page: Page = doc[0]  # single-page assumption
        page.insert_image(rect, stream=img_bytes)  # type: ignore

    doc.save(output_pdf_path)
    doc.close()
    return {
        DocLogKey.visual_elements_insertion_success: ve_generated
        and len(missing_ves) == 0,
        DocLogKey.visual_elements_were_generated: ve_generated,
        DocLogKey.visual_elements_missing_images: missing_ves,
    }


def pipeline_insert_visual_elements(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    valid_document_ids = []
    total_documents_count = 0

    for doclog in dsdef.get_document_logs():
        total_documents_count += 1

        if doclog.pdf_num_pages == 1:
            # Already copy each PDF to pdf_final, those which have vis elems inserted are later overridden
            src = dsfiles.pdf_with_handwriting_directory / f"{doclog.document_id}.pdf"
            dst = dsfiles.final_pdf_directory / f"{doclog.document_id}.pdf"
            shutil.copy(src, dst)

            if (
                doclog.visual_elements_num_elements > 0
                and len(doclog.visual_elements_extraction_errors) == 0
            ):
                valid_document_ids.append(doclog.document_id)
    print(
        f"{len(valid_document_ids)} of {total_documents_count} documents valid for visual element insertion."
    )

    with get_progress_bar() as progress:
        insert_task = progress.add_task(
            "[red]Inserting visual elements into pdfs...", total=len(valid_document_ids)
        )
        success = 0
        examples = list()
        for docid in valid_document_ids:
            visual_element_def_file = (
                dsfiles.visual_element_definitions_directory / f"{docid}.json"
            )
            visual_element_definitions = json.loads(
                visual_element_def_file.read_text(encoding="utf-8")
            )

            insertion_logs = insert_visual_elements(
                veds=visual_element_definitions, docid=docid, dsfiles=dsfiles
            )
            dsdef.write_to_document_log(document_id=docid, vals=insertion_logs)
            if insertion_logs[DocLogKey.visual_elements_insertion_success]:
                success += 1
                examples.append(
                    {
                        "docid": docid,
                        "types": sorted(
                            {v["type"] for v in visual_element_definitions}
                        ),
                    }
                )
            progress.update(insert_task, advance=1)

        print(
            f"""Inserted visual elements in {success} PDFs and {len(valid_document_ids) - success} errors occur.
    Examples: {examples[:3]}"""
        )
