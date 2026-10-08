"""
Handwriting insertion with left-alignment only (no region-aware scaling).
"""

from collections import Counter, defaultdict
from io import BytesIO
import json
import pathlib
import random
import shutil
from PIL import Image

import fitz  # PyMuPDF
from fitz import Page
from docdjinn import ENV
from docdjinn.generation.constants import (
    FIXED_HANDWRITING_X_OFFSET,
    MAX_HANDWRITING_RAND_DEG_ROT,
    MAX_HANDWRITING_RAND_X_OFFSET_LEFT,
    MAX_HANDWRITING_RAND_X_OFFSET_RIGHT,
    MAX_HANDWRITING_RAND_Y_OFFSET_DOWN,
    MAX_HANDWRITING_RAND_Y_OFFSET_UP,
    PIPELINE_04_3_SCALE_UP_FACTOR,
)
from docdjinn.generation.models import (
    DocLogKey,
    OCRBox,
    PipelineParameters,
    SyntheticDatasetFileStructure,
)

from docdjinn.generation.utils.bboxes import (
    draw_bboxes_on_pdf,
    read_syn_dataset_bbox_str,
)
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def resize_to_bbox_highres(img, bbox_width, bbox_height, scale_up=3):
    """Resize with preserved aspect ratio, pad to bbox, upscale for sharpness."""
    bbox_width = round(bbox_width)
    bbox_height = round(bbox_height)

    # Aspect Ratio
    iw, ih = img.size
    scale = min(bbox_width / iw, bbox_height / ih)

    new_w = int(iw * scale * scale_up)
    new_h = int(ih * scale * scale_up)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGBA")
    final_img = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 0))
    final_img.paste(img_resized, (0, 0), mask=img_resized)

    return final_img


def group_handwriting_bboxes_by_block_line(entry: dict):
    """Group handwriting bboxes by block and line."""
    groupedbboxes = defaultdict(list)

    for seg_idx, bbox in enumerate(entry["bboxes"]):
        box = read_syn_dataset_bbox_str(bbox)
        groupedbboxes[(box.block_no, box.line_no)].append(box)

    for key, bboxes in groupedbboxes.items():
        first = bboxes[0]
        # x0, y0 = first.x0, first.y0
        x0 = min([b.x0 for b in bboxes])
        y0 = min([b.y0 for b in bboxes])
        x2 = max([b.x2 for b in bboxes])
        y2 = max([b.y2 for b in bboxes])
        # last = bboxes[-1]
        # x2, y2 = last.x2, last.y2
        txt = " ".join(b.text for b in bboxes)
        yield OCRBox(
            x0=x0,
            y0=y0,
            x2=x2,
            y2=y2,
            text=txt,
            block_no=key[0],
            line_no=key[1],
            word_no=first.word_no,
        )


def insert_handwriting_images(
    docid: str, dsfiles: SyntheticDatasetFileStructure, scale_up: int, debug: bool
):
    """
    Insert handwriting images with LEFT-ALIGNMENT at rect.x position.
    Uses original bbox height, no region-aware scaling.
    """
    images_path = dsfiles.handwritten_text_images_directory / "sentences" / docid
    images_generated = images_path.exists()

    json_path = dsfiles.handwritten_bboxes_directory / f"{docid}.json"
    handwriting_bboxes = json.loads(json_path.read_text(encoding="utf-8"))
    pdf_path = dsfiles.pdf_without_handwriting_placeholder_directory / f"{docid}.pdf"
    doc = fitz.open(pdf_path)

    missing_images = []
    inserted_bboxes = []

    for entry in handwriting_bboxes:
        hw_id = entry["id"]
        rect = entry["rect"]

        for seg_idx, bbox in enumerate(group_handwriting_bboxes_by_block_line(entry)):
            img_name_prefix = f"{hw_id}_block{bbox.block_no}_line{bbox.line_no}"

            if not images_generated:
                if img_name_prefix not in missing_images:
                    missing_images.append(img_name_prefix)
                continue

            img_path = images_path / f"{img_name_prefix}.png"

            if not img_path.exists():
                if img_name_prefix not in missing_images:
                    missing_images.append(img_name_prefix)
                continue

            img = Image.open(img_path)
            bbox_w, bbox_h = bbox.x2 - bbox.x0, bbox.y2 - bbox.y0

            # Resize using original logic
            # print(f"{docid=} {img_name_prefix=} {bbox_w=} {bbox_h=}")
            img_resized = resize_to_bbox_highres(img, bbox_w, bbox_h, scale_up=scale_up)

            # Random rotation
            rnddeg = 0  # random.random() * 1.5 - (1.5 / 2)
            img_resized = img_resized.rotate(rnddeg)

            # Convert to bytes
            img_bytes = BytesIO()
            img_resized.save(img_bytes, format="png")
            img_bytes = img_bytes.getvalue()

            # LEFT-ALIGN at rect.x instead of bbox.x0
            y_padding = 50
            offset_x = (
                random.randint(
                    -MAX_HANDWRITING_RAND_X_OFFSET_LEFT,
                    MAX_HANDWRITING_RAND_X_OFFSET_RIGHT,
                )
                + FIXED_HANDWRITING_X_OFFSET
            )
            offset_y = random.randint(
                -MAX_HANDWRITING_RAND_Y_OFFSET_UP, MAX_HANDWRITING_RAND_Y_OFFSET_DOWN
            )
            x0 = rect["x"] + offset_x
            y0 = bbox.y0 + offset_y - y_padding
            x2 = min(x0 + img_resized.size[0] / scale_up, bbox.x2) + offset_x
            y2 = (
                min(y0 + img_resized.size[1] / scale_up, bbox.y2)
                + offset_y
                + 2 * y_padding
            )

            # print(
            #     f"{bbox=} {offset_x=} {x0=} {x2=} {img_resized.size[0] / scale_up=} {docid=} {img_name_prefix=}"
            # )

            rect_fitz = fitz.Rect(x0, y0, x2, y2)

            assert len(doc) == 1
            page: Page = doc[0]
            page.insert_image(rect_fitz, stream=img_bytes)

            # Store for debug
            debug_bbox = OCRBox(
                x0=x0,
                y0=y0,
                x2=x2,
                y2=y2,
                text=bbox.text,
                block_no=bbox.block_no,
                line_no=bbox.line_no,
                word_no=bbox.word_no,
            )
            inserted_bboxes.append(debug_bbox)

    output_path = dsfiles.pdf_with_handwriting_directory / f"{docid}.pdf"
    doc.save(output_path)
    doc.close()

    # Debug
    if debug:
        draw_bboxes_on_pdf(
            dsfiles.pdf_with_handwriting_directory / f"{docid}.pdf",
            dsfiles.debug_pdf_handwriting_directory / f"{docid}.pdf",
            inserted_bboxes,
            color=(1, 0, 0),  # handwriting red
        )

    return {
        DocLogKey.handwriting_insertion_success: images_generated
        and len(missing_images) == 0,
        DocLogKey.handwriting_images_were_generated: images_generated,
        DocLogKey.handwriting_missing_images: missing_images,
    }


def pipeline_handwritten_text_insertion(params: PipelineParameters, scale_up: int = 3):
    """Pipeline for inserting handwritten text with left-alignment."""
    log_pipeline_level()

    scale_up = PIPELINE_04_3_SCALE_UP_FACTOR

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    valid_document_ids = []
    total_documents_count = 0
    cnt = Counter()

    for doclog in dsdef.get_document_logs():
        total_documents_count += 1
        if doclog.pdf_num_pages == 1:
            cnt["pdf_num_pages"] += 1

            # Copy each PDF to pdf_with_handwriting_directory
            src = (
                dsfiles.pdf_without_handwriting_placeholder_directory
                / f"{doclog.document_id}.pdf"
            )
            dst = dsfiles.pdf_with_handwriting_directory / f"{doclog.document_id}.pdf"
            shutil.copy(src, dst)

            if doclog.handwriting_num_elements > 0:
                cnt["has_handwriting"] += 1
                if len(doclog.handwriting_element_extraction_errors) == 0:
                    cnt["no_errors"] += 1
                    valid_document_ids.append(doclog.document_id)
                else:
                    print(
                        doclog.document_id, doclog.handwriting_element_extraction_errors
                    )
            else:
                dsdef.write_to_document_log(
                    document_id=doclog.document_id,
                    vals={
                        DocLogKey.handwriting_insertion_success: True,
                        DocLogKey.handwriting_images_were_generated: True,
                        DocLogKey.handwriting_missing_images: [],
                    },
                )

    print(
        f"{len(valid_document_ids)} out of {total_documents_count} Documents valid for handwriting image insertion: {cnt}"
    )

    with get_progress_bar() as progress:
        insert_task = progress.add_task(
            "[red]Inserting text into pdfs...", total=len(valid_document_ids)
        )

        success = 0
        all_logs = []
        for docid in valid_document_ids:
            insertion_log = insert_handwriting_images(
                docid=docid, dsfiles=dsfiles, scale_up=scale_up, debug=params.debug
            )

            dsdef.write_to_document_log(document_id=docid, vals=insertion_log)
            all_logs.append(insertion_log)

            if insertion_log[DocLogKey.handwriting_insertion_success]:
                success += 1

            progress.update(insert_task, advance=1)

        print(
            f"""Inserted handwriting images in {success} PDFs
    {len(valid_document_ids) - success} errors:
    {len([1 for insertlog in all_logs if not insertlog[DocLogKey.handwriting_images_were_generated]])} documents dont have images generated
    {sum([len(insertlog[DocLogKey.handwriting_missing_images]) for insertlog in all_logs if insertlog[DocLogKey.handwriting_images_were_generated]])} images missing for documents where images were generated"""
        )
