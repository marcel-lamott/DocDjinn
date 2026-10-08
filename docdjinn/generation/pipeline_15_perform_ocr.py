from collections import defaultdict
import json
from typing import Literal
from docdjinn.generation.constants import IMAGE_RENDER_EXT, PDF_DPI
from docdjinn.generation.models import DocLogKey, OCRBox, PipelineParameters
from docdjinn.generation.models._syndatadef import SynDatasetDefinition
from docdjinn.generation.utils.bboxes import read_syn_dataset_bboxes, save_bboxes
from docdjinn.generation.utils.debug import draw_geos_and_bboxes_on_pdf
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.ocr import call_ocr_service_from_file
from docdjinn.generation.utils.status import get_progress_bar
from docdjinn.utils.ocr import MicrosoftOCR, MicrosoftOCRWord


def _convert_ms_ocr_to_ocrbox(
    ocr: MicrosoftOCR, level: Literal["word", "lines"]
) -> list[OCRBox]:
    res = list()
    word: MicrosoftOCRWord
    collection = ocr.words if level == "word" else ocr.lines
    for word in collection:
        (left, top, width, height) = tuple(word.geo)
        box = OCRBox(
            x0=left,
            y0=top,
            x2=left + width,
            y2=top + height,
            text=word.text,
            block_no=-1,  # not supplied
            line_no=-1,  # not supplied
            word_no=-1,  # not supplied
        )
        res.append(box)

    return res


def _convert_word_level_to_line_level_bboxes(bboxes: list[OCRBox]) -> list[OCRBox]:
    grouped = defaultdict(list)
    for b in bboxes:
        grouped[(b.block_no, b.line_no)].append(b)

    result = []
    for (block_no, line_no), boxes in grouped.items():
        first: OCRBox = boxes[0]
        last: OCRBox = boxes[-1]
        txt = " ".join([b.text for b in boxes])
        result.append(
            OCRBox(
                x0=first.x0,
                y0=first.y0,
                x2=last.x2,
                y2=last.y2,
                text=txt,
                block_no=block_no,
                line_no=line_no,
                word_no=first.word_no,
            )
        )

    return result


def draw_bbox_debug(dsdef: SynDatasetDefinition, docid: str, bboxes: list[OCRBox]):
    dsfiles = dsdef.get_file_structure()

    pdf_path = dsfiles.final_pdf_directory / f"{docid}.pdf"

    geo_path = dsfiles.geometries_directory / f"{docid}.json"
    geos = json.loads(geo_path.read_text(encoding="utf-8"))

    outpath = dsfiles.debug_ocr_bboxes_and_geos_directory / f"{docid}.pdf"

    # for g in geos:
    #     g["rect"] = pdf_region_to_image(g["rect"])

    bboxes = [b.scale(72.0 / PDF_DPI) for b in bboxes]

    try:
        draw_geos_and_bboxes_on_pdf(
            pdf_in=pdf_path,
            pdf_out=outpath,
            bboxes_=bboxes,
            geos=geos,
            verbose=False,
        )
    except Exception as err:
        print(f"[ERROR]: Skipping debug PDF: {str(err)}")


def pipeline_perform_ocr(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    documents_requiring_ocr = []
    documents_not_requiring_ocr = []
    for doclog in dsdef.get_document_logs():
        has_valid_handwriting = (
            doclog.handwriting_num_elements > 0
            and len(doclog.handwriting_element_extraction_errors) == 0
        )
        has_valid_visual_elements = (
            doclog.visual_elements_num_elements > 0
            and len(doclog.visual_elements_extraction_errors) == 0
        )
        is_valid_document = doclog.pdf_num_pages == 1
        if not is_valid_document:
            continue

        if has_valid_handwriting or has_valid_visual_elements:
            documents_requiring_ocr.append(doclog.document_id)
        else:
            documents_not_requiring_ocr.append(doclog.document_id)

    total_valid_documents = len(documents_requiring_ocr) + len(
        documents_not_requiring_ocr
    )
    print(
        f"{len(documents_requiring_ocr)} out of {total_valid_documents} valid documents require OCR."
    )

    """
    We collect BBoxes and supply segment-level bounding boxes in the end
    """

    # First copy BBoxes extracted from PDF for those documents that don't require OCR
    with get_progress_bar() as progress:
        task = progress.add_task(
            "[white]Copy BBoxes for documents not requiring OCR...",
            total=len(documents_not_requiring_ocr),
        )

        for docid in documents_not_requiring_ocr:
            pdf_bbox_path = dsfiles.get_pdf_bbox_path(level="word", doc_id=docid)
            word_bboxes = read_syn_dataset_bboxes(pdf_bbox_path)
            result_path = dsfiles.get_final_bbox_path(level="word", doc_id=docid)
            save_bboxes(word_bboxes, result_path)

            line_bboxes = _convert_word_level_to_line_level_bboxes(word_bboxes)
            result_path = dsfiles.get_final_bbox_path(level="segment", doc_id=docid)
            save_bboxes(line_bboxes, result_path)

            dsdef.write_to_document_log(
                document_id=docid,
                vals={
                    DocLogKey.ocr_required: False,
                    DocLogKey.ocr_found: True,
                    DocLogKey.ocr_num_bboxes_words: len(word_bboxes),
                    DocLogKey.ocr_num_bboxes_lines: len(line_bboxes),
                },
            )

            progress.update(task, advance=1)

    with get_progress_bar() as progress:
        task = progress.add_task(
            "[white]Performing OCR for documents...", total=len(documents_requiring_ocr)
        )

        # Then parse OCR results for other documents
        ocr_not_found_count = 0
        for docid in documents_requiring_ocr:
            image_file = dsfiles.img_directory / f"{docid}.{IMAGE_RENDER_EXT}"
            ocr_result_file = (
                dsfiles.ocr_results_directory
                / f"{docid}.{IMAGE_RENDER_EXT}.0.MicrosoftOcrService.json"
            )

            ocr_error = None
            try:
                if ocr_result_file.exists():
                    ocr_result = MicrosoftOCR.load_from_file(ocr_result_file)
                else:
                    ocr_result: MicrosoftOCR = call_ocr_service_from_file(
                        image_file, client_caching=False
                    )
                    ocr_result.save_to_file(ocr_result_file)
            except Exception as e:
                ocr_error = str(e)

            ocr_found = ocr_result_file.exists()

            num_bboxes_words = -1
            num_bboxes_lines = -1
            if ocr_found:
                bboxes: list[OCRBox] = _convert_ms_ocr_to_ocrbox(
                    ocr=ocr_result,  # type: ignore
                    level="word",
                )

                # Write to file
                result_path = dsfiles.get_final_bbox_path(level="word", doc_id=docid)
                save_bboxes(
                    bboxes=bboxes,
                    bbox_path=result_path,
                )
                num_bboxes_words = len(bboxes)

                if params.debug:
                    draw_bbox_debug(dsdef=dsdef, docid=docid, bboxes=bboxes)

                # Parse Microsoft OCR for lines
                ocr_result: MicrosoftOCR = MicrosoftOCR.load_from_file(ocr_result_file)
                bboxes: list[OCRBox] = _convert_ms_ocr_to_ocrbox(
                    ocr=ocr_result, level="lines"
                )

                # Write to file
                result_path = dsfiles.get_final_bbox_path(level="segment", doc_id=docid)
                save_bboxes(
                    bboxes=bboxes,
                    bbox_path=result_path,
                )
                num_bboxes_lines = len(bboxes)
            else:
                ocr_not_found_count += 1

            dsdef.write_to_document_log(
                document_id=docid,
                vals={
                    DocLogKey.ocr_required: True,
                    DocLogKey.ocr_found: ocr_found,
                    DocLogKey.ocr_num_bboxes_words: num_bboxes_words,
                    DocLogKey.ocr_num_bboxes_lines: num_bboxes_lines,
                    DocLogKey.ocr_error: ocr_error,
                },
            )

            progress.update(task, advance=1)

    print(
        f"{ocr_not_found_count} of {len(documents_requiring_ocr)} OCR results documents missing."
    )
