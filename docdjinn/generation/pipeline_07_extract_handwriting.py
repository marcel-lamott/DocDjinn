from collections import defaultdict
from curses.ascii import isspace
import json
import pathlib

from bs4 import BeautifulSoup
from docdjinn.generation.constants import (
    BBOX_TO_GEO_MATCHING_THRESHOLD,
    HANDWRITING_CLASS_NAME,
    PIPELINE_06_EXTRACT_HANDWRITING__MAX_WORD_LEN,
    SIGNATURE_CLASS_NAME,
)
from docdjinn.generation.models import (
    DocLogKey,
    OCRBox,
    PipelineParameters,
    SynDatasetDefinition,
)

from docdjinn.generation.utils.bboxes import is_in_rect, read_syn_dataset_bboxes
from docdjinn.generation.utils.geos import read_handwriting_elements_from_geos
from docdjinn.generation.utils.handwriting import get_author_id
from docdjinn.generation.utils.html import get_field_text
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar

__BS_PARSER = "lxml"  # "html.parser"


"""
Need to work with char level here as well and split words longer than N characters
"""


def decompose_word_bboxes(word_bboxes, char_bboxes, N: int = 7, verbose=False):
    bbox: OCRBox
    word_no_to_chars = defaultdict(list)
    for bbox in char_bboxes:
        word_no_to_chars[bbox.key].append(bbox)

    result = []
    for bbox in word_bboxes:
        # result.append([])
        if N > 0 and len(bbox.text) > N:
            # Split bbox -> collect char bboxes
            chars = word_no_to_chars[bbox.key]
            # print(bbox.text, ''.join([c.text for c in chars]))
            for i in range(0, len(bbox.text), N):
                subword_chars: list[OCRBox] = chars[i : i + N]
                if verbose:
                    for c in subword_chars:
                        print(c)
                subword_bbox = OCRBox(
                    x0=subword_chars[0].x0,
                    y0=subword_chars[0].y0,
                    x2=subword_chars[-1].x2,
                    y2=subword_chars[-1].y2,
                    text=bbox.text[i : i + N],
                    block_no=bbox.block_no,
                    line_no=bbox.line_no,
                    word_no=bbox.word_no,
                )
                # print(subword_bbox)
                # result[-1].append(subword_bbox)
                result.append(subword_bbox)
        else:
            # result[-1].append(bbox)
            result.append(bbox)

    # print(result)
    # input()
    return result


def extract_handwritten_fields(
    dsdef: SynDatasetDefinition, doc_id: str, max_word_len: int = -1
) -> list[dict]:
    paths = dsdef.get_file_structure()
    word_bbox_path = paths.get_pdf_bbox_path(level="word", doc_id=doc_id)
    word_bboxes = read_syn_dataset_bboxes(word_bbox_path)
    char_bbox_path = paths.get_pdf_bbox_path(level="char", doc_id=doc_id)
    char_bboxes = read_syn_dataset_bboxes(char_bbox_path)

    geo_path = paths.geometries_directory / f"{doc_id}.json"
    geos = read_handwriting_elements_from_geos(geo_path=geo_path)
    geos = list(geos)

    # Extract text content
    result = []
    taken_bbox_indices = set()
    for i, geo in enumerate(geos):
        field_text = geo["text"]

        # Get author ID
        classes = geo["classes"].split(" ")
        author_id = get_author_id(classes)  # type: ignore

        is_signature = SIGNATURE_CLASS_NAME in classes  # type: ignore

        if author_id is None:
            value = {
                "id": f"hw{i}",
                "text": field_text,
                "author-id": None,
                "bboxes": None,
                "rect": geo["rect"],
                "is_signature": is_signature,
                "error": "no-authorid",
            }
            result.append(value)
            continue

        if not field_text or field_text.isspace():
            value = {
                "id": f"hw{i}",
                "text": field_text,
                "author-id": None,
                "bboxes": None,
                "rect": geo["rect"],
                "is_signature": is_signature,
                "error": "no-text",
            }
            result.append(value)
            continue

        startidx, stopidx = find_bbox_indices(
            word_bboxes,
            query=field_text,
            taken_indices=taken_bbox_indices,
            rect=geo["rect"],
            verbose=False,
        )
        taken_bbox_indices.add((startidx, stopidx))
        if startidx is None or stopidx is None:
            value = {
                "id": f"hw{i}",
                "text": field_text,
                "author-id": None,
                "bboxes": None,
                "rect": geo["rect"],
                "is_signature": is_signature,
                "error": "not-found",
            }
            result.append(value)
            continue

        corresponding_boxes = word_bboxes[startidx:stopidx]
        extracted_text = " ".join([b.text for b in corresponding_boxes])
        extracted_text = extracted_text.strip()
        # assert field_text == extracted_text, f'{field_text=} {extracted_text=}'

        # Split words to max len for diffusion model
        if max_word_len > 1:
            split_bboxes = decompose_word_bboxes(
                word_bboxes=corresponding_boxes,
                char_bboxes=char_bboxes,
                N=max_word_len,
                verbose=False,
            )
            bboxes = split_bboxes
        else:
            bboxes = corresponding_boxes

        value = {
            "id": f"hw{i}",
            "text": field_text,
            "author-id": author_id,
            "bboxes": [b.as_string() for b in bboxes],
            "rect": geo["rect"],
            "is_signature": is_signature,
            "error": None,
        }
        result.append(value)

    return result


def find_bbox_indices(
    bboxes: list[OCRBox],
    query: str,
    taken_indices: set[tuple[int, int]],
    rect: dict,
    verbose: bool,
) -> tuple[int | None, int | None]:
    """
    Find consecutive bounding boxes matching the full query string.

    Parameters:
        bboxes (list of tuples): [(x1, y1, x2, y2, text), ...]
        query (str): The full string to search for (words separated by spaces)

    Returns:
        list of tuples: The matching sublist of bounding boxes, or [] if not found
    """
    words = query.split()
    n = len(words)

    for i in range(len(bboxes) - n + 1):
        # Extract the text from a consecutive slice
        slice_texts = [b.text for b in bboxes[i : i + n]]
        start, stop = i, i + n

        if slice_texts == words:
            if (start, stop) not in taken_indices:
                start_in_rect = is_in_rect(
                    rect=rect,
                    bbox=bboxes[start],
                    threshold=BBOX_TO_GEO_MATCHING_THRESHOLD,
                )
                stop_in_rect = is_in_rect(
                    rect=rect,
                    bbox=bboxes[stop - 1],
                    threshold=BBOX_TO_GEO_MATCHING_THRESHOLD,
                )
                # # start_in_rect = True
                # # stop_in_rect = True
                # if query == "K. Thompson":
                #     print(
                #         f"{bboxes[start]=} {bboxes[stop]=} {rect=} {start_in_rect=} {stop_in_rect=}"
                #     )
                #     input()
                if start_in_rect and stop_in_rect:
                    return (start, stop)

    return (None, None)


def pipeline_extract_handwritten_fields(params: PipelineParameters):
    log_pipeline_level()

    max_word_len = PIPELINE_06_EXTRACT_HANDWRITING__MAX_WORD_LEN

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    valid_document_ids = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1 and doclog.can_map_chars_to_words:
            bbox_path = dsfiles.get_pdf_bbox_path(
                level="char", doc_id=doclog.document_id
            )
            if bbox_path.exists():
                valid_document_ids.append(doclog.document_id)

    print(
        f"{len(valid_document_ids)} out of {total_pdfs_count} PDFs valid for handwritten&signature extraction."
    )

    with get_progress_bar() as progress:
        hw_task = progress.add_task(
            f"[red]Extracting Handwriting from {len(valid_document_ids)} PDFs...",
            total=len(valid_document_ids),
        )

        for document_id in valid_document_ids:
            data = extract_handwritten_fields(
                dsdef=dsdef, doc_id=document_id, max_word_len=max_word_len
            )

            errors = [
                f'{d["id"]}: {d["error"]}, text: "{d["text"]}"'
                for d in data
                if d["error"] is not None
            ]

            if len(data) > 0:
                result_path = (
                    dsfiles.handwritten_bboxes_directory / f"{document_id}.json"
                )
                json_str = json.dumps(data, indent=4)
                result_path.write_text(json_str, encoding="utf-8")

            dsdef.write_to_document_log(
                document_id=document_id,
                vals={
                    DocLogKey.handwriting_num_elements: len(data),
                    DocLogKey.handwriting_element_extraction_errors: errors,
                },
            )

            progress.update(hw_task, advance=1)
