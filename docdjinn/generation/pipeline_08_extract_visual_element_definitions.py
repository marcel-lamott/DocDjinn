import json
import pathlib
import re

from bs4 import BeautifulSoup
import cssutils

from docdjinn.generation.constants import (
    BS_PARSER,
    VISUAL_ELEMENT_TYPE_SYNONYMS,
    VISUAL_ELEMENT_TYPES,
)
from docdjinn.generation.models import (
    DocLogKey,
    OCRBox,
    PipelineParameters,
    SynDatasetDefinition,
)
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from docdjinn.generation.utils.bboxes import draw_bboxes_on_pdf, read_syn_dataset_bboxes
from docdjinn.generation.utils.geos import read_visual_elements_from_geos
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar
from docdjinn.generation.utils.visualelement import get_visual_element_id


def extract_dimensions(style: str) -> tuple[int | None, int | None]:
    """
    Returns width,height in milimeters or None,None
    """
    # Parse width and height
    width = None
    height = None

    for prop in style.split(";"):
        if ":" not in prop:
            continue
        key, value = prop.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "width":
            width = value
        elif key == "height":
            height = value

    def normalize(val: str | None):
        if val is None:
            return None

        if val.endswith("mm"):
            return int(val.replace("mm", ""))
        elif val.endswith("cm"):
            return int(float(val.replace("cm", "")) * 10)
        else:
            print(f'Encountered an unknown size unit "{val}". (Setting size to `None`)')
            return None

    return normalize(width), normalize(height)


def parse_2d_rotation(transform_str):
    """
    Extracts the 2D rotation angle in degrees from a CSS transform string.
    Returns None if no rotation is found.
    """
    # Regex to match rotate(<angle>deg)
    match = re.search(r"rotate\(\s*([-+]?\d*\.?\d+)\s*deg\s*\)", transform_str)
    if match:
        return float(match.group(1))
    return None


def extract_rotation_from_transform(style: str) -> float | None | None:
    if not style:
        return None
    style = cssutils.parseStyle(style)
    return parse_2d_rotation(style["transform"])  # type: ignore


def extract_visual_elements_from_geos(
    dsdef: SynDatasetDefinition, doc_id: str
) -> list[dict]:
    files = dsdef.get_file_structure()
    geo_path = files.geometries_directory / f"{doc_id}.json"
    geos = read_visual_elements_from_geos(geo_path=geo_path)
    geos = list(geos)

    result = []
    for i, geo in enumerate(geos):
        data_type = geo["dataPlaceholder"]

        # Map using type synonyms
        valid_type = data_type in VISUAL_ELEMENT_TYPES
        type_mapped = data_type
        if not valid_type:
            if data_type in VISUAL_ELEMENT_TYPE_SYNONYMS:
                type_mapped = VISUAL_ELEMENT_TYPE_SYNONYMS[data_type]  # type: ignore
            else:
                type_mapped = None

        data_content = geo["dataContent"]

        style = geo["style"]
        # width, height = extract_dimensions(style)  # type: ignore
        rotation = extract_rotation_from_transform(style)  # type: ignore
        invalid_size = geo["rect"]["width"] == 0 or geo["rect"]["height"] == 0
        if type_mapped is None:
            value = {
                "id": f"ve{i}",
                "type": None,
                "type_unmapped": data_type,
                "content": data_content,
                "rect": geo["rect"],
                "rotation": rotation,
                "error": "unknown-type",
            }
        elif invalid_size:
            value = {
                "id": f"ve{i}",
                "type": type_mapped,
                "type_unmapped": data_type,
                "content": data_content,
                "rect": geo["rect"],
                "rotation": rotation,
                "error": "invalid-size",
            }
        else:
            value = {
                "id": f"ve{i}",
                "type": type_mapped,
                "type_unmapped": data_type,
                "content": data_content,
                "rect": geo["rect"],
                "rotation": rotation,
                "error": None,
            }

        # print(value)
        result.append(value)

    return result


def mm_to_px(mm):
    return mm * 72 / 25.4


def pipeline_extract_visual_element_definitions(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    valid_document_ids = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1:
            valid_document_ids.append(doclog.document_id)

    print(
        f"{len(valid_document_ids)} out of {total_pdfs_count} PDFs valid for visual element extraction."
    )

    with get_progress_bar() as progress:
        vee_task = progress.add_task(
            f"[red]Extracting visual elements from {len(valid_document_ids)} PDFs...",
            total=len(valid_document_ids),
        )

        total_visual_elements_count = 0
        for document_id in valid_document_ids:
            data = extract_visual_elements_from_geos(dsdef=dsdef, doc_id=document_id)

            errors = [
                f"{d['id']}: {d['error']}" for d in data if d["error"] is not None
            ]

            dsdef.write_to_document_log(
                document_id=document_id,
                vals={
                    DocLogKey.visual_elements_num_elements: len(data),
                    DocLogKey.visual_elements_extraction_errors: errors,
                },
            )

            # data is None if there were no bboxes extracted for visual elements
            if data is None or len(data) == 0:  # type: ignore
                progress.update(vee_task, advance=1)
                continue

            total_visual_elements_count += len(data)

            result_path = (
                dsfiles.visual_element_definitions_directory / f"{document_id}.json"
            )
            json_str = json.dumps(data, indent=4)
            result_path.write_text(json_str, encoding="utf-8")

            progress.update(vee_task, advance=1)

        print(f"{total_visual_elements_count=}")
