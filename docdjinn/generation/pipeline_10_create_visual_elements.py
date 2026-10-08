"""
TODO: latent diffusion model inference
"""

import pathlib
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.stamp import (
    create_stamp,
)
import json
from docdjinn import ENV
import random
from pathlib import Path
from PIL import Image
import io
from barcode import Code128
from barcode.writer import ImageWriter
from docdjinn.generation.models import (
    DocLogKey,
    PipelineParameters,
    SyntheticDatasetFileStructure,
    SynDatasetDefinition,
    LLMType,
)
from docdjinn.generation.utils.status import get_progress_bar

__LOGO_PREFABS__ = ENV.VISUAL_ELEMENT_PREFABS_DIR / "logo"
__FIGURE_PREFABS__ = ENV.VISUAL_ELEMENT_PREFABS_DIR / "figure"
__PHOTO_PREFABS__ = ENV.VISUAL_ELEMENT_PREFABS_DIR / "photo"
_LOGO_CACHE = None
_PHOTO_CACHE = None
_CHART_CACHE = None


def _get_prefabs_paths(image_type: str) -> list[Path]:
    """Cache logo paths to avoid repeated directory scans."""
    global _LOGO_CACHE, _PHOTO_CACHE, _CHART_CACHE

    image_type_lower = image_type.lower()

    if image_type_lower == "logo":
        if _LOGO_CACHE is None:
            _LOGO_CACHE = _scan_directory(__LOGO_PREFABS__, "logo")
        return _LOGO_CACHE
    elif image_type_lower == "photo":
        if _PHOTO_CACHE is None:
            _PHOTO_CACHE = _scan_directory(__PHOTO_PREFABS__, "photo")
        return _PHOTO_CACHE
    elif image_type_lower == "figure":
        if _CHART_CACHE is None:
            _CHART_CACHE = _scan_directory(__FIGURE_PREFABS__, "figure")
        return _CHART_CACHE
    else:
        raise ValueError(
            f"Invalid image_type: {image_type}. Must be 'logo', 'photo', or 'figure'"
        )


def _scan_directory(directory, image_type):
    """Helper to scan directory for images."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(directory.glob(ext))

    if not paths:
        raise FileNotFoundError(f"No {image_type} images found in {directory}")

    return paths


"""
{
        "id": "ve0",
        "type": "stamp",
        "type_unmapped": "stamp",
        "content": "CONFIDENTIAL",
        "rect": {
            "x": 766.7671508789062,
            "y": 100.63824462890625,
            "width": 138.8602294921875,
            "height": 138.8602294921875
        },
        "rotation": -15.0,
        "error": null
    }
"""


def _prepare_stamp(
    result_path: Path, ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
):
    content = ved["content"]
    rotation = ved["rotation"]
    width = ved["rect"]["width"]
    height = ved["rect"]["height"]
    # we dont pass rotation here, each stamp has a slight random rotation, we apply rotation in insertion
    stamp = create_stamp(text=content, width=width, height=height, rot_angle=None)
    stamp.save(result_path)


def _prepare_logo(
    result_path: Path, ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
):
    logo_paths = _get_prefabs_paths("logo")  # getting chached logo paths here

    selected_logo_image_path = random.choice(logo_paths)
    logo_image = Image.open(selected_logo_image_path).convert(
        "RGBA"
    )  # check this conversion if face any issues
    """If anyone want to do any processing on image do it here->like text insertion"""
    logo_image.save(result_path)


# Generate barcode with transparent background
writer = ImageWriter()
writer.set_options(
    {  # I think we have to play around with these numbers
        "module_width": 0.3,
        "module_height": 15.0,
        "quiet_zone": 6.5,
        "font_size": 7,
        "text_distance": 5,
        "background": "rgba(255, 255, 255, 0)",  # Transparent background
        "foreground": "black",
    }
)


def _prepare_barcode(
    result_path: Path, ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
):
    content = ved["content"]
    if content and content.strip().isdigit():
        barcode_content = content.strip()
    else:
        # Generate random number if content is invalid or empty
        barcode_content = str(
            random.randint(100000000000, 999999999999)
        )  # 12-digit number

    code128 = Code128(barcode_content, writer=writer)

    # Save to buffer first to handle transparency
    buffer = io.BytesIO()
    code128.write(buffer, options={"format": "PNG"})
    buffer.seek(0)

    barcode_image = Image.open(buffer).convert("RGBA")  # Transparent background
    barcode_image.save(result_path)


def _prepare_photo(
    result_path: Path, ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
):
    photo_paths = _get_prefabs_paths("photo")  # getting chached photo paths here

    selected_photo_image_path = random.choice(photo_paths)
    photo_image = Image.open(
        selected_photo_image_path
    )  # check this conversion if face any issues
    photo_image.save(result_path)


def _prepare_figure(
    result_path: Path, ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
):
    chart_paths = _get_prefabs_paths("figure")  # getting chached charts paths here

    selected_chart_image_path = random.choice(chart_paths)
    chart_image = Image.open(
        selected_chart_image_path
    )  # check this conversion if face any issues
    chart_image.save(result_path)


def process_visual_element_definition(
    ved: dict, docid: str, dsfiles: SyntheticDatasetFileStructure
) -> dict:
    content = ved["content"]
    ved_id = ved["id"]
    error = ved["error"]
    log = {
        "id": ved_id,
        "type": ved["type"],
        "type_unmapped": ved["type_unmapped"],
        "content": content,
        "error": error,
    }

    document_visual_elements_dir = dsfiles.visual_elements_directory / docid
    document_visual_elements_dir.mkdir(parents=True, exist_ok=True)
    result_path = document_visual_elements_dir / f"{ved_id}.png"

    # Skip already generated vis elements
    if error is None and not result_path.exists():
        match ved["type"]:
            case "stamp":
                _prepare_stamp(
                    result_path=result_path,
                    ved=ved,
                    docid=docid,
                    dsfiles=dsfiles,
                )
            case "logo":
                _prepare_logo(
                    result_path=result_path,
                    ved=ved,
                    docid=docid,
                    dsfiles=dsfiles,
                )
            case "barcode":
                _prepare_barcode(
                    result_path=result_path,
                    ved=ved,
                    docid=docid,
                    dsfiles=dsfiles,
                )
            case "photo":
                _prepare_photo(
                    result_path=result_path,
                    ved=ved,
                    docid=docid,
                    dsfiles=dsfiles,
                )
            case "figure":
                _prepare_figure(
                    result_path=result_path,
                    ved=ved,
                    docid=docid,
                    dsfiles=dsfiles,
                )
            case _:
                log["error"] = "unknown-type"

    log["image_path"] = str(result_path) if result_path is not None else None

    return log


def prepare_visual_elements(
    defs: list[dict], docid: str, dsfiles: SyntheticDatasetFileStructure
) -> list[dict]:
    logs = []

    random.seed(docid)
    for ved in defs:
        log = process_visual_element_definition(ved, docid=docid, dsfiles=dsfiles)
        logs.append(log)

    return logs


def pipeline_create_visual_elements(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid documents
    valid_documents = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1:
            has_visual_elements = doclog.visual_elements_num_elements > 0
            if has_visual_elements:
                valid_documents.append(doclog.document_id)

    print(
        f"{len(valid_documents)} of {total_pdfs_count} documents valid for visual element generation."
    )

    with get_progress_bar() as progress:
        insert_task = progress.add_task(
            "[red]Creating visual elements...", total=len(valid_documents)
        )
        for docid in valid_documents:
            visual_element_def_file = (
                dsfiles.visual_element_definitions_directory / f"{docid}.json"
            )
            visual_element_definitions = json.loads(
                visual_element_def_file.read_text(encoding="utf-8")
            )
            insertion_logs = prepare_visual_elements(
                defs=visual_element_definitions, docid=docid, dsfiles=dsfiles
            )

            errors = [
                f"{d['id']}: {d['error']}"
                for d in insertion_logs
                if d["error"] is not None
            ]
            dsdef.write_to_document_log(
                document_id=docid,
                vals={
                    DocLogKey.visual_elements_generation_logs: insertion_logs,
                    DocLogKey.visual_elements_generation_errors: errors,
                },
            )
            progress.update(insert_task, advance=1)
