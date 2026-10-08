import pathlib
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from docdjinn.generation.constants import IMAGE_RENDER_EXT
from docdjinn.generation.models import (
    PipelineParameters,
)
from docdjinn.generation.pipeline_05.pdftoimage import convert_from_path_singlepage
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar

# In a PDF, the default coordinate system uses points as its unit of measurement, and a point is defined as 1/72 of an inch.
# This means the coordinate system is effectively 72 DPI (dots per inch).


def pipeline_render_image(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    valid_document_ids = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1:
            final_pdf_path = dsfiles.final_pdf_directory / f"{doclog.document_id}.pdf"
            img_path = dsfiles.img_directory / f"{doclog.document_id}.png"
            if final_pdf_path.exists() and not img_path.exists():
                valid_document_ids.append(doclog.document_id)

    print(
        f"{len(valid_document_ids)} out of {total_pdfs_count} PDFs valid for image conversion."
    )

    with get_progress_bar() as progress:
        img_task = progress.add_task(
            f"[red]Converting {len(valid_document_ids)} PDFs to images...",
            total=len(valid_document_ids),
        )

        for document_id in valid_document_ids:
            # Convert PDF to list of PIL images
            """Changing pdf locattion to final_pdf directory"""
            pdf_path = dsfiles.final_pdf_directory / f"{document_id}.pdf"
            img = convert_from_path_singlepage(pdf_path)

            img_path = dsfiles.img_directory / f"{document_id}.{IMAGE_RENDER_EXT}"
            img.save(img_path, IMAGE_RENDER_EXT.upper())

            # bboxes_path = dsfiles.bboxes_directory / f'{sample_id}.txt'
            # _draw_bboxes(img_path, bboxes_path)

            progress.update(img_task, advance=1)
