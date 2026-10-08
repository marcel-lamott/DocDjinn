import json
from docdjinn.generation.models import (
    DocLogKey,
    OCRBox,
    PipelineParameters,
)
from docdjinn.generation.models._syndatadef import SynDatasetDefinition
from docdjinn.generation.pipeline_04.extract_bbox import (
    extract_bboxes_from_pdf,
    validate_char_bbox_word_mapping,
)
from docdjinn.generation.utils.bboxes import (
    draw_bboxes_on_pdf,
    read_syn_dataset_bboxes,
    save_bboxes,
)
from docdjinn.generation.utils.debug import draw_geos_and_bboxes_on_pdf
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def draw_bbox_debug(dsdef: SynDatasetDefinition, docid: str):
    dsfiles = dsdef.get_file_structure()

    bbox_norm_path = dsfiles.get_pdf_bbox_path(level="word", doc_id=docid)
    bbox_unnorm = read_syn_dataset_bboxes(bbox_norm_path)

    pdf_path = dsfiles.pdf_initial_directory / f"{docid}.pdf"
    outpath = dsfiles.debug_pdf_bboxes_directory / f"{docid}.pdf"

    geo_path = dsfiles.geometries_directory / f"{docid}.json"
    geos = json.loads(geo_path.read_text(encoding="utf-8"))

    outpath2 = dsfiles.debug_pdf_bboxes_and_geos_directory / f"{docid}.pdf"

    try:
        draw_bboxes_on_pdf(pdf_path=pdf_path, outpath=outpath, bboxes=bbox_unnorm)
        draw_geos_and_bboxes_on_pdf(
            pdf_in=pdf_path,
            pdf_out=outpath2,
            bboxes_=bbox_unnorm,
            geos=geos,
            verbose=False,
        )
    except Exception as err:
        print(f"[ERROR]: Skipping debug PDF: {str(err)}")


def pipeline_extract_bboxes(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    valid_document_ids = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1:
            bbox_path = dsfiles.get_pdf_bbox_path(
                level="word", doc_id=doclog.document_id
            )
            if not bbox_path.exists():
                valid_document_ids.append(doclog.document_id)

    print(
        f"{len(valid_document_ids)} out of {total_pdfs_count} PDFs valid for BBox extraction."
    )

    with get_progress_bar() as progress:
        bbox_task = progress.add_task(
            f"[red]Extracting BBoxes from {len(valid_document_ids)} PDFs...",
            total=len(valid_document_ids),
        )

        for document_id in valid_document_ids:
            pdf_path = dsfiles.pdf_initial_directory / f"{document_id}.pdf"

            word_bboxes = extract_bboxes_from_pdf(pdf_path=pdf_path, level="word")
            # Save word level bounding boxes
            save_bboxes(
                bboxes=word_bboxes,
                bbox_path=dsfiles.get_pdf_bbox_path(level="word", doc_id=document_id),
            )

            if params.debug:
                draw_bbox_debug(dsdef=dsdef, docid=document_id)

            # Save character level bounding boxes for splitting handwritting text
            # before inputting to difussion model (they support only short text)
            char_bboxes = extract_bboxes_from_pdf(pdf_path=pdf_path, level="char")
            can_map_chars_to_words = validate_char_bbox_word_mapping(
                char_bboxes=char_bboxes, word_bboxes=word_bboxes
            )

            if can_map_chars_to_words:
                save_bboxes(
                    bboxes=char_bboxes,
                    bbox_path=dsfiles.get_pdf_bbox_path(
                        level="char", doc_id=document_id
                    ),
                )

            dsdef.write_to_document_log(
                document_id=document_id,
                vals={
                    DocLogKey.num_word_bboxes: len(word_bboxes),
                    DocLogKey.num_char_bboxes: len(char_bboxes),
                    DocLogKey.can_map_chars_to_words: can_map_chars_to_words,
                },
            )

            progress.update(bbox_task, advance=1)
