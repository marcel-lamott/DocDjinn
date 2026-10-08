from dataclasses import asdict
import json
from docdjinn.generation.models import (
    DocLogKey,
    PipelineParameters,
    SynDatasetDefinition,
    SynDocumentLog,
)
from docdjinn.generation.models._bbox import LayoutBox
from docdjinn.generation.models._consts import DatasetTask
from docdjinn.generation.models._file import SyntheticDatasetFileStructure
from docdjinn.generation.utils.debug import draw_geos_on_pdf
from docdjinn.generation.utils.geos import (
    read_custom_elements_from_geos,
    read_layout_elements_from_geos,
)
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def extract_layout_elements_from_geos(
    dsdef: SynDatasetDefinition, doc_id: str, debug: bool
) -> list[dict]:
    files = dsdef.get_file_structure()
    geo_path = files.geometries_directory / f"{doc_id}.json"
    geos_gen = read_layout_elements_from_geos(geo_path=geo_path)
    geos: list[dict] = list(geos_gen)

    # Add temporary unique ID to each layout element.
    results = []
    for i, geo in enumerate(geos):
        r = geo["rect"]
        if not geo["classes"]:
            result = {
                "id": f"le{i}",
                "class": None,
                "content": None,
                "rect": r,
                "error": "no-class",
            }
        else:
            classes = geo["classes"].split(" ")
            # layout_node_class cannot be None, since the elementes are selected by class in pipeline_04
            layout_node_class = next(
                (cls for cls in classes if cls.startswith("LE-")), None
            )
            result = {
                "id": f"le{i}",
                "class": layout_node_class,
                "content": layout_node_class,
                "rect": r,
                "error": None,
            }

            if r["width"] == 0 or r["height"] == 0:
                result["error"] = "invalid-size"

        results.append(result)

    if debug:
        debug_pdf_file = files.debug_pdf_layout_directory / f"{doc_id}.pdf"
        draw_geos_on_pdf(
            geos=geos,
            pdf_in=files.pdf_initial_directory / f"{doc_id}.pdf",
            pdf_out=debug_pdf_file,
        )

    return results


def process_dla(
    document_log: SynDocumentLog,
    dsdef: SynDatasetDefinition,
    dsfiles: SyntheticDatasetFileStructure,
    debug: bool,
):
    document_id = document_log.document_id
    data = extract_layout_elements_from_geos(
        dsdef=dsdef, doc_id=document_id, debug=debug
    )
    valid_data = [d for d in data if d["error"] is None]
    errors = [f"{d['id']}: {d['error']}" for d in data if d["error"] is not None]

    dsdef.write_to_document_log(
        document_id=document_id,
        vals={
            DocLogKey.layout_elements_num_elements: len(data),
            DocLogKey.layout_elements_extraction_errors: errors,
            DocLogKey.layout_elements_generation_logs: data,
            DocLogKey.raw_annotation_gt_found: len(valid_data) > 0,
            DocLogKey.raw_annotation_gt_extraction_errors: errors,
            DocLogKey.raw_gt_or_annotation_annotations_count: len(data),
        },
    )

    # data is None if there were no bboxes extracted for layout elements
    if data is None or len(data) == 0:  # type: ignore
        return 0

    result_path = dsfiles.layout_element_definitions_directory / f"{document_id}.json"
    result_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    # Write GT for DLA: raw_annotations are layout bboxes and gt is normalized layout bboxes (created later in pipeline where bboxes are normalized)
    raw_annotations_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
    layout_bboxes = []
    for d in data:
        x0 = d["rect"]["x"]
        y0 = d["rect"]["y"]
        x2 = d["rect"]["x"] + d["rect"]["width"]
        y2 = d["rect"]["y"] + d["rect"]["height"]
        layout_bboxes.append(LayoutBox(x0=x0, y0=y0, x2=x2, y2=y2, label=d["content"]))

    boxes_dicts = [asdict(b) for b in layout_bboxes]
    raw_annotations_path.write_text(json.dumps(boxes_dicts, indent=4), encoding="utf-8")

    return len(data)


def extract_kie_elements_from_geos(
    dsdef: SynDatasetDefinition, doc_id: str
) -> list[dict]:
    files = dsdef.get_file_structure()
    geo_path = files.geometries_directory / f"{doc_id}.json"
    geos_gen = read_custom_elements_from_geos(geo_path=geo_path)
    geos: list[dict] = list(geos_gen)

    # Add temporary unique ID to each layout element.
    results = []
    for i, geo in enumerate(geos):
        classes = geo["classes"].split(" ")
        # kie_label cannot be None, since the elementes are selected by class in pipeline_04
        all_kie_labels = [c for c in classes if c in dsdef.valid_labels]

        kie_label = all_kie_labels[0]
        # kie_secondary_label can be none, depending on the task
        all_secondary_labels = [
            c for c in classes if c in (dsdef.valid_secondary_labels or [])
        ]
        kie_secondary_label = (
            all_secondary_labels[0] if len(all_secondary_labels) > 0 else None
        )

        # print(f"{classes=} {kie_label=} {kie_secondary_label=}")
        # input()

        result = {
            "id": f"{i}_{kie_label}_{kie_secondary_label}",
            "group": kie_secondary_label,
            "key": kie_label,
            "value": geo["text"],
            "rect": geo["rect"],
            "error": None,
        }

        # Don't allow empty key
        if not geo["text"] or not geo["text"].strip():
            result["error"] = "missing-value"

        if len(all_kie_labels) > 1:
            result["error"] = "multiple-labels"

        results.append(result)

    return results


def process_kie(
    document_log: SynDocumentLog,
    dsdef: SynDatasetDefinition,
    dsfiles: SyntheticDatasetFileStructure,
):
    document_id = document_log.document_id
    data = extract_kie_elements_from_geos(dsdef=dsdef, doc_id=document_id)
    valid_data = [d for d in data if d["error"] is None]
    errors = [f"{d['id']}: {d['error']}" for d in data if d["error"] is not None]

    dsdef.write_to_document_log(
        document_id=document_id,
        vals={
            DocLogKey.layout_elements_num_elements: 0,
            DocLogKey.layout_elements_extraction_errors: [],
            DocLogKey.layout_elements_generation_logs: [],
            DocLogKey.raw_annotation_gt_found: len(valid_data) > 0,
            DocLogKey.raw_annotation_gt_extraction_errors: errors,
            DocLogKey.raw_gt_or_annotation_annotations_count: len(valid_data),
        },
    )

    # data is None if there were no bboxes extracted for layout elements
    if data is None or len(data) == 0:  # type: ignore
        return 0

    # Write GT for KIE: raw_annotations are extracted annotation and gt is with value mapped to word bboxes (created later)
    raw_annotations_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
    raw_annotations_path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    return len(data)


def pipeline_extract_layout_element_definitions_and_annotation_gt(
    params: PipelineParameters,
):
    log_pipeline_level()

    if params.dsdef.prompt_task != "annotation":
        return

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()

    # Get valid PDF paths (single page, not processed yet)
    document_logs = {}
    valid_document_ids = []
    total_pdfs_count = 0
    for doclog in dsdef.get_document_logs():
        total_pdfs_count += 1
        if doclog.pdf_num_pages == 1:
            valid_document_ids.append(doclog.document_id)
            document_logs[doclog.document_id] = doclog

    total_document_count = 0
    total_layout_elements_count = 0
    dataset_task = DatasetTask(dsdef.task)

    with get_progress_bar() as progress:
        annotation_task = progress.add_task(
            description=f"[red]Extracting layout elements from {len(valid_document_ids)} PDFs...",
            total=len(valid_document_ids),
        )

        for document_id, document_log in document_logs.items():
            if dataset_task == DatasetTask.DLA:
                found_annotations = process_dla(
                    document_log=document_log,
                    dsdef=dsdef,
                    dsfiles=dsfiles,
                    debug=params.debug,
                )
                total_document_count += 1 if found_annotations > 0 else 0
                total_layout_elements_count += found_annotations

            # This whole pipeline step is only executed if prompt_task == 'annotation', thus KIE raw_annotations are not overridden if its modelled as prompt_task=='json'
            elif dataset_task == DatasetTask.KIE:
                found_annotations = process_kie(
                    document_log=document_log, dsdef=dsdef, dsfiles=dsfiles
                )
                total_document_count += 1 if found_annotations > 0 else 0
                total_layout_elements_count += found_annotations

            progress.update(annotation_task, advance=1)

        print(f"Extracted {total_layout_elements_count=} from {total_document_count=}")
