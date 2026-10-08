import json
import re
from dataclasses import asdict
from itertools import combinations

import Levenshtein
import fitz

from docdjinn.generation.constants import (
    BBOX_TO_GEO_MATCHING_THRESHOLD,
    PIPELINE_06_GT_VERIFICATION__GT_SIMILARITY_CUTOFF,
    PDF_DPI,
)
from docdjinn.generation.models import (
    DocLogKey,
    OCRBox,
    PipelineParameters,
    SynDatasetDefinition,
    DatasetTask,
    SyntheticDatasetFileStructure,
)
from docdjinn.generation.models._bbox import LayoutBox
from docdjinn.generation.models._log import SynDocumentLog
from docdjinn.generation.utils.bboxes import is_in_rect, read_syn_dataset_bboxes
from docdjinn.generation.utils.documentsize import get_document_size_for_bbox_unnormalization, get_image_size_px, get_pdf_size_pt
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar

__KEY_SEPERATOR = "<<%?"  # some value that will surely never be part of a key in KIE


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _find_best_fuzzy_match_span(
    original_text: str,
    pattern: str,
    cutoff: float,
    text_positions: list[tuple[int, int]],
):
    """
    Returns (best_candidate_text, best_score, found, [bbox_indices])
    """
    clean_text = normalize(original_text)
    clean_text_lower = clean_text.lower()
    clean_pattern = normalize(pattern).lower()
    pat_len = len(clean_pattern)

    best_candidate = ""
    best_score = -1
    best_span = (0, 0)

    for i in range(0, len(clean_text) - pat_len + 1):
        candidate = clean_text_lower[i : i + pat_len]
        dist = Levenshtein.distance(candidate, clean_pattern)
        clen = max(len(clean_pattern), len(candidate))
        if clen == 0:
            continue
        score = 1 - dist / clen
        if score > best_score:
            best_score = score
            best_candidate = clean_text[i : i + pat_len]
            best_span = (i, i + pat_len)

    found = best_score >= cutoff

    # Map char span → bbox indices
    bbox_indices = []
    if found:
        span_start, span_end = best_span
        for idx, (start, end) in enumerate(text_positions):
            if end < span_start:
                continue
            if start > span_end:
                break
            bbox_indices.append(idx)

    return best_candidate, best_score, found, bbox_indices


def _find_best_fuzzy_match_span_restriced(
    original_text: str,
    pattern: str,
    cutoff: float,
    allowed_bbox_indices: list[int] | None,
    text_positions: list[tuple[int, int]],
):
    """
    Returns (best_candidate_text, best_score, found, [bbox_indices])
    """
    clean_text = normalize(original_text)
    clean_text_lower = clean_text.lower()
    clean_pattern = normalize(pattern).lower()
    pat_len = len(clean_pattern)

    best_candidate = ""
    best_score = -1
    best_span = (0, 0)

    # Determine which character ranges are allowed
    if allowed_bbox_indices is not None:
        allowed_char_ranges = [
            text_positions[i]
            for i in allowed_bbox_indices
            # if 0 <= i < len(text_positions)
        ]
        # Merge them into one list of allowed character indices
        allowed_chars = set()
        for start, end in allowed_char_ranges:
            allowed_chars.update(range(start, end + 1))
    else:
        allowed_chars = set(range(len(clean_text)))

    # Scan only candidate windows where *all chars* fall within allowed ranges
    for i in range(0, len(clean_text) - pat_len + 1):
        window_range = set(range(i, i + pat_len))
        # if allowed_bbox_indices is not None:
        #     input(
        #         f"{i=} {window_range=} {window_range.issubset(allowed_chars)=} {allowed_chars=}"
        #     )
        if not window_range.issubset(allowed_chars):
            continue  # skip if this substring crosses disallowed areas

        candidate = clean_text_lower[i : i + pat_len]
        dist = Levenshtein.distance(candidate, clean_pattern)
        clen = max(len(clean_pattern), len(candidate))
        if clen == 0:
            continue
        score = 1 - dist / clen
        if score > best_score:
            best_score = score
            best_candidate = clean_text[i : i + pat_len]
            best_span = (i, i + pat_len)

    found = best_score >= cutoff

    # Map char span → bbox indices
    bbox_indices = []
    if found:
        span_start, span_end = best_span
        for idx, (start, end) in enumerate(text_positions):
            if end < span_start:
                continue
            if start > span_end:
                break
            # Only include bbox if it's allowed (if restricted)
            if allowed_bbox_indices is None or idx in allowed_bbox_indices:
                bbox_indices.append(idx)

    return best_candidate, best_score, found, bbox_indices


def _verify_dla_valid_labels(
    layout_bboxes: list[LayoutBox], valid_labels: list[str]
) -> bool:
    """
    Checks that all labels on the layout elements are valid.
    Returns True if all labels are valid. False otherwise.
    """
    for b in layout_bboxes:
        if b.label not in valid_labels:
            return False

    return True


def _verify_dla_has_containment_or_overlap(
    layout_bboxes: list[LayoutBox], overlap_threshold: float
) -> bool:
    """
    Checks if there are layout elements contained within other layout elements and if there are strong overlaps.
    Returns True if there are elements contained within another or if there are strong overlaps. False otherwise.
    """
    for box1, box2 in combinations(layout_bboxes, 2):
        if LayoutBox.box_contains(box1, box2) or LayoutBox.box_contains(box2, box1):
            return True

        overlap_ratio = LayoutBox.calculate_overlap_ratio(box1, box2)
        if overlap_ratio > overlap_threshold:
            return True

    return False


def _dla_load_visual_elements(
    document_id: str, dsfiles: SyntheticDatasetFileStructure
) -> list[LayoutBox]:
    """
    Load all available visual elements for a document.
    Returns the visual elements as a list of layout boxes.
    """
    visual_elements: list[LayoutBox] = []
    ve_data_path = dsfiles.visual_element_definitions_directory / f"{document_id}.json"
    if not ve_data_path.exists():
        return []

    data = json.loads(ve_data_path.read_text(encoding="utf-8"))
    for d in data:
        if d["error"] is not None:
            continue

        rect = d["rect"]
        label = d["type"]
        visual_elements.append(
            LayoutBox(
                x0=rect["x"],
                y0=rect["y"],
                x2=rect["x"] + rect["width"],
                y2=rect["y"] + rect["height"],
                label=label.lower(),
            )
        )

    return visual_elements


def _dla_merge_visual_elements_into_dla_annotations(
    document_id: str,
    dsdef: SynDatasetDefinition,
    layout_bboxes: list[LayoutBox],
    visual_elements: list[LayoutBox],
    overlap_threshold: float,
) -> list[LayoutBox]:
    """
    Merge visual elements into dla annotations if they are missing in the dla annotations.
    We currently only merge figure/picture elements.
    Returns a new list of layout boxes that is extended.
    """
    ds_has_figures = "LE-FIGURE" in dsdef.valid_labels
    ds_has_pictures = "LE-PICTURE" in dsdef.valid_labels
    visual_elements_figures = [
        element
        for element in visual_elements
        if element.label == "figure" and (ds_has_figures or ds_has_pictures)
    ]

    result = list(layout_bboxes)
    for figure in visual_elements_figures:
        has_strong_overlap = any(
            LayoutBox.calculate_overlap_ratio(figure, layout_box) > overlap_threshold
            for layout_box in layout_bboxes
        )
        if has_strong_overlap:
            # Already contained
            continue

        _label = "LE-FIGURE"
        if ds_has_pictures and not ds_has_figures:
            _label = "LE-PICTURE"
        _element = LayoutBox(
            x0=figure.x0, y0=figure.y0, x2=figure.x2, y2=figure.y2, label=_label
        )
        result.append(_element)
        # print(f'[NOTE] {document_id}: Inserting {_label} visual element into layout element GT because it was missing!')

    return result


def _dla_get_pdf_size_pt(
    document_id: str, dsfiles: SyntheticDatasetFileStructure
) -> tuple[float, float]:
    pdf_path = dsfiles.final_pdf_directory / f"{document_id}.pdf"
    doc = fitz.open(pdf_path)
    page = doc[0]
    width_pt, height_pt = page.rect.width, page.rect.height
    doc.close()
    return width_pt, height_pt


def prepare_and_verify_gt_dla(
    document_id: str,
    dsfiles: SyntheticDatasetFileStructure,
    dsdef: SynDatasetDefinition,
    params: PipelineParameters,
) -> dict:
    # written in pipeline_16_normalize_bboxes
    gt_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
    data: list[dict] = json.loads(gt_path.read_text(encoding="utf-8"))
    layout_bboxes: list[LayoutBox] = [LayoutBox(**d) for d in data]

    # Check if we only have valid Layout element labels.
    all_labels_known = _verify_dla_valid_labels(layout_bboxes, dsdef.valid_labels)

    # Check that there are no layout elements contained within others or are strongly overlapping.
    # has_containment_or_overlap = False
    # if all_labels_known:
    #     has_containment_or_overlap = _verify_dla_has_containment_or_overlap(
    #         layout_bboxes, overlap_threshold=0.5
    #     )
    #     # if has_containment_or_overlap:
    #     #     print(f'[ERROR]: Skipping {document_id} due to containment or overlap in the layout elements.')

    gt_verification_passed = all_labels_known

    # Merge layout elements generated by the LLM with visual elements generated by the LLM.
    if gt_verification_passed:
        visual_elements: list[LayoutBox] = _dla_load_visual_elements(
            document_id, dsfiles
        )
        layout_bboxes = _dla_merge_visual_elements_into_dla_annotations(
            document_id, dsdef, layout_bboxes, visual_elements, overlap_threshold=0.8
        )

        # Save post-processed GTs as new RAW-GT
        gt_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
        gt_path.write_text(
            json.dumps([asdict(box) for box in layout_bboxes], indent=2),
            encoding="utf-8",
        )

        # Save post-processed GTs as new Normalized-GT
        pdf_width_pt, pdf_height_pt = _dla_get_pdf_size_pt(document_id, dsfiles)
        layout_bboxes_normalized = [
            LayoutBox.normalize_to_pdf(
                box, width_pt=pdf_width_pt, height_pt=pdf_height_pt, dpi=PDF_DPI
            )
            for box in layout_bboxes
        ]

        # Filter out problematic Bboxes
        def is_valid_bbox(b: LayoutBox):
            def val(v: float):
                return v >= 0 and v <= 1

            w = b.x2 - b.x0
            h = b.y2 - b.y0
            return (
                val(w)
                and val(h)
                and val(b.x0)
                and val(b.x2)
                and val(b.y0)
                and val(b.y2)
            )

        layout_bboxes_normalized_filtered = []
        problematic_bboxes = []
        for box in layout_bboxes_normalized:
            if is_valid_bbox(box):
                layout_bboxes_normalized_filtered.append(box)
            else:
                problematic_bboxes.append(box)

        if len(problematic_bboxes) == 0:
            # ...
            # print(
            #     f"Removed {len(problematic_bboxes)} buggy bboxes, remaining: {len(layout_bboxes_normalized_filtered)}!"
            # )
            # print(dsfiles.debug_pdf_layout_directory / f"{document_id}.pdf")
            # input(document_id)

            gt_path = dsfiles.gt_directory / f"{document_id}.json"
            gt_path.write_text(
                json.dumps(
                    [asdict(box) for box in layout_bboxes_normalized_filtered], indent=2
                ),
                encoding="utf-8",
            )

            # Update the debug PDF
            update_debug_pdfs = params.debug
            if update_debug_pdfs:
                from docdjinn.generation.utils.debug import draw_geos_on_pdf

                debug_pdf_file = (
                    dsfiles.debug_pdf_layout_directory / f"{document_id}.pdf"
                )
                print(f"Updating: {debug_pdf_file}")
                draw_geos_on_pdf(
                    geos=[
                        {
                            "rect": {
                                "x": box.x0,
                                "y": box.y0,
                                "width": box.x2 - box.x0,
                                "height": box.y2 - box.y0,
                            }
                        }
                        for box in layout_bboxes
                    ],
                    pdf_in=dsfiles.pdf_initial_directory / f"{document_id}.pdf",
                    pdf_out=debug_pdf_file,
                )
        else:
            gt_verification_passed = False

    gt_validation_log = {
        DocLogKey.gt_verification_confirmed_keys: [],
        DocLogKey.gt_verification_similarities: [],
        DocLogKey.gt_verification_passed: gt_verification_passed,
        DocLogKey.gt_verification_skipped: False,
    }
    return gt_validation_log


def prepare_and_verify_gt_classification(
    document_id: str,
    raw_annotations: dict,
    dsfiles: SyntheticDatasetFileStructure,
    dsdef: SynDatasetDefinition,
) -> dict:
    _, cls = next(iter(raw_annotations.items()), (None, None))
    gt_data = {"label": cls}
    gt_path = dsfiles.gt_directory / f"{document_id}.json"
    gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")

    valid_label = cls in dsdef.valid_labels
    if not valid_label:
        print(f'Not a valid label "{cls}", not in {dsdef.valid_labels}')

    gt_validation_log = {
        DocLogKey.gt_verification_confirmed_keys: [],
        DocLogKey.gt_verification_similarities: [],
        DocLogKey.gt_verification_passed: valid_label,
        DocLogKey.gt_verification_skipped: False,
    }
    return gt_validation_log


def _postprocess_qa_gt_search_answer_indices(
    gts: dict, document_text, cutoff: float, bboxes: list[OCRBox], text_positions
):
    verbatim_gts = dict()
    similarities = dict()
    keys_with_values_found = list()
    bbox_indices_per_key = dict()

    # Build document text and map each word's char span
    document_text = ""
    text_positions = []
    pos = 0
    for b in bboxes:
        start = pos
        document_text += b.text + " "
        end = len(document_text) - 1
        text_positions.append((start, end))
        pos = len(document_text)

    for k, v in gts.items():
        if isinstance(v, dict):
            for qa_key, qa_value in v.items():
                best_text, similarity, found, bbox_indices = (
                    _find_best_fuzzy_match_span(
                        document_text,
                        qa_value,
                        cutoff=cutoff,
                        text_positions=text_positions,
                    )
                )

                full_key = f"{k}{__KEY_SEPERATOR}{qa_key}"
                if found:
                    keys_with_values_found.append(full_key)

                verbatim_gts[full_key] = best_text.strip()
                similarities[full_key] = similarity
                bbox_indices_per_key[full_key] = bbox_indices

        else:
            best_text, similarity, found, bbox_indices = _find_best_fuzzy_match_span(
                document_text,
                v,
                cutoff=cutoff,
                text_positions=text_positions,
            )

            if found:
                keys_with_values_found.append(k)

            verbatim_gts[k] = best_text.strip()
            similarities[k] = similarity
            bbox_indices_per_key[k] = bbox_indices

    return verbatim_gts, keys_with_values_found, similarities, bbox_indices_per_key


def prepare_and_verify_gt_qa(
    dsdef: SynDatasetDefinition,
    dsfiles: SyntheticDatasetFileStructure,
    document_id: str,
    verbatim_gts: dict,
    bbox_indices_per_key: dict,
    keys_with_values_found: list,
    similarities: dict,
):
    gt_data = []

    for i, q in enumerate(keys_with_values_found):
        answer_indices = bbox_indices_per_key[q]
        a = verbatim_gts[q]
        gt_data.append(
            {
                "question": q,
                "answer": a,
                "answer_bbox_indices": answer_indices,
            }
        )

    # Save postprocessed GTs
    gt_path = dsfiles.gt_directory / f"{document_id}.json"
    gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")

    # Return GT validation log + bbox info
    gt_validation_log = {
        DocLogKey.gt_verification_confirmed_keys: keys_with_values_found,
        DocLogKey.gt_verification_similarities: similarities,
        DocLogKey.gt_verification_passed: len(keys_with_values_found) > 0,
        DocLogKey.gt_verification_skipped: False,
    }

    return gt_validation_log

def pdf_region_to_image(r):
    scale = PDF_DPI / 72.0
    x_px = r["x"] * scale
    #y_px = (page_height_pt - y_pt - h_pt) * scale
    y_px = r["y"] * scale
    w_px = r["width"] * scale
    h_px = r["height"] * scale
    return{ "x": x_px, "y": y_px, "width": w_px, "height": h_px }


def _postprocess_kie_gt_search_key_indices(
    gts: list,
    document_text,
    cutoff: float,
    bboxes: list[OCRBox],
    text_positions,
    doclog: SynDocumentLog,
    dsfiles: SyntheticDatasetFileStructure,
    is_annotation_task: bool,
):
    verbatim_gts = dict()
    similarities = dict()
    keys_with_values_found = list()
    bbox_indices_per_key = dict()
    key_to_label = dict()

    for d in gts:
        if d["error"]:
            continue

        g = d["group"]
        k = d["key"]
        lbl = k
        if g is not None:
            k = f"{k}_{g}"

        key_to_label[k] = lbl
        v = d["value"]
        r = d["rect"]

        if is_annotation_task and doclog.ocr_required:
            r = pdf_region_to_image(r)

        found = False
        best_text = ""
        similarity = -1
        bbox_indices = None

        if v:
            bbox_indices_in_rect = (
                [
                    i
                    for i, b in enumerate(bboxes)
                    if is_in_rect(
                        rect=r, bbox=b, threshold=BBOX_TO_GEO_MATCHING_THRESHOLD
                    )
                ]
                if is_annotation_task
                else None
            )

            best_text, similarity, found, bbox_indices = (
                _find_best_fuzzy_match_span_restriced(
                    document_text,
                    v,
                    cutoff=cutoff,
                    allowed_bbox_indices=bbox_indices_in_rect,  # is None for JSON tasks und thus unrestricted
                    text_positions=text_positions,
                )
            )

            # if doclog.document_id == '74577486-4e36-425e-b733-8a745ca840f1_0':
            #     print(f'{is_annotation_task=} {k=} {v=} {best_text=} {similarity=} {found=} {bbox_indices=} {bbox_indices_in_rect=}')
            #     input()

        # if not found:
        #     print(
        #         f"RESTRICTED\n{bbox_indices_in_rect=} {v=} {best_text=} {similarity=} {found=} {bbox_indices=}"
        #     )
        #     print(" ".join([bboxes[i].text for i in bbox_indices_in_rect]))
        #     input()

        if found:
            keys_with_values_found.append(k)

        verbatim_gts[k] = best_text.strip()
        similarities[k] = similarity
        bbox_indices_per_key[k] = bbox_indices

    return (
        verbatim_gts,
        keys_with_values_found,
        similarities,
        bbox_indices_per_key,
        key_to_label,
    )


def prepare_and_verify_gt_kie(
    dsdef: SynDatasetDefinition,
    dsfiles: SyntheticDatasetFileStructure,
    document_id: str,
    verbatim_gts: dict,
    key_to_label: dict,
    word_bboxes: list[OCRBox],
    bbox_indices_per_key: dict,
    keys_with_values_found: list,
    similarities: dict,
):
    gt_data = dict()
    gt_data["entities"] = []

    # BIO Tagging: first collect all B- and I-
    non_o = dict()
    known_keys = set()
    key: str
    for key in keys_with_values_found:
        k = key
        lbl = key_to_label[k]
        g = k.replace(f"{lbl}_", "")

        # grouping is only relevant for entity linking tasks

        known_keys.add(k)
        answer_indices = bbox_indices_per_key[key]
        value = verbatim_gts[key]

        label_mapped = lbl
        if dsdef.label_mapping is not None and len(dsdef.label_mapping) > 0:
            label_mapped = dsdef.label_mapping[lbl]

        gt_data["entities"].append(
            {
                "key": label_mapped,
                "value": value,
                "group": g,  # is '' when no group given, not None
                "bbox_indices": answer_indices,
                "similarity": similarities[k],
            }
        )

        # print(f"{key=} {value=} {answer_indices=}")
        for i, bidx in enumerate(answer_indices):
            prefix = "B-" if i == 0 else "I-"
            non_o[bidx] = f"{prefix}{lbl}"

    # Then add all O Tags
    word_labels = [non_o.get(i, "O") for i in range(len(word_bboxes))]
    gt_data["word_labels"] = word_labels

    # Save postprocessed GTs
    gt_path = dsfiles.gt_directory / f"{document_id}.json"
    gt_path.write_text(json.dumps(gt_data, indent=2), encoding="utf-8")

    # Return GT validation log + bbox info
    gt_validation_log = {
        DocLogKey.gt_verification_confirmed_keys: keys_with_values_found,
        DocLogKey.gt_verification_similarities: similarities,
        DocLogKey.gt_verification_passed: len(keys_with_values_found) > 0,
        DocLogKey.gt_verification_skipped: False,
    }

    return gt_validation_log


def prepare_and_verify_gt(
    dsdef: SynDatasetDefinition,
    document_id: str,
    cutoff: float,
    params: PipelineParameters,
) -> dict:
    dsfiles = dsdef.get_file_structure()
    dataset_task = DatasetTask(dsdef.task)
    if dataset_task == DatasetTask.DLA:
        return prepare_and_verify_gt_dla(
            document_id=document_id, dsfiles=dsfiles, dsdef=dsdef, params=params
        )

    raw_annotations_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
    raw_annotations = json.loads(raw_annotations_path.read_text(encoding="utf-8"))

    if dataset_task == DatasetTask.CLASSIFICATION:
        # Classification labels do not need to be searched in the OCR.
        # Currently, we do not check if the generated labels are valid, it might contain hallucinations.
        return prepare_and_verify_gt_classification(
            document_id=document_id,
            raw_annotations=raw_annotations,
            dsfiles=dsfiles,
            dsdef=dsdef,
        )

    bbox_path = dsfiles.get_final_bbox_path(level="word", doc_id=document_id)
    bboxes: list[OCRBox] = read_syn_dataset_bboxes(box_path=bbox_path)
    # Rect ist das Problem nicht bbox
    #bboxes = [b.unnormalize(width_px=width_px, height_px=height_px) for b in bboxes_normalized]

    # Build document text and map each word's char span
    document_text = ""
    text_positions = []
    pos = 0
    for b in bboxes:
        start = pos
        document_text += b.text + " "
        end = len(document_text) - 1
        text_positions.append((start, end))
        pos = len(document_text)

    if dataset_task == DatasetTask.QA:
        verbatim_gts, keys_with_values_found, similarities, bbox_indices_per_key = (
            _postprocess_qa_gt_search_answer_indices(
                gts=raw_annotations,
                document_text=document_text,
                cutoff=cutoff,
                bboxes=bboxes,
                text_positions=text_positions,
            )
        )

        return prepare_and_verify_gt_qa(
            dsdef=dsdef,
            dsfiles=dsfiles,
            document_id=document_id,
            verbatim_gts=verbatim_gts,
            bbox_indices_per_key=bbox_indices_per_key,
            keys_with_values_found=keys_with_values_found,
            similarities=similarities,
        )

    if dataset_task == DatasetTask.KIE:
        # SROIE is modeled as JSON, but CORD and FUNSD as annotation task
        is_annotation_task = dsdef.prompt_task == "annotation"
        (
            verbatim_gts,
            keys_with_values_found,
            similarities,
            bbox_indices_per_key,
            key_to_label,
        ) = _postprocess_kie_gt_search_key_indices(
            gts=raw_annotations,
            document_text=document_text,
            cutoff=cutoff,
            bboxes=bboxes,
            text_positions=text_positions,
            doclog=SynDocumentLog(document_id, dsfiles.document_logs_directory),
            dsfiles=dsfiles,
            is_annotation_task=is_annotation_task,
        )

        return prepare_and_verify_gt_kie(
            dsdef=dsdef,
            dsfiles=dsfiles,
            document_id=document_id,
            verbatim_gts=verbatim_gts,
            key_to_label=key_to_label,
            bbox_indices_per_key=bbox_indices_per_key,
            word_bboxes=bboxes,
            keys_with_values_found=keys_with_values_found,
            similarities=similarities,
            # is_annotation_task=is_annotation_task,
        )

    raise ValueError(f"Unknown synthetic dataset task: {dataset_task}")


def pipeline_ground_truth_verification(params: PipelineParameters):
    log_pipeline_level()

    cutoff = PIPELINE_06_GT_VERIFICATION__GT_SIMILARITY_CUTOFF

    dsdef = params.dsdef

    # Get valid PDF paths (single page, not processed yet)
    valid_document_ids = []
    total_annotations_count = 0
    for doclog in dsdef.get_document_logs():
        total_annotations_count += 1
        gt_valid = (doclog.raw_json_gt_found and doclog.raw_json_gt_valid_json) or (
            doclog.raw_annotation_gt_found
            #and len(doclog.raw_annotation_gt_extraction_errors) == 0
        )
        if doclog.pdf_num_pages == 1 and doclog.ocr_found and gt_valid:
            # annotations_path = dsfiles.gt_directory / f"{doclog.document_id}.json"
            # if not annotations_path.exists():
            valid_document_ids.append(doclog.document_id)

    print(
        f"{len(valid_document_ids)} out of {total_annotations_count} documents valid for GT preparation and verification."
    )

    with get_progress_bar() as progress:
        verification_task = progress.add_task(
            f"[red]Preparing and verifying {len(valid_document_ids)} document annotations...",
            total=len(valid_document_ids),
        )

        for document_id in valid_document_ids:
            dsfiles = dsdef.get_file_structure()
            gt_log = prepare_and_verify_gt(
                dsdef=dsdef, document_id=document_id, cutoff=cutoff, params=params
            )
            dsdef.write_to_document_log(
                document_id=document_id,
                vals=gt_log,
            )
            progress.update(verification_task, advance=1)
