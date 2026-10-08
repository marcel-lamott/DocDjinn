from collections import Counter
import json
from docdjinn.generation.models import (
    DocLogKey,
    MessageProcessingLogKey,
    PipelineParameters,
    PromptMsgResultLogKey,
)
from docdjinn.generation.pipeline_01.cost import print_cost_report
from docdjinn.generation.utils.log import log_pipeline_level


def pipeline_analyze(params: PipelineParameters):
    log_pipeline_level()

    dsdef = params.dsdef

    # Count results of each step
    # STEP 1 - Prompt Message Results
    dsfiles = dsdef.get_file_structure()
    batches_count = sum(
        [1 for f in dsfiles.prompt_batches_directory.iterdir() if f.is_file()]
    )
    messages_count = sum(
        [1 for f in dsfiles.message_results_directory.iterdir() if f.is_file()]
    )
    errored_messages_count = 0
    succeeded_messages_count = 0
    total_usage_output_tokens = 0
    total_usage_input_tokens = 0
    for f in dsfiles.message_results_directory.iterdir():
        msg_result = json.loads(f.read_text(encoding="utf-8"))

        total_usage_input_tokens += msg_result[PromptMsgResultLogKey.usage_input_tokens]
        total_usage_output_tokens += msg_result[
            PromptMsgResultLogKey.usage_output_tokens
        ]

        if msg_result[PromptMsgResultLogKey.result_type] == "errored":
            errored_messages_count += 1
        elif msg_result[PromptMsgResultLogKey.result_type] == "succeeded":
            succeeded_messages_count += 1

    # STEP 2 - Message Processing Logs
    num_documents_expected = 0
    num_documents_found = 0
    for f in dsfiles.message_processing_logs_directory.iterdir():
        msg_processing_log = json.loads(f.read_text(encoding="utf-8"))
        if msg_processing_log[MessageProcessingLogKey.result_type] == "succeeded":
            num_documents_expected += msg_processing_log[
                MessageProcessingLogKey.num_documents_expected
            ]
            num_documents_found += msg_processing_log[
                MessageProcessingLogKey.num_documents_found
            ]

    prompting_log = {
        "batches_count": batches_count,
        "messages_count": messages_count,
        "total_usage_input_tokens": total_usage_input_tokens,
        "total_usage_output_tokens": total_usage_output_tokens,
        "succeeded_messages_count": succeeded_messages_count,
        "errored_messages_count": errored_messages_count,
        "num_documents_expected": num_documents_expected,
        "num_documents_found": num_documents_found,
    }

    # STEP 3 - Document Logs
    def list_to_entry(items: set):
        return {"total": len(items), "items": sorted(items)}

    class DocumentError:
        is_multipage = "is_multipage"
        invalid_raw_gt = "invalid_raw_gt"
        cannot_map_chars_to_words = "cannot_map_chars_to_words"
        visual_elements_extraction_error = "visual_elements_extraction_error"
        handwriting_extraction_error = "handwriting_extraction_error"
        missing_handwriting_images = "missing_handwriting_images"
        missing_ocr = "missing_ocr"
        gt_verification_failed = "gt_verification_failed"
        no_text = "no_text"

    has_no_valid_gt = set()
    has_multiple_pages_pass1 = set()
    cannot_map_chars_to_words = set()
    has_visual_element_extraction_errors = set()
    has_handwriting_extraction_errors = set()
    has_missing_handwriting = set()
    has_missing_ocr = set()
    has_failed_gt_verification = set()
    has_no_text = set()

    has_handwriting = set()
    has_no_handwriting = set()
    has_ve = set()
    has_no_ve = set()
    has_handwriting_and_ve = set()

    doc_level_stats_counter = (
        Counter()
    )  # handwriting_num_elements visual_elements_num_elements
    valid_samples = set()
    document_errors = dict()

    # Fetch perfect documents
    total_documents = 0
    min_annotation_count = 99999
    max_annotation_count = 0
    for doclog in dsdef.get_document_logs():
        did = doclog.document_id
        total_documents += 1

        gt_valid = (doclog.raw_json_gt_found and doclog.raw_json_gt_valid_json) or (
            doclog.raw_annotation_gt_found
        )
        if not gt_valid:
            has_no_valid_gt.add(did)
            document_errors[did] = DocumentError.invalid_raw_gt
            continue

        if not doclog.pdf_num_pages == 1:
            has_multiple_pages_pass1.add(did)
            document_errors[did] = DocumentError.is_multipage
            continue

        if not doclog.can_map_chars_to_words:
            cannot_map_chars_to_words.add(did)
            document_errors[did] = DocumentError.cannot_map_chars_to_words
            continue

        if len(doclog.visual_elements_extraction_errors) != 0:
            has_visual_element_extraction_errors.add(did)
            document_errors[did] = DocumentError.visual_elements_extraction_error
            continue

        if len(doclog.handwriting_element_extraction_errors) != 0:
            has_handwriting_extraction_errors.add(did)
            document_errors[did] = DocumentError.handwriting_extraction_error
            continue

        if len(doclog.handwriting_missing_images) != 0:
            has_missing_handwriting.add(did)
            document_errors[did] = DocumentError.missing_handwriting_images
            continue

        if not doclog.ocr_found:
            has_missing_ocr.add(did)
            document_errors[did] = DocumentError.missing_ocr
            continue

        if not doclog.gt_verification_passed:
            has_failed_gt_verification.add(did)
            document_errors[did] = DocumentError.gt_verification_failed
            continue

        if doclog.num_word_bboxes == 0:
            has_no_text.add(did)
            document_errors[did] = DocumentError.no_text
            continue

        if doclog.handwriting_num_elements > 0:
            doc_level_stats_counter[DocLogKey.handwriting_num_elements] += (
                doclog.handwriting_num_elements
            )
            has_handwriting.add(did)
        else:
            has_no_handwriting.add(did)

        if doclog.visual_elements_num_elements > 0:
            doc_level_stats_counter[DocLogKey.visual_elements_num_elements] += (
                doclog.visual_elements_num_elements
            )
            has_ve.add(did)

            if doclog.handwriting_num_elements > 0:
                has_handwriting_and_ve.add(did)
        else:
            has_no_ve.add(did)

        if doclog.num_word_bboxes != -1:
            doc_level_stats_counter["num_words"] += doclog.num_word_bboxes

        if doclog.num_char_bboxes != -1:
            doc_level_stats_counter["num_chars"] += doclog.num_char_bboxes

        annotations_count = doclog.annotations_count
        doc_level_stats_counter["annotations_count"] += annotations_count
        min_annotation_count = min(min_annotation_count, annotations_count)
        max_annotation_count = max(max_annotation_count, annotations_count)

        valid_samples.add(did)

    # Divide all counts by divisor (keep as float)
    normalized = {k: v / len(valid_samples) for k, v in doc_level_stats_counter.items()}

    total_cost_summary = print_cost_report(
        batch_data_directory=dsfiles.prompt_batches_directory,
        dataset_log_path=dsfiles.ds_log_path,
    )

    dataset_log = {
        "prompting": prompting_log,
        "total_cost_summary": total_cost_summary,
        "valid_samples_stats": {
            "total": doc_level_stats_counter,
            "avg": normalized,
            "min_annotation_count": min_annotation_count,
            "max_annotation_count": max_annotation_count,
        },
        "total_samples": total_documents,
        "valid_samples": list_to_entry(valid_samples),
        "valid_samples_by_category": {
            "has_handwriting": list_to_entry(has_handwriting),
            "has_visual_elements": list_to_entry(has_ve),
            "has_handwriting_and_visual_elements": list_to_entry(
                has_handwriting_and_ve
            ),
            "no_handwriting": list_to_entry(has_no_handwriting),
            "no_visual_elements": list_to_entry(has_no_ve),
        },
        "errors": {
            "has_no_valid_gt": list_to_entry(has_no_valid_gt),
            "has_multiple_pages_pass1": list_to_entry(has_multiple_pages_pass1),
            "cannot_map_chars_to_words": list_to_entry(cannot_map_chars_to_words),
            "has_visual_element_extraction_errors": list_to_entry(
                has_visual_element_extraction_errors
            ),
            "has_handwriting_extraction_errors": list_to_entry(
                has_handwriting_extraction_errors
            ),
            "has_missing_handwriting": list_to_entry(has_missing_handwriting),
            "has_missing_ocr": list_to_entry(has_missing_ocr),
            "has_failed_gt_verification": list_to_entry(has_failed_gt_verification),
            "has_no_text": list_to_entry(has_no_text),
        },
        "docid_to_error": document_errors,
    }
    print(f"Valid samples: {len(valid_samples)}, errors: {len(document_errors)}")

    dsfiles.ds_log_path.write_text(json.dumps(dataset_log, indent=2), encoding="utf-8")
