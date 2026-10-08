import json
import pathlib
import re

from bs4 import BeautifulSoup

from docdjinn.generation.constants import BS_PARSER
from docdjinn.generation.models import (
    DocLogKey,
    MessageProcessingLogKey,
    PipelineParameters,
    PromptMsgResultLogKey,
    PromptParameters,
    SyntheticDatasetFileStructure,
    SynDatasetDefinition,
    DatasetTask,
)
from docdjinn.generation.utils.log import log_pipeline_level


def extract_gt(html: str):
    # Parse HTML
    soup = BeautifulSoup(html, BS_PARSER)

    # Find the script tag with id="GT"
    script_tag = soup.find("script", {"id": "GT"})

    # Extract and parse the JSON content
    if script_tag:
        raw_json = script_tag.string.strip()  # type: ignore

        # Remove the script tag from the HTML
        script_tag.decompose()

        # Return the JSON content and modified HTML
        return raw_json, str(soup), soup
    else:
        return None, html, soup


def extract_html_documents_from_text(text: str):
    # Split response HTMLs based on regex
    # Use regex to find all <html>...</html> blocks
    html_docs = re.findall(
        r"<!DOCTYPE html>.*?</html>", text, re.DOTALL | re.IGNORECASE
    )

    htmls = []
    for i, doc in enumerate(html_docs):
        # Insert invisible debugging utils to be able to inspect generated HTML.
        doc = doc.replace("</body>", '<script src="debug.js"></script></body>')

        # Prettify the minified HTML for easier inspection
        pretty_html = None
        try:
            soup = BeautifulSoup(doc, BS_PARSER)
            pretty_html = soup.prettify()
        except Exception:
            pretty_html = None

        if pretty_html is not None:
            htmls.append(pretty_html)

    return htmls


def _prepare_kie_json_gt(gts: dict) -> list[dict]:
    res = []
    for k, v in gts.items():
        r = {"id": k, "group": None, "key": k, "value": v, "rect": None, "error": None}

        # Don't allow empty key
        if not v or not v.strip():
            r["error"] = "missing-value"

        res.append(r)

    return res


def _parse_and_save_json_gt(
    dsdef: SynDatasetDefinition,
    html: str,
    gt_path: pathlib.Path,
) -> dict:
    dataset_task = DatasetTask(dsdef.task)

    if dsdef.prompt_task == "annotation":
        # DLA/KIE datasets do not generate JSON GT, this GT is extracted in pipeline_06_extract_layout_element_definitions_and_annotation_gt
        return {
            DocLogKey.raw_json_gt_found: False,
            DocLogKey.raw_json_gt_valid_json: False,
        }
    elif dsdef.prompt_task == "json":
        # ! We assume the raw gt extracted is always a dict, never a list

        # Extract GT from HTML and parse it (enclosed in script tag with ID 'GT')
        gts_json, html_without_scripttag, soup = extract_gt(html=html)
        gt_log = dict()
        gt_log[DocLogKey.raw_json_gt_found] = gts_json is not None
        gt_log[DocLogKey.raw_json_gt_valid_json] = False

        try:
            gts: dict = json.loads(gts_json)  # type: ignore
            gt_log[DocLogKey.raw_json_gt_valid_json] = True

            # For QA, we just save the json with question as key and answer as value,
            # but for KIE tasks modelled as JSON-prompt, we should stick to the same format as annoation-prompt KIE.
            if dataset_task == DatasetTask.KIE:
                gts = _prepare_kie_json_gt(gts)

            gt_log[DocLogKey.raw_gt_or_annotation_annotations_count] = len(gts)

            with open(gt_path, "w", encoding="utf-8") as f:
                json.dump(gts, f, indent=2)
        except Exception as e:
            # print(f"[ERROR]: {str(e)}")
            gt_log[DocLogKey.raw_json_gt_valid_json] = False

        return gt_log
    else:
        raise ValueError(f"Unknown prompt_task: {dsdef.prompt_task}")


def _process_message_response(
    dsdef: SynDatasetDefinition,
    message_results: dict,
    prompt_params: PromptParameters,
    dsfiles: SyntheticDatasetFileStructure,
):
    message_custom_id = message_results["custom_id"]
    message_processing_log_path = (
        dsfiles.message_processing_logs_directory / f"{message_custom_id}.json"
    )
    message_processing_log = {
        MessageProcessingLogKey.custom_id: message_custom_id,
        MessageProcessingLogKey.result_type: message_results[
            PromptMsgResultLogKey.result_type
        ],
        MessageProcessingLogKey.num_documents_expected: -1,
        MessageProcessingLogKey.num_documents_found: -1,
        MessageProcessingLogKey.document_ids: [],
    }

    # Only process succeeded files
    # Policy violations have result_type == "succeeded" but an error message
    if (
        message_results[PromptMsgResultLogKey.result_type] == "succeeded"
        and not message_results[PromptMsgResultLogKey.error]
    ):
        # Extract HTMLs from LLM response
        response = message_results[PromptMsgResultLogKey.response]
        htmls = extract_html_documents_from_text(text=response)

        if len(htmls) != prompt_params.num_solutions:
            print(
                f"Expected {prompt_params.num_solutions} HTML files but found only {len(htmls)} in {message_custom_id}!"
            )

        message_processing_log[MessageProcessingLogKey.num_documents_expected] = (
            prompt_params.num_solutions
        )
        message_processing_log[MessageProcessingLogKey.num_documents_found] = len(htmls)

        # Save HTMLs and parse GT
        for i, html in enumerate(htmls):
            document_id = f"{message_custom_id}_{i}"
            document_log: dict[str, str | int] = {DocLogKey.document_id: document_id}

            # Save raw HTML
            html_path = dsfiles.raw_html_directory / f"{document_id}.html"
            html_path.write_text(html, encoding="utf-8")
            document_log[DocLogKey.html_len] = len(html)

            # Extract GT from HTML and parse it (enclosed in script tag with ID 'GT')
            gt_path = dsfiles.raw_annotations_directory / f"{document_id}.json"
            gt_log = _parse_and_save_json_gt(dsdef=dsdef, html=html, gt_path=gt_path)
            document_log.update(gt_log)

            # Append sample log to message processing log
            message_processing_log[MessageProcessingLogKey.document_ids].append(
                document_id
            )

            dsdef.write_to_document_log(document_id=document_id, vals=document_log)

    # Write log file, indicating finished processing of message results
    with open(message_processing_log_path, "w", encoding="utf-8") as f:
        json.dump(message_processing_log, f, indent=2)


def pipeline_process_response_extract_html_and_gt(params: PipelineParameters):
    """
    Extract raw HTML and GT, write logs
    """
    log_pipeline_level()

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()
    prompt_params = dsdef.prompt_params

    total_message_results_count = 0
    messages_to_process = []
    if not params.message_custom_id:
        for f in dsfiles.message_results_directory.iterdir():
            if f.is_file():
                total_message_results_count += 1
                # logfile = dsfiles.message_processing_logs_directory / f"{f.stem}.json"
                # if not logfile.exists():
                messages_to_process.append(f)
    else:
        total_message_results_count = -1
        f = dsfiles.message_results_directory / f"{params.message_custom_id}.json"
        messages_to_process.append(f)

    print(
        f"{len(messages_to_process)} out of {total_message_results_count} message responses need to be processed."
    )
    for f in messages_to_process:
        message_results = json.loads(f.read_text(encoding="utf-8"))
        _process_message_response(
            dsdef=dsdef,
            message_results=message_results,
            prompt_params=prompt_params,
            dsfiles=dsfiles,
        )
