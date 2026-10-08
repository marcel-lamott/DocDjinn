import argparse
import os
import pathlib
import shutil

from docdjinn import ENV
from docdjinn.generation.constants import HANDWRITING_DEFAULT_BATCH_SIZE
from docdjinn.generation.models import LLMType, DatasetTask
from docdjinn.generation.models import PipelineParameters, SynDatasetDefinition
from docdjinn.data.interface import load_dataset
from docdjinn.generation.pipeline_01_select_seeds import pipeline_select_seeds
from docdjinn.generation.pipeline_02_prompt_llm import (
    pipeline_retrieve_document_html_seed_based,
)
from docdjinn.generation.pipeline_03_process_response import (
    pipeline_process_response_extract_html_and_gt,
)
from docdjinn.generation.pipeline_04_render_pdf_and_extract_geos import (
    pipeline_render_pdf_and_extract_geos_parallel,
)
from docdjinn.generation.pipeline_05_extract_bboxes_from_pdf import (
    pipeline_extract_bboxes,
)
from docdjinn.generation.pipeline_06_extract_layout_element_definitions_and_annotation_gt import (
    pipeline_extract_layout_element_definitions_and_annotation_gt,
)
from docdjinn.generation.pipeline_08_extract_visual_element_definitions import (
    pipeline_extract_visual_element_definitions,
)
from docdjinn.generation.pipeline_07_extract_handwriting import (
    pipeline_extract_handwritten_fields,
)
from docdjinn.generation.pipeline_09_create_handwriting_images import (
    pipeline_create_handwriting_images,
)
from docdjinn.generation.pipeline_11_render_pdf_second_pass import (
    pipeline_render_pdf_second_pass,
)
from docdjinn.generation.pipeline_12_insert_handwriting_images import (
    pipeline_handwritten_text_insertion,
)
from docdjinn.generation.pipeline_10_create_visual_elements import (
    pipeline_create_visual_elements,
)
from docdjinn.generation.pipeline_13_insert_visual_elements import (
    pipeline_insert_visual_elements,
)
from docdjinn.generation.pipeline_16_normalize_bboxes import pipeline_normalize_bboxes
from docdjinn.generation.pipeline_15_perform_ocr import (
    pipeline_perform_ocr,
)
from docdjinn.generation.pipeline_14_render_image import (
    pipeline_render_image,
)
from docdjinn.generation.pipeline_17_gt_preparation_verification import (
    pipeline_ground_truth_verification,
)
from docdjinn.generation.pipeline_19_create_debug_data import pipeline_create_debug_data
from docdjinn.generation.pipeline_18_analyze import pipeline_analyze


def parse_args():
    parser = argparse.ArgumentParser(
        description="DocDjinn Synthetic Document Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "SynDatasetDefinition",
        type=str,
        help="Filename without extension of the SynDatasetDefinition in data/syn_dataset_definitions",
    )

    parser.add_argument(
        "--reset",
        "-r",
        action="store_true",
        help="If set, all previous data is deleted prior to execution, except: prompt batches, prompt responses and seed images.",
    )

    parser.add_argument(
        "--entry",
        "-e",
        type=int,
        default=None,
        help="If set, starts the pipeline at this step",
    )

    # parser.add_argument(
    #     "--docids",
    #     type=str,
    #     default=None,
    #     help="Define document ids to which restrict the pipeline",
    # )

    parser.add_argument(
        "--hwbs",
        type=int,
        default=HANDWRITING_DEFAULT_BATCH_SIZE,
        help="Handwriting batch size",
    )

    parser.add_argument(
        "--nohw",
        action="store_true",
        help="Runs the pipeline without creating handwriting",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Runs the pipeline and creates debug data",
    )

    parser.add_argument(
        "--seedsonly",
        "-s",
        action="store_true",
        help="If set, the pipeline only collects seed images and then aborts",
    )

    parser.add_argument(
        "--apikey",
        type=str,
        default=None,
        help="If given, use the env variable with this name to retrieve the anthropic API key",
    )

    parser.add_argument(
        "--LLMType",
        type=str,
        choices=[e.value for e in LLMType],
        default=LLMType.CLAUDE.value,
        help="Define the whether to use closed source model or open source (currently just Qwen2.5-32B)",
    )

    parser.add_argument(
        "--message_custom_id",
        type=str,
        default=None,
        help="If specified, the pipeline is run only for this message and ignores existing results.",
    )

    args = parser.parse_args()
    args.LLMType = LLMType(args.LLMType)
    assert args.SynDatasetDefinition

    if args.apikey:
        assert os.getenv(args.apikey)
        print(f"Using Anthropic API Key from {args.apikey}")

    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    deffile = ENV.SYN_DATA_DEFINITIONS_DIR / f"{args.SynDatasetDefinition}.yaml"
    dsdef: SynDatasetDefinition = SynDatasetDefinition.from_file(deffile)
    dsfiles = dsdef.get_file_structure()

    if args.reset:
        print(f"""Parameter --reset has been passed. All existing data from {dsdef.name} will be deleted, except:
    - {dsfiles.prompt_batches_directory}
    - {dsfiles.message_results_directory}
    - {dsfiles.ocr_results_directory}
    - {dsfiles.preprocessed_seed_images_directory}
""")
        dsdef.reset_data_except_prompt_and_seeds()

    dataset = load_dataset(dsdef.base_dataset_name, split="train")

    print(f"The LLM will be used is: {args.LLMType}")

    # Copy used syn dataset defintion to output directory
    dst = dsfiles.base_path / f"{args.SynDatasetDefinition}.yaml"
    shutil.copy2(deffile, dst)

    params = PipelineParameters(
        dsdef=dsdef,
        llmtype=args.LLMType,
        message_custom_id=args.message_custom_id,
        seedsonly=args.seedsonly,
        debug=args.debug,
        handwriting_batch_size=args.hwbs,
        generate_handwriting=not args.nohw,
        api_key_env_variable_name=args.apikey,
    )

    entry = args.entry or 0

    # Execute pipeline
    if entry <= 1:
        pipeline_select_seeds(params=params)

    if args.seedsonly:
        exit(0)

    if entry <= 2:
        pipeline_retrieve_document_html_seed_based(params=params)

    if entry <= 3:
        pipeline_process_response_extract_html_and_gt(params=params)

    if entry <= 4:
        pipeline_render_pdf_and_extract_geos_parallel(params=params)

    if entry <= 5:
        pipeline_extract_bboxes(params=params)

    if entry <= 6:
        if dsdef.prompt_task == "annotation":
            pipeline_extract_layout_element_definitions_and_annotation_gt(params=params)

    if entry <= 7:
        pipeline_extract_handwritten_fields(params=params)

    if entry <= 8:
        pipeline_extract_visual_element_definitions(params=params)

    if entry <= 9:
        pipeline_create_handwriting_images(params=params)

    if entry <= 10:
        pipeline_create_visual_elements(params=params)

    if entry <= 11:
        pipeline_render_pdf_second_pass(params=params)

    if entry <= 12:
        pipeline_handwritten_text_insertion(params=params)

    if entry <= 13:
        pipeline_insert_visual_elements(params=params)

    if entry <= 14:
        pipeline_render_image(params=params)

    if entry <= 15:
        pipeline_perform_ocr(params=params)

    if entry <= 16:
        pipeline_normalize_bboxes(params=params)

    if entry <= 17:
        pipeline_ground_truth_verification(params=params)

    if entry <= 18:
        pipeline_analyze(params=params)

    if entry <= 19 and params.debug:
        pipeline_create_debug_data(params=params)
