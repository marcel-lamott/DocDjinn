import json
import math
import os

import pandas as pd
from docdjinn import GENERATION
from docdjinn.generation.constants import (
    PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_MAX_SIZE,
    PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_POLL_SLEEP_SECONDS_BETWEEN_BATCH,
    PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_POLL_SLEEP_SECONDS_BETWEEN_ITERATION,
)
from docdjinn.generation.utils.serialization import image_to_base64
from docdjinn.generation.pipeline_01.claude_batching import ClaudeBatchedClient
from docdjinn.generation.models import (
    PipelineParameters,
    SynDatasetDefinition,
    LLMType,
    SyntheticDatasetFileStructure,
)
from docdjinn.generation.utils.log import log_pipeline_level


def _create_batch(
    prompt: str,
    cur_msg_index: int,
    seeds_df: pd.DataFrame,
    batch_size: int,
    dsfiles: SyntheticDatasetFileStructure,
):
    def prompt_gen():
        for _ in range(batch_size):
            yield prompt

    def seed_doc_ids_gen():
        for i in range(batch_size):
            index = (cur_msg_index + i) % len(
                seeds_df
            )  # Start all over again when we have more messages than seeds
            seeds_row = seeds_df.iloc[index]
            doc_ids = seeds_row.tolist()
            yield doc_ids

    batch_seed_doc_ids = list(seed_doc_ids_gen())

    def imgs_gen():
        for i in range(batch_size):
            doc_ids = batch_seed_doc_ids[i]
            img_base64 = [
                image_to_base64(
                    imgpath=dsfiles.preprocessed_seed_images_directory / f"{docid}.jpg"
                )
                for docid in doc_ids
            ]
            yield img_base64

    return cur_msg_index + batch_size, prompt_gen(), imgs_gen(), batch_seed_doc_ids


def get_remaining_prompt_calls_count(
    dsdef: SynDatasetDefinition, awaited_messages_total: int
):
    dsfiles = dsdef.get_file_structure()

    probable_message_responses = awaited_messages_total
    total_message_responses = awaited_messages_total
    for f in dsfiles.message_results_directory.iterdir():
        if f.is_file():
            total_message_responses += 1
            msg_result = json.loads(f.read_text(encoding="utf-8"))
            if msg_result["result_type"] == "succeeded":
                probable_message_responses += 1

    # Each message response should contain N HTML documents
    # TODO: better count actual number of responses here in case num_solutions was changed afterwards
    probable_raw_documents = (
        probable_message_responses * dsdef.prompt_params.num_solutions
    )
    remaining_documents_count = dsdef.documents_count - probable_raw_documents
    print(
        f"{dsdef.documents_count=} {probable_message_responses=} {probable_raw_documents=} {remaining_documents_count=}"
    )

    # Create batches
    remaining_prompt_calls = (
        remaining_documents_count / dsdef.prompt_params.num_solutions
    )

    return (
        total_message_responses,
        remaining_documents_count,
        remaining_prompt_calls,
    )


def pipeline_retrieve_document_html_seed_based(
    params: PipelineParameters,
):
    log_pipeline_level()

    prompt_batch_max_size = PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_MAX_SIZE
    prompt_batch_poll_sleep_seconds_between_batch = PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_POLL_SLEEP_SECONDS_BETWEEN_BATCH
    prompt_batch_poll_sleep_seconds_between_iteration = PIPELINE_01_RETRIEVE_DOCUMENT_HTML__PROMPT_BATCH_POLL_SLEEP_SECONDS_BETWEEN_ITERATION

    dsdef = params.dsdef
    # Check which LLM Type sanity
    assert params.llmtype.value in [e.value for e in LLMType], (
        f"Invalid model:{params.llmtype.value}"
    )

    dsfiles = dsdef.get_file_structure()
    prompt = dsdef.get_prompt()

    seeds_path = dsfiles.base_path / "seeds.csv"
    seeds_df = pd.read_csv(seeds_path)
    assert len(seeds_df.columns) == dsdef.seed_images_count

    # TODO Saifullah: implement opensource VLMs here

    api_key_env_variable_name = params.api_key_env_variable_name or "ANTHROPIC_API_KEY"
    api_key = os.getenv(api_key_env_variable_name)
    print(f"{api_key_env_variable_name=} len: {len(api_key)}")  # type: ignore
    if params.api_key_env_variable_name:
        input("PRESS ENTER TO CONFIRM")

    client = ClaudeBatchedClient(api_key=api_key)  # type: ignore
    running_batches, awaited_messages_total = client.get_running_batches(
        batch_data_directory=dsfiles.prompt_batches_directory
    )
    total_message_responses, remaining_documents_count, remaining_prompt_calls = (
        get_remaining_prompt_calls_count(
            dsdef=dsdef, awaited_messages_total=awaited_messages_total
        )
    )

    cur_msg_index = total_message_responses
    if remaining_documents_count > 0:
        num_batches = math.ceil(remaining_prompt_calls / prompt_batch_max_size)
        print(f"{remaining_prompt_calls=} {num_batches=}")

        for i_batch in range(num_batches):
            # For each batch, we select <seed_images_count> images from all seed images
            this_batch_size = min(
                math.ceil(
                    remaining_documents_count / dsdef.prompt_params.num_solutions
                ),
                prompt_batch_max_size,
            )
            cur_msg_index, batch_prompts, batch_imgs, batch_seed_docids = _create_batch(
                prompt=prompt,
                cur_msg_index=cur_msg_index,
                seeds_df=seeds_df,
                batch_size=this_batch_size,
                dsfiles=dsfiles,
            )

            # Batch is sent to LLM and batch infos are stored locally on disk
            client.send_batch(
                model=GENERATION.LLM,
                prompts=batch_prompts,
                images_base64=batch_imgs,
                image_docids=batch_seed_docids,
                batch_data_directory=dsfiles.prompt_batches_directory,
                max_tokens=GENERATION.MAX_TOKENS,  # up to 32k
            )

            remaining_documents_count -= this_batch_size

    # Await all previously sent batches
    # LLM Responses are saved to disk
    client.await_batches(
        batch_data_directory=dsfiles.prompt_batches_directory,
        message_data_directory=dsfiles.message_results_directory,
        sleep_seconds_between_batch=prompt_batch_poll_sleep_seconds_between_batch,
        sleep_seconds_iteration=prompt_batch_poll_sleep_seconds_between_iteration,
    )
