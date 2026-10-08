from dataclasses import dataclass
from datetime import datetime
import time
import os
import pathlib
import uuid
from typing import Iterable, Literal
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types import (
    MessageParam,
    ImageBlockParam,
    TextBlockParam,
    Base64ImageSourceParam,
    TextBlock,
)
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages.message_batch import MessageBatch
from anthropic.types.messages.message_batch_individual_response import (
    MessageBatchIndividualResponse,
)
from anthropic.types.messages.message_batch_succeeded_result import (
    MessageBatchSucceededResult,
)

from docdjinn import ENV, LLM, GENERATION

import json

from docdjinn.generation.models import PromptMsgResultLogKey, SynDatasetDefinition
from docdjinn.generation.utils.serialization import image_to_base64
from docdjinn.generation.utils.status import StatusLine
from docdjinn.generation.pipeline_01.cost import (
    calculate_message_cost,
    get_total_cost,
    print_cost_report,
)


def create_batch(
    client: anthropic.Anthropic,
    id_to_message: dict[str, MessageParam],
    model=GENERATION.LLM,
    max_tokens=GENERATION.MAX_TOKENS,
):
    requests = []
    for msg_id, msg in id_to_message.items():
        requests.append(
            Request(
                custom_id=msg_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[msg],
                ),
            )
        )
    message_batch = client.messages.batches.create(requests=requests)

    # print(message_batch)
    return message_batch.id


def create_message(prompt: str, images_base64: list[str]):
    content = []
    # Only prompt is cached, images not (because they come after) as they change with each call
    content.append(
        TextBlockParam(text=prompt, type="text", cache_control={"type": "ephemeral"})
    )
    if images_base64:
        for img_base64 in images_base64:
            content.append(
                ImageBlockParam(
                    source=Base64ImageSourceParam(
                        media_type="image/jpeg", type="base64", data=img_base64
                    ),
                    type="image",
                )
            )

    return MessageParam(
        role="user",
        content=content,
    )


"""
3.7.
Claude-Sonnet 3.7 [2] is employed as the underlying
MLLM for HTML-based document generation. For each
document category, a set of S = 10 real documents is
selected as seed samples to guide the generation process.
The MLLM is prompted with the seed samples and document category to generate N = 10 synthetic documents per
category. Each model call generates 4 HTML-based documents per iteration, repeated until the total target is reached
"""


class ClaudeBatchedClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def send_batch(
        self,
        model: str,
        prompts: Iterable[str],
        images_base64: Iterable[list[str]],
        image_docids: Iterable[list[str]],
        batch_data_directory: pathlib.Path,
        max_tokens: int = 8192,
    ):
        # assert len(prompts) == len(images_base64)

        # Collect batch data
        id_to_message = dict()
        id_to_message_seed_docids = dict()
        for prompt, image_base64s, seed_docids in zip(
            prompts, images_base64, image_docids
        ):
            # Create GUID message ID
            message_id = str(uuid.uuid4())
            message = create_message(prompt=prompt, images_base64=image_base64s)
            id_to_message[message_id] = message
            id_to_message_seed_docids[message_id] = seed_docids

        # Send batch
        batch_id = create_batch(
            client=self.client,
            id_to_message=id_to_message,
            model=model,
            max_tokens=max_tokens,
        )

        # Store batch data
        batch_data_file = batch_data_directory / f"{batch_id}.json"
        batch_metadata = {
            "id": batch_id,
            "model": model,
            "processing_status": "in_progress",
            "message_ids": list(id_to_message.keys()),
            "message_id_to_seed_docids": id_to_message_seed_docids,
            "created_at": datetime.now().isoformat(),
            "ended_at": "",
            "cost_tracking": {
                "total_cost_usd": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cache_creation_tokens": 0,
                "total_cache_read_tokens": 0,
            },
        }
        batch_metadata_json = json.dumps(batch_metadata, indent=2)
        batch_data_file.write_text(batch_metadata_json, encoding="utf-8")

    def get_running_batches(self, batch_data_directory: pathlib.Path):
        # Get metadata for all batches that are currently running
        running_batches = []
        awaited_messages_total = 0
        for f in batch_data_directory.iterdir():
            if f.is_file():
                batch_metadata = json.loads(f.read_text())
                if batch_metadata["processing_status"] == "in_progress":
                    running_batches.append(batch_metadata)
                    awaited_messages_total += len(batch_metadata["message_ids"])

        return running_batches, awaited_messages_total

    def await_batches(
        self,
        batch_data_directory: pathlib.Path,
        message_data_directory: pathlib.Path,
        sleep_seconds_between_batch: float = 2,
        sleep_seconds_iteration: float = 30,
    ):
        running_batches, awaited_messages_total = self.get_running_batches(
            batch_data_directory=batch_data_directory
        )
        running_batches_count = len(running_batches)
        print(
            f"Found {running_batches_count} batches with {awaited_messages_total} messages in total."
        )

        status = StatusLine()
        status.start()

        while any(running_batches):
            finished_batches = []
            # print(f"Awaiting {len(running_batches)} batches...")
            status.update_message(f"Awaiting {len(running_batches)} batches...")

            for batch_metadata in running_batches:
                batch_id = batch_metadata["id"]
                message_batch = self.client.messages.batches.retrieve(
                    message_batch_id=batch_id
                )

                if message_batch.processing_status != "in_progress":
                    # Batch has finished or was canceled
                    # print(f"Batch {message_batch.id} processing status is now {message_batch.processing_status}")
                    status.log(
                        f"Batch {message_batch.id} processing status is now {message_batch.processing_status}"
                    )

                    # Retrieve batch results if batch was processed
                    if message_batch.processing_status == "ended":
                        cost_tracking = self._finalize_batch(
                            message_batch=message_batch,
                            batch_id=batch_id,
                            message_ids=set(batch_metadata["message_ids"]),
                            message_data_directory=message_data_directory,
                            model=batch_metadata.get("model", GENERATION.LLM),
                        )
                        batch_metadata["cost_tracking"] = cost_tracking

                    # Update batch metadata
                    batch_metadata["processing_status"] = (
                        message_batch.processing_status
                    )
                    batch_metadata["ended_at"] = datetime.now().isoformat()
                    batch_metadata_json = json.dumps(batch_metadata, indent=2)
                    batch_data_file = batch_data_directory / f"{batch_id}.json"
                    batch_data_file.write_text(batch_metadata_json, encoding="utf-8")

                    # Dont keep polling this batch
                    finished_batches.append(batch_metadata)

                time.sleep(sleep_seconds_between_batch)

            for batch_metadata in finished_batches:
                running_batches.remove(batch_metadata)

            time.sleep(sleep_seconds_iteration)

        status.stop()

        print(f"Finished awaiting {running_batches_count} batches.")

    def get_total_cost(self, batch_data_directory: pathlib.Path) -> dict:
        return get_total_cost(batch_data_directory)

    def print_cost_report(
        self, batch_data_directory: pathlib.Path, dataset_log_path: pathlib.Path = None
    ):
        print_cost_report(batch_data_directory, dataset_log_path)

    def _finalize_batch(
        self,
        message_batch: MessageBatch,
        batch_id: str,
        message_ids: set[str],
        message_data_directory: pathlib.Path,
        model: str,
    ) -> dict:
        """
        Finalize a batch by processing results and calculating costs.

        Returns:
            Dictionary with cost tracking information
        """
        assert message_batch.processing_status == "ended"

        # Initialize cost tracking
        cost_tracking = {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_cache_read_tokens": 0,
        }

        # Stream results file in memory-efficient chunks, processing one at a time
        result: MessageBatchIndividualResponse
        for result in self.client.messages.batches.results(message_batch_id=batch_id):
            # Ensure we know this message in this batch
            assert result.custom_id in message_ids, (
                f"Unknown message '{result.custom_id}' in batch '{batch_id}'"
            )

            message_data = {
                PromptMsgResultLogKey.custom_id: result.custom_id,
                PromptMsgResultLogKey.id: "",
                PromptMsgResultLogKey.result_type: result.result.type,
                PromptMsgResultLogKey.error: "",
                PromptMsgResultLogKey.response: "",
                PromptMsgResultLogKey.usage_input_tokens: -1,
                PromptMsgResultLogKey.usage_output_tokens: -1,
            }

            match result.result.type:
                case "succeeded":
                    res: MessageBatchSucceededResult = result.result
                    message_data["id"] = res.message.id

                    # Extract token usage
                    input_tokens = res.message.usage.input_tokens
                    output_tokens = res.message.usage.output_tokens
                    cache_creation_tokens = getattr(
                        res.message.usage, "cache_creation_input_tokens", 0
                    )
                    cache_read_tokens = getattr(
                        res.message.usage, "cache_read_input_tokens", 0
                    )

                    message_data[PromptMsgResultLogKey.usage_input_tokens] = (
                        input_tokens
                    )
                    message_data[PromptMsgResultLogKey.usage_output_tokens] = (
                        output_tokens
                    )

                    # Calculate cost for this message
                    message_cost = calculate_message_cost(
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_creation_input_tokens=cache_creation_tokens,
                        cache_read_input_tokens=cache_read_tokens,
                    )

                    # Update batch totals
                    cost_tracking["total_cost_usd"] += message_cost
                    cost_tracking["total_input_tokens"] += input_tokens
                    cost_tracking["total_output_tokens"] += output_tokens
                    cost_tracking["total_cache_creation_tokens"] += (
                        cache_creation_tokens
                    )
                    cost_tracking["total_cache_read_tokens"] += cache_read_tokens

                    if res.message.stop_reason == "refusal":
                        # The LLM refused to process the request because of a policy violation
                        url = "https://docs.claude.com/en/docs/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals"
                        print(f"[SKIPPING] Policy Violation error ({url})")
                        message_data[PromptMsgResultLogKey.error] = "refusal"
                        message_data[PromptMsgResultLogKey.response] = None
                    else:
                        # raise Exception(f"Policy violation from Claude API ({url})")

                        # Assert that content is of expected shape and type
                        assert len(res.message.content) == 1 and isinstance(
                            res.message.content[0], TextBlock
                        ), (
                            f"Content validation failed: len={len(res.message.content)}, "
                            f"content={res.message.content}, "
                            f"type={type(res.message.content[0]).__name__ if res.message.content else 'empty'}"
                        )

                        # Fetch actual response
                        response: TextBlock = res.message.content[0]
                        message_data[PromptMsgResultLogKey.response] = response.text

                case "errored":
                    if result.result.error.type == "invalid_request":
                        # Request body must be fixed before re-sending request
                        print(f"Validation error {result.custom_id}")
                        raise Exception(
                            f"Validation error from Claude API: {result.result.error}"
                        )
                    else:
                        # Request can be retried directly
                        print(f"Server error {result.custom_id} {result.result.error}")
                        message_data[PromptMsgResultLogKey.error] = (
                            f"{result.result.error}"
                        )

            # Save message to disk
            message_data_file = message_data_directory / f"{result.custom_id}.json"
            message_data_json = json.dumps(message_data, indent=2)
            message_data_file.write_text(message_data_json, encoding="utf-8")

        return cost_tracking
