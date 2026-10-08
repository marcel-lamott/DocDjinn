from __future__ import annotations

import inspect
from typing import Any

from pydantic import Field
from transformers import (
    AutoProcessor,
    BatchEncoding,
    BertTokenizerFast,
    RobertaTokenizerFast,
)

from docdjinn.data._transforms._generics._base import BaseTransform

# add custom models
from docdjinn.logging import get_logger

logger = get_logger(__name__)


class HuggingfaceProcessor(BaseTransform[BatchEncoding]):
    _TOKENIZERS_REQUIRING_SPLIT_TEXT = (BertTokenizerFast, RobertaTokenizerFast)

    tokenizer_name: str = "microsoft/layoutlmv3-base"
    init_kwargs: dict = Field(default_factory=dict)
    call_kwargs: dict = Field(default_factory=dict)
    cache_dir: str = "./cache"
    overflow_sampling: str = "return_all"

    @property
    def tokenizer(self):
        return (
            self._hf_processor.tokenizer
            if hasattr(self._hf_processor, "tokenizer")
            else self._hf_processor
        )

    @property
    def all_special_ids(self) -> set[int]:
        return set(self.tokenizer.all_special_ids)

    def model_post_init(self, context) -> None:
        assert self.overflow_sampling in [
            "return_all",
            "return_random_n",
            "no_overflow",
            "return_first_n",
        ], f"Overflow sampling strategy {self.overflow_sampling} is not supported."

        self._hf_processor = self._initialize_transform()

    def _get_default_call_kwargs(self):
        return {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "stride": 0,
            "pad_to_multiple_of": 8,
            "is_split_into_words": True,
            "return_overflowing_tokens": self.overflow_sampling
            != "no_overflow",  # set some arguments that we need to stay fixed for our case
            "return_token_type_ids": None,
            "return_attention_mask": True,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_length": False,
            "return_tensors": "pt",
            "verbose": True,
        }

    def _initialize_transform(self):
        processor = AutoProcessor.from_pretrained(
            self.tokenizer_name,
            cache_dir=self.cache_dir,
            local_files_only=False,
            apply_ocr=False,
            add_prefix_space=True,
            do_lower_case=True,
            do_normalize=False,
            do_resize=False,
            do_rescale=False,
            **self.init_kwargs,
        )

        self.call_kwargs = {**self._get_default_call_kwargs(), **self.call_kwargs}
        self._possible_args = inspect.signature(processor.__call__).parameters
        for key in list(self.call_kwargs.keys()):
            if key not in self._possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                self.call_kwargs.pop(key)
        return processor

    def get_config(self):
        return {
            "tokenizer_name": self.tokenizer_name,
            "init_kwargs": self.init_kwargs,
            "call_kwargs": self.call_kwargs,
        }

    def get_output_data_model(self) -> type[BatchEncoding]:
        return BatchEncoding

    def _convert_text_to_list(self, text: Any) -> list[str]:
        if isinstance(text, str):
            return text.split()
        elif isinstance(text, list):
            return text
        else:
            raise ValueError("Input text must be a string or a list of strings.")

    def __call__(self, **inputs) -> BatchEncoding:
        if isinstance(self.tokenizer, self._TOKENIZERS_REQUIRING_SPLIT_TEXT):
            text = inputs.get("text", None)
            text_pair = inputs.get("text_pair", None)

            if text is not None and text_pair is not None:
                inputs["text"] = self._convert_text_to_list(text)
                inputs["text_pair"] = self._convert_text_to_list(text_pair)

                assert isinstance(inputs["text"], list), (
                    "Input 'text' must be a list of strings."
                )
                assert isinstance(inputs["text_pair"], list), (
                    "Input 'text_pair' must be a list of strings."
                )
        filtered_inputs = {k: v for k, v in inputs.items() if k in self._possible_args}
        return self._hf_processor(**filtered_inputs, **self.call_kwargs)
