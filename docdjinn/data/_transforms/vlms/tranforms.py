from abc import ABC, abstractmethod

from atria_core.utilities.repr import RepresentationMixin
from pydantic import BaseModel

from docdjinn.data._core._data_types import (
    ConditionalGenerationModelInput,
    DatasetLabels,
    DocumentInstance,
)
from docdjinn.data._transforms._tokenizers._utilities import _extract_annotations
from docdjinn.logging import get_logger

logger = get_logger(__name__)


class BaseVLMTokenizer(RepresentationMixin, BaseModel, ABC):
    """Base class for VLM tokenizers"""

    tokenizer_name: str = "deepseek-community/deepseek-vl-1.3b-base"
    tokenizer_cache_dir: str = "./cache"
    is_training: bool = True
    dataset_labels: DatasetLabels

    def model_post_init(self, context) -> None:
        self._default_init_kwargs = {
            "cache_dir": self.tokenizer_cache_dir,
            "local_files_only": False,
            "apply_ocr": False,
        }
        self._default_call_kwargs = {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 1024,
            "stride": 0,
            "pad_to_multiple_of": 8,
            "return_tensors": "pt",
        }

        self._setup_processor()
        self._tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )

    def _setup_processor(self):
        """Setup processor - can be overridden by child classes"""
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.tokenizer_name, **self._default_init_kwargs
        )

    def _get_common_kwargs(self, document_instance: DocumentInstance) -> tuple:
        """Extract common data from document instance"""
        image = document_instance.image.load().content.convert("RGB")
        words = (
            document_instance.content.words
            if document_instance.content is not None
            else []
        )
        boxes = (
            document_instance.content.word_bboxes.value
            if document_instance.content is not None
            else []
        )

        if not words:
            words = ["None"]
            boxes = [[0, 0, 0, 0]]

        return image, words, boxes

    def _tokenize_target(self, target_text: str, max_length: int = 128):
        """Common target tokenization logic"""

        target_token_ids = self._tokenizer.encode(
            target_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )[0]

        # Set padding token IDs to -100 to ignore in loss computation
        target_token_ids[target_token_ids == 0] = -100
        return target_token_ids

    @abstractmethod
    def _prepare_instances(
        self, document_instance: DocumentInstance, annotations
    ) -> ConditionalGenerationModelInput:
        """Prepare instances for the specific task type"""
        pass

    def __call__(self, document_instance: DocumentInstance):
        annotations = _extract_annotations(document_instance)
        return self._prepare_instances(document_instance, annotations)


class SequenceClassificationVLMTokenizer(BaseVLMTokenizer):
    """Tokenizer for sequence classification tasks"""

    def _prepare_instances(
        self, document_instance: DocumentInstance, annotations
    ) -> ConditionalGenerationModelInput:
        import torch

        possible_labels = self.dataset_labels.classification or []
        image, words, boxes = self._get_common_kwargs(document_instance)

        prompt = f"Document Classification. Classify the document into one of these categories: {', '.join(possible_labels)}. Document: "
        target_text = annotations.label.name

        # Tokenize input
        if self.tokenizer_name == "microsoft/udop-large":
            tokenized_instance = self._processor(
                image, prompt, text_pair=words, boxes=boxes, **self._default_call_kwargs
            )
        elif self.tokenizer_name in ["google-t5/t5-large", "google-t5/t5-base"]:
            tokenized_instance = self._processor(
                prompt, text_pair=" ".join(words), **self._default_call_kwargs
            )

        for key, value in tokenized_instance.items():
            tokenized_instance[key] = value.squeeze(0)

        target_token_ids = self._tokenize_target(target_text, max_length=16)

        return ConditionalGenerationModelInput(
            **tokenized_instance,
            index=torch.tensor(document_instance.index),
            sample_id=document_instance.sample_id,
            words=words,
            target_text=target_text,
            target_token_ids=target_token_ids,
            _tokenizer_name=self.tokenizer_name,
            _tokenizer_init_kwargs=self._default_init_kwargs,
        )
