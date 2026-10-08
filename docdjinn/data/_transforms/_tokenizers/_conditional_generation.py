from __future__ import annotations

import json

from docdjinn.data._transforms._tokenizers._document_processors import BaseTransform
from docdjinn.data._transforms._tokenizers._udop_processor import CustomUdopProcessor
from docdjinn.logging import get_logger

from ..._core._data_types import (
    AnnotatedObjectList,
    ConditionalGenerationModelInput,
    DatasetLabels,
    DocumentInstance,
    ExtractiveQAPair,
    Label,
    LabelList,
)
from ..._core._utilities import TaskType
from ._utilities import _extract_annotations

logger = get_logger(__name__)


class ConditionalGenerationTokenizer(BaseTransform):
    task_type: TaskType
    tokenizer_name: str = "microsoft/udop-large"
    tokenizer_cache_dir: str = "./cache"
    is_training: bool = True
    generate_entity_vocabulary: bool = True
    dataset_labels: DatasetLabels

    def get_output_data_model(self) -> type:
        return ConditionalGenerationModelInput

    def model_post_init(self, context) -> None:
        from transformers import AutoProcessor

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
        if self.task_type == TaskType.token_classification:
            self._default_call_kwargs["return_overflowing_tokens"] = True
            self._default_call_kwargs["return_offsets_mapping"] = True
            self._default_call_kwargs["stride"] = 128
            self._default_call_kwargs["max_length"] = 512
            self._processor = CustomUdopProcessor.from_pretrained(
                self.tokenizer_name,
                **self._default_init_kwargs,
                clean_up_tokenization_spaces=False,
            )
        else:
            self._processor = AutoProcessor.from_pretrained(
                self.tokenizer_name, **self._default_init_kwargs
            )
        self._tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )

        # if self.task_type == TaskType.token_classification and self.generate_entity_vocabulary:
        #     possible_labels = (
        #         self.dataset_labels.ser
        #         if self.dataset_labels.ser is not None
        #         else []
        #     )
        # possible_labels = [f"<{lbl}>" for lbl in possible_labels]
        # num_added_tokens = self._tokenizer.add_special_tokens({"additional_special_tokens": possible_labels})
        # logger.info(f"Added {num_added_tokens} special tokens for entity labels: {possible_labels}")

    def _get_common_kwargs(self, document_instance: DocumentInstance) -> tuple:
        # get pil image from the document instance
        image = document_instance.image.load().content.convert("RGB")

        # get words from the document instance
        words = (
            document_instance.content.words
            if document_instance.content is not None
            else []
        )

        # get bounding boxes from the document instance
        boxes = (
            document_instance.content.word_bboxes.value
            if document_instance.content is not None
            else []
        )

        return image, words, boxes

    def _prepare_instances_for_sequence_classification(
        self, document_instance: DocumentInstance, label: Label
    ) -> ConditionalGenerationModelInput:
        import torch

        possible_labels = (
            self.dataset_labels.classification
            if self.dataset_labels.classification is not None
            else []
        )
        image, words, boxes = self._get_common_kwargs(document_instance)
        prompt = f"Document Classification. Classify the document into one of these categories: {', '.join(possible_labels)}. Document: "
        target_text = label.name

        if not words:
            # Supply a dummy token and box so UDOP doesn't crash
            words = ["None"]
            boxes = [[0, 0, 0, 0]]

        tokenized_instance = {}
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

        # # for debugging decode the input ids
        # decoded_input = self._processor.decode(tokenized_instance['input_ids'], skip_special_tokens=True)
        # print('Decoded input:', decoded_input)

        # Tokenize target text to get target_token_ids
        target_token_ids = self._tokenizer.encode(  # this takes text but returns a batch, truly a garbage design
            target_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=16,
            truncation=True,
            padding="max_length",
        )[0]

        # decoded_target_text = self._processor.decode(target_token_ids, skip_special_tokens=True)
        # print('Decoded target_text:', decoded_target_text)

        # Set padding token IDs to -100 to ignore in loss computation
        target_token_ids[target_token_ids == 0] = -100

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

    def _generate_target_text_for_token_classification(
        self,
        words: list[str],
        word_labels: list[str],
        target_text_type: str = "key_value_pairs",
    ) -> str:
        # entities = {}

        target_text = ""
        for word_idx, (word, word_label) in enumerate(
            zip(words, word_labels, strict=True)
        ):
            target_text += f"{word} {word_label} "
        target_text = target_text.strip()
        return target_text

        # if word_label == "O" or word_label.startswith("I-"):
        #     continue

        # if word_label.startswith("B-"):
        #     entity_words = [word]
        #     for next_word, next_label in zip(
        #         words[word_idx + 1 :], word_labels[word_idx + 1 :]
        #     ):
        #         if next_label == f"I-{word_label[2:]}":
        #             entity_words.append(next_word)
        #         else:
        #             break

        #     if word_label[2:] not in entities:
        #         entities[word_label[2:]] = []
        #     entities[word_label[2:]].append(" ".join(entity_words))

        if len(entities) == 0:
            return None

        if target_text_type == "csv":
            lines = []
            separator = "|"
            for key, values in entities.items():
                for value in values:
                    line = f"{key}={value}{separator}"
                    lines.append(line)
            lines[-1] = lines[-1].rstrip(f"{separator}")  # remove sep from last line
            return "".join(lines)
        elif target_text_type == "json":
            return json.dumps(entities)
        else:
            raise NotImplementedError(
                f"Target text type {target_text_type} not supported."
            )

    def _prepare_instances_for_token_classification(
        self, document_instance: DocumentInstance, word_labels: LabelList
    ) -> list[ConditionalGenerationModelInput]:
        import torch

        image, words, boxes = self._get_common_kwargs(document_instance)
        prompt = "Information Extraction. Extract all the entities present in this document: Document: "

        if not words:
            words = ["None"]
            boxes = [[0, 0, 0, 0]]
            word_labels.name = ["O"]

        features, encoded_batch = None, None
        if self.tokenizer_name == "microsoft/udop-large":
            features, encoded_batch = self._processor(
                image, prompt, text_pair=words, boxes=boxes, **self._default_call_kwargs
            )
        elif self.tokenizer_name in ["google-t5/t5-large", "google-t5/t5-base"]:
            raise NotImplementedError(
                "Token classification not implemented for T5 models yet."
            )

        sequence_ids = []
        word_ids = []
        for i in range(len(encoded_batch["input_ids"])):
            sequence_ids_per_overflow = encoded_batch.sequence_ids(i)
            word_ids_per_overflow = encoded_batch.word_ids(i)

            # filter sequence_ids
            sequence_ids_per_overflow = [
                -100 if x is None else x for x in sequence_ids_per_overflow
            ]
            word_ids_per_overflow = [
                -100 if x is None else x for x in word_ids_per_overflow
            ]
            if max(sequence_ids_per_overflow) > 0:
                word_ids_per_overflow = [
                    -100 if sequence_id == 0 else word_id
                    for word_id, sequence_id in zip(
                        word_ids_per_overflow, sequence_ids_per_overflow
                    )
                ]
            sequence_ids.append(sequence_ids_per_overflow)
            word_ids.append(word_ids_per_overflow)

        sequence_ids = torch.tensor(sequence_ids)
        word_ids = torch.tensor(word_ids)

        # to compare the targets we need to know where the start of next overlfow sequence is after stride
        last_max_word_id = -1
        instances = []
        for overflow_idx in range(len(encoded_batch["input_ids"])):
            # find min max word ids
            input_ids_per_per_overflow = encoded_batch["input_ids"][overflow_idx]
            word_ids_per_overflow = word_ids[overflow_idx]
            min_word_id = min(
                [wid for wid in word_ids_per_overflow.tolist() if wid != -100]
            )
            max_word_id = max(
                [wid for wid in word_ids_per_overflow.tolist() if wid != -100]
            )

            # words in this overflow
            words_in_this_overflow = words[min_word_id : max_word_id + 1]
            word_labels_in_this_overflow = word_labels.name[
                min_word_id : max_word_id + 1
            ]

            target_text = self._generate_target_text_for_token_classification(
                words=words_in_this_overflow,
                word_labels=word_labels_in_this_overflow,
                target_text_type="csv",
            )

            if target_text is None:
                continue

            target_token_ids = self._tokenizer.encode(
                target_text,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding="max_length",
            )[0]

            # word labels after stride
            word_to_extract_in_this_overflow = words[
                last_max_word_id + 1 : max_word_id + 1
            ]
            word_labels_to_extract_in_this_overflow = word_labels.name[
                last_max_word_id + 1 : max_word_id + 1
            ]
            # decoded_target_text = tokenizer.decode(target_token_ids, skip_special_tokens=True)
            # decoded_input_text = tokenizer.decode(input_ids_per_overflow, skip_special_tokens=True)
            last_max_word_id = max_word_id

            # index: Optional["torch.Tensor"] = None
            # sample_id: Optional[str] = None
            # input_ids: Optional["torch.Tensor"] = None
            # bbox: Optional["torch.Tensor"] = None
            # attention_mask: Optional["torch.Tensor"] = None
            # pixel_values: Optional["torch.Tensor"] = None
            # question_text: Optional[str] = None
            # target_text: Optional[str] = None
            # target_token_ids: Optional["torch.Tensor"] = None
            # words: Optional[list[str]] = None
            # word_labels: Optional[list[str]] = None
            # label: Optional["torch.Tensor"] = None
            # _tokenizer_name: Optional[str] = None
            # _tokenizer_init_kwargs: Optional[dict] = None

            # Set padding token IDs to -100 to ignore in loss computation
            target_token_ids[target_token_ids == 0] = -100

            instance = ConditionalGenerationModelInput(
                index=torch.tensor(document_instance.index),
                sample_id=document_instance.sample_id,
                input_ids=input_ids_per_per_overflow,
                attention_mask=features["attention_mask"][overflow_idx],
                pixel_values=features["pixel_values"][overflow_idx],
                bbox=features["bbox"][overflow_idx],
                words=word_to_extract_in_this_overflow,
                word_labels=word_labels_to_extract_in_this_overflow,
                target_text=target_text,
                target_token_ids=target_token_ids,
                _tokenizer_name=self.tokenizer_name,
                _tokenizer_init_kwargs=self._default_init_kwargs,
            )
            instances.append(instance)

        if self.is_training:
            random_index = int(torch.randint(0, len(instances), (1,)).item())
            return instances[random_index]

        return instances

    def _prepare_instances_for_question_answering(
        self, document_instance: DocumentInstance, qa_pairs: list[ExtractiveQAPair]
    ) -> list[ConditionalGenerationModelInput]:
        import torch

        image, words, boxes = self._get_common_kwargs(document_instance)

        instances = []
        for qa_pair in qa_pairs:
            # since we can have multiple answers per question, we need to handle that here and just take one which is not
            # -1 # we don't need to remove no answer indices in conditional generation setting as we always have the answer anyway
            # word_ans_start, word_ans_end = -1, -1
            # for ans_start, ans_end in zip(qa_pair.answer_start, qa_pair.answer_end):
            #     if ans_start != -1 and ans_end != -1:
            #         word_ans_start = ans_start
            #         word_ans_end = ans_end
            #         break

            # if word_ans_start == -1 or word_ans_end == -1:
            #     if self.is_training:
            #         logger.warning(f"Skipping QA pair with no answer during training: {qa_pair}")
            #         continue

            prompt = f"Question answering. {qa_pair.question_text}"
            target_text = qa_pair.answer_text[0]

            tokenized_instance = {}
            if self.tokenizer_name == "microsoft/udop-large":
                tokenized_instance = self._processor(
                    image,
                    prompt,
                    text_pair=words,
                    boxes=boxes,
                    **self._default_call_kwargs,
                )
            elif self.tokenizer_name in ["google-t5/t5-large", "google-t5/t5-base"]:
                tokenized_instance = self._processor(
                    prompt, text_pair=" ".join(words), **self._default_call_kwargs
                )

            for key, value in tokenized_instance.items():
                tokenized_instance[key] = value.squeeze(0)

            # # # for debugging decode the input ids
            # decoded_input = self._processor.decode(tokenized_instance['input_ids'], skip_special_tokens=True)
            # print('Decoded input:', decoded_input)

            # Tokenize target text to get target_token_ids
            target_token_ids = self._tokenizer.encode(  # this takes text but returns a batch, truly a garbage design
                target_text,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            )[0]

            # decoded_target_text = self._processor.decode(target_token_ids, skip_special_tokens=True)
            # print('Decoded target_text:', decoded_target_text)

            # Set padding token IDs to -100 to ignore in loss computation
            target_token_ids[target_token_ids == 0] = -100

            instance = ConditionalGenerationModelInput(
                **tokenized_instance,
                index=torch.tensor(document_instance.index),
                sample_id=document_instance.sample_id,
                words=words,
                target_text=target_text,
                question_text=qa_pair.question_text,
                target_token_ids=target_token_ids,
                _tokenizer_name=self.tokenizer_name,
                _tokenizer_init_kwargs=self._default_init_kwargs,
            )

            instances.append(instance)

        if self.is_training:
            random_index = int(torch.randint(0, len(instances), (1,)).item())
            return instances[random_index]

        return instances

    def _prepare_instances_for_layout_analysis(
        self,
        document_instance: DocumentInstance,
        annotated_objects: AnnotatedObjectList,
    ) -> str:
        possible_labels = (
            self.dataset_labels.layout if self.dataset_labels.layout is not None else []
        )
        image, words, boxes = self._get_common_kwargs(document_instance)
        prompt = f"Layout Analysis. Extract the layout entities present in the document into one of these categories: {', '.join(possible_labels)}. Document: "
        bbox_labels_concatenated = []
        for label, bbox in zip(
            annotated_objects.label.name,
            annotated_objects.bbox,
        ):
            bbox = [int(x * 1000) for x in bbox]
            bbox_str = "".join([f"<{coord}>" for coord in bbox])
            bbox_labels_concatenated.append(f"{bbox_str}<{label}>")

        target_text = ",".join(bbox_labels_concatenated)

        tokenized_instance = {}
        if self.tokenizer_name == "microsoft/udop-large":
            tokenized_instance = self._processor(
                image, prompt, **self._default_call_kwargs
            )
        elif self.tokenizer_name in ["google-t5/t5-large", "google-t5/t5-base"]:
            tokenized_instance = self._processor(
                prompt, text_pair=" ".join(words), **self._default_call_kwargs
            )

        for key, value in tokenized_instance.items():
            tokenized_instance[key] = value.squeeze(0)

        # for debugging decode the input ids
        # decoded_input = self._processor.decode(tokenized_instance['input_ids'], skip_special_tokens=True)
        # print('Decoded input:', decoded_input)

        # Tokenize target text to get target_token_ids
        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        target_token_ids = tokenizer.encode(  # this takes text but returns a batch, truly a garbage design
            target_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length",
        )[0]

        # debugging
        # print(tokenizer.special_tokens_map)
        # print(tokenizer.additional_special_tokens)
        # print(tokenizer.additional_special_tokens_ids)
        # print('target_token_ids', target_token_ids)
        # for idx, token in enumerate(target_token_ids):
        #     decoded_token = tokenizer.decode([token.item()])
        #     print(f'Token ID: {token.item()} -> Decoded Token: "{decoded_token}"')
        #     if idx > 10:
        #         break
        # decoded_input = self._processor.decode(target_token_ids, skip_special_tokens=True)
        # print('Decoded target_text:', target_text)

        # Set padding token IDs to -100 to ignore in loss computation
        target_token_ids[target_token_ids == 0] = -100

        return ConditionalGenerationModelInput(
            **tokenized_instance,
            index=torch.tensor(document_instance.index),
            sample_id=document_instance.sample_id,
            words=words,
            target_text=target_text,
            target_token_ids=target_token_ids,
            image_size=document_instance.image.load().content.size,
            _tokenizer_name=self.tokenizer_name,
            _tokenizer_init_kwargs=self._default_init_kwargs,
        )

    def __call__(
        self, document_instance: DocumentInstance
    ) -> ConditionalGenerationModelInput | list[ConditionalGenerationModelInput]:
        # prepare prompt based on task type
        annotations = _extract_annotations(document_instance)
        if self.task_type == TaskType.sequence_classification:
            return self._prepare_instances_for_sequence_classification(
                document_instance, annotations.label
            )
        elif self.task_type == TaskType.token_classification:
            return self._prepare_instances_for_token_classification(
                document_instance, annotations.word_labels
            )
        elif self.task_type == TaskType.extractive_qa:
            return self._prepare_instances_for_question_answering(
                document_instance, annotations.qa_pairs
            )
        elif self.task_type == TaskType.layout_analysis:
            return self._prepare_instances_for_layout_analysis(
                document_instance, annotations.annotated_objects
            )
        else:
            raise NotImplementedError(f"Task type {self.task_type} not supported.")

    def __repr__(self) -> str:
        return f"ConditionalGenerationTokenizer(task_type={self.task_type}, is_training={self.is_training})"

    def __str__(self) -> str:
        return f"ConditionalGenerationTokenizer(task_type={self.task_type}, is_training={self.is_training})"
