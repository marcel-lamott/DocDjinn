from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any

import torch
from rapidfuzz import fuzz, process

from docdjinn.data._core._data_types import (
    ConditionalGenerationModelInput,
)
from docdjinn.evaluation.model_pipeline._core._data_types import (
    ClassificationModelOutput,
    ModelOutput,
    QAModelOutput,
    QAPair,
    TokenClassificationModelOutput,
)
from docdjinn.logging import get_logger

from ._base import ModelPipeline

logger = get_logger(__name__)


def convert_predictions_to_labels(predicted_text: str, words: list[str]) -> list[str]:
    predicted_entities = []
    for item in predicted_text:
        parts = item.split("|", 1)
        if len(parts) == 2:
            predicted_entities.append((parts[1], parts[0]))

    predicted_labels = ["O"] * len(words)
    last_word_index = -1
    for predicted_word, predicted_entity in predicted_entities:
        try:
            original_word_idx = words.index(predicted_word.strip(), last_word_index + 1)
            predicted_labels[original_word_idx] = predicted_entity
            last_word_index = original_word_idx
        except ValueError:
            continue  # skip if the word is not found

    return predicted_labels


def parse_key_value_pairs(entity_string, valid_tags):
    """
    Parses a string where each token is followed by its tag,
    e.g. "Hello typea world typeb my typea guy typec"

    Returns two lists: keys (tokens) and values (tags).
    Any tag not in valid_tags is replaced by "O".
    """

    tags = [x.replace("B-", "").replace("I-", "") for x in valid_tags]
    tags_str = "|".join(tags)
    pairs = re.findall(rf"(\S+?)\s((?:B-|I-)(?:{tags_str})|O)", entity_string)
    keys, values = [], []
    for key, value in pairs:
        if key.strip() == "":
            continue
        if value not in valid_tags:
            value = "O"
        keys.append(key)
        values.append(value)

    return keys, values


def _convert_entity_string_to_dict(
    entity_string: str, entity_str_type: str = "csv"
) -> dict:
    if entity_str_type == "json":
        import json

        return json.loads(entity_string)
    elif entity_str_type == "csv":
        predictions = entity_string.split("|")
        predicted_entities = {}
        for prediction in predictions:
            if "=" not in prediction:
                continue
            result = prediction.split("=")
            if len(result) != 2:
                continue
            entity_type, entity_value = result[0], result[1]
            if entity_type.strip() not in predicted_entities:
                predicted_entities[entity_type.strip()] = []
            predicted_entities[entity_type.strip()].append(entity_value.strip())
        return predicted_entities
    else:
        raise ValueError(f"Unsupported entity_str_type: {entity_str_type}")


# def convert_entity_to_word_labels(entities, words):
#     word_labels = ["O"] * len(words)
#     for _, key, text in entities:
#         result, _ = _find_text_spans_in_text_list(
#             [text.lower()], [w.lower() for w in words]
#         )
#         if len(result) == 0:
#             continue
#         result = result[0]
#         start_index = result["answer_start_index"]
#         end_index = result["answer_end_index"]
#         word_labels[start_index] = f"B-{key}"
#         word_labels[start_index + 1 : end_index + 1] = [f"I-{key}"] * (
#             end_index - start_index
#         )
#     return word_labels


class ConditionalGenerationPipeline(ModelPipeline):
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        dataset_metadata: DatasetMetadata,
    ):
        self._tokenizer = None

        super().__init__(
            model_name=model_name,
            dataset_metadata=dataset_metadata,
            model_cache_dir=model_cache_dir,
        )

    def _build_model(
        self,
        model_name: str,
        model_cache_dir: str,
        **kwargs,
    ) -> Module:
        from transformers import (
            AutoProcessor,
            T5ForConditionalGeneration,
            UdopForConditionalGeneration,
        )

        self._processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=model_cache_dir,
        )
        if model_name == "microsoft/udop-large":
            self._tokenizer = self._processor.tokenizer
            return UdopForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=model_cache_dir,
            )
        elif model_name in ["google-t5/t5-large", "google-t5/t5-base"]:
            self._tokenizer = self._processor
            return T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=model_cache_dir,
            )
        else:
            raise ValueError(
                f"Unsupported model name for ConditionalGenerationPipeline: {model_name}"
            )

    def training_step(
        self, batch: ConditionalGenerationModelInput, **kwargs
    ) -> ModelOutput:
        # assume token bboxes are in [0, 1] range
        token_bboxes = (
            (batch.bbox * 1000.0).clip(0, 1000).long()
            if batch.bbox is not None
            else None
        )
        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "bbox": token_bboxes,
            "pixel_values": batch.pixel_values,
            "labels": batch.target_token_ids,
        }
        # for key, value in inputs.items():
        #     print(key, value.shape if isinstance(value, torch.Tensor) else value)
        # test = inputs["labels"].clone()
        # test[test == -100] = self._tokenizer.pad_token_id
        # decoded_labels = self._tokenizer.batch_decode(
        #     test, skip_special_tokens=False
        # )
        # print('decoded labels:', decoded_labels)

        for key in list(inputs.keys()):
            if key not in self._possible_args:
                inputs.pop(key)

        assert inputs["attention_mask"] is not None, "Attention mask cannot be None"
        output = self._model(**inputs)
        return ModelOutput(loss=output.loss)

    def evaluation_step(
        self, batch: ConditionalGenerationModelInput, **kwargs
    ) -> ModelOutput:
        # assume token bboxes are in [0, 1] range
        token_bboxes = (
            (batch.bbox * 1000.0).clip(0, 1000).long()
            if batch.bbox is not None
            else None
        )
        inputs = {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "bbox": token_bboxes,
            "pixel_values": batch.pixel_values,
        }
        for key in list(inputs.keys()):
            if key not in self._possible_args:
                inputs.pop(key)
        # print('input text',inputs['input_ids'])
        # decoded_input_texts = self._processor.batch_decode(
        #     inputs['input_ids'], skip_special_tokens=False
        # )
        # print('decoded input texts:', decoded_input_texts)
        assert inputs["attention_mask"] is not None, "Attention mask cannot be None"
        predicted_texts = self._generate_predictions(inputs)
        print("Evaluation step, predicted_texts:", predicted_texts)
        return self._prepare_evaluation_output(batch, predicted_texts)

    def _generate_predictions(self, inputs: dict[str, Any], **kwargs) -> list[str]:
        predicted_ids = self._model.generate(**inputs, max_length=2048)
        predicted_texts = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return predicted_texts

    @abstractmethod
    def _prepare_evaluation_output(
        self, batch: ConditionalGenerationModelInput, predicted_texts: list[str]
    ) -> ModelOutput:
        pass


class GenerativeSequenceClassificationPipeline(ConditionalGenerationPipeline):
    def _prepare_evaluation_output(
        self, batch: ConditionalGenerationModelInput, predicted_texts: list[str]
    ) -> ModelOutput:
        # for classification the expected output is document class name
        possible_labels = self._dataset_metadata.dataset_labels.classification
        assert possible_labels is not None, (
            "Possible labels cannot be None for classification task"
        )
        assert batch.target_text is not None, (
            "Target text cannot be None for classification task"
        )
        target_label_values = [possible_labels.index(t) for t in batch.target_text]
        predicted_label_values = []
        for predicted_text in predicted_texts:
            predicted_label_name = predicted_text.strip()
            if predicted_label_name in possible_labels:
                predicted_label_value = possible_labels.index(predicted_text.strip())
            else:
                predicted_label_value = torch.randint(
                    0, len(possible_labels), (1,)
                ).item()
            predicted_label_values.append(predicted_label_value)

        print("predicted_label_values", predicted_label_values)
        print("target_label_values", target_label_values)

        # create logits from ground truth label values
        # initialize logits with small random values
        # this is a hack to pass the logits for evaluation metrics
        logits = torch.zeros((len(predicted_label_values), len(possible_labels)))

        # Set high probability for the correct labels
        for i, target_idx in enumerate(predicted_label_values):
            logits[i, target_idx] = 1.0

        return ClassificationModelOutput(
            loss=0.0,  # just pass a dummy value
            gt_label_value=torch.tensor(target_label_values),
            logits=logits,
        )

    def _generate_predictions(self, inputs: dict[str, Any], **kwargs) -> list[str]:
        predicted_ids = self._model.generate(**inputs, max_length=32)
        predicted_texts = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return predicted_texts


class GenerativeTokenClassificationPipeline(ConditionalGenerationPipeline):
    def _prepare_evaluation_output(
        self,
        batch,
        predicted_texts: list[str],
        match_threshold: int = 80,
    ):
        predicted_labels = []
        target_labels = []
        for words_per_sample, predicted_text_per_sample, word_labels_per_sample in zip(
            batch.words, predicted_texts, batch.word_labels
        ):
            predicted_words_per_sample, predicted_labels_per_sample = (
                parse_key_value_pairs(
                    predicted_text_per_sample, valid_tags=self._get_token_labels()
                )
            )

            predicted_labels_per_sample_cleaned = ["O"] * len(word_labels_per_sample)
            for word_idx, word in enumerate(words_per_sample):
                # Fuzzy match the predicted word to the remaining reference words
                result = process.extractOne(
                    query=word, choices=predicted_words_per_sample, scorer=fuzz.ratio
                )
                if result is None:
                    continue

                match, score, predicted_word_idx = result
                if score >= match_threshold:
                    predicted_words_per_sample.pop(
                        predicted_word_idx
                    )  # remove to prevent re-matching
                    predicted_labels_per_sample_cleaned[word_idx] = (
                        predicted_labels_per_sample.pop(predicted_word_idx)
                    )
                    print(
                        "Matched word:",
                        word,
                        "at index:",
                        word_idx,
                        "with label:",
                        predicted_labels_per_sample_cleaned[word_idx],
                        "target label:",
                        word_labels_per_sample[word_idx],
                    )
                else:
                    # No good match found, skip
                    continue
            # we remap all the target entities to the text to get the target labels in BIO format
            # this is necessary because the words are now per chunk of the input and not the whole document
            # therefore we find all entities in this chunk and assign the labels accordingly
            # target_labels_per_sample = ["O"] * len(words_per_sample)
            # for key, value in target_entities.items():
            #     for v in value:
            #         output = _find_text_spans_in_text_list(v, words_per_sample)
            #         start_idx, end_idx = (
            #             output["answer_start_index"],
            #             output["answer_end_index"],
            #         )
            #         target_labels_per_sample[start_idx] = "B-" + key
            #         target_labels_per_sample[start_idx + 1 : end_idx + 1] = ["I-" + key] * (
            #             end_idx - start_idx
            #         )

            # predicted_labels_per_sample = ["O"] * len(words_per_sample)
            # for key, value in extracted_entities.items():
            #     for v in value:
            #         output = _find_text_spans_in_text_list(v, words_per_sample)
            #         start_idx, end_idx = (
            #             output["answer_start_index"],
            #             output["answer_end_index"],
            #         )
            #         predicted_labels_per_sample[start_idx] = "B-" + key
            #         predicted_labels_per_sample[start_idx + 1 : end_idx + 1] = ["I-" + key] * (
            #             end_idx - start_idx
            #         )

            # # now we match the extracted entities with ground truth entities using fuzzy matching
            # # if the match is found above the threshold, we use the ground truth entity text for labeling
            # matches = []
            # for key in target_entities.keys():
            #     if key not in extracted_entities:
            #         continue
            #     for ext in extracted_entities[key]:
            #         # Find the best match in ground truth
            #         match, score, idx = process.extractOne(
            #             ext, target_entities[key], scorer=fuzz.ratio
            #         )
            #         if score >= match_threshold:
            #             matches.append((key, match))  # matched

            # predicted_labels_per_sample = ["O"] * len(words_per_sample)
            # for m in matches:
            #     output = _find_text_spans_in_text_list(m[1], words_per_sample)
            #     start_idx, end_idx = (
            #         output["answer_start_index"],
            #         output["answer_end_index"],
            #     )
            #     predicted_labels_per_sample[start_idx] = "B-" + m[0]
            #     predicted_labels_per_sample[start_idx + 1 : end_idx + 1] = ["I-" + m[0]] * (
            #         end_idx - start_idx
            #     )

            # predicted_labels_per_sample_padded = ["O"] * len(target_labels_per_sample)
            # for idx, label in enumerate(predicted_labels_per_sample):
            #     if idx >= len(predicted_labels_per_sample_padded):
            #         break
            #     predicted_labels_per_sample_padded[idx] = label

            target_labels.append(word_labels_per_sample)
            predicted_labels.append(predicted_labels_per_sample_cleaned)

        print("p_labels", predicted_labels)
        print("t_labels", target_labels)
        return TokenClassificationModelOutput(
            loss=0.0,
            predicted_label_names=predicted_labels,
            target_label_names=target_labels,
        )

    # def _prepare_evaluation_output(
    #     self, batch: ConditionalGenerationModelInput, predicted_texts: list[str]
    # ) -> ModelOutput:
    #     predicted_labels = []
    #     id_to_label_map = {
    #         idx: label for idx, label in enumerate(self._get_token_labels())
    #     }
    #     for words_per_sample, predicted_text_per_sample in zip(
    #         batch.words, predicted_texts
    #     ):
    #         predicted_labels_per_sample = convert_predictions_to_labels(
    #             predicted_text_per_sample,
    #             words_per_sample,
    #         )
    #         predicted_labels.append(predicted_labels_per_sample)
    #     return TokenClassificationModelOutput(
    #         loss=0.0,
    #         predicted_label_names=predicted_labels,
    #         target_label_names=batch.word_labels,
    #     )

    def _get_token_labels(self) -> list[str]:
        return self._dataset_metadata.dataset_labels.ser

    # def _build_model(
    #     self,
    #     model_name: str,
    #     model_cache_dir: str,
    #     **kwargs,
    # ) -> Module:
    #     model = super()._build_model(
    #         model_name=model_name, model_cache_dir=model_cache_dir, **kwargs
    #     )
    #     possible_labels = self._get_token_labels()
    #     possible_labels = [f"<{lbl}>" for lbl in possible_labels]
    #     num_added_tokens = self._tokenizer.add_special_tokens({"additional_special_tokens": possible_labels})
    #     logger.info(f"Added {num_added_tokens} special tokens for entity labels: {possible_labels}")
    #     model.resize_token_embeddings(len(self._tokenizer))
    #     return model

    def _generate_predictions(self, inputs: dict[str, Any], **kwargs) -> list[str]:
        predicted_ids = self._model.generate(**inputs, max_length=1024)
        predicted_texts = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return predicted_texts


class GenerativeQuestionAnsweringPipeline(ConditionalGenerationPipeline):
    def _prepare_evaluation_output(
        self, batch: ConditionalGenerationModelInput, predicted_texts: list[str]
    ) -> ModelOutput:
        qa_outputs = []
        for sample_id, predicted_answer, question_text in zip(
            batch.sample_id, predicted_texts, batch.question_text
        ):
            if "_page_" in sample_id:
                sample_id = sample_id.split("_page_")[0]  # get the original sample id
            else:
                sample_id = sample_id.split("_subsample_")[
                    0
                ]  # get the original sample id
            qa_outputs.append(
                QAPair(
                    sample_id=sample_id, question=question_text, answer=predicted_answer
                )
            )

        return QAModelOutput(loss=0.0, qa_pairs=qa_outputs)

    def _generate_predictions(self, inputs: dict[str, Any], **kwargs) -> list[str]:
        predicted_ids = self._model.generate(**inputs, max_length=128)
        predicted_texts = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return predicted_texts


class GenerativeLayoutAnalysisPipeline(ConditionalGenerationPipeline):
    def _prepare_evaluation_output(
        self, batch: ConditionalGenerationModelInput, predicted_texts: list[str]
    ) -> ModelOutput:
        # # we extract all entities from the predicted text
        # extracted_layout_entities = []

        # # get image shape
        # image_width, image_height = batch.image.size[0], batch.image.size[1]
        # for predicted_text_per_sample in predicted_texts:
        #     pattern = r"<(\d+)><(\d+)><(\d+)><(\d+)><([^>]+)>"
        #     entities_per_sample = re.findall(pattern, predicted_text_per_sample)
        #     for (x1, y1, x2, y2, label) in entities_per_sample:
        #         extracted_layout_entities.append(
        #             (int(x1), int(y1), int(x2), int(y2), label)
        #         )

        #     print('matches',matches)
        return ModelOutput(loss=0.0)
