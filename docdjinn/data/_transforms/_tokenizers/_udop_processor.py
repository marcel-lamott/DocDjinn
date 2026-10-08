# verify input
# patch udop processor to return word and sequence ids 
from transformers.models.udop.processing_udop import UdopProcessor, UdopProcessorKwargs
from typing import List, Optional, Union

from transformers import logging

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput



class CustomUdopProcessor(UdopProcessor):
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        *args,
        audio=None,
        videos=None,
        **kwargs: Unpack[UdopProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            UdopProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )

        boxes = output_kwargs["text_kwargs"].pop("boxes", None)
        word_labels = output_kwargs["text_kwargs"].pop("word_labels", None)
        text_pair = output_kwargs["text_kwargs"].pop("text_pair", None)
        return_overflowing_tokens = output_kwargs["text_kwargs"].get("return_overflowing_tokens", False)
        return_offsets_mapping = output_kwargs["text_kwargs"].get("return_offsets_mapping", False)
        text_target = output_kwargs["text_kwargs"].get("text_target", None)

        if self.image_processor.apply_ocr and (boxes is not None):
            raise ValueError(
                "You cannot provide bounding boxes if you initialized the image processor with apply_ocr set to True."
            )

        if self.image_processor.apply_ocr and (word_labels is not None):
            raise ValueError(
                "You cannot provide word labels if you initialized the image processor with apply_ocr set to True."
            )

        if return_overflowing_tokens and not return_offsets_mapping:
            raise ValueError("You cannot return overflowing tokens without returning the offsets mapping.")

        if text_target is not None:
            # use the processor to prepare the targets of UDOP
            return self.tokenizer(
                **output_kwargs["text_kwargs"],
            )

        else:
            # use the processor to prepare the inputs of UDOP
            # first, apply the image processor
            features = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            features_words = features.pop("words", None)
            features_boxes = features.pop("boxes", None)

            output_kwargs["text_kwargs"].pop("text_target", None)
            output_kwargs["text_kwargs"].pop("text_pair_target", None)
            output_kwargs["text_kwargs"]["text_pair"] = text_pair
            output_kwargs["text_kwargs"]["boxes"] = boxes if boxes is not None else features_boxes
            output_kwargs["text_kwargs"]["word_labels"] = word_labels

            # second, apply the tokenizer
            if text is not None and self.image_processor.apply_ocr and text_pair is None:
                if isinstance(text, str):
                    text = [text]  # add batch dimension (as the image processor always adds a batch dimension)
                output_kwargs["text_kwargs"]["text_pair"] = features_words

            encoded_inputs = self.tokenizer(
                text=text if text is not None else features_words,
                **output_kwargs["text_kwargs"],
            )

            # add pixel values
            if return_overflowing_tokens is True:
                features["pixel_values"] = self.get_overflowing_images(
                    features["pixel_values"], encoded_inputs["overflow_to_sample_mapping"]
                )
            features.update(encoded_inputs)

            return features, encoded_inputs