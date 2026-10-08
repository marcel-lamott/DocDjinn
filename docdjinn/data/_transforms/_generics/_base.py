from __future__ import annotations

from typing import Generic, TypeVar

import torch
from atria_core.utilities.repr import RepresentationMixin
from PIL.Image import Image as PILImage
from pydantic import BaseModel

from docdjinn.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class ToRGB(object):
    def __call__(self, image: PILImage | torch.Tensor) -> PILImage | torch.Tensor:
        if isinstance(image, torch.Tensor):
            if image.shape[0] == 3:
                return image
            return image.repeat(3, 1, 1)
        else:
            return image.convert("RGB")


class BaseTransform(RepresentationMixin, BaseModel, Generic[T]):
    def get_output_data_model(self) -> type[T]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> T | list[T]:
        raise NotImplementedError
