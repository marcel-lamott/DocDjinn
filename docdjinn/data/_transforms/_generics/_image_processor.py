from __future__ import annotations

from PIL.Image import Image as PILImage

from docdjinn.data._transforms._generics._base import BaseTransform, ToRGB
from docdjinn.logging import get_logger

logger = get_logger(__name__)


class ImageProcessor(BaseTransform[PILImage]):
    do_normalize: bool = True  # Normalize the image to ImageNet mean and std
    do_resize: bool = True  # Resize the image to 224x224
    use_imagenet_mean_std: bool = False
    resize_height: int = 224
    resize_width: int = 224
    image_mean: list[float] | None = None
    image_std: list[float] | None = None

    def model_post_init(self, context) -> None:
        from transformers.utils.constants import (
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD,
            IMAGENET_STANDARD_MEAN,
            IMAGENET_STANDARD_STD,
        )

        self.image_mean = self.image_mean or IMAGENET_STANDARD_MEAN
        self.image_std = self.image_std or IMAGENET_STANDARD_STD
        if self.use_imagenet_mean_std:
            self.image_mean = IMAGENET_DEFAULT_MEAN
            self.image_std = IMAGENET_DEFAULT_STD

        # prepare image transform
        self._transform = self._prepare_image_transform()

    def _prepare_image_transform(self):
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor

        transform = [ToRGB(), ToTensor()]
        if self.do_resize:
            transform += [
                Resize(
                    (self.resize_height, self.resize_width),
                    interpolation=2,  # type: ignore[attr-defined]
                    antialias=True,  # type: ignore[attr-defined]
                ),
            ]
        if self.do_normalize:
            transform += [
                Normalize(mean=self.image_mean, std=self.image_std),
            ]
        transform = Compose(transform)
        return transform

    def get_output_data_model(self) -> type[PILImage]:
        return PILImage

    def __call__(self, image: PILImage) -> PILImage:
        return self._transform(image)
