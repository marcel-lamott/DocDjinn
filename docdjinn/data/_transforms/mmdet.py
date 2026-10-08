from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.registry import TRANSFORMS
from PIL.Image import Image as PILImage
from pydantic import Field

from docdjinn.data._core._data_types import (
    DocumentInstance,
    LayoutAnalysisAnnotation,
    MMDetInput,
)
from docdjinn.data._transforms._generics._base import (
    BaseTransform as DocDjinnBaseTransform,
)
from docdjinn.logging import get_logger

logger = get_logger(__name__)


@TRANSFORMS.register_module()
class RandomChoiceResize(BaseTransform):
    def __init__(self, scales: Sequence[int | tuple], **resize_kwargs) -> None:
        super().__init__()

        import mmengine
        from mmdet.datasets.transforms import Resize

        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert mmengine.is_seq_of(self.scales, (tuple, int))
        self.resize = Resize(scale=0, backend="pillow", **resize_kwargs)

    @cache_randomness
    def _random_select(self) -> tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()
        self.resize.scale = target_scale
        results = self.resize(results)
        results["scale_idx"] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scales={self.scales}"
        repr_str += f", resize={self.resize})"
        return repr_str


class DocumentInstanceMMDetTransform(DocDjinnBaseTransform[MMDetInput]):
    train_scale: list[tuple[int, int]] | tuple[int, int] = Field(
        default=[
            (480, 1333),
            (512, 1333),
            (800, 1333),
        ],
        description="Scale for training images.",
    )
    test_scale: list[tuple[int, int]] | tuple[int, int] = Field(
        default=(1333, 800), description="Scale for testing images."
    )
    is_training: bool = Field(
        default=False, description="Whether the transform is used for training."
    )
    use_test_time_augmentation: bool = Field(
        default=False, description="Whether to use test time augmentation."
    )
    use_flip: bool = Field(
        default=False, description="Whether to use flip augmentation during testing."
    )
    use_fixed_size: bool = Field(
        default=False, description="Whether to use fixed size resizing."
    )
    fixed_size: int = Field(
        default=800, description="Fixed size to resize the shorter side to."
    )

    def get_output_data_model(self) -> type[MMDetInput]:
        return MMDetInput

    def model_post_init(self, context) -> None:
        import torchvision.transforms as T

        # self._transform = T.Compose([])
        # return
        from mmdet.datasets.transforms import (
            LoadAnnotations,
            PackDetInputs,
            RandomFlip,
            Resize,
        )

        if self.is_training:
            # from mmcv.transforms import RandomChoiceResize, TestTimeAug
            from mmcv.transforms import TestTimeAug

            train_scale = self.train_scale
            if isinstance(self.train_scale, tuple):
                train_scale = [self.train_scale]

            self._transform = T.Compose(
                [
                    LoadAnnotations(with_bbox=True, with_mask=False, box_type=None),
                    Resize(scale=self.fixed_size, keep_ratio=False)
                    if self.use_fixed_size
                    else RandomChoiceResize(
                        scales=train_scale, keep_ratio=self.use_fixed_size is False
                    ),
                    *([RandomFlip(prob=0.5)] if self.use_flip else []),
                    PackDetInputs(
                        meta_keys=(
                            "id",
                            "img_id",
                            "img_path",
                            "ori_shape",
                            "img_shape",
                            "scale_factor",
                            "flip",
                            "flip_direction",
                        )
                    ),
                ]
            )
        else:
            from mmcv.transforms import TestTimeAug

            if self.use_test_time_augmentation:
                if isinstance(self.test_scale, tuple):
                    test_scale = [self.test_scale]
                self._transform = T.Compose(
                    [
                        LoadAnnotations(with_bbox=True, with_mask=False, box_type=None),
                        TestTimeAug(
                            transforms=[
                                [
                                    RandomChoiceResize(
                                        scales=test_scale, keep_ratio=True
                                    )
                                ],
                                [RandomFlip(prob=0.0), RandomFlip(prob=1.0)],
                                [
                                    PackDetInputs(
                                        meta_keys=(
                                            "__key__",
                                            "__index__",
                                            "img_id",
                                            "img_path",
                                            "ori_shape",
                                            "img_shape",
                                            "scale_factor",
                                            "flip",
                                            "flip_direction",
                                        )
                                    )
                                ],
                            ]
                        ),
                    ]
                )
            else:
                import torchvision.transforms as T
                from mmcv.transforms import TestTimeAug

                if isinstance(self.test_scale, list):
                    test_scale = self.test_scale[0]
                else:
                    test_scale = self.test_scale

                self._transform = T.Compose(
                    [
                        LoadAnnotations(with_bbox=True, with_mask=False, box_type=None),
                        Resize(
                            scale=self.fixed_size,
                            keep_ratio=self.use_fixed_size is False,
                        ),
                        PackDetInputs(
                            meta_keys=(
                                "__key__",
                                "__index__",
                                "img_id",
                                "img_path",
                                "ori_shape",
                                "img_shape",
                                "scale_factor",
                                "flip",
                                "flip_direction",
                            )
                        ),
                    ]
                )

    def _extract_annotated_objects(self, document_instance: DocumentInstance):
        assert document_instance.annotations is not None, (
            f"Document instance must have annotations for {self.__class__} ."
        )
        layout_annotations = None
        for annotation in document_instance.annotations:
            if isinstance(annotation, LayoutAnalysisAnnotation):
                layout_annotations = annotation.annotated_objects
                break
        assert layout_annotations is not None, (
            f"Document instance must have layout annotations for {self.__class__}."
        )
        return layout_annotations

    def _get_image(self, document_instance: DocumentInstance) -> PILImage:
        assert document_instance.image is not None, (
            "DocumentInstance image must be loaded before applying transforms."
        )
        assert isinstance(document_instance.image.content, PILImage), (
            "DocumentInstance image content must be a PIL Image."
        )
        return document_instance.image.content

    def _is_valid_bbox(
        self, bbox: list[float], image_width: int, image_height: int
    ) -> bool:
        x1, y1, x2, y2 = bbox
        if 0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height:
            return True
        if (x2 - x1) > 1 and (y2 - y1) > 1:
            return True
        return False

    def _unnormalize_bbox(
        self, bbox: list[float], image_width: int, image_height: int
    ) -> list[float]:
        x1, y1, x2, y2 = bbox
        return [
            x1 * image_width,
            y1 * image_height,
            x2 * image_width,
            y2 * image_height,
        ]

    def _clip_bbox(
        self, bbox: list[float], image_width: int, image_height: int
    ) -> list[float]:
        x1, y1, x2, y2 = bbox
        x1 = min(max(x1, 0), image_width - 1)
        x2 = min(max(x2, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        y2 = min(max(y2, 0), image_height - 1)
        return [x1, y1, x2, y2]

    def _prepare_instances(
        self, document_instance: DocumentInstance, image_width: int, image_height: int
    ) -> list[dict]:
        annotated_objects = self._extract_annotated_objects(document_instance)

        instances = []
        is_bbox_normalized = annotated_objects.bbox.normalized
        for bbox, label, iscrowd in zip(
            annotated_objects.bbox.value,
            annotated_objects.label.value,
            annotated_objects.iscrowd,
            strict=True,
        ):
            if is_bbox_normalized:
                bbox = self._unnormalize_bbox(bbox, image_width, image_height)

            # first clip the bbox to be within image bounds, then check validity
            bbox = self._clip_bbox(bbox, image_width, image_height)

            if not self._is_valid_bbox(bbox, image_width, image_height):
                logger.warning(
                    f"Invalid bbox {bbox} for image of size ({image_width}, {image_height}) in document instance {document_instance.sample_id}. Skipping this bbox."
                )
                continue

            instance = {
                "bbox": [float(coord) for coord in bbox],
                "bbox_label": label,
                "ignore_flag": 1 if iscrowd else 0,
            }
            instances.append(instance)

        return instances

    def _apply_transforms(self, document_instance: DocumentInstance) -> MMDetInput:
        image = self._get_image(document_instance)
        output = self._transform(
            {
                "id": document_instance.sample_id,
                "img_id": document_instance.index,
                "instances": self._prepare_instances(
                    document_instance,
                    image_width=image.width,
                    image_height=image.height,
                ),
                "img": np.array(image),
                "img_shape": (
                    image.height,
                    image.width,
                ),
                "ori_shape": (
                    image.height,
                    image.width,
                ),
            }
        )
        output = MMDetInput(**output)

        return output

    def __call__(self, document_instance: DocumentInstance):
        return self._apply_transforms(document_instance)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  is_training={self.is_training},\n"
            f"  use_test_time_augmentation={self.use_test_time_augmentation},\n"
            f"  transform={self._transform},\n"
            f")"
        )
