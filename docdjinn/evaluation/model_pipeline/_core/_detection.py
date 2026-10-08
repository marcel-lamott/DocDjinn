from __future__ import annotations

import os

import torch
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules
from mmengine.config import Config
from torch.nn import Module

from docdjinn.data._core._data_types import DatasetMetadata, MMDetInput
from docdjinn.evaluation.model_pipeline._core._data_types import MMDetEvaluationOutput
from docdjinn.logging import get_logger

from ._base import ModelPipeline

logger = get_logger(__name__)

register_all_modules()


class DetectionPipeline(ModelPipeline):
    def __init__(
        self,
        model_name: str,
        model_cache_dir: str,
        dataset_metadata: DatasetMetadata,
    ):
        super().__init__(
            model_name=model_name,
            model_cache_dir=model_cache_dir,
            dataset_metadata=dataset_metadata,
        )

    def _build_model(
        self, model_name: str, model_cache_dir: str, pretrained: bool = True, **kwargs
    ) -> Module:
        from mmengine.runner import load_checkpoint

        cfg_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mmdet_cfgs/configs"
        )
        available_cfgs = {
            os.path.splitext(f)[0]: os.path.join(cfg_dir, f)
            for f in os.listdir(cfg_dir)
            if f.endswith(".py")
        }

        if model_name not in available_cfgs:
            raise ValueError(
                f"Unsupported model name: {model_name}. "
                f"Available options: {list(available_cfgs.keys())}"
            )

        path = available_cfgs[model_name]

        config = Config.fromfile(path)
        config.model.data_preprocessor.bgr_to_rgb = False

        # reconfigure given the dataset
        num_classes = len(self._dataset_metadata.dataset_labels.layout)
        if "roi_head" in config.model:
            if isinstance(config.model.roi_head.bbox_head, list):
                # cascade-rcnn
                for head in config.model.roi_head.bbox_head:
                    head.num_classes = num_classes
            else:
                # faster-rcnn
                config.model.roi_head.bbox_head.num_classes = num_classes
        elif "bbox_head" in config.model:
            # yolov3
            config.model.bbox_head.num_classes = num_classes
        else:
            raise Exception(
                "Could not automatically locate bbox_head — check model structure."
            )

        logger.info("Building model from config:")
        logger.info(config.pretty_text)

        model = MODELS.build(config.model)
        load_checkpoint(model, config.model_path)
        model._is_init = True

        # # reinit head layers
        # if "roi_head" in config.model:
        #     if isinstance(config.model.roi_head.bbox_head, list):
        #         # cascade-rcnn
        #         for head in model.roi_head.bbox_head:
        #             head.init_weights()
        #     else:
        #         # faster-rcnn
        #         model.roi_head.bbox_head.init_weights()
        # elif "bbox_head" in config.model:
        #     # yolov3
        #     model.bbox_head.init_weights()

        return model

    def training_step(self, batch: MMDetInput, **kwargs) -> MMDetEvaluationOutput:  # type: ignore
        batch_dict = {
            "inputs": batch.inputs,
            "data_samples": batch.data_samples,
        }
        batch_dict = self._model.data_preprocessor(batch_dict, training=True)
        losses = self._model._run_forward(batch_dict, mode="loss")
        loss, loss_dict = self._model.parse_losses(losses)

        return MMDetEvaluationOutput(
            loss=loss,
            loss_dict=loss_dict,
            det_data_samples=batch.data_samples,
            class_labels=[
                label.upper() for label in self._dataset_metadata.dataset_labels.layout
            ],
        )

    def evaluation_step(  # type: ignore
        self, batch: MMDetInput, stage="validation", **kwargs
    ) -> MMDetEvaluationOutput:
        batch_dict = {
            "inputs": batch.inputs,
            "data_samples": batch.data_samples,
        }

        if stage == "validation":
            det_data_samples = self._model.val_step(batch_dict)
        elif stage == "test":
            det_data_samples = self._model.test_step(batch_dict)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        return MMDetEvaluationOutput(
            loss=torch.tensor(0.0),
            det_data_samples=det_data_samples,
            class_labels=[
                label.upper() for label in self._dataset_metadata.dataset_labels.layout
            ],
        )
