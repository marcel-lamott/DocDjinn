from __future__ import annotations

import torch
from ignite.engine import Engine

from docdjinn.data._core._data_types import BaseModelInput
from docdjinn.evaluation.model_pipeline import ModelPipeline


class EvaluationStep:
    def __init__(
        self, model_pipeline: ModelPipeline, device: str, stage: str = "validation"
    ):
        self.model_pipeline = model_pipeline
        self.device = device
        self.stage = stage

    def __call__(self, engine: "Engine", batch: "BaseModelInput"):
        self.model_pipeline.eval()

        with torch.no_grad():
            batch = batch.to(self.device)
            return self.model_pipeline.evaluation_step(
                batch=batch,
                stage=self.stage,
            )
