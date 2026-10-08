from __future__ import annotations

import torch
from ignite.engine import Engine
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from docdjinn.data._core._data_types import BaseModelInput
from docdjinn.evaluation.model_pipeline import ModelPipeline


class TrainingStep:
    def __init__(
        self,
        model_pipeline: ModelPipeline,
        optimizer: "torch.optim.Optimizer",
        lr_scheduler: "torch.optim.lr_scheduler.LRScheduler",
        device: str,
        with_amp: bool,
        gradient_accumulation_steps: int,
        enable_grad_clipping: bool,
        max_grad_norm: float,
    ):
        self.model_pipeline = model_pipeline
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.with_amp = with_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_grad_clipping = enable_grad_clipping
        self.max_grad_norm = max_grad_norm

        # setup grad scaler for mixed precision training
        self.scaler = GradScaler(enabled=self.with_amp)

    def __call__(self, engine: "Engine", batch: "BaseModelInput"):
        from torch.nn.utils import clip_grad_norm_

        self.model_pipeline.train()
        batch = batch.to(self.device)

        if (engine.state.iteration - 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        with autocast(device_type=str(self.device), enabled=self.with_amp):
            outputs = self.model_pipeline.training_step(batch=batch)
            loss = outputs.loss

            # accumulate loss if required
            assert loss is not None, "Loss is None in training step."
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

        # backward pass with grad scaler
        self.scaler.scale(loss).backward()

        if engine.state.iteration % self.gradient_accumulation_steps == 0:
            if self.enable_grad_clipping:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(
                    self.model_pipeline.model.parameters(),
                    max_norm=self.max_grad_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
        return outputs
