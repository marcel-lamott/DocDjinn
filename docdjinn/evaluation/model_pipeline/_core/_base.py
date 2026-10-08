from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

from atria_core.types import DatasetMetadata
from atria_core.utilities.repr import RepresentationMixin

from docdjinn.data._core._data_types import (
    BaseModelInput,
)
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    import torch
    from atria_core.types import (
        DatasetMetadata,
        ModelOutput,
    )

logger = get_logger(__name__)


class ModelPipeline(ABC, RepresentationMixin):
    def __init__(self, dataset_metadata: DatasetMetadata | None = None, **kwargs):
        import ignite.distributed as idist

        super().__init__()
        self._dataset_metadata = dataset_metadata

        if idist.get_rank() > 0:  # Stop all ranks > 0
            idist.barrier()

        self._model_name = kwargs.get("model_name", "unknown_model")
        self._model = self._build_model(dataset_metadata=dataset_metadata, **kwargs)
        self._possible_args = inspect.signature(self._model.forward).parameters

        if idist.get_rank() == 0:
            idist.barrier()

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def to(self, device: str | torch.device) -> Self:
        self._model.to(device)
        return self

    def train(self):
        self._model.train()
        return self

    def eval(self):
        self._model.eval()
        return self

    def half(self):
        self._model.half()
        return self

    def state_dict(self) -> dict:
        state_dict = {}
        if self._dataset_metadata is not None:
            state_dict["dataset_metadata"] = self._dataset_metadata.state_dict()
        state_dict["model"] = self.model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        if "dataset_metadata" in state_dict:
            self._dataset_metadata.load_state_dict(state_dict["dataset_metadata"])
        logger.info("Loading model state dict.")
        self._model.load_state_dict(state_dict["model"], strict=True)

    @abstractmethod
    def training_step(self, batch: BaseModelInput, **kwargs) -> ModelOutput:
        pass

    @abstractmethod
    def evaluation_step(
        self,
        batch: BaseModelInput,
        **kwargs,
    ) -> ModelOutput:
        pass

    @abstractmethod
    def _build_model(self, *args, **kwargs) -> torch.nn.Module:
        pass

    def __repr__(self):
        from torchinfo import summary

        return f"{self.__class__.__name__} ({self._model_name}):\n{str(summary(self._model, verbose=0, depth=3))}"

    def __str__(self):
        return self.__repr__()
