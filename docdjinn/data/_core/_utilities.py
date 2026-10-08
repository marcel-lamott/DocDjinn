import enum
from typing import Any

from docdjinn.data._core._data_types import BaseModelInput
from docdjinn.logging import get_logger

logger = get_logger(__name__)


class TaskType(str, enum.Enum):
    generate_embeddings = "generate_embeddings"
    sequence_classification = "sequence_classification"
    token_classification = "token_classification"
    extractive_qa = "extractive_qa"
    layout_analysis = "layout_analysis"
    table_extraction = "table_extraction"
    table_detection = "table_detection"


def auto_dataloader(dataset: Any, **kwargs: Any) -> Any:
    """
    Automatically configures a DataLoader for distributed training.

    This function adjusts DataLoader settings based on the distributed training configuration,
    including rank, world size, and device type. It supports XLA devices and provides warnings
    for incompatible configurations.

    Args:
        iterator (Iterator): The dataset split iterator to load data from.
        **kwargs (Any): Additional arguments for configuring the DataLoader.

    Returns:
        DataLoader: A configured DataLoader instance.

    Raises:
        ValueError: If incompatible configurations are detected.
    """
    from ignite.distributed import DistributedProxySampler
    from ignite.distributed import utils as idist
    from ignite.distributed.comp_models import xla as idist_xla
    from torch.utils.data import DataLoader, IterableDataset
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.sampler import Sampler

    rank = idist.get_rank()
    world_size = idist.get_world_size()

    if world_size > 1:
        if "batch_size" in kwargs and kwargs["batch_size"] >= world_size:
            kwargs["batch_size"] //= world_size

        nproc = idist.get_nproc_per_node()
        if "num_workers" in kwargs and kwargs["num_workers"] >= nproc:
            kwargs["num_workers"] = (kwargs["num_workers"] + nproc - 1) // nproc

        if "batch_sampler" not in kwargs:
            if isinstance(dataset, IterableDataset):
                logger.info(
                    "Found iterable dataset, dataloader will be created without any distributed sampling. "
                    "Please, make sure that the dataset itself produces different data on different ranks."
                )
            else:
                sampler: DistributedProxySampler | DistributedSampler | Sampler | None
                sampler = kwargs.get("sampler", None)
                if isinstance(sampler, DistributedSampler):
                    if sampler.rank != rank:
                        logger.warning(
                            f"Found distributed sampler with rank={sampler.rank}, but process rank is {rank}"
                        )
                    if sampler.num_replicas != world_size:
                        logger.warning(
                            f"Found distributed sampler with num_replicas={sampler.num_replicas}, "
                            f"but world size is {world_size}"
                        )
                elif sampler is None:
                    shuffle = kwargs.pop("shuffle", True)
                    sampler = DistributedSampler(
                        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
                    )
                else:
                    sampler = DistributedProxySampler(
                        sampler, num_replicas=world_size, rank=rank
                    )
                kwargs["sampler"] = sampler
        else:
            logger.warning(
                "Found batch_sampler in provided kwargs. Please, make sure that it is compatible "
                "with distributed configuration"
            )

    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and kwargs.get("pin_memory", False)
    ):
        logger.warning(
            "Found incompatible options: xla support and pin_memory args equal True. "
            "Argument `pin_memory=False` will be used to construct data loader."
        )
        kwargs["pin_memory"] = False
    else:
        kwargs["pin_memory"] = kwargs.get("pin_memory", "cuda" in idist.device().type)

    dataloader = DataLoader(dataset, **kwargs)
    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and world_size > 1
    ):
        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        from torch_xla.distributed.parallel_loader import MpDeviceLoader  # type: ignore

        mp_device_loader_cls = MpDeviceLoader
        mp_dataloader = mp_device_loader_cls(dataloader, idist.device())
        mp_dataloader.sampler = dataloader.sampler  # type: ignore[attr-defined]
        return mp_dataloader

    return dataloader


def default_collate(list_of_inputs: list[BaseModelInput] | list[list[BaseModelInput]]):
    if isinstance(list_of_inputs[0], list):
        list_of_inputs = [item for sublist in list_of_inputs for item in sublist]
    if isinstance(list_of_inputs, list) and len(list_of_inputs) > 0:
        return list_of_inputs[0].batch(list_of_inputs)
    else:
        raise ValueError("Batch is empty or not a list.")


def mmdet_pseudo_collate(batch: list["MMDetInput"]):
    from atria_datasets.core.transforms.mmdet import MMDetInput
    from mmengine.dataset.utils import pseudo_collate

    return MMDetInput.model_construct(
        **pseudo_collate([sample.model_dump() for sample in batch])
    )
