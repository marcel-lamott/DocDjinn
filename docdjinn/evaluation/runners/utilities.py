from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import Metric

from docdjinn.data._core._data_types import DatasetLabels
from docdjinn.data._core._utilities import TaskType
from docdjinn.data.interfaces.dataset import get_dataset_config
from docdjinn.evaluation.model_pipeline import ModelPipeline
from docdjinn.evaluation.runners._config import RunnerConfig
from docdjinn.evaluation.runners.wandb_logger import WandBLogger
from docdjinn.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME_KEY = "experiment_name"
METRICS_KEY = "metrics"
TRAINING_ENGINE_KEY = "training_engine"
MODEL_PIPELINE_CHECKPOINT_KEY = "model_pipeline"
RUN_CONFIG_KEY = "run_config"


def _reset_random_seeds(seed):
    """
    Resets random seeds for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set for random number generation.

    Libraries affected:
        - random: Python's built-in random module.
        - numpy: NumPy library for numerical computations.
        - torch: PyTorch library for deep learning.
    """
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _initialize_torch(seed: int = 0, deterministic: bool = False):
    """
    Initializes PyTorch settings, including random seeds and deterministic behavior.

    Args:
        seed (int, optional): The base seed value for random number generation. Defaults to 0.
        deterministic (bool, optional): Whether to enforce deterministic behavior for reproducibility. Defaults to False.

    Behavior:
        - Sets the global seed for reproducibility.
        - Configures PyTorch's CuDNN backend for deterministic or performance-optimized behavior.
    """

    import torch

    _reset_random_seeds(seed)

    # Configure CuDNN backend for deterministic behavior if required
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return seed


def _log_system_info():
    """Log system information including hardware, memory, and GPU details."""
    import platform
    import sys

    import ignite.distributed as idist
    import psutil
    import torch

    logger.info("SYSTEM INFORMATION")
    logger.info("-" * 30)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info(
        f"CPU count: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )

    # Memory information
    memory = psutil.virtual_memory()
    logger.info(f"Total RAM: {memory.total / 1024**3:.2f} GB")
    logger.info(f"Available RAM: {memory.available / 1024**3:.2f} GB")
    logger.info(f"RAM usage: {memory.percent}%")

    # GPU information
    if torch.cuda.is_available():
        logger.info("World size: %d", idist.get_world_size())
        logger.info("Rank: %d", idist.get_rank())
        logger.info("CUDA available: True")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {gpu_props.name} ({gpu_props.total_memory / 1024**3:.2f} GB)"
            )
    else:
        logger.info("CUDA available: False")

    logger.info(f"PyTorch version: {torch.__version__}")


def _log_run_configuration(config: RunnerConfig):
    """Log the run configuration details."""
    import yaml

    logger.info("-" * 30)
    logger.info("RUN CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Run name: {config.run_name}")
    logger.info(f"Output directory: {config.runs_dir}")
    logger.info(f"Full config:\n{yaml.dump(config.dict(), indent=4)}")


def _initialize_wandb(
    project: str,
    id: str,
    name: str,
    config: dict,
    tags: list[str],
    runs_dir: str,
    resume: bool = False,
) -> WandBLogger | None:
    """Initialize wandb logging if available."""
    import ignite.distributed as idist

    if idist.get_rank() == 0:
        try:
            wandb_logger = WandBLogger(
                project=project,
                id=id,
                name=name,
                config=config,
                tags=tags,
                dir=runs_dir,  # Add this line to specify wandb directory
                resume="must" if resume else None,
            )
            logger.info("Wandb logging initialized")
            return wandb_logger
        except ImportError:
            logger.warning("wandb not available, skipping initialization")


def _find_resume_checkpoint_in_dir(output_dir: str | Path) -> str | None:
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return

    checkpoint_files = [
        str(ckpt).split("_")[-1].replace(".pt", "")
        for ckpt in list(output_dir.glob("epoch_checkpoint_*.pt"))
    ]

    epochs = [int(ckpt.split("_")[-1].replace(".pt", "")) for ckpt in checkpoint_files]

    checkpoint_files = [ckpt for _, ckpt in sorted(zip(epochs, checkpoint_files))]

    if len(checkpoint_files) == 0:
        return

    resume_checkpoint = str(output_dir / f"epoch_checkpoint_{checkpoint_files[-1]}.pt")
    return resume_checkpoint


def _find_best_checkpoint_in_dir(
    output_dir: str | Path, checkpoint_prefix: str = "epoch_checkpoint"
) -> str | None:
    output_dir = Path(output_dir)

    if not output_dir.exists():
        return

    checkpoint_files = [ckpt for ckpt in list(output_dir.glob("best_checkpoint_*.pt"))]
    scores = [
        float(str(ckpt).split("=")[-1].replace(".pt", "")) for ckpt in checkpoint_files
    ]
    checkpoint_files = [ckpt for _, ckpt in sorted(zip(scores, checkpoint_files))]
    if len(checkpoint_files) == 0:
        return
    return str(output_dir / checkpoint_files[-1])


def _load_optimizer(
    optimizer: str, parameters, lr: float, weight_decay: float, momentum: float
):
    from torch.optim import SGD, Adam, AdamW

    if optimizer.lower() == "sgd":
        return SGD(
            parameters,
            lr=lr,
            momentum=momentum,
        )
    elif optimizer.lower() == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented.")


def _load_metrics(
    dataset_name: str,
    stage: str,
    task_type: TaskType,
    device: str,
    dataset_labels: DatasetLabels,
) -> dict[str, Metric]:
    from docdjinn.evaluation.runners._metrics import (
        load_classification_metrics,
        load_detection_metrics,
        load_extractive_qa_metrics,
        load_token_classification_metrics,
    )

    if task_type == TaskType.sequence_classification:
        assert dataset_labels is not None, (
            "dataset_labels cannot be None when loading metrics."
        )
        assert dataset_labels.classification is not None, (
            "dataset_labels.classification cannot be None when loading classification metrics."
        )
        num_labels = len(dataset_labels.classification)
        return load_classification_metrics(
            device=device,
            num_classes=num_labels,
        )
    elif task_type == TaskType.token_classification:
        return load_token_classification_metrics(
            device=device,
        )
    elif task_type == TaskType.extractive_qa:
        return load_extractive_qa_metrics(
            dataset_name=dataset_name,
            stage=stage,
            device=device,
        )
    elif task_type in [TaskType.layout_analysis, TaskType.table_extraction]:
        return load_detection_metrics(device=device)
    else:
        raise NotImplementedError(f"Metrics not implemented for task type: {task_type}")


def _get_linear_schedule_with_min_lr(
    optimizer, num_warmup_steps, num_training_steps, initial_lr, final_lr
):
    import torch

    # Calculate the decay factor based on the desired final learning rate
    decay_factor = (initial_lr - final_lr) / initial_lr

    # Adjusted lambda function for custom linear decay with final LR
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Linearly decrease to final_lr instead of 0
        return max(final_lr / initial_lr, 1.0 - decay_factor * progress)

    # Use the LambdaLR scheduler with this custom lambda function
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def configure_engine(
    engine: "Engine",
    stage: str,
    model_name: str = "",
    dataset_name: str = "",
    wandb_logger: WandBLogger | None = None,
    metrics: dict | None = None,
    optimizer: Any | None = None,
    parent_engine: Engine | None = None,
    task_type: TaskType | None = None,
    output_dir: str | Path | None = None,
) -> None:
    import ignite.distributed as idist
    from ignite.engine import Events
    from ignite.handlers import ProgressBar
    from ignite.metrics import EpochWise, RunningAverage

    # attach runninge average of loss
    logger.info("Setting up running average loss callback")

    is_layout_analysis = task_type is not None and task_type in [
        TaskType.layout_analysis,
        TaskType.table_extraction,
    ]
    if is_layout_analysis:
        if stage == "train":
            loss_metric_names = [
                "loss",
                "loss_rpn_cls",
                "loss_rpn_bbox",
                "loss_cls",
                "loss_bbox",
                "acc",
            ]
            for key in loss_metric_names:
                RunningAverage(
                    alpha=0.95,
                    output_transform=lambda x, k=key: x.loss_dict[k]
                    if k in x.loss_dict
                    else 0.0,
                    epoch_bound=True,
                ).attach(engine, key)
        else:
            loss_metric_names = []
    else:
        loss_metric_names = [f"{stage}/running_avg_loss"]
        RunningAverage(
            alpha=0.95,
            output_transform=lambda x: x.loss,
            epoch_bound=True,
        ).attach(engine, loss_metric_names[0])

    if metrics is not None:
        # attach metrics
        for metric_name, metric in metrics.items():
            logger.info(f"Attaching metric {metric_name} to engine")
            metric.attach(
                engine,
                f"{stage}/{metric_name}",
                usage=EpochWise(),
            )

    # attach progress bar only on rank 0
    if idist.get_rank() == 0:
        logger.info(f"Setting up progress bar with loss metrics: {loss_metric_names}")
        progress_bar = ProgressBar(
            desc=f"Running stage={stage} on model={model_name} and dataset={dataset_name}",
            persist=True,
        )
        progress_bar.attach(
            engine,
            event_name=Events.ITERATION_COMPLETED(every=10),
            metric_names=loss_metric_names,
        )

        @engine.on(Events.EPOCH_COMPLETED)
        def progress_on_epoch_completed(engine: Engine) -> None:
            logger.info(
                "Epoch %d - Evaluation time: %.2fs - %s metrics: EpochResult:",
                engine.state.epoch,
                engine.state.times["EPOCH_COMPLETED"],
                stage,
            )
            for k, v in engine.state.metrics.items():
                logger.info(f"\t{k}: {v}")

        @engine.on(Events.TERMINATE | Events.INTERRUPT)
        def progress_on_terminate(engine: Engine) -> None:
            logger.info(
                f"Engine [{stage}] terminated after {engine.state.epoch} epochs."
            )
            progress_bar.close()

    if wandb_logger is not None and idist.get_rank() == 0:
        wandb_logger.attach_output_handler(
            engine,
            event_name=Events.EPOCH_COMPLETED | Events.COMPLETED,
            tag=stage,
            metric_names="all",
            global_step_transform=lambda *_: parent_engine.state.iteration
            if parent_engine
            else engine.state.iteration,
        )

        if stage == "train":
            wandb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=50),
                tag=stage,
                metric_names=loss_metric_names,
            )

            if optimizer is not None:
                wandb_logger.attach_opt_params_handler(
                    engine,
                    event_name=Events.ITERATION_STARTED(every=20),
                    optimizer=optimizer,
                )

        if is_layout_analysis:
            assert output_dir is not None, (
                "output_dir cannot be None for visualization."
            )

            @engine.on(Events.EPOCH_COMPLETED)
            def log_images(engine):
                import mmcv
                from mmdet.structures.bbox import scale_boxes
                from mmdet.visualization import DetLocalVisualizer

                # Determine output directory
                epoch = (
                    parent_engine.state.epoch
                    if stage == "validation"
                    else engine.state.epoch
                )
                _output_dir = Path(output_dir) / "visualizations" / stage / f"{epoch}"
                _output_dir.mkdir(parents=True, exist_ok=True)
                batch = engine.state.batch
                batch = {
                    "inputs": batch.inputs,
                    "data_samples": batch.data_samples,
                }
                det_data_samples = engine.state.output.det_data_samples
                class_labels = engine.state.output.class_labels

                visualizer = DetLocalVisualizer()
                visualizer.dataset_meta["classes"] = class_labels

                logger.info(f"Saving visualizations to {_output_dir}")
                for idx, data_sample in enumerate(det_data_samples):
                    img = batch["inputs"][idx].cpu().numpy().transpose(1, 2, 0)

                    scale_factor = data_sample.metainfo.get("scale_factor")
                    if "gt_instances" in data_sample and stage != "train":
                        data_sample.gt_instances.bboxes = scale_boxes(
                            data_sample.gt_instances.bboxes, scale_factor
                        )

                    if "pred_instances" in data_sample:
                        data_sample.pred_instances.bboxes = scale_boxes(
                            data_sample.pred_instances.bboxes, scale_factor
                        )

                    # Draw predictions
                    visualizer.add_datasample(
                        name=f"{engine.state.iteration}_sample_{idx}",
                        image=img,
                        data_sample=data_sample,
                        draw_gt=True,
                        draw_pred=True,
                    )

                    # Save visualization
                    output_path = (
                        _output_dir / f"{engine.state.iteration}_sample_{idx}.png"
                    )
                    logger.info(f"Saving visualization to {output_path}")
                    mmcv.imwrite(visualizer.get_image(), str(output_path))


def configure_model_checkpointer(
    config: RunnerConfig,
    training_engine: Engine,
    model_pipeline: ModelPipeline,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    validation_engine: Engine | None = None,
    resume_checkpoint: str | None = None,
) -> dict | None:
    """Configure model checkpointing for training and validation engines."""
    import torch
    from ignite.contrib.handlers import global_step_from_engine
    from ignite.engine import Events
    from ignite.handlers import Checkpoint, ModelCheckpoint

    logger.info("=" * 50)
    logger.info("CONFIGURING MODEL CHECKPOINTER")
    logger.info("=" * 50)

    logger.info(f"Checkpoint directory: {config.output_dir}")
    logger.info(f"Save checkpoint every: {config.save_ckpt_every_n_epochs} epochs")
    logger.info(f"Keep n checkpoints: {config.keep_n_checkpoints}")

    # setup checkpoint saving if required
    checkpoint_state_dict = {
        "config": config,
        "training_engine": training_engine,
        "model_pipeline": model_pipeline,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    model_checkpoint = ModelCheckpoint(
        config.output_dir,
        filename_prefix="epoch",
        n_saved=1,
        include_self=True,
        global_step_transform=lambda *_: training_engine.state.epoch,
        require_empty=False,
    )
    training_engine.add_event_handler(
        Events.EPOCH_COMPLETED(every=1),
        model_checkpoint,
        checkpoint_state_dict,
    )

    # Configure best model checkpointing if monitored metric is specified
    if validation_engine is not None and config.monitored_metric is not None:
        logger.info(
            f"Configuring best model checkpointing with monitored metric: {config.monitored_metric}"
        )
        logger.info(f"Monitored metric mode: {config.monitored_metric_mode}")

        best_model_saver = ModelCheckpoint(
            config.output_dir,
            filename_prefix="best",
            n_saved=1,
            global_step_transform=global_step_from_engine(training_engine),
            score_name=config.monitored_metric.replace("/", "-"),
            score_function=Checkpoint.get_default_score_fn(
                config.monitored_metric,
                -1 if config.monitored_metric_mode == "min" else 1.0,
            ),
            require_empty=False,
        )
        validation_engine.add_event_handler(
            Events.COMPLETED, best_model_saver, checkpoint_state_dict
        )
        logger.info("Best model checkpoint saving configured.")

    if resume_checkpoint is not None:
        logger.info("=" * 50)
        logger.info("RESUMING FROM CHECKPOINT")
        logger.info("=" * 50)
        logger.info(f"Checkpoint detected, resuming training from: {resume_checkpoint}")

        try:
            import torch
            from ignite.handlers import Checkpoint

            resume_checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
            Checkpoint.load_objects(
                to_load=checkpoint_state_dict,
                checkpoint=resume_checkpoint_data,
                strict=True,
            )
            logger.info(f"Resumed at epoch: {training_engine.state.epoch}")
            return resume_checkpoint_data
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            logger.info("Starting training from scratch...")
            return None
    else:
        logger.info("No existing checkpoints found, starting from scratch.")
        return None


def prepare_transform_kwargs(config: RunnerConfig) -> dict:
    transform_kwargs: dict[str, Any] = {
        "tokenizer_name": config.tokenizer_name,
    }
    task_type = get_dataset_config(config.dataset_name).task_type
    if task_type in [TaskType.sequence_classification, TaskType.token_classification]:
        transform_kwargs["use_segment_level_bboxes"] = config.use_segment_level_bboxes
    elif task_type == TaskType.extractive_qa:
        transform_kwargs["use_segment_level_bboxes"] = config.use_segment_level_bboxes
        transform_kwargs["ignore_samples_with_no_answer"] = (
            config.ignore_samples_with_no_answer
        )
    if config.resize_width is not None:
        transform_kwargs["resize_width"] = config.resize_width
    if config.resize_height is not None:
        transform_kwargs["resize_height"] = config.resize_height
    if config.use_imagenet_mean_std:
        transform_kwargs["use_imagenet_mean_std"] = config.use_imagenet_mean_std
    if config.add_segment_level_info:
        transform_kwargs["add_segment_level_info"] = config.add_segment_level_info
    if config.model_name.startswith("microsoft/udop") or config.model_name.startswith(
        "google-t5/t5"
    ):
        transform_kwargs["transform_type"] = "conditional_generation"
    if task_type in [TaskType.layout_analysis, TaskType.table_extraction]:
        transform_kwargs["use_fixed_size"] = config.use_fixed_size
        transform_kwargs["fixed_size"] = config.fixed_size
        transform_kwargs["use_flip"] = config.use_flip

    return transform_kwargs
