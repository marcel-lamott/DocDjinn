from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pydantic_argparse
from ignite.engine import Events
from ignite.handlers import Checkpoint, EarlyStopping

from docdjinn.data._core._utilities import TaskType
from docdjinn.data.interfaces.dataset import get_dataset_config
from docdjinn.evaluation.runners._config import RunnerConfig
from docdjinn.evaluation.runners._evaluation_step import EvaluationStep
from docdjinn.evaluation.runners._training_step import TrainingStep
from docdjinn.evaluation.runners.utilities import (
    _find_best_checkpoint_in_dir,
    _find_resume_checkpoint_in_dir,
    _get_linear_schedule_with_min_lr,
    _initialize_torch,
    _initialize_wandb,
    _load_metrics,
    _load_optimizer,
    _log_run_configuration,
    _log_system_info,
    configure_engine,
    configure_model_checkpointer,
    prepare_transform_kwargs,
)
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    from ignite.engine import Engine


logger = get_logger(__name__)


class Runner:
    def __init__(self, config: RunnerConfig):
        self.config = config

    def setup_logger(self):
        logger.info("=" * 50)
        logger.info("BUILDING LOGGER")
        logger.info("=" * 50)
        resume_checkpoint = _find_resume_checkpoint_in_dir(self.config.output_dir)
        logger.info("Found checkpoint: {}".format(resume_checkpoint))
        return resume_checkpoint, _initialize_wandb(
            project=self.config.project_name,
            id=self.config.run_name,
            name=self.config.run_name,
            config=self.config.dict(),
            tags=[self.config.dataset_name, self.config.model_name],
            runs_dir=self.config.runs_dir,
            resume=resume_checkpoint is not None,
        )

    def setup_data_pipeline(self):
        """Setup and return the data pipeline."""
        from docdjinn.data import load_data_pipeline, load_preprocessed_data_pipeline

        logger.info("=" * 50)
        logger.info("BUILDING DATA PIPELINE")
        logger.info("=" * 50)
        task_type = get_dataset_config(self.config.dataset_name).task_type
        if task_type == TaskType.extractive_qa:
            assert self.config.use_preprocessed_dataset, (
                "Extractive QA datasets require preprocessed datasets. Pass the --use-preprocessed-dataset flag"
            )

        transform_kwargs = prepare_transform_kwargs(self.config)
        if self.config.use_preprocessed_dataset:
            data_pipeline = load_preprocessed_data_pipeline(
                dataset_name=self.config.dataset_name,
                dataset_splitting_enabled=self.config.dataset_splitting_enabled,
                split_ratio=self.config.split_ratio,
                **transform_kwargs,
            )
        else:
            data_pipeline = load_data_pipeline(
                dataset_name=self.config.dataset_name,
                dataset_splitting_enabled=self.config.dataset_splitting_enabled,
                split_ratio=self.config.split_ratio,
                **transform_kwargs,
            )

        # Log detailed data pipeline information
        logger.info(f"Dataset: {data_pipeline.dataset}")
        logger.info(f"Dataset metadata: {data_pipeline.dataset_metadata}")
        logger.info(f"Dataset labels: {data_pipeline.dataset_metadata.dataset_labels}")

        return data_pipeline

    def setup_model_pipeline(self):
        """Setup and return the model pipeline."""
        import torch

        from docdjinn.evaluation.model_pipeline import load_model_pipeline

        logger.info("=" * 50)
        logger.info("BUILDING MODEL PIPELINE")
        logger.info("=" * 50)
        model_pipeline = load_model_pipeline(
            task_type=self.data_pipeline.dataset.task_type,
            model_name=self.config.model_name,
            model_cache_dir=self.config.model_cache_dir,
            dataset_metadata=self.data_pipeline.dataset_metadata,
        )
        logger.info(f"Model architecture: {model_pipeline}")
        logger.info(f"Moving model to device: {self.device}")
        model_pipeline = model_pipeline.to(self.device)

        if self.config.pretrained_checkpoint is not None:
            logger.info(
                f"Loading pretrained model from checkpoint: {self.config.pretrained_checkpoint}"
            )
            checkpoint_data = torch.load(
                self.config.pretrained_checkpoint, map_location="cpu"
            )
            Checkpoint.load_objects(
                to_load={"model_pipeline": model_pipeline},
                checkpoint=checkpoint_data,
                strict=True,
            )
        return model_pipeline

    def setup_optimizer_and_scheduler(self):
        logger.info("=" * 50)
        logger.info("SETTING UP OPTIMIZER AND LR SCHEDULER")
        logger.info("=" * 50)

        optimizer = _load_optimizer(
            optimizer=self.config.optimizer,
            parameters=self.model_pipeline.model.parameters(),
            lr=self.config.lr_start,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
        )

        # total number of training steps
        num_training_steps = self.config.num_epochs * len(self.train_dataloader)
        # build lr scheduler
        warmup_steps = int(
            num_training_steps * self.config.lr_schedule_warmup_steps_frac_of_total
        )
        logger.info(f"Total training epochs: {self.config.num_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_dataloader)}")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Initial LR: {self.config.lr_start}")
        logger.info(f"Final LR: {self.config.lr_end}")
        logger.info(
            f"Warmup steps: {warmup_steps} ({self.config.lr_schedule_warmup_steps_frac_of_total:.1%} of total)"
        )

        lr_scheduler = _get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            initial_lr=self.config.lr_start,
            final_lr=self.config.lr_end,
        )

        # create optimizer and lr scheduler
        logger.info("Using optimizer:")
        logger.info(optimizer)

        # create optimizer and lr scheduler
        logger.info("Using lr_scheduler:")
        logger.info(lr_scheduler)

        return optimizer, lr_scheduler

    def setup_dataloaders(self):
        """Setup and return the dataloaders."""
        train_dataloader = self.data_pipeline.train_dataloader(
            batch_size=self.config.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_dataloader = self.data_pipeline.validation_dataloader(
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_dataloader = self.data_pipeline.test_dataloader(
            batch_size=self.config.eval_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        # setup dataloader
        if test_dataloader is None:
            logger.warning(
                "This dataset does not have a test split, using validation set for evaluation."
            )
            test_dataloader = self.data_pipeline.validation_dataloader(
                batch_size=self.config.eval_batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )

        assert train_dataloader is not None, "Train dataloader is None."
        assert val_dataloader is not None, "Validation dataloader is None."
        assert test_dataloader is not None, "Test dataloader is None."

        return train_dataloader, val_dataloader, test_dataloader

    def setup(self):
        # Log system information
        _log_system_info()

        # Log run configuration
        _log_run_configuration(self.config)

        # initialize training
        _initialize_torch(
            seed=self.config.seed, deterministic=self.config.deterministic
        )

        # initialize torch device (cpu or gpu)
        self.device = "cuda" if self.config.device_id >= 0 else "cpu"
        logger.info(f"Selected device: {self.device}")

        # setup wandb logger
        self.resume_checkpoint, self.wandb_logger = self.setup_logger()

        # setup data pipeline
        self.data_pipeline = self.setup_data_pipeline()

        # if only_prepare_data flag is set, exit after data preparation
        if self.config.only_prepare_data:
            logger.info("Only data preparation flag is set. Exiting after data setup.")
            logger.info(
                "Total samples in train set: {}".format(
                    len(self.data_pipeline.dataset.train)
                )
            )  # type: ignore
            logger.info(
                "Total samples in validation set: {}".format(
                    len(self.data_pipeline.dataset.validation)
                )
            )  # type: ignore
            logger.info(
                "Total samples in test set: {}".format(
                    len(self.data_pipeline.dataset.test)
                )
            )  # type: ignore

            # print first ids of val samples
            if self.data_pipeline.dataset.validation is not None:
                val_sample_ids = [
                    sample.sample_id for sample in self.data_pipeline.dataset.validation
                ]
                logger.info(f"First 10 validation sample IDs: {val_sample_ids[:10]}")

            sys.exit(0)

        # set task type
        self.task_type = self.data_pipeline.dataset.task_type

        # setup model pipeline
        self.model_pipeline = self.setup_model_pipeline()

        # setup dataloaders
        self.train_dataloader, self.val_dataloader, self.test_dataloader = (
            self.setup_dataloaders()
        )

    def train(
        self,
    ) -> None:
        from ignite.engine import Engine

        assert self.model_pipeline is not None, "Model pipeline is not initialized."
        assert self.data_pipeline is not None, "Data pipeline is not initialized."

        logger.info("=" * 50)
        logger.info("SETTING UP TRAINING")
        logger.info("=" * 50)

        # create optimizer and lr scheduler
        optimizer, lr_scheduler = self.setup_optimizer_and_scheduler()

        # setup model to devoce
        logger.info(f"Moving model to device: {self.device}")
        self.model_pipeline = self.model_pipeline.to(self.device)
        self.model_pipeline.train()

        # make training step
        training_step = TrainingStep(
            model_pipeline=self.model_pipeline,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=self.device,
            with_amp=self.config.with_amp,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            enable_grad_clipping=self.config.enable_grad_clipping,
            max_grad_norm=self.config.max_grad_norm,
        )

        # make validation step
        validation_step = EvaluationStep(
            model_pipeline=self.model_pipeline, device=self.device, stage="validation"
        )

        # make and create the training engine
        training_engine = Engine(training_step)

        # configure training engine
        configure_engine(
            engine=training_engine,
            stage="train",
            metrics=None,  # no metrics for training stage
            optimizer=optimizer,
            wandb_logger=self.wandb_logger,
            model_name=self.config.model_name,
            dataset_name=self.config.dataset_name,
            task_type=self.task_type,
            output_dir=self.config.output_dir,
        )

        validation_engine = None
        if self.config.do_validation:
            # configure validation engine
            validation_engine = Engine(validation_step)

            # configure validation engine
            configure_engine(
                engine=validation_engine,
                stage="validation",
                metrics=_load_metrics(
                    stage="validation",
                    dataset_name=self.config.dataset_name,
                    task_type=self.task_type,
                    device=self.device,
                    dataset_labels=self.data_pipeline.dataset_metadata.dataset_labels,
                ),
                parent_engine=training_engine,
                wandb_logger=self.wandb_logger,
                model_name=self.config.model_name,
                dataset_name=self.config.dataset_name,
                task_type=self.task_type,
                output_dir=self.config.output_dir,
            )

            if self.config.enable_early_stopping:
                assert self.config.monitored_metric is not None, (
                    "monitored_metric must be set for early stopping."
                )
                es_handler = EarlyStopping(
                    patience=10,
                    score_function=Checkpoint.get_default_score_fn(
                        self.config.monitored_metric,
                        -1 if self.config.monitored_metric_mode == "min" else 1.0,
                    ),
                    trainer=training_engine,
                )
                validation_engine.add_event_handler(Events.COMPLETED, es_handler)

            def run_validation(engine: Engine) -> None:
                logger.info(
                    f"Running validation engine on total samples {len(self.val_dataloader.dataset)} "  # type: ignore
                    f"with batch size [{self.val_dataloader.batch_size}]"
                )
                validation_engine.run(
                    self.val_dataloader,
                    epoch_length=1 if engine.state.epoch == 0 else None,
                )

            if self.config.validate_every_n_epochs > 1.0:
                training_engine.add_event_handler(
                    Events.EPOCH_COMPLETED(
                        every=int(self.config.validate_every_n_epochs)
                    )
                    | Events.STARTED,
                    run_validation,
                )
            else:
                assert self.config.validate_every_n_epochs > 0.0, (
                    "validate_every_n_epochs must be positive."
                )
                validate_every_n_iterations = int(
                    self.config.validate_every_n_epochs * len(self.train_dataloader)
                )
                logger.info(
                    f"Validating every {validate_every_n_iterations} iterations."
                )
                training_engine.add_event_handler(
                    Events.ITERATION_COMPLETED(every=validate_every_n_iterations)
                    | Events.STARTED,
                    run_validation,
                )

        # Configure model checkpointing
        configure_model_checkpointer(
            config=self.config,
            training_engine=training_engine,
            validation_engine=validation_engine,
            model_pipeline=self.model_pipeline,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            resume_checkpoint=self.resume_checkpoint,
        )

        resume_epoch = training_engine.state.epoch
        if (
            training_engine._is_done(training_engine.state)
            and resume_epoch >= self.config.num_epochs
        ):  # if we are resuming from last checkpoint and training is already finished
            logger.warning(
                "Training has already been finished! Either increase the number of "
                f"epochs (current={self.config.num_epochs}) >= {resume_epoch} "
                "OR reset the training from start."
            )
            return

        # setup dataloader
        logger.info(
            f"Running training engine on total samples {len(self.train_dataloader.dataset)} "  # type: ignore
            f"with batch size [{self.train_dataloader.batch_size}] and output_dir: {self.config.output_dir}"  # type: ignore
        )

        if self.config.test_run:
            logger.info("Test run enabled, running only 1 epoch...")
            training_engine.run(self.train_dataloader, max_epochs=1, epoch_length=10)
        else:
            training_engine.run(
                self.train_dataloader, max_epochs=self.config.num_epochs
            )

    def get_test_output_path(self) -> Path:
        return (
            Path("data")
            / "results"
            / self.config.dataset_name
            / self.config.model_name
            / str(self.config.seed)
            / f"{self.config.project_name}"
            / f"{self.config.run_name}.json"
        )

    def test(
        self,
        resume_checkpoint: str | None = None,
    ) -> None:
        import numpy as np
        import torch
        from ignite.engine import Engine

        if self.config.use_best:
            resume_checkpoint = _find_best_checkpoint_in_dir(
                output_dir=self.config.output_dir
            )
        else:
            resume_checkpoint = _find_resume_checkpoint_in_dir(
                output_dir=self.config.output_dir
            )
        if resume_checkpoint is not None:
            logger.info(f"Loading model from checkpoint: {resume_checkpoint}")
            import torch
            from ignite.handlers import Checkpoint

            checkpoint_data = torch.load(resume_checkpoint, map_location="cpu")
            Checkpoint.load_objects(
                to_load={"model_pipeline": self.model_pipeline},
                checkpoint=checkpoint_data,
                strict=True,
            )

        # setup model to devoce
        model_pipeline = self.model_pipeline.to(self.device)

        # load task metrics
        metrics = _load_metrics(
            dataset_name=self.config.dataset_name,
            stage="test",
            task_type=self.task_type,
            device=self.device,
            dataset_labels=self.data_pipeline.dataset_metadata.dataset_labels,
        )

        # create evaluation engine
        evaluation_engine = Engine(
            EvaluationStep(
                model_pipeline=model_pipeline, device=self.device, stage="test"
            )
        )

        # configure engine
        configure_engine(
            engine=evaluation_engine,
            stage="test",
            metrics=metrics,
            wandb_logger=self.wandb_logger,
            model_name=self.config.model_name,
            dataset_name=self.config.dataset_name,
            task_type=self.task_type,
            output_dir=self.config.output_dir,
        )

        logger.info(
            f"Running evaluation engine on total samples {len(self.test_dataloader.dataset)} "  # type: ignore
            f"with batch size [{self.test_dataloader.batch_size}]"
        )
        state = evaluation_engine.run(
            self.test_dataloader, epoch_length=1 if self.config.test_run else None
        )

        # get the test metrics and save the results in a file
        test_metrics = state.metrics

        # recursively go throughg metric dict converting tensors or numpy arays to floats
        def _convert_metric_values(obj):
            if isinstance(obj, dict):
                return {k: _convert_metric_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_metric_values(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_convert_metric_values(item) for item in obj)
            elif isinstance(obj, torch.Tensor):
                if obj.numel() == 1:
                    return obj.item()
                else:
                    return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, "item") and callable(getattr(obj, "item")):
                return obj.item()
            else:
                return obj

        test_metrics = _convert_metric_values(state.metrics)

        output_file_path = self.get_test_output_path()
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            import json

            json.dump({
                **test_metrics,
                "checkpoint": str(resume_checkpoint),
                "config": self.config.dict(),

            }, f, indent=4, sort_keys=True)
        logger.info(f"Test results saved to: {output_file_path}")

    def run(self):
        output_file_path = self.get_test_output_path()
        if output_file_path.exists() and not self.config.do_eval:
            logger.info(
                f"Output file {output_file_path} already exists and overwrite_output is set to False. Skipping run."
            )
            return

        self.setup()
        if self.config.do_train:
            self.train()

        if self.config.do_eval:
            self.test()
        self.close()

    def close(self):
        if self.wandb_logger is not None:
            self.wandb_logger.close()


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=RunnerConfig,
    )
    Runner(parser.parse_typed_args()).run()
