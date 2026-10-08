from __future__ import annotations

from pathlib import Path
from typing import Literal

import pydantic.v1 as pydantic

from docdjinn import ENV


class RunnerConfig(pydantic.BaseModel):
    # do eval
    do_train: bool = True
    do_eval: bool = True

    # runtime args
    project_name: str = "docdjinn-experiments"
    device_id: int = 0
    run_name: str
    runs_dir: str = ENV.RUNS_DIR
    seed: int = 42
    deterministic: bool = False
    backend: str | None = "nccl"
    n_devices: int = 1
    # dataset args
    dataset_name: str
    root_datasets_dir: str = ENV.BASE_DATASETS_DIR
    # dataloader args
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    # dataset split args
    dataset_splitting_enabled: bool = False
    split_ratio: float = 0.9
    # preprocessed dataset
    use_preprocessed_dataset: bool = False
    # tokenizer args
    tokenizer_name: str | None = None
    use_segment_level_bboxes: bool = False
    resize_width: int | None = None
    resize_height: int | None = None
    use_imagenet_mean_std: bool = False
    ignore_samples_with_no_answer: bool = False
    add_segment_level_info: bool = False  # this is only used in geolayoutlm
    # only_prepare data
    only_prepare_data: bool = False
    # mmdet detection pipeline args
    use_flip: bool = True
    use_fixed_size: bool = False
    fixed_size: int = 800

    # evaluator configs
    with_amp: bool = False
    use_best: bool = True
    do_validation: bool = True

    # model args
    model_checkpoint_path: str | None = None
    model_name: str = "microsoft/layoutlmv3-base"
    model_cache_dir: str = ENV.MODELS_DIR / "pretrained"
    pretrained_checkpoint: str | None = None
    # training args
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    momentum: float = 0.9
    num_epochs: int = 50
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    enable_grad_clipping: bool = False
    max_grad_norm: float = 1.0
    # lr scheduler args
    lr_start: float = 1e-5
    lr_end: float = 1e-8
    lr_schedule_warmup_steps_frac_of_total: float = 0.1
    # validation args
    validate_every_n_epochs: float = 1.0
    enable_early_stopping: bool = True
    # checkpoint args
    save_ckpt_every_n_epochs: int = 1
    keep_n_checkpoints: int = 1
    monitored_metric: str | None = None
    monitored_metric_mode: str = "max"  # "min" or "max"
    save_weights_only: bool = False
    test_run: bool = False
    # Per sample eval
    split_for_per_sample_eval: Literal["train", "validation", "test"] = "test"

    @property
    def output_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_name

    def state_dict(self) -> dict:
        return self.dict()

    def load_state_dict(self, state_dict: dict) -> None:
        for key, value in state_dict.items():
            if key in self.__fields__:
                setattr(self, key, value)

    def dict(self, *args, **kwargs):
        self.model_cache_dir = str(self.model_cache_dir)
        self.root_datasets_dir = str(self.root_datasets_dir)
        self.runs_dir = str(self.runs_dir)
        return super().dict(*args, **kwargs)


class MixedRunnerConfig(RunnerConfig):
    synthetic_dataset_name: str
    num_real_samples: int = -1
    num_synthetic_samples: int = -1
