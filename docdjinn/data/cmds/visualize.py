from __future__ import annotations

import pydantic.v1 as pydantic
import pydantic_argparse

from docdjinn.data._core._visualization_utilities import _save_visualization
from docdjinn.logging import get_logger

logger = get_logger(__name__)


def main(
    cfg: VisualizeDataset,
):
    from docdjinn.data import load_dataset

    # setup data pipeline and dataloaders
    logger.info("Saving samples from split [train]...")
    dataset = load_dataset(
        dataset_name=cfg.dataset_name,
        is_synthetic=cfg.synthetic,
    )

    dataset_labels = dataset.metadata.dataset_labels
    for split, dataset in dataset.split_iterators.items():
        if dataset is not None:
            for idx, sample in enumerate(dataset):
                if cfg.sample_id is not None and sample.sample_id != cfg.sample_id:
                    continue

                _save_visualization(
                    sample,
                    cfg.dataset_name,
                    cfg.output_dir,
                    split,
                    dataset_labels=dataset_labels,
                    visualize_gt_only=cfg.visualize_gt_only,
                )
                if cfg.n_samples is not None and idx + 1 >= cfg.n_samples:
                    break


class VisualizeDataset(pydantic.BaseModel):
    """
    Configuration for visualizing dataset samples.
    """

    dataset_name: str
    output_dir: str = "data/visualizations/"
    n_samples: int = 20
    synthetic: bool = False
    visualize_gt_only: bool = True
    sample_id: str | None = None


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=VisualizeDataset,
    )
    main(parser.parse_typed_args())
