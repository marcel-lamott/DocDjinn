import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic.v1 as pydantic
import pydantic_argparse
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoModel, AutoProcessor

from docdjinn.data._core._data_types import DocumentInstance
from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader
from docdjinn.data.interface import (
    load_dataset,
    load_synthetic_dataset,
)
from docdjinn.logging import get_logger

import torchvision.transforms.functional as TF
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def get_activations(
    dataset, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    model.eval()

    dataset.set_transform(lambda sample: TF.to_tensor(sample.image.content.convert("RGB").resize((1024, 1024))))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(dataset), dims))

    start_idx = 0

    for batch in tqdm.tqdm(dataloader):
        batch = batch.to(device)
        print('batch',batch.shape)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(
    dataset, model, batch_size=50, dims=2048, device="cpu", num_workers=1
):
    act = get_activations(dataset, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_fid_given_datasets(real_dataset, syn_dataset, batch_size, device, dims, num_workers=1):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(
        real_dataset, model, batch_size, dims, device, num_workers
    )
    m2, s2 = calculate_activation_statistics(
        syn_dataset, model, batch_size, dims, device, num_workers
    )
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


class FIDCalculatorConfig(pydantic.BaseModel):
    """
    Configuration for clustering operations.
    """

    seed: int = 42
    real_dataset_name: str
    synth_dataset_name: str
    batch_size: int = 50
    limit_sizes_to_smallest: bool = True


def main(
    cfg: FIDCalculatorConfig,
):
    """Example usage of FID calculator."""

    # load the results csv
    output_df_path = Path("data/results/fid.csv")

    # load the results csv and check if row with same real and synth dataset exists
    if output_df_path.exists():
        output_df = pd.read_csv(output_df_path)
        existing_row = output_df[
            (output_df["real_dataset"] == cfg.real_dataset_name)
            & (output_df["synth_dataset"] == cfg.synth_dataset_name)
        ]
        if not existing_row.empty:
            logger.info(
                f"FID already calculated for real dataset '{cfg.real_dataset_name}' and synthetic dataset '{cfg.synth_dataset_name}'. Skipping calculation."
            )
            logger.info(
                f"Existing FID Score: {existing_row['fid'].values[0]:.4f}"
            )
            return
    else:
        output_df = pd.DataFrame(
            columns=["real_dataset", "synth_dataset", "fid", "num_samples"]
        )

    # torch manual seed for reproducibility
    torch.manual_seed(42)

    # logging config
    logger.info("Calculating FID with config:")
    logger.info(cfg.json(indent=4))

    # load real dataset pipeline
    real_dataset = load_dataset(
        dataset_name=cfg.real_dataset_name,
        create_train_val_splits=False,
    ).train

    synth_dataset = load_synthetic_dataset(
        dataset_name=cfg.synth_dataset_name,
    ).train

    # assert datasets are not None
    assert real_dataset is not None, "Real dataset train split is None"
    assert synth_dataset is not None, "Synthetic dataset train split is None"

    # log dataset sizes
    logger.info(f"Real dataset size: {len(real_dataset)}")
    logger.info(f"Synthetic dataset size: {len(synth_dataset)}")

    # limit both datasets to smallest size
    if cfg.limit_sizes_to_smallest:
        real_size = len(real_dataset)  # type: ignore
        synth_size = len(synth_dataset)  # type: ignore

        if real_size > synth_size:
            logger.info(
                f"Real dataset is bigger ({real_size} samples) than synthetic dataset ({synth_size} samples)."
            )
            random_indices = torch.randperm(real_size)[:synth_size]
            real_dataset.set_subset_indices(random_indices.tolist())
        else:
            logger.info(
                f"Synthetic dataset is bigger ({synth_size} samples) than real dataset ({real_size} samples)."
            )
            random_indices = torch.randperm(synth_size)[:real_size]
            synth_dataset.set_subset_indices(random_indices.tolist())

    total_real_dataset_samples = len(real_dataset)  # type: ignore
    total_synth_dataset_samples = len(synth_dataset)  # type: ignore
    assert total_real_dataset_samples == total_synth_dataset_samples, (
        "FID calculation requires both datasets to have the same number of samples. "
        f"Got {total_real_dataset_samples} real and {total_synth_dataset_samples} synthetic samples."
    )

    num_samples = total_real_dataset_samples
    fid = calculate_fid_given_datasets(real_dataset, synth_dataset, cfg.batch_size, device="cuda", dims=2048)
    logger.info(f"\FID Score: {fid:.4f} over {num_samples} samples")

    # append result to csv
    new_row = {
        "real_dataset": cfg.real_dataset_name,
        "synth_dataset": cfg.synth_dataset_name,
        "fid": fid,
        "num_samples": len(real_dataset),
    }
    output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    output_df.to_csv("data/results/fid.csv", index=False)
    logger.info("FID score saved to data/results/fid.csv")


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=FIDCalculatorConfig,
    )
    main(parser.parse_typed_args())
