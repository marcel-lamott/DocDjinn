import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from docdjinn import ENV
from docdjinn.analyzation.clustering.core._embeddings import _load_embeddings
import pydantic.v1 as pydantic
import pydantic_argparse
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from docdjinn.data._core._data_types import DocumentInstance
from docdjinn.data._core._msgpack_dataset_reader import MsgpackDatasetReader
from docdjinn.data.interface import (
    load_dataset,
    load_synthetic_dataset,
)
from docdjinn.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


class LayoutFIDCalculator:
    """
    GPU-accelerated LayoutFID score calculator using LayoutLMv3 embeddings.
    """

    def __init__(
        self, device: str = "cuda", model_name: str = "microsoft/layoutlmv3-base"
    ):
        """
        Initialize LayoutFID calculator.

        Args:
            device: 'cuda' or 'cpu'
            model_name: HuggingFace model identifier for LayoutLMv3
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load LayoutLMv3 model and processor
        self.processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _get_embeddings(
        self,
        dataset: MsgpackDatasetReader,
        batch_size: int,
        use_image_only: bool = False,
    ) -> np.ndarray:
        """
        Extract LayoutLMv3 embeddings for images.

        Args:
            image_paths: List of paths to document images
            batch_size: Batch size for processing

        Returns:
            Embeddings array of shape (n_images, embedding_dim)
        """

        embeddings_list = []

        with torch.no_grad():
            dataloader = DataLoader(
                dataset,  # type: ignore
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=lambda x: x,
            )
            for batch in tqdm(
                dataloader,
                desc=f"Extracting embeddings batch_size=[{batch_size}]",
                total=len(dataloader),
            ):
                batch: list[DocumentInstance]

                # get images, words, boxes from batch
                words, word_bboxes, images = [], [], []
                for sample in batch:
                    assert sample.image is not None, "Sample image is None"
                    assert isinstance(sample.image.content, Image.Image), (
                        "Sample image content is not PIL Image"
                    )
                    images.append(sample.image.content.convert("RGB"))
                    if use_image_only:
                        continue
                    assert sample.content is not None, "Sample content is None"
                    assert sample.content.word_bboxes is not None, (
                        "Sample word bboxes are None"
                    )

                    words.append(sample.content.words)
                    word_bboxes.append(sample.content.word_bboxes.value)

                # Process images with LayoutLMv3 processor
                inputs = self.processor(
                    text=words,
                    boxes=word_bboxes,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                # layoutlmv3 expects bboxes in range [0, 1000]
                # we assume to get normalized bboxes in [0, 1]
                # scale bboxes
                # if (
                #     inputs["bbox"].max() > 1.01 or inputs["bbox"].min() < -0.01
                # ):  # 1.1 to account for any floating point precision issues
                #     raise ValueError(
                #         f"Expected normalized bounding boxes in range [0, 1], Got max value {inputs['bbox'].max()}"
                #     )

                inputs["bbox"] = (inputs["bbox"].clip(0.0, 1.0) * 1000).long()

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model output
                outputs = self.model(**inputs, output_hidden_states=True)

                # Use last hidden state (CLS token or mean pooling)
                # Extract the [CLS] token representation (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(batch_embeddings)

        embeddings = np.concatenate(embeddings_list, axis=0)
        return embeddings

    def _compute_statistics(self, embeddings: np.ndarray) -> tuple:
        """
        Compute mean and covariance of embeddings.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)

        Returns:
            Tuple of (mean, covariance)
        """
        mu = np.mean(embeddings, axis=0)
        sigma = np.cov(embeddings.T)

        # Ensure sigma is 2D (handle 1D case)
        if sigma.ndim == 1:
            sigma = np.diag(sigma)

        return mu, sigma

    # def _compute_fid( # this works same as calculate_frechet_distance but i kept the original as its taken from well-known FID implementation
    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/inception.py
    #     self, mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
    # ) -> float:
    #     """
    #     Compute Fréchet Inception Distance.

    #     Args:
    #         mu1, sigma1: Mean and covariance of real embeddings
    #         mu2, sigma2: Mean and covariance of generated embeddings

    #     Returns:
    #         FID score
    #     """
    #     # Euclidean distance between means
    #     diff = mu1 - mu2
    #     diff_norm = np.sum(diff**2)

    #     # Trace of covariance matrices
    #     trace_cov = np.trace(sigma1 + sigma2)

    #     # Matrix square root of product of covariances
    #     # Using eigenvalue decomposition for numerical stability
    #     sqrt_cov_prod = self._sqrtm(sigma1 @ sigma2)
    #     trace_sqrt_prod = np.trace(sqrt_cov_prod)

    #     # FID = ||µr - µg||^2 + Tr(Σr + Σg - 2√(ΣrΣg))
    #     fid = diff_norm + trace_cov - 2 * trace_sqrt_prod

    #     return float(np.real(fid))

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, (
            "Training and test mean vectors have different lengths"
        )
        assert sigma1.shape == sigma2.shape, (
            "Training and test covariances have different dimensions"
        )

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            logger.info(msg)
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

    @staticmethod
    def _sqrtm(matrix: np.ndarray) -> np.ndarray:
        """
        Compute matrix square root using eigenvalue decomposition.
        More numerically stable than scipy.linalg.sqrtm for this use case.
        """
        try:
            # Use scipy's sqrtm for general case
            sqrt_m = linalg.sqrtm(matrix)
            # Return real part if imaginary component is negligible
            if np.iscomplexobj(sqrt_m):
                sqrt_m = np.real(sqrt_m)
            return sqrt_m
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
            sqrt_m = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            return np.real(sqrt_m)

    def calculate_layoutfid(
        self,
        real_embeddings: "np.ndarray",
        synth_embeddings: "np.ndarray",
        limit_sizes_to_smallest: bool = True,
    ) -> tuple[float, int]:
        # limit both datasets to smallest size
        if limit_sizes_to_smallest:
            real_size = len(real_embeddings)  # type: ignore
            synth_size = len(synth_embeddings)  # type: ignore

            # layout fix see which dataset is smaller in size
            if real_size > synth_size:
                logger.info(
                    f"Real embeddings is bigger ({real_size} samples) than synthetic dataset ({synth_size} samples)."
                )
                random_indices = torch.randperm(real_size)[:synth_size]
                real_embeddings = real_embeddings[random_indices.tolist()]
            else:
                logger.info(
                    f"Synthetic dataset is bigger ({synth_size} samples) than real dataset ({real_size} samples)."
                )
                random_indices = torch.randperm(synth_size)[:real_size]
                synth_embeddings = synth_embeddings[random_indices.tolist()]

        total_real_dataset_samples = len(real_embeddings)  # type: ignore
        total_synth_dataset_samples = len(synth_embeddings)  # type: ignore
        assert total_real_dataset_samples == total_synth_dataset_samples, (
            "FID calculation requires both datasets to have the same number of samples. "
            f"Got {total_real_dataset_samples} real and {total_synth_dataset_samples} synthetic samples."
        )
        logger.info("Calculating real statistics...")
        mu_real, sigma_real = self._compute_statistics(real_embeddings)
        logger.info("Calculating synthetic statistics...")
        mu_gen, sigma_gen = self._compute_statistics(synth_embeddings)
        layoutfid = self.calculate_frechet_distance(
            mu_real, sigma_real, mu_gen, sigma_gen
        )
        return layoutfid, real_embeddings.shape[0]


class LayoutFIDCalculatorConfig(pydantic.BaseModel):
    """
    Configuration for clustering operations.
    """

    seed: int = 42
    real_dataset_name: str
    synth_dataset_name: str
    limit_sizes_to_smallest: bool = True
    embedding_src: str = "layout"


def main(
    cfg: LayoutFIDCalculatorConfig,
):
    """Example usage of LayoutFID calculator."""

    # load the results csv
    output_df_path = Path("data/results/layout_fid_embeddings.csv")

    # load the results csv and check if row with same real and synth dataset exists
    if output_df_path.exists():
        output_df = pd.read_csv(output_df_path)
        existing_row = output_df[
            (output_df["real_dataset"] == cfg.real_dataset_name)
            & (output_df["synth_dataset"] == cfg.synth_dataset_name)
            & (output_df["embedding_src"] == cfg.embedding_src)
        ]
        if not existing_row.empty:
            logger.info(
                f"LayoutFID already calculated for real dataset '{cfg.real_dataset_name}' and synthetic dataset '{cfg.synth_dataset_name}'. Skipping calculation."
            )
            logger.info(
                f"Existing LayoutFID Score: {existing_row['layoutfid_score'].values[0]:.4f}"
            )
            return
    else:
        output_df = pd.DataFrame(
            columns=["real_dataset", "synth_dataset", "layoutfid_score", "num_samples"]
        )

    # torch manual seed for reproducibility
    torch.manual_seed(42)

    # logging config
    logger.info("Calculating LayoutFID with config:")
    logger.info(cfg.json(indent=4))

    # load the real embeddings
    real_embeddings, _ = _load_embeddings(
        file_path=ENV.EMBEDDINGS_DIR / cfg.real_dataset_name / f"{cfg.embedding_src}.h5"
    )

    # load the synthetic embeddings
    synthetic_embeddings, _ = _load_embeddings(
        file_path=ENV.EMBEDDINGS_DIR
        / 'synth' 
        / cfg.synth_dataset_name
        / f"{cfg.embedding_src}.h5",
    )

    # Initialize calculator
    calculator = LayoutFIDCalculator(device="cuda")

    # Calculate LayoutFID
    layoutfid_score, num_samples = calculator.calculate_layoutfid(
        real_embeddings,
        synthetic_embeddings,
        limit_sizes_to_smallest=cfg.limit_sizes_to_smallest,
    )
    logger.info(f"\nLayoutFID Score: {layoutfid_score:.4f} over {num_samples} samples")

    # append result to csv
    new_row = {
        "real_dataset": cfg.real_dataset_name,
        "synth_dataset": cfg.synth_dataset_name,
        "layoutfid_score": layoutfid_score,
        "num_samples": len(real_embeddings),
        "embedding_src": cfg.embedding_src,
    }
    output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
    output_df_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_df_path, index=False)
    logger.info("LayoutFID score saved to data/results/layout_fid.csv")


if __name__ == "__main__":
    parser = pydantic_argparse.ArgumentParser(
        model=LayoutFIDCalculatorConfig,
    )
    main(parser.parse_typed_args())
