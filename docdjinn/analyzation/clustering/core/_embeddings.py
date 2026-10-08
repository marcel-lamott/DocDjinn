from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import tqdm

from docdjinn.analyzation.clustering.core._utilities import EmbeddingType
from docdjinn.data._core._data_types import DocumentInstanceModelInput
from docdjinn.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _iterate_dataset(
    model_fn: Callable,
    embedding_fn: Callable,
    dataloader: "DataLoader",
    device: str = "cpu",
):
    """Inner function that actually generates the embeddings."""
    import torch

    model = model_fn()
    model.to(device)
    model.eval()

    sample_ids = []
    embeddings = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Extracting embeddings"):
            batch: DocumentInstanceModelInput
            batch = batch.select_first_overflow_samples()
            batch = batch.to(device)

            token_bboxes = batch.token_bboxes
            if token_bboxes is not None:
                if token_bboxes.min() >= 0 and token_bboxes.max() <= 1.0:
                    # if bboxes are normalized to [0, 1], convert to [0, 1000] as expected by layoutlmv3
                    token_bboxes = (token_bboxes * 1000).long()
                else:
                    logger.warning(
                        f"Token bboxes must be in the range [0, 1], but got min {token_bboxes.min()} and max {token_bboxes.max()}"
                    )
                    token_bboxes = (token_bboxes.clip(0, 1.0) * 1000).long()

                # assert check
                assert token_bboxes.min() >= 0 and token_bboxes.max() <= 1000, (
                    f"Token bboxes must be in the range [0, 1000], but got min {token_bboxes.min()} and max {token_bboxes.max()}"
                )

            # make sure if image is normlized 0-1 as in layoutlm we renormalize using clip stats
            assert batch.image.min() >= -1.1 and batch.image.max() <= 1.1, (
                f"Image pixel values must be in the range [0, 1], but got min {batch.image.min()} and max {batch.image.max()}"
            )

            # make inputs
            inputs = dict(
                input_ids=batch.token_ids,
                bbox=token_bboxes,
                attention_mask=batch.attention_mask,
                pixel_values=batch.image,
                words=batch.words,
            )

            embeddings.append(embedding_fn(model, inputs))

            # in our preprocessed dataset indices are always unqiue
            # but sample_ids may not be always unique in some rare cases
            sample_ids.extend(batch.sample_id)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings.cpu().numpy(), sample_ids


def _extract_layoutlm_embeddings(
    dataloader: "DataLoader",
    device: str = "cpu",
):
    """Inner function that actually generates the embeddings."""

    def model_fn():
        from transformers import (
            LayoutLMv3Model,
        )

        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        model.to(device)
        model.eval()
        return model

    def embedding_fn(model, inputs):
        outputs = model(
            input_ids=inputs["input_ids"],
            bbox=inputs["bbox"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )
        return outputs.last_hidden_state[:, 0, :]

    embeddings, sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return embeddings, sample_ids


def _extract_text_embeddings(
    dataloader: "DataLoader",
    device: str = "cpu",
):
    """Inner function that actually generates the embeddings."""

    def model_fn():
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-mpnet-base-v2")
        model.to(device)
        model.eval()
        return model

    def embedding_fn(model, inputs):
        sentences = [" ".join(words_per_sample) for words_per_sample in inputs["words"]]
        return model.encode(sentences, convert_to_tensor=True)

    embeddings, sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return embeddings, sample_ids


def _extract_image_embeddings(
    dataloader: "DataLoader",
    device: str = "cpu",
):
    """Inner function that actually generates the embeddings."""
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    def model_fn():
        from transformers import (
            CLIPModel,
        )

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to(device)
        model.eval()
        return model

    def embedding_fn(model, inputs):
        from torchvision.transforms.functional import normalize

        # make sure if image is normlized 0-1 as in layoutlm we renormalize using clip stats
        inputs["pixel_values"] = inputs["pixel_values"] * 0.5 + 0.5  # -1 to 1 to [0, 1]
        inputs["pixel_values"] = normalize(
            inputs["pixel_values"], mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD
        )
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
        return outputs.cpu()

    embeddings, sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return embeddings, sample_ids


def _extract_paper_embeddings(
    dataloader: "DataLoader",
    device: str = "cpu",
    paper_embedding_kernel_size: int = 4,
):
    """Inner function that actually generates the embeddings."""

    def model_fn():
        from transformers import (
            LayoutLMv3Model,
        )

        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        model.to(device)
        model.eval()
        return model

    def embedding_fn(model, inputs):
        import torch
        from torch import nn

        # do layoutlmv3 forward
        outputs = model(
            input_ids=inputs["input_ids"],
            bbox=inputs["bbox"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )

        # get last last_hidden_state
        last_hidden_state_batch = outputs.last_hidden_state

        # now apply paper embedding logic
        pad_token_id = model.config.pad_token_id
        num_image_tokens = (model.config.input_size // model.config.patch_size) ** 2
        embeddings = []
        for idx in range(last_hidden_state_batch.shape[0]):
            last_hidden_state = last_hidden_state_batch[idx, :, :]  # (L, D)
            Lt = (inputs["input_ids"][idx] != pad_token_id).sum()  # its a 1D tensor
            text_embedding = last_hidden_state[:Lt, :]  # (Lt, D)
            # image_embedding_with_padding = last_hidden_state[Lt:, :]     # (Lv, D)
            image_embedding = last_hidden_state[-num_image_tokens:, :]  # (Lv, D)

            # Step 1: Mean pooling of text embeddings
            vt = text_embedding.mean(dim=0)  # shape: (D,)

            # Step 2: 1D max-pooling on image embeddings to reduce feature dimension
            # Reshape Hv to (Lv, 1, D) to apply 1D max-pooling along the feature dimension
            Hv_reshaped = image_embedding.unsqueeze(1)  # (Lv, 1, D)
            maxpool = nn.MaxPool1d(
                kernel_size=paper_embedding_kernel_size,
                stride=paper_embedding_kernel_size,
            )
            Hv_pooled = maxpool(Hv_reshaped)  # (Lv, 1, N), N < D
            Hv_pooled = Hv_pooled.squeeze(1)  # shape: (Lv, N)

            # Step 3: Mean pooling of pooled image embeddings
            vv = Hv_pooled.mean(dim=0)  # shape: (N,)

            # Step 4: Concatenate text and pooled image embeddings
            v = torch.cat([vt, vv], dim=0)  # shape: (D + N,)
            embeddings.append(v)
        return torch.stack(embeddings, dim=0)  # (B, D + N)

    embeddings, sample_ids = _iterate_dataset(
        model_fn=model_fn,
        embedding_fn=embedding_fn,
        dataloader=dataloader,
        device=device,
    )

    return embeddings, sample_ids


def embedding_extraction_with_cache(
    dataloader: "DataLoader",
    output_dir: str | Path,
    embedding_type: EmbeddingType,
    device: str = "cpu",
    cache_outputs: bool = True,
):
    """Generic cacher function that handles caching logic for any embedding type."""
    cache_file = Path(output_dir) / f"{embedding_type.value}.h5"
    if cache_outputs and cache_file.exists():
        logger.info(
            f"Loading cached {embedding_type.value} embeddings from {cache_file}"
        )
        return _load_embeddings(cache_file)

    # Generate new embeddings using the provided extraction function
    if embedding_type == EmbeddingType.layout:
        extraction_func = _extract_layoutlm_embeddings
        embeddings, sample_ids = extraction_func(dataloader, device)
    elif embedding_type == EmbeddingType.text:
        extraction_func = _extract_text_embeddings
        embeddings, sample_ids = extraction_func(dataloader, device)
    elif embedding_type == EmbeddingType.image:
        extraction_func = _extract_image_embeddings
        embeddings, sample_ids = extraction_func(dataloader, device)
    elif embedding_type == EmbeddingType.paper:
        extraction_func = _extract_paper_embeddings
        embeddings, sample_ids = extraction_func(dataloader, device)
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    if cache_outputs:
        assert len(sample_ids) == embeddings.shape[0], (
            f"Number of sample IDs ({len(sample_ids)}) must match number of embeddings ({embeddings.shape[0]})"
        )
        assert len(set(sample_ids)) == len(sample_ids), "Sample IDs must be unique"
        _save_embeddings(
            embeddings=embeddings,
            sample_ids=sample_ids,
            file_path=Path(output_dir) / f"{embedding_type.value}.h5",
        )
        return _load_embeddings(cache_file)

    return embeddings, sample_ids


def _save_embeddings(embeddings: "np.ndarray", sample_ids: list[str], file_path: Path):
    import h5py

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("sample_ids", data=sample_ids)


def _load_embeddings(file_path: Path):
    import h5py

    with h5py.File(file_path, "r") as f:
        sample_ids = f["sample_ids"][:]
        embeddings = f["embeddings"][:]
    return embeddings, [
        s.decode("utf-8") if isinstance(s, bytes) else s for s in sample_ids
    ]


def _load_sample_ids_from_embeddings(file_path: Path):
    import h5py

    with h5py.File(file_path, "r") as f:
        sample_ids = f["sample_ids"][:]
    return [  # decode and remove the index suffx
        s.decode("utf-8") if isinstance(s, bytes) else s for s in sample_ids
    ]
