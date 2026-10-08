#!/usr/bin/env python3
"""
Diffusion-based handwriting token generator with intelligent word splitting and stitching.

This script:
    - Reads handwriting bbox JSON files with format: "x1,y1,x2,y2,text,block_no,line_no,word_no"
    - Intelligently splits long words internally based on --split-length parameter
    - Splits numeric sequences within tokens into configurable chunk sizes (default: 2)
    - Generates handwriting using HuggingFace diffusion model with text conditioning
    - Stitches split word segments horizontally with baseline alignment
    - Supports sentence-level reconstruction using line metadata
    - Outputs transparent RGBA images with tight cropping
    - Maintains consistent writer styles per document
    - Supports batched generation for GPU efficiency

Usage example:
  python scripts/generate_handwriting_diffusion_raw.py \
      --input-dir docvqa-handwritten-sizes4/handwriting_bbox \
      --output-dir docvqa-handwritten-sizes4/handwriting_raw_tokens \
      --run-dir model/experiments/hf_conditional_latent \
      --checkpoint latest.pt \
      --steps 30 --split-length 7 --batch-size 8 --device cuda

With sentence stitching and custom baseline:
  python scripts/generate_handwriting_diffusion_raw.py \
    --input-dir docvqa-handwritten-sizes4/handwriting_bbox \
    --output-dir docvqa-handwritten-sizes4/handwriting_raw_tokens \
    --run-dir model/experiments/hf_conditional_latent \
    --checkpoint latest.pt \
    --steps 30 --split-length 7 --stitch-sentences \
    --baseline-percentile 85.0 --device cuda

Install requirements:
  pip install torch diffusers transformers Pillow PyYAML

Mapping file (raw_token_map.json) structure:
{
  "backend": "diffusion-hf",
  "split_length": 7,
  "entries": [
     {
       "source_json": "example.json",
       "hw_id": "hw0",
       "author_id": "author1",
       "words": [
         {
           "block_no": 22,
           "line_no": 0,
           "word_no": 0,
           "image": "example/hw0_0.png",
           "style_id": 123,
           "width": 250,
           "height": 64,
           "segments": [
             {"token": "genera", "bbox": [x1,y1,x2,y2]},
             {"token": "tion", "bbox": [x1,y1,x2,y2]}
           ]
         }
       ]
     }
  ],
  "file_author_styles": {"example.json": {"author1": {"style_id": 123}}}
}
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .tokenizer import CharTokenizer
from .text_encoder import TextEncoder

try:
    import torch
    import torch.nn as nn
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        UNet2DConditionModel,
    )
    from diffusers.training_utils import EMAModel
    import numpy as np
    from PIL import Image
    import yaml
    from rich.progress import Progress
except Exception as e:
    print(
        "[ERROR] Missing dependencies. Install: torch diffusers transformers Pillow PyYAML",
        file=sys.stderr,
    )
    raise


BBox = Tuple[float, float, float, float]


@dataclass
class WordSegment:
    """Represents a segment of a word after splitting."""

    token: str
    bbox: BBox
    original_index: (
        int  # Track which part of the word this is (0=first, 1=second, etc.)
    )
    space_before: bool = (
        False  # True if this segment had a space before it in the original word
    )


@dataclass
class WordTask:
    """Represents a complete word (possibly split into segments)."""

    source_json: str
    hw_id: str
    author_id: str
    block_no: int
    line_no: int
    word_no: int
    segments: List[WordSegment]  # List of segments if word was split
    original_bbox: BBox  # Original bbox before splitting
    include_in_sentence: bool = (
        True  # Whether this word should be considered for sentence stitching
    )
    sentence_exclusion_reason: Optional[str] = (
        None  # Reason for omitting from sentence stitching
    )


# ---------------------------- util ----------------------------


def list_json_files(p: Path) -> List[Path]:
    return sorted([x for x in p.glob("*.json") if x.is_file()])


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_bbox_record(rec: str) -> Tuple[BBox, str, int, int, int]:
    """Parse bbox record in format: x1,y1,x2,y2,text,block_no,line_no,word_no"""
    parts = rec.split(",")
    if len(parts) < 8:
        raise ValueError(f"Invalid bbox record (expected at least 8 parts): {rec}")
    x1, y1, x2, y2 = map(float, parts[:4])
    block_no = int(parts[-3])
    line_no = int(parts[-2])
    word_no = int(parts[-1])
    # Text is everything between coordinates and the last 3 indices
    token = ",".join(parts[4:-3])
    return (x1, y1, x2, y2), token, block_no, line_no, word_no


def split_word(word: str, split_length: int) -> List[str]:
    """
    Split a word into segments where each segment is AT MOST split_length characters.
    All segments will have equal or nearly equal length, with no segment exceeding split_length.

    Args:
        word: The word to split
        split_length: Maximum length for each segment

    Returns:
        List of word segments (all <= split_length)

    Examples:
        split_word("generation", 4) -> ["gen", "era", "tio", "n"] (3, 3, 3, 1)
        split_word("generation", 5) -> ["gener", "ation"] (5, 5)
        split_word("extraordinary", 7) -> ["extraor", "dinary"] (7, 7)
        split_word("extraordinary", 5) -> ["extra", "ordin", "ary"] (5, 5, 3)
        split_word("hello", 10) -> ["hello"] (5)

    Strategy:
    - Calculate minimum number of segments needed (ceil(len/split_length))
    - Distribute characters as evenly as possible
    - Ensure no segment exceeds split_length
    """
    if split_length <= 0:
        return [word]

    word_len = len(word)

    if word_len <= split_length:
        return [word]

    # Calculate minimum number of segments needed
    num_segments = (word_len + split_length - 1) // split_length  # Ceiling division

    # Calculate base length for each segment (will be <= split_length)
    base_length = word_len // num_segments
    remainder = word_len % num_segments

    # Verify base_length doesn't exceed split_length
    # (This should always be true given our calculation, but being safe)
    assert base_length <= split_length, (
        f"base_length {base_length} exceeds split_length {split_length}"
    )

    # Build segments: first 'remainder' segments get base_length+1, rest get base_length
    segments = []
    start = 0

    for i in range(num_segments):
        # First 'remainder' segments get one extra character
        seg_length = base_length + (1 if i < remainder else 0)
        segments.append(word[start : start + seg_length])
        start += seg_length

    # Verify all segments are <= split_length
    for seg in segments:
        assert len(seg) <= split_length, (
            f"Segment '{seg}' (len={len(seg)}) exceeds split_length {split_length}"
        )

    return segments


def split_token_preserving_digit_chunks(
    token: str, split_length_words: int, split_length_numeric: int
) -> List[str]:
    """
    Split a token while keeping numeric sequences in configurable chunk sizes.

    Args:
        token: The token to split.
        split_length_words: Maximum length for each non-numeric segment.
        split_length_numeric: Maximum length for numeric sequences (<=0 disables special handling).

    Returns:
        List of token segments in the original order.
    """
    if split_length_numeric <= 0:
        return split_word(token, split_length_words)

    segments: List[str] = []
    idx = 0
    token_len = len(token)

    while idx < token_len:
        if token[idx].isdigit():
            start = idx
            while idx < token_len and token[idx].isdigit():
                idx += 1
            digits = token[start:idx]
            effective_chunk = max(1, split_length_numeric)
            if split_length_words > 0:
                effective_chunk = min(effective_chunk, split_length_words)
            for chunk_start in range(0, len(digits), effective_chunk):
                segments.append(digits[chunk_start : chunk_start + effective_chunk])
        else:
            start = idx
            while idx < token_len and not token[idx].isdigit():
                idx += 1
            alpha = token[start:idx]
            if alpha:
                segments.extend(split_word(alpha, split_length_words))

    return segments or [token]


def split_word_with_spaces(
    word: str, split_length_words: int, split_length_numeric: int
) -> List[Tuple[str, bool]]:
    """
    Split a word into segments, handling spaces first, then applying length-based splitting.

    Args:
        word: The word to split (may contain spaces)
        split_length_words: Maximum length for each segment
        split_length_numeric: Maximum length for numeric sequences within each token (<=0 disables special handling)

    Returns:
        List of tuples (segment_text, space_before) where space_before indicates if this
        segment was separated by a space in the original word.

    Examples:
        split_word_with_spaces("hello world", 10) -> [("hello", False), ("world", True)]
        split_word_with_spaces("very long phrase", 5) -> [("very", False), ("long", True), ("phras", True), ("e", False)]
        split_word_with_spaces("hello", 3) -> [("hel", False), ("lo", False)]

    Strategy:
    1. Split at spaces first
    2. Apply length-based splitting (with digit chunking) to each space-separated part
    3. Mark segments that were separated by spaces with space_before=True
    """
    if not word:
        return []

    # Split at spaces first
    space_parts = word.split(" ")

    result = []
    for part_idx, part in enumerate(space_parts):
        if not part:  # Skip empty parts (from consecutive spaces)
            continue

        # Apply length-based splitting to this part
        length_segments = split_token_preserving_digit_chunks(
            part, split_length_words, split_length_numeric
        )

        for seg_idx, seg in enumerate(length_segments):
            # First segment of non-first parts had a space before it
            space_before = part_idx > 0 and seg_idx == 0
            result.append((seg, space_before))

    return result


def extract_tasks(
    json_path: Path,
    data: List[Dict[str, Any]],
    split_length_words: int,
    split_length_numeric: int,
) -> Tuple[List[WordTask], List[Dict[str, Any]]]:
    """
    Extract word tasks from JSON data, splitting long words internally.

    Args:
        json_path: Path to the JSON file
        data: Parsed JSON data
        split_length_words: Maximum word length before splitting
        split_length_numeric: Maximum length for numeric sequences within tokens (<=0 disables special handling)

    Returns:
        Tuple of (word tasks, extraction log entries)
    """
    tasks: List[WordTask] = []
    extraction_logs: List[Dict[str, Any]] = []
    fallback_counters: Dict[str, int] = defaultdict(int)
    zero_bbox: BBox = (0.0, 0.0, 0.0, 0.0)

    for obj in data:
        # Skip entries without valid data
        if obj is None:
            continue

        hw_id = obj.get("id")
        author_id = obj.get("author-id") or obj.get("author_id")
        bboxes = obj.get("bboxes")
        text_content = (obj.get("text") or "").strip()

        # Skip entries with None or empty bboxes
        if bboxes is None or not bboxes:
            if not text_content:
                extraction_logs.append(
                    {
                        "type": "extraction_skip",
                        "source_json": json_path.name,
                        "hw_id": hw_id,
                        "reason": "missing_bbox_no_text",
                    }
                )
                continue

            fallback_words = [w for w in text_content.split() if w]
            if not fallback_words:
                extraction_logs.append(
                    {
                        "type": "extraction_skip",
                        "source_json": json_path.name,
                        "hw_id": hw_id,
                        "reason": "missing_bbox_no_tokens",
                    }
                )
                continue

            for fallback_idx, raw_word in enumerate(fallback_words):
                word_segments_with_flags = split_word_with_spaces(
                    raw_word, split_length_words, split_length_numeric
                )
                if not word_segments_with_flags:
                    continue

                segments: List[WordSegment] = []
                for seg_idx, (seg_text, space_before) in enumerate(
                    word_segments_with_flags
                ):
                    segments.append(
                        WordSegment(
                            token=seg_text,
                            bbox=zero_bbox,
                            original_index=seg_idx,
                            space_before=space_before,
                        )
                    )

                fallback_counter = fallback_counters[hw_id]
                fallback_counters[hw_id] += 1
                tasks.append(
                    WordTask(
                        source_json=json_path.name,
                        hw_id=hw_id,
                        author_id=author_id,
                        block_no=-1,
                        line_no=-1,
                        word_no=100000 + fallback_counter,
                        segments=segments,
                        original_bbox=zero_bbox,
                        include_in_sentence=False,
                        sentence_exclusion_reason="missing_bbox",
                    )
                )

            extraction_logs.append(
                {
                    "type": "extraction_notice",
                    "source_json": json_path.name,
                    "hw_id": hw_id,
                    "reason": "missing_bbox_generated",
                    "num_words": len(fallback_words),
                }
            )
            continue

        for idx, rec in enumerate(bboxes):
            bbox, token, block_no, line_no, word_no = parse_bbox_record(rec)

            # Split word with space-awareness (splits at spaces first, then by length)
            word_segments_with_flags = split_word_with_spaces(
                token, split_length_words, split_length_numeric
            )

            # Create WordSegment objects for each part
            segments = []
            for seg_idx, (seg_text, space_before) in enumerate(
                word_segments_with_flags
            ):
                segments.append(
                    WordSegment(
                        token=seg_text,
                        bbox=bbox,  # Use same bbox for all segments (will adjust proportionally if needed)
                        original_index=seg_idx,
                        space_before=space_before,
                    )
                )

            tasks.append(
                WordTask(
                    source_json=json_path.name,
                    hw_id=hw_id,
                    author_id=author_id,
                    block_no=block_no,
                    line_no=line_no,
                    word_no=word_no,
                    segments=segments,
                    original_bbox=bbox,
                )
            )
    return tasks, extraction_logs


def style_id_for_file(json_name: str, author_id: str, seed: int, vocab: int) -> int:
    """Deterministically derive a style id for (json_name, author_id) combo."""
    composite = f"{json_name}::{author_id}"
    return (hash(composite) ^ seed) % vocab


def build_word_filename(task: WordTask) -> str:
    """Create a unique filename for a word using hw_id, block, line, and word numbers."""
    block_part = f"b{task.block_no}" if task.block_no is not None else "bX"
    line_part = f"l{task.line_no}" if task.line_no is not None else "lX"
    word_part = f"w{task.word_no}"
    return f"{task.hw_id}_{block_part}_{line_part}_{word_part}.png"


# ------------------------ generation -------------------------


def load_experiment(
    run_dir: Path, checkpoint_name: str, device: torch.device
) -> Dict[str, Any]:
    """
    Load model components from experiment directory.
    Based on inference_hf.ipynb load_experiment function.
    """
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config at {config_path}.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    vocab_path = Path(config["data"]["vocab_path"])
    if not vocab_path.is_absolute():
        vocab_path = run_dir / vocab_path
    if not vocab_path.exists():
        vocab_path = run_dir.parent / config["data"]["vocab_path"]

    tokenizer = CharTokenizer.load(str(vocab_path))

    # Load writer_id_map
    writer_map_path = run_dir / "writer_id_map.json"
    if not writer_map_path.exists():
        raise FileNotFoundError(f"Expected writer mapping at {writer_map_path}.")
    with open(writer_map_path, "r", encoding="utf-8") as f:
        raw_writer_map = json.load(f)
    writer_id_map = {str(k): int(v) for k, v in raw_writer_map.items()}
    num_writers = len(writer_id_map)

    # Load text encoder
    text_cfg = config["model"]["text_encoder"]
    text_encoder = TextEncoder(
        vocab_size=len(tokenizer),
        d_model=text_cfg["d_model"],
        num_layers=text_cfg["num_layers"],
        num_heads=text_cfg["num_heads"],
        d_ff=text_cfg["d_ff"],
        dropout=text_cfg["dropout"],
        max_length=text_cfg["max_length"],
        output_dim=text_cfg.get("output_dim", text_cfg["d_model"]),
    ).to(device)
    text_encoder.eval()

    # Load UNet
    unet_cfg = deepcopy(config["model"]["unet"])
    pretrained_path = unet_cfg.pop("pretrained_model_name_or_path", None)

    # Ensure tuple types
    for key in ("down_block_types", "up_block_types", "block_out_channels"):
        if key in unet_cfg and isinstance(unet_cfg[key], list):
            unet_cfg[key] = tuple(unet_cfg[key])

    if "sample_size" in unet_cfg and isinstance(unet_cfg["sample_size"], list):
        unet_cfg["sample_size"] = tuple(unet_cfg["sample_size"])

    # Set num_class_embeds from writer_id_map
    unet_cfg["num_class_embeds"] = num_writers

    if pretrained_path:
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_path, num_class_embeds=num_writers
        ).to(device)
    else:
        unet = UNet2DConditionModel(**unet_cfg).to(device)

    unet.eval()

    # Load scheduler - using DPM-Solver++ with order 3 for fast, high-quality sampling
    scheduler_cfg = config["model"]["scheduler"]
    noise_scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
        beta_start=scheduler_cfg["beta_start"],
        beta_end=scheduler_cfg["beta_end"],
        beta_schedule=scheduler_cfg["beta_schedule"],
        prediction_type=scheduler_cfg.get("prediction_type", "epsilon"),
        algorithm_type="dpmsolver++",
        solver_order=3,  # Higher order = better quality
        use_karras_sigmas=scheduler_cfg.get("use_karras_sigmas", False),
    )
    # Add timestep_spacing if specified in config
    if "timestep_spacing" in scheduler_cfg:
        noise_scheduler.config.timestep_spacing = scheduler_cfg["timestep_spacing"]

    # Load VAE if latent mode
    mode = config["training"].get("mode", "latent")
    vae = None
    vae_scale_factor = 0.18215
    if mode == "latent":
        vae_config = config["model"].get("vae")
        if vae_config is None:
            raise KeyError("Latent mode requires 'model.vae' configuration.")
        vae_model_name = vae_config["model_name"]

        vae_cache_dir = run_dir / "cached_vae"
        if vae_cache_dir.exists():
            vae = AutoencoderKL.from_pretrained(vae_cache_dir).to(device)
        else:
            vae = AutoencoderKL.from_pretrained(vae_model_name).to(device)
            vae_cache_dir.mkdir(parents=True, exist_ok=True)
            vae.save_pretrained(vae_cache_dir)

        vae.eval()

    # Load checkpoint
    checkpoint_path = run_dir / checkpoint_name

    print(checkpoint_path)
    if not checkpoint_path.exists():
        checkpoint_path = Path(checkpoint_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_name} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    text_encoder.load_state_dict(checkpoint["text_encoder"])
    unet.load_state_dict(checkpoint["unet"], strict=False)

    # Load EMA if available
    ema_model = None
    if "ema" in checkpoint:
        training_cfg = config.get("training", {})
        use_warmup = training_cfg.get("ema_use_warmup", False)
        ema_model = EMAModel(
            unet.parameters(),
            decay=training_cfg.get("ema_decay", 0.9999),
            use_ema_warmup=use_warmup,
            inv_gamma=training_cfg.get("ema_inv_gamma", 1.0),
            power=training_cfg.get("ema_power", 1.0),
            min_decay=training_cfg.get("ema_min_decay", 0.0),
            device=device,
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )
        ema_model.load_state_dict(checkpoint["ema"])
        ema_model.to(device)
        ema_model.copy_to(unet.parameters())

    latent_shape = config["model"].get("latent_shape")
    image_shape = config["model"].get("image_shape")
    if mode == "latent":
        sample_shape = tuple(latent_shape)
    else:
        sample_shape = tuple(image_shape)

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "vae": vae,
        "vae_scale_factor": vae_scale_factor,
        "writer_id_map": writer_id_map,
        "device": device,
        "config": config,
        "sample_shape": sample_shape,
        "mode": mode,
    }


def diffusion_generate_batch(
    tokens: List[str],
    style_ids: List[int],
    components: Dict[str, Any],
    steps: int,
    temperature: float = 1.0,
) -> List[Image.Image]:
    """
    Generate batch of handwriting images using diffusion model.
    Based on sample_diffusion from inference_hf.ipynb.
    """
    if not tokens:
        return []

    device = components["device"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    unet = components["unet"]
    noise_scheduler = components["noise_scheduler"]
    sample_shape = components["sample_shape"]
    mode = components["mode"]
    vae = components.get("vae")
    vae_scale_factor = components.get("vae_scale_factor", 0.18215)

    # Encode text
    encodings = tokenizer.encode_batch(tokens)
    input_ids = torch.tensor(encodings["input_ids"], device=device, dtype=torch.long)
    attention_mask = torch.tensor(
        encodings["attention_mask"], device=device, dtype=torch.float32
    )

    # Convert writer style IDs to class indices
    writer_indices = torch.tensor(style_ids, device=device, dtype=torch.long)

    # Set timesteps
    noise_scheduler.set_timesteps(steps, device=device)
    timesteps = noise_scheduler.timesteps

    # Initialize latents
    batch_shape = (len(tokens),) + tuple(sample_shape)
    latents = torch.randn(batch_shape, device=device) * temperature

    # Generate text features
    with torch.no_grad():
        text_features = text_encoder(input_ids, attention_mask=attention_mask)

        # Sampling loop
        for timestep in timesteps:
            t_batch = torch.full(
                (len(tokens),), int(timestep), device=device, dtype=torch.long
            )

            model_output = unet(
                latents,
                t_batch,
                encoder_hidden_states=text_features,
                encoder_attention_mask=attention_mask,
                class_labels=writer_indices,
            )
            noise_pred = (
                model_output.sample if hasattr(model_output, "sample") else model_output
            )

            scheduler_step = noise_scheduler.step(noise_pred, int(timestep), latents)
            latents = scheduler_step.prev_sample

        # Decode if latent mode
        if mode == "latent" and vae is not None:
            latents = latents / vae_scale_factor
            decoded = vae.decode(latents).sample
        else:
            decoded = latents

        images = (decoded / 2 + 0.5).clamp(0.0, 1.0)

    # Convert to PIL images with cropping and transparency
    results: List[Image.Image] = []
    imgs = images.cpu().numpy()

    for i in range(len(tokens)):
        arr = imgs[i]
        if arr.shape[0] == 1:
            arr = arr[0]  # Remove channel dim if grayscale
        else:
            arr = arr.transpose(1, 2, 0)  # CHW -> HWC

        arr8 = (arr * 255).round().astype("uint8")

        # Binarize
        if arr8.ndim == 3:
            arr8 = arr8.mean(axis=2).astype("uint8")

        thresh = otsu_threshold(arr8)
        bin_arr = (arr8 > thresh).astype("uint8") * 255

        # Crop to content
        cropped, crop_box = crop_to_content(bin_arr)

        # Convert to RGBA
        rgba = binary_to_rgba(cropped)
        rgba.info["crop_box"] = crop_box
        results.append(rgba)

    return results


# ---------------------- binarization utils -------------------


def otsu_threshold(arr8):
    hist = np.bincount(arr8.ravel(), minlength=256).astype(np.float64)
    total = arr8.size
    sum_total = (hist * np.arange(256)).sum()
    weight_bg = 0.0
    sum_bg = 0.0
    max_between = -1.0
    thresh = 0
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            thresh = i
    return thresh


# ---------------------- cropping & alpha --------------------


def crop_to_content(bin_arr: np.ndarray, pad: int = 0):
    """Crop binary array (0=ink,255=bg) to tight bounding box. Returns (cropped_array, (x1,y1,x2,y2))."""
    h, w = bin_arr.shape
    ink_mask = bin_arr < 255
    if not ink_mask.any():
        # No ink; return 1x1 transparent placeholder
        return bin_arr[:1, :1], (0, 0, 1, 1)
    rows = np.where(ink_mask.any(axis=1))[0]
    cols = np.where(ink_mask.any(axis=0))[0]
    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]
    if pad:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)
    cropped = bin_arr[y1 : y2 + 1, x1 : x2 + 1]
    return cropped, (
        int(x1),
        int(y1),
        int(x2) + 1,
        int(y2) + 1,
    )  # x2,y2 exclusive for convenience


def binary_to_rgba(bin_arr: np.ndarray) -> Image.Image:
    """Convert binary (0 ink, 255 bg) to RGBA with transparent background."""
    h, w = bin_arr.shape
    # Ink black RGB (0,0,0), alpha 255 where ink, 0 where bg
    alpha = (bin_arr == 0).astype("uint8") * 255
    rgb = np.zeros((h, w, 3), dtype="uint8")  # already black
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


def pad_tokens_to_equal_length(tokens: List[str]) -> List[str]:
    """Pad tokens to equal length by appending spaces to shorter tokens."""
    if not tokens:
        return tokens
    max_len = max(len(t) for t in tokens)
    print([t.ljust(max_len) for t in tokens])
    return [t.ljust(max_len) for t in tokens]


def calculate_baseline_info(
    img: Image.Image, baseline_percentile: float = 85.0
) -> Dict[str, Any]:
    """
    Calculate baseline information for an RGBA image.

    Args:
        img: RGBA PIL Image
        baseline_percentile: Percentile for baseline detection (default: 85.0)

    Returns:
        Dictionary with baseline metrics:
        - baseline_y: Absolute baseline position (pixels from top)
        - baseline_ratio: Baseline as ratio of height (0.0-1.0)
        - height_above: Pixels above baseline
        - height_below: Pixels below baseline
        - ascender_ratio: Ratio of height above baseline
        - descender_ratio: Ratio of height below baseline
    """
    arr = np.array(img)
    height = img.height

    if arr.shape[2] == 4:  # RGBA
        alpha = arr[:, :, 3]
    else:
        alpha = np.ones((height, img.width), dtype=np.uint8) * 255

    ink_mask = alpha > 200

    if not ink_mask.any():
        # No ink, use bottom as baseline
        baseline_y = height - 1
    else:
        # Find bottom-most ink pixels for each column
        bottom_candidates = []
        cols_with_ink = np.where(ink_mask.any(axis=0))[0]
        for col_idx in cols_with_ink:
            ink_rows = np.where(ink_mask[:, col_idx])[0]
            if ink_rows.size > 0:
                bottom_candidates.append(int(ink_rows[-1]))

        if bottom_candidates:
            baseline_y = int(np.percentile(bottom_candidates, baseline_percentile))
        else:
            baseline_y = height - 1

    height_above = baseline_y
    height_below = height - 1 - baseline_y

    return {
        "baseline_y": baseline_y,
        "baseline_ratio": baseline_y / height if height > 0 else 0.0,
        "height_above": height_above,
        "height_below": height_below,
        "ascender_ratio": height_above / height if height > 0 else 0.0,
        "descender_ratio": height_below / height if height > 0 else 0.0,
    }


def concatenate_images_horizontal(
    images: List[Image.Image],
    gap: int = 0,
    baseline_align: bool = True,
    baseline_percentile: float = 75.0,
) -> Image.Image:
    """
    Horizontally concatenate a list of RGBA images with baseline alignment.

    Args:
        images: List of RGBA images to concatenate
        gap: Spacing between images in pixels
        baseline_align: If True, align by baseline; if False, center vertically
        baseline_percentile: Percentile for baseline detection (default: 85.0)

    Returns:
        Concatenated RGBA image
    """
    if not images:
        raise ValueError("Cannot concatenate empty image list")
    if len(images) == 1:
        return images[0]

    if baseline_align:
        # Calculate baseline for each image
        baselines = []
        max_above_baseline = 0
        max_below_baseline = 0

        for img in images:
            # Convert to grayscale array
            arr = np.array(img)
            if arr.shape[2] == 4:  # RGBA
                alpha = arr[:, :, 3]
            else:
                alpha = np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255

            # Find ink pixels
            ink_mask = alpha > 200

            if not ink_mask.any():
                # No ink, use bottom as baseline
                baseline = img.height - 1
            else:
                # Find bottom-most ink pixels for each column (optimized: only iterate columns with ink)
                bottom_candidates = []
                cols_with_ink = np.where(ink_mask.any(axis=0))[0]
                for col_idx in cols_with_ink:
                    ink_rows = np.where(ink_mask[:, col_idx])[0]
                    if ink_rows.size > 0:
                        bottom_candidates.append(int(ink_rows[-1]))

                if bottom_candidates:
                    baseline = int(
                        np.percentile(bottom_candidates, baseline_percentile)
                    )
                else:
                    baseline = img.height - 1

            baselines.append(baseline)

            # Calculate space above and below baseline
            above = baseline
            below = img.height - 1 - baseline
            max_above_baseline = max(max_above_baseline, above)
            max_below_baseline = max(max_below_baseline, below)

        # Total height needed
        canvas_height = max_above_baseline + 1 + max_below_baseline
        total_width = sum(img.width for img in images) + gap * (len(images) - 1)

        # Create canvas
        result = Image.new("RGBA", (total_width, canvas_height), (0, 0, 0, 0))

        # Paste images aligned by baseline
        x_offset = 0
        for img, baseline in zip(images, baselines):
            # Calculate y position to align baselines
            y_offset = max_above_baseline - baseline
            result.paste(img, (x_offset, y_offset), img)
            x_offset += img.width + gap
    else:
        # Simple vertical centering
        max_height = max(img.height for img in images)
        total_width = sum(img.width for img in images) + gap * (len(images) - 1)

        result = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))

        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2
            result.paste(img, (x_offset, y_offset), img)
            x_offset += img.width + gap

    return result


def concatenate_segments_with_variable_gaps(
    images: List[Image.Image],
    segments: List[WordSegment],
    segment_gap: int = 2,
    word_gap: int = 20,
    baseline_percentile: float = 75.0,
) -> Image.Image:
    """
    Concatenate word segments with variable gaps based on whether they were separated by spaces.

    Args:
        images: List of RGBA segment images (same length as segments)
        segments: List of WordSegment objects with space_before flags
        segment_gap: Gap for length-split segments (no space in original)
        word_gap: Gap for space-separated segments
        baseline_percentile: Percentile for baseline detection

    Returns:
        Concatenated RGBA image with appropriate gaps
    """
    if not images:
        raise ValueError("Cannot concatenate empty image list")
    if len(images) == 1:
        return images[0]
    if len(images) != len(segments):
        raise ValueError(f"Mismatch: {len(images)} images but {len(segments)} segments")

    # Calculate baseline for each image
    baselines = []
    max_above_baseline = 0
    max_below_baseline = 0

    for img in images:
        arr = np.array(img)
        if arr.shape[2] == 4:  # RGBA
            alpha = arr[:, :, 3]
        else:
            alpha = np.ones((arr.shape[0], arr.shape[1]), dtype=np.uint8) * 255

        ink_mask = alpha > 200

        if not ink_mask.any():
            baseline = img.height - 1
        else:
            bottom_candidates = []
            cols_with_ink = np.where(ink_mask.any(axis=0))[0]
            for col_idx in cols_with_ink:
                ink_rows = np.where(ink_mask[:, col_idx])[0]
                if ink_rows.size > 0:
                    bottom_candidates.append(int(ink_rows[-1]))

            if bottom_candidates:
                baseline = int(np.percentile(bottom_candidates, baseline_percentile))
            else:
                baseline = img.height - 1

        baselines.append(baseline)
        above = baseline
        below = img.height - 1 - baseline
        max_above_baseline = max(max_above_baseline, above)
        max_below_baseline = max(max_below_baseline, below)

    # Calculate total width based on variable gaps
    canvas_height = max_above_baseline + 1 + max_below_baseline
    total_width = sum(img.width for img in images)
    for i in range(1, len(images)):
        # Use word_gap if this segment had a space before it, else segment_gap
        gap = word_gap if segments[i].space_before else segment_gap
        total_width += gap

    # Create canvas and paste images
    result = Image.new("RGBA", (total_width, canvas_height), (0, 0, 0, 0))

    x_offset = 0
    for i, (img, baseline, segment) in enumerate(zip(images, baselines, segments)):
        y_offset = max_above_baseline - baseline
        result.paste(img, (x_offset, y_offset), img)
        x_offset += img.width

        # Add appropriate gap before next image
        if i < len(images) - 1:
            gap = word_gap if segments[i + 1].space_before else segment_gap
            x_offset += gap

    return result


# -------------------------- main -----------------------------


def generate_handwriting(
    input_dir: Path,
    output_dir: Path,
    run_dir: Path,
    checkpoint: str = "latest.pt",
    progress: Progress | None = None,
    steps: int = 30,
    split_length_words: int = 6,
    split_length_numeric: int = 2,
    temperature: float = 0.5,
    seed: int = 42,
    device: str = "cuda",
    overwrite: bool = False,
    mapping_file: Optional[Path] = None,
    log_file: Optional[Path] = None,
    batch_size: int = 32,
    stitch_sentences: bool = True,
    segment_gap: int = 2,
    word_gap: int = 20,
    baseline_percentile: float = 75.0,
    allowed_writers: Optional[List[str]] = None,
) -> None:
    """Generate handwriting images and metadata using configured diffusion models."""
    random.seed(seed)
    torch.manual_seed(seed)
    device_obj = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    run_dir = Path(run_dir)
    mapping_file = Path(mapping_file) if mapping_file is not None else None
    log_file = Path(log_file) if log_file is not None else None

    # Load model components
    print(f"Loading model from {run_dir}...")
    components = load_experiment(run_dir, checkpoint, device_obj)
    print(f"✓ Model loaded successfully")
    print(f"  Mode: {components['mode']}")
    print(f"  Sample shape: {components['sample_shape']}")
    print(f"  Writers: {len(components['writer_id_map'])}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON files
    json_files = list_json_files(input_dir)
    if not json_files:
        print("[ERROR] No JSON files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files")

    # Extract tasks with word splitting
    tasks: List[WordTask] = []
    extraction_logs: List[Dict[str, Any]] = []
    for jf in json_files:
        data = load_json(jf)
        extracted_tasks, extracted_log_entries = extract_tasks(
            jf, data, split_length_words, split_length_numeric
        )
        tasks.extend(extracted_tasks)
        extraction_logs.extend(extracted_log_entries)

    print(f"Extracted {len(tasks)} word tasks")
    if split_length_words > 0:
        total_segments = sum(len(t.segments) for t in tasks)
        print(
            f"  Split into {total_segments} segments (split_length={split_length_words}, digit_chunk_length={split_length_numeric})"
        )

    # Per-file author style mapping
    file_author_style_ids: Dict[str, Dict[str, int]] = {}
    writer_id_map = components["writer_id_map"]

    # Filter to allowed writers if specified
    allowed_writer_ids = None
    if allowed_writers is not None:
        allowed_writer_ids = []
        for w in allowed_writers:
            try:
                writer_id = int(w)
                if 0 <= writer_id < len(writer_id_map):
                    allowed_writer_ids.append(writer_id)
                else:
                    print(
                        f"[WARNING] Writer ID {writer_id} out of range (0-{len(writer_id_map) - 1}), ignoring"
                    )
            except ValueError:
                print(f"[WARNING] Invalid writer ID '{w}', must be integer, ignoring")

        if not allowed_writer_ids:
            print("[ERROR] No valid writer IDs provided in --allowed-writers")
            sys.exit(1)

        print(
            f"Using {len(allowed_writer_ids)} allowed writer(s): {sorted(allowed_writer_ids)}"
        )

    # Set up RNG for random writer selection if needed
    rng = random.Random(seed)

    for t in tasks:
        file_author_style_ids.setdefault(t.source_json, {})
        if t.author_id not in file_author_style_ids[t.source_json]:
            # Map author_id to writer index from the model's writer_id_map
            if t.author_id in writer_id_map:
                style_id = writer_id_map[t.author_id]
                # If allowed_writers specified and this author's style not in list, randomly pick from allowed
                if (
                    allowed_writer_ids is not None
                    and style_id not in allowed_writer_ids
                ):
                    style_id = rng.choice(allowed_writer_ids)
            else:
                # Author not in map: use allowed writers if specified, else fallback to hashing
                if allowed_writer_ids is not None:
                    style_id = rng.choice(allowed_writer_ids)
                else:
                    style_id = style_id_for_file(
                        t.source_json, t.author_id, seed, len(writer_id_map)
                    )
            file_author_style_ids[t.source_json][t.author_id] = style_id

    results: List[Dict[str, Any]] = []
    generation_skip_log: List[Dict[str, Any]] = []
    generation_error_log: List[Dict[str, Any]] = []
    sentence_exclusion_log: List[Dict[str, Any]] = []
    total_words = len(tasks)
    effective_batch_size = max(1, batch_size)
    progress = progress or Progress(transient=True)
    generation_task_id = progress.add_task("Generating words", total=total_words)

    for word_idx in range(0, total_words, effective_batch_size):
        batch_tasks = tasks[word_idx : word_idx + effective_batch_size]

        # Process each word task
        for task in batch_tasks:
            json_stem = Path(task.source_json).stem
            doc_dir = output_dir / json_stem
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Output filename includes block and line numbers to avoid collisions across lines
            out_name = build_word_filename(task)
            relative_image_path = f"{json_stem}/{out_name}"
            out_path = doc_dir / out_name

            if out_path.exists() and not overwrite:
                # Load existing metadata
                try:
                    existing_img = Image.open(out_path)
                    w, h = existing_img.size
                    baseline_info = calculate_baseline_info(
                        existing_img, baseline_percentile=baseline_percentile
                    )
                    results.append(
                        {
                            "image": relative_image_path,
                            "hw_id": task.hw_id,
                            "author_id": task.author_id,
                            "style_id": file_author_style_ids[task.source_json][
                                task.author_id
                            ],
                            "source_json": task.source_json,
                            "block_no": task.block_no,
                            "line_no": task.line_no,
                            "word_no": task.word_no,
                            "segments": [
                                {
                                    "token": seg.token,
                                    "bbox": list(seg.bbox),
                                    "space_before": seg.space_before,
                                }
                                for seg in task.segments
                            ],
                            "skipped": True,
                            "skip_reason": "existing_output",
                            "include_in_sentence": task.include_in_sentence,
                            "sentence_exclusion_reason": task.sentence_exclusion_reason,
                            "width": w,
                            "height": h,
                            "baseline": baseline_info,
                        }
                    )
                    generation_skip_log.append(
                        {
                            "type": "existing_output",
                            "source_json": task.source_json,
                            "hw_id": task.hw_id,
                            "word_no": task.word_no,
                            "block_no": task.block_no,
                            "line_no": task.line_no,
                            "image": relative_image_path,
                        }
                    )
                    if not task.include_in_sentence:
                        sentence_exclusion_log.append(
                            {
                                "source_json": task.source_json,
                                "hw_id": task.hw_id,
                                "word_no": task.word_no,
                                "block_no": task.block_no,
                                "line_no": task.line_no,
                                "image": relative_image_path,
                                "reason": task.sentence_exclusion_reason
                                or "manual_exclusion",
                            }
                        )
                except Exception as e:
                    print(f"[WARN] Could not load existing {out_path}: {e}")
                continue

            # Generate all segments for this word
            try:
                tokens_batch = [seg.token for seg in task.segments]
                style_id = file_author_style_ids[task.source_json][task.author_id]
                style_ids_batch = [style_id] * len(tokens_batch)

                segment_images = diffusion_generate_batch(
                    tokens_batch,
                    style_ids_batch,
                    components,
                    steps,
                    temperature=temperature,
                )

                # Concatenate segments with variable gaps (word-gap for spaces, segment-gap for length splits)
                if len(segment_images) > 1:
                    final_image = concatenate_segments_with_variable_gaps(
                        segment_images,
                        task.segments,
                        segment_gap=segment_gap,
                        word_gap=word_gap,
                        baseline_percentile=baseline_percentile,
                    )
                else:
                    final_image = segment_images[0]

                # Save
                w, h = final_image.size
                final_image.save(out_path)

                # Calculate baseline information for alignment
                baseline_info = calculate_baseline_info(
                    final_image, baseline_percentile=baseline_percentile
                )

                results.append(
                    {
                        "image": relative_image_path,
                        "hw_id": task.hw_id,
                        "author_id": task.author_id,
                        "style_id": style_id,
                        "source_json": task.source_json,
                        "block_no": task.block_no,
                        "line_no": task.line_no,
                        "word_no": task.word_no,
                        "segments": [
                            {
                                "token": seg.token,
                                "bbox": list(seg.bbox),
                                "space_before": seg.space_before,
                            }
                            for seg in task.segments
                        ],
                        "skipped": False,
                        "skip_reason": None,
                        "include_in_sentence": task.include_in_sentence,
                        "sentence_exclusion_reason": task.sentence_exclusion_reason,
                        "width": w,
                        "height": h,
                        "baseline": baseline_info,
                    }
                )
                if not task.include_in_sentence:
                    sentence_exclusion_log.append(
                        {
                            "source_json": task.source_json,
                            "hw_id": task.hw_id,
                            "word_no": task.word_no,
                            "block_no": task.block_no,
                            "line_no": task.line_no,
                            "image": relative_image_path,
                            "reason": task.sentence_exclusion_reason
                            or "manual_exclusion",
                        }
                    )
            except Exception as e:
                print(
                    f"[ERROR] Generation failed for {task.hw_id} word {task.word_no}: {e}",
                    file=sys.stderr,
                )
                import traceback

                traceback.print_exc()
                generation_error_log.append(
                    {
                        "type": "generation_error",
                        "source_json": task.source_json,
                        "hw_id": task.hw_id,
                        "word_no": task.word_no,
                        "block_no": task.block_no,
                        "line_no": task.line_no,
                        "reason": str(e),
                        "traceback": traceback.format_exc(),
                    }
                )

        if progress and generation_task_id is not None:
            progress.advance(generation_task_id, len(batch_tasks))

    # Sentence-level stitching (if requested)
    if stitch_sentences:
        print("\nStitching words into sentences...")
        sentences_dir = output_dir / "sentences"
        sentences_dir.mkdir(exist_ok=True)

        # Group results by (source_json, hw_id, block_no, line_no)
        line_groups: Dict[Tuple[str, str, int, int], List[Dict[str, Any]]] = {}
        for r in results:
            if r["skipped"]:
                continue
            if not r.get("include_in_sentence", True):
                continue
            key = (r["source_json"], r["hw_id"], r["block_no"], r["line_no"])
            line_groups.setdefault(key, []).append(r)

        # Sort words within each line by word_no
        for key in line_groups:
            line_groups[key].sort(key=lambda x: x["word_no"])

        sentence_results: List[Dict[str, Any]] = []
        sentence_progress = progress
        sentence_task_id = sentence_progress.add_task(
            "Stitching sentences", total=len(line_groups)
        )

        for (source_json, hw_id, block_no, line_no), word_list in line_groups.items():
            if not word_list:
                continue

            json_stem = Path(source_json).stem
            sent_doc_dir = sentences_dir / json_stem
            sent_doc_dir.mkdir(parents=True, exist_ok=True)

            # Output filename: hw{id}_block{block}_line{line}.png
            sent_name = f"{hw_id}_block{block_no}_line{line_no}.png"
            sent_relative_path = f"sentences/{json_stem}/{sent_name}"
            sent_path = sent_doc_dir / sent_name

            if sent_path.exists() and not overwrite:
                if sentence_progress and sentence_task_id is not None:
                    sentence_progress.advance(sentence_task_id, 1)
                continue

            try:
                # Load all word images for this line
                word_images = []
                for word_data in word_list:
                    word_img_path = output_dir / word_data["image"]
                    if word_img_path.exists():
                        word_images.append(Image.open(word_img_path))

                if not word_images:
                    continue

                # Stitch words together with larger gap
                sentence_image = concatenate_images_horizontal(
                    word_images,
                    gap=word_gap,
                    baseline_align=True,
                    baseline_percentile=baseline_percentile,
                )

                # Save sentence image
                sentence_image.save(sent_path)

                # Collect text for this line
                line_text = " ".join(
                    [
                        "".join([seg["token"] for seg in w["segments"]])
                        for w in word_list
                    ]
                )

                sentence_results.append(
                    {
                        "image": sent_relative_path,
                        "source_json": source_json,
                        "hw_id": hw_id,
                        "block_no": block_no,
                        "line_no": line_no,
                        "text": line_text,
                        "num_words": len(word_list),
                        "width": sentence_image.width,
                        "height": sentence_image.height,
                    }
                )

            except Exception as e:
                print(
                    f"[ERROR] Failed to stitch sentence {hw_id} block{block_no} line{line_no}: {e}",
                    file=sys.stderr,
                )

            if sentence_progress and sentence_task_id is not None:
                sentence_progress.advance(sentence_task_id, 1)

        # Save sentence mapping
        sentence_mapping_file = sentences_dir / "sentence_map.json"
        with sentence_mapping_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "backend": "diffusion-hf-sentences",
                    "word_gap": word_gap,
                    "sentences": sentence_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"✓ Generated {len(sentence_results)} sentence images")
        print(f"✓ Sentence mapping saved: {sentence_mapping_file}")

    # Build mapping structure
    entries_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in results:
        key = (r["source_json"], r["hw_id"])
        entries_map.setdefault(key, []).append(r)

    # Export file author styles
    file_author_styles_export = {
        fname: {aid: {"style_id": sid} for aid, sid in inner.items()}
        for fname, inner in sorted(file_author_style_ids.items())
    }

    consolidated = {
        "backend": "diffusion-hf",
        "split_length": split_length_words,
        "digit_chunk_length": split_length_numeric,
        "temperature": temperature,
        "steps": steps,
        "segment_gap": segment_gap,
        "word_gap": word_gap if stitch_sentences else None,
        "baseline_percentile": baseline_percentile,
        "entries": [
            {
                "source_json": src,
                "hw_id": hw,
                "author_id": words[0]["author_id"] if words else None,
                "words": [
                    {
                        "block_no": w["block_no"],
                        "line_no": w["line_no"],
                        "word_no": w["word_no"],
                        "image": w["image"],
                        "style_id": w["style_id"],
                        "width": w["width"],
                        "height": w["height"],
                        "baseline": w["baseline"],
                        "segments": w["segments"],
                    }
                    for w in sorted(
                        words, key=lambda x: (x["block_no"], x["line_no"], x["word_no"])
                    )
                ],
            }
            for (src, hw), words in sorted(entries_map.items())
        ],
        "file_author_styles": file_author_styles_export,
    }

    mapping_path = mapping_file or (output_dir / "raw_token_map.json")
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False, indent=2)

    generated_count = sum(1 for r in results if not r["skipped"])
    reused_count = sum(1 for r in results if r["skipped"])
    log_file_path = log_file or (output_dir / "generation_log.json")
    log_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_tasks": len(tasks),
            "extraction_skips": len(
                [
                    entry
                    for entry in extraction_logs
                    if entry.get("type") == "extraction_skip"
                ]
            ),
            "words_generated": generated_count,
            "words_reused": reused_count,
            "generation_errors": len(generation_error_log),
            "sentence_exclusions": len(sentence_exclusion_log),
        },
        "details": {
            "extraction": extraction_logs,
            "generation_skips": generation_skip_log,
            "generation_errors": generation_error_log,
            "sentence_exclusions": sentence_exclusion_log,
        },
    }
    with log_file_path.open("w", encoding="utf-8") as log_fp:
        json.dump(log_payload, log_fp, ensure_ascii=False, indent=2)

    print(f"\n✓ Generated {len(results)} word images")
    print(f"✓ Mapping saved: {mapping_path}")
    print(f"✓ Log saved: {log_file_path}")
    print("[DONE] Freeing up memory..")
    for k, v in components.items():
        del v
    del components
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diffusion-based handwriting token generator with intelligent word splitting."
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing bbox JSON files",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for generated images",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Model experiment directory (e.g., model/experiments/hf_conditional_latent)",
    )
    ap.add_argument(
        "--checkpoint", type=str, default="latest.pt", help="Checkpoint filename"
    )
    ap.add_argument("--steps", type=int, default=30, help="Number of diffusion steps")
    ap.add_argument(
        "--split-length-words",
        type=int,
        default=6,
        help="Maximum word length before splitting (0 = no splitting)",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.5, help="Sampling temperature"
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing images"
    )
    ap.add_argument(
        "--mapping-file", type=Path, default=None, help="Output mapping JSON path"
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path for JSON log output (default: output_dir/generation_log.json)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for generation"
    )
    ap.add_argument(
        "--stitch-sentences",
        default=True,
        action="store_true",
        help="Generate sentence-level stitched images in separate folder",
    )
    ap.add_argument(
        "--segment-gap",
        type=int,
        default=2,
        help="Gap between word segments (split parts) in pixels",
    )
    ap.add_argument(
        "--word-gap",
        type=int,
        default=20,
        help="Gap between words in sentence stitching in pixels",
    )
    ap.add_argument(
        "--baseline-percentile",
        type=float,
        default=75.0,
        help="Percentile for baseline detection (0-100, default: 85.0)",
    )
    ap.add_argument(
        "--allowed-writers",
        type=str,
        nargs="+",
        default=None,
        help="List of allowed writer IDs to choose from (e.g., --allowed-writers 0 5 10 25)",
    )
    args = ap.parse_args()

    generate_handwriting(**vars(args))


if __name__ == "__main__":
    main()
