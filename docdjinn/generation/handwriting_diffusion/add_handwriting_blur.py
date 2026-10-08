#!/usr/bin/env python3
"""
Post-process generated handwriting token images (e.g. from `generate_handwriting_diffusion_raw.py`) to reduce pixelation
and add a natural soft edge via Gaussian blur + optional scale anti-aliasing.

Features:
  * Recursively scans an input root directory for PNG images (expects per-document subfolders)
  * Applies a randomized (or fixed) Gaussian blur radius (on RGB while preserving alpha)
  * Optional downscale+upscale anti-alias pass before blur to smooth jagged edges
    * Advanced edge refinement options (erosion/dilation/feather of alpha, contrast/gamma, noise, unsharp mask)
  * Writes results either (a) in-place with a suffix before extension or (b) into a mirror output directory tree
  * Can update an existing mapping JSON (e.g. raw_token_map.json) by appending `blurred_image` for each segment

Typical usage (mirror output tree):
  python scripts/add_handwriting_blur.py \
      --input-root syn_docvqa/handwriting_raw_tokens \
      --output-root syn_docvqa/handwriting_raw_tokens_blurred \
      --mapping-json syn_docvqa/handwriting_raw_tokens/raw_token_map.json \
      --append-mapping \
      --radius-min 0.6 --radius-max 1.8 --antialias

In-place variant (adds suffix _b):
  python scripts/add_handwriting_blur.py \
      --input-root syn_docvqa/handwriting_raw_tokens \
      --in-place --suffix _soft --radius 1.2 --append-mapping

Key Arguments:
  --radius:        Fixed Gaussian blur radius (overrides min/max if set)
  --radius-min/max Range for random uniform blur radius per image when --radius not given
  --antialias      Enable a downscale+upscale pass before blur (slower but smoother)
  --scale-factor   Downscale factor when antialiasing (default 0.75)
  --suffix         Filename suffix (only used in --in-place mode)
  --append-mapping Update mapping JSON adding a 'blurred_image' key per segment (keeps original).
  --skip-existing  Skip processing if blurred file already exists
  --extensions     Comma separated list of extensions to process (default: .png)

Mapping Update Behavior:
  - Loads JSON, finds segments with an 'image' field
  - If a blurred counterpart is produced, adds 'blurred_image' (relative path analogous to original)
  - Writes updated mapping next to original unless --mapping-output specified

Limitations:
  - Mapping update assumes relative paths in JSON remain valid under original root. If you mirror into a different
    --output-root, the blurred path is generated accordingly.
  - Only PNG RGBA expected; non-RGBA images are converted.

Requires: Pillow, numpy (for advanced options)
(Optional) tqdm for progress bar.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
from types import SimpleNamespace

from PIL import Image, ImageFilter
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional
    tqdm = None  # type: ignore


@dataclass
class BlurConfig:
    radius: Optional[float]
    radius_min: float
    radius_max: float
    antialias: bool
    scale_factor: float
    suffix: str
    skip_existing: bool
    alpha_erosion: int
    alpha_dilation: int
    feather: float
    contrast: float
    ink_gamma: float
    add_noise: float
    unsharp: Optional[str]
    max_alpha: Optional[int]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Apply soft blur to handwriting token images."
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing per-document subfolders with images.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        help="Root to write blurred images (mirrors structure). Omit with --in-place.",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="Blur in-place (creates new files with suffix).",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="_b",
        help="Suffix to append before extension in in-place mode.",
    )
    ap.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Fixed blur radius (overrides min/max).",
    )
    ap.add_argument(
        "--radius-min",
        type=float,
        default=0.35,
        help="Min random blur radius when --radius not set (slight blur).",
    )
    ap.add_argument(
        "--radius-max",
        type=float,
        default=0.85,
        help="Max random blur radius when --radius not set (slight blur).",
    )
    ap.add_argument(
        "--antialias",
        action="store_true",
        help="Apply downscale+upscale anti-alias pass before blur.",
    )
    ap.add_argument(
        "--scale-factor",
        type=float,
        default=0.75,
        help="Downscale factor for anti-alias pass.",
    )
    # Advanced edge / tone controls
    ap.add_argument(
        "--alpha-erosion",
        type=int,
        default=0,
        help="Erode alpha mask this many pixels before feather (default off).",
    )
    ap.add_argument(
        "--alpha-dilation",
        type=int,
        default=0,
        help="Dilate alpha mask this many pixels after erosion (default off).",
    )
    ap.add_argument(
        "--feather",
        type=float,
        default=0.6,
        help="Feather (Gaussian blur) radius for alpha edges (subtle).",
    )
    ap.add_argument(
        "--contrast",
        type=float,
        default=1.02,
        help="Contrast multiplier for RGB (slight).",
    )
    ap.add_argument(
        "--ink-gamma",
        type=float,
        default=0.98,
        help="Gamma adjustment for ink intensity (<1 darkens mid-tones slightly).",
    )
    ap.add_argument(
        "--add-noise",
        type=float,
        default=0.35,
        help="Std dev of Gaussian noise (0-10) added to RGB pre-blur (subtle grain).",
    )
    ap.add_argument(
        "--unsharp",
        type=str,
        default="0.5,30,2",
        help="Unsharp mask params radius,percent,threshold (mild crisp restore).",
    )
    ap.add_argument(
        "--max-alpha",
        type=int,
        default=None,
        help="Clamp final alpha to at most this (0-255).",
    )
    ap.add_argument("--mapping-json", type=Path, help="Path to mapping JSON to update.")
    ap.add_argument(
        "--mapping-output",
        type=Path,
        help="Path to write updated mapping (default overwrites original when --append-mapping).",
    )
    ap.add_argument(
        "--append-mapping",
        action="store_true",
        help="Append blurred_image field to mapping JSON segments.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if output blurred file already exists.",
    )
    ap.add_argument(
        "--extensions",
        type=str,
        default=".png",
        help="Comma-separated list of extensions to process.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible radius sampling.",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar.")
    args = ap.parse_args()

    if args.in_place and args.output_root:
        ap.error("Cannot specify --output-root with --in-place.")
    if not args.in_place and not args.output_root:
        ap.error("Either provide --output-root or use --in-place.")
    if args.radius is not None and args.radius <= 0:
        ap.error("--radius must be > 0")
    if args.radius is None and args.radius_min <= 0:
        ap.error("--radius-min must be > 0")
    if args.radius is None and args.radius_max < args.radius_min:
        ap.error("--radius-max must be >= --radius-min")
    if args.scale_factor <= 0 or args.scale_factor >= 1 and args.antialias:
        # Allow >1? Not necessary here.
        pass
    return args


def iter_images(root: Path, exts: List[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def choose_radius(cfg: BlurConfig) -> float:
    if cfg.radius is not None:
        return cfg.radius
    return random.uniform(cfg.radius_min, cfg.radius_max)


def anti_alias(im: Image.Image, scale_factor: float) -> Image.Image:
    if scale_factor >= 1 or scale_factor <= 0:
        return im
    w, h = im.size
    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))
    if new_w == w or new_h == h:
        return im
    # Downscale (BOX) then upscale (BICUBIC) to soften
    small = im.resize((new_w, new_h), Image.Resampling.BOX)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def process_image(src: Path, dst: Path, cfg: BlurConfig) -> bool:
    if cfg.skip_existing and dst.exists():
        return False
    try:
        im = Image.open(src).convert("RGBA")
        # Separate alpha
        r, g, b, a = im.split()
        rgb = Image.merge("RGB", (r, g, b))
        # Tone / noise adjustments
        if cfg.contrast != 1.0 or cfg.ink_gamma != 1.0 or cfg.add_noise > 0:
            arr = np.array(rgb).astype(np.float32) / 255.0
            if cfg.contrast != 1.0:
                arr = (arr - 0.5) * cfg.contrast + 0.5
            arr = np.clip(arr, 0, 1)
            if cfg.ink_gamma != 1.0:
                arr = np.power(arr, cfg.ink_gamma)
            if cfg.add_noise > 0:
                noise = np.random.normal(0, cfg.add_noise / 255.0, arr.shape).astype(
                    np.float32
                )
                arr = np.clip(arr + noise, 0, 1)
            rgb = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
        if cfg.antialias:
            rgb = anti_alias(rgb, cfg.scale_factor)
        radius = choose_radius(cfg)
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
        # Optional unsharp mask
        if cfg.unsharp:
            try:
                parts = [p.strip() for p in cfg.unsharp.split(",")]
                if len(parts) == 3:
                    u_radius, u_percent, u_threshold = (
                        float(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                    )
                    rgb = rgb.filter(
                        ImageFilter.UnsharpMask(
                            radius=u_radius, percent=u_percent, threshold=u_threshold
                        )
                    )
            except Exception:
                pass
        # Alpha refinement (erosion / dilation / feather / clamp)
        if (
            cfg.alpha_erosion > 0
            or cfg.alpha_dilation > 0
            or cfg.feather > 0
            or cfg.max_alpha is not None
        ):
            a_np = np.array(a).astype(np.uint8)
            mask = (a_np > 0).astype(np.uint8) * 255

            def morph(mask_arr: np.ndarray, iters: int, op: str) -> np.ndarray:
                if iters <= 0:
                    return mask_arr
                for _ in range(iters):
                    padded = np.pad(mask_arr, 1, mode="constant", constant_values=0)
                    out = mask_arr.copy()
                    h, w = mask_arr.shape
                    if op == "erode":
                        for y in range(h):
                            for x in range(w):
                                region = padded[y : y + 3, x : x + 3]
                                out[y, x] = 255 if np.all(region == 255) else 0
                    else:  # dilate
                        for y in range(h):
                            for x in range(w):
                                region = padded[y : y + 3, x : x + 3]
                                out[y, x] = 255 if np.any(region == 255) else 0
                    mask_arr = out
                return mask_arr

            if cfg.alpha_erosion > 0:
                mask = morph(mask, cfg.alpha_erosion, "erode")
            if cfg.alpha_dilation > 0:
                mask = morph(mask, cfg.alpha_dilation, "dilate")
            if cfg.feather > 0:
                mask = np.array(
                    Image.fromarray(mask, "L").filter(
                        ImageFilter.GaussianBlur(radius=cfg.feather)
                    )
                )
            if cfg.max_alpha is not None:
                mask = np.minimum(mask, cfg.max_alpha).astype(np.uint8)
            a = Image.fromarray(mask, "L")
        out = Image.merge("RGBA", (*rgb.split(), a))
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to blur {src}: {e}", file=sys.stderr)
        return False


def map_destination(src: Path, args, input_root: Path) -> Path:
    if args.in_place:
        return src.with_name(src.stem + args.suffix + src.suffix)
    # Mirror path inside output_root
    rel = src.relative_to(input_root)
    return args.output_root / rel


def update_mapping(
    mapping_path: Path, output_path: Path, input_root: Path, args
) -> None:
    try:
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not read mapping JSON: {e}", file=sys.stderr)
        return

    # Determine if mapping uses 'file_author_styles' or old style; keep untouched.
    entries = data.get("entries", [])
    changed = False
    for entry in entries:
        for seg in entry.get("segments", []):
            img_rel = seg.get("image")
            if not img_rel:
                continue
            # Destination blurred relative path
            src_abs = input_root / img_rel
            if args.in_place:
                blurred_rel = str(
                    Path(img_rel).with_name(
                        Path(img_rel).stem + args.suffix + Path(img_rel).suffix
                    )
                )
            else:
                # Mirror relative path but under output_root
                blurred_rel = img_rel  # Same relative name inside mirrored root
            blurred_abs = (
                (args.output_root / blurred_rel)
                if not args.in_place
                else src_abs.with_name(Path(blurred_rel).name)
            )
            if blurred_abs.exists():
                seg["blurred_image"] = blurred_rel
                changed = True
    if changed:
        out_path = args.mapping_output or mapping_path
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[INFO] Updated mapping JSON with blurred_image fields: {out_path}")
    else:
        print(
            "[INFO] No mapping changes applied (maybe images not yet generated or already present)."
        )


def blur_handwriting(
    input_root: Path,
    output_root: Optional[Path] = None,
    in_place: bool = False,
    suffix: str = "_b",
    radius: Optional[float] = None,
    radius_min: float = 0.35,
    radius_max: float = 0.85,
    antialias: bool = False,
    scale_factor: float = 0.75,
    alpha_erosion: int = 0,
    alpha_dilation: int = 0,
    feather: float = 0.6,
    contrast: float = 1.02,
    ink_gamma: float = 0.98,
    add_noise: float = 0.35,
    unsharp: Optional[str] = "0.5,30,2",
    max_alpha: Optional[int] = None,
    mapping_json: Optional[Path] = None,
    mapping_output: Optional[Path] = None,
    append_mapping: bool = False,
    skip_existing: bool = False,
    extensions: str = ".png",
    seed: int = 42,
    no_progress: bool = False,
) -> int:
    """Apply soft blur to handwriting token images.

    Mirrors the command-line behavior while exposing a reusable API.

    Returns the number of processed images.
    """
    # Basic validation to mirror CLI expectations
    if in_place and output_root is not None:
        raise ValueError("Cannot specify output_root with in_place=True.")
    if not in_place and output_root is None:
        raise ValueError("Either provide output_root or set in_place=True.")
    if radius is not None and radius <= 0:
        raise ValueError("--radius must be > 0")
    if radius is None and radius_min <= 0:
        raise ValueError("--radius-min must be > 0")
    if radius is None and radius_max < radius_min:
        raise ValueError("--radius-max must be >= --radius-min")

    random.seed(seed)

    exts = [e if e.startswith(".") else f".{e}" for e in extensions.split(",")]

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not in_place and output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    cfg = BlurConfig(
        radius=radius,
        radius_min=radius_min,
        radius_max=radius_max,
        antialias=antialias,
        scale_factor=scale_factor,
        suffix=suffix,
        skip_existing=skip_existing,
        alpha_erosion=alpha_erosion,
        alpha_dilation=alpha_dilation,
        feather=feather,
        contrast=contrast,
        ink_gamma=ink_gamma,
        add_noise=add_noise,
        unsharp=unsharp,
        max_alpha=max_alpha,
    )

    images = list(iter_images(input_root, exts))
    if not images:
        print("[WARN] No images found to process.")

    iterator = images
    if not no_progress and tqdm is not None:
        iterator = tqdm(images, desc="Blurring tokens", unit="img")  # type: ignore

    processed = 0
    # Minimal namespace to reuse map_destination/update_mapping without changing their signatures
    ns = SimpleNamespace(
        in_place=in_place,
        suffix=suffix,
        output_root=output_root,
        mapping_output=mapping_output,
    )

    for img_path in iterator:  # type: ignore
        dst = map_destination(img_path, ns, input_root)
        if process_image(img_path, dst, cfg):
            processed += 1

    print(f"[INFO] Blurred {processed} / {len(images)} images.")

    if append_mapping and mapping_json:
        update_mapping(mapping_json, mapping_json, input_root, ns)

    return processed


def main():
    args = parse_args()
    try:
        blur_handwriting(
            input_root=args.input_root,
            output_root=args.output_root,
            in_place=args.in_place,
            suffix=args.suffix,
            radius=args.radius,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            antialias=args.antialias,
            scale_factor=args.scale_factor,
            alpha_erosion=args.alpha_erosion,
            alpha_dilation=args.alpha_dilation,
            feather=args.feather,
            contrast=args.contrast,
            ink_gamma=args.ink_gamma,
            add_noise=args.add_noise,
            unsharp=args.unsharp,
            max_alpha=args.max_alpha,
            mapping_json=args.mapping_json,
            mapping_output=args.mapping_output,
            append_mapping=args.append_mapping,
            skip_existing=args.skip_existing,
            extensions=args.extensions,
            seed=args.seed,
            no_progress=args.no_progress,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
