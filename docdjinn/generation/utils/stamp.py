from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import PIL
import PIL.Image
import numpy as np
import math
import random
from typing import Tuple, Optional


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    lv = len(hex_color)
    return tuple(int(hex_color[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def _make_noise_image(size, mean=0.7, std=0.22, blur=2, contrast=1.0):
    """Return L-mode noise image (0-255)."""
    w, h = size
    arr = np.clip(np.random.normal(loc=mean, scale=std, size=(h, w)), 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if contrast != 1.0:
        a = np.asarray(img).astype(np.float32)
        a = 128 + (a - 128) * contrast
        a = np.clip(a, 0, 255).astype(np.uint8)
        img = Image.fromarray(a, mode="L")
    return img


def _bias_noise_towards_opaque(noise_img: Image.Image, min_val=200):
    """
    Bias a noise image so values fall in [min_val..255], preserving local variation
    but ensuring the noise doesn't make the stamp too transparent.
    """
    assert 0 <= min_val <= 255
    return noise_img.point(lambda p: min_val + (p * (255 - min_val) // 255))


def _draw_text_on_arc(
    target_img: Image.Image,
    text: str,
    center: Tuple[int, int],
    radius: float,
    font: ImageFont.FreeTypeFont,
    color: Tuple[int, int, int, int],
    start_angle_deg: float = 0.0,
    inward: bool = False,
):
    """
    Draw text along an arc centered at `center` with given `radius`.
    Characters are placed and rotated tangentially for realism.
    """
    draw = ImageDraw.Draw(target_img)
    # measure each character width using textbbox
    char_widths = []
    for ch in text:
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = bbox[2] - bbox[0]
        char_widths.append(max(w, 1))

    angs = [(w / radius) * (180.0 / math.pi) for w in char_widths]
    total_arc = sum(angs)
    angle = start_angle_deg - total_arc / 2.0

    cx, cy = center
    for i, ch in enumerate(text):
        char_ang = angs[i]
        angle += char_ang / 2.0
        theta = math.radians(angle)

        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)

        bbox = draw.textbbox((0, 0), ch, font=font)
        cw = bbox[2] - bbox[0]
        chh = bbox[3] - bbox[1]
        pad = int(max(cw, chh) * 1.6) + 6
        char_img = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
        cd = ImageDraw.Draw(char_img)
        cd.text((pad // 2, pad // 2), ch, font=font, fill=color, anchor="mm")

        rot_angle = -angle + 90
        if inward:
            rot_angle += 180

        rot = char_img.rotate(rot_angle, resample=Image.BICUBIC, expand=True)
        px = int(x - rot.width / 2)
        py = int(y - rot.height / 2)
        target_img.paste(rot, (px, py), rot)

        angle += char_ang / 2.0


def _wrap_text_to_fit(text, font, max_width):
    """
    Automatically wrap text by inserting line breaks to fit within max_width.
    Returns text with line breaks inserted.
    """
    # If text already has line breaks, process each line separately
    existing_lines = text.split("\n")
    wrapped_lines = []

    temp_img = Image.new("RGBA", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    for line in existing_lines:
        words = line.split()
        if not words:
            wrapped_lines.append("")
            continue

        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    wrapped_lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word is too long, just add it anyway
                    wrapped_lines.append(word)
                    current_line = []

        if current_line:
            wrapped_lines.append(" ".join(current_line))

    return "\n".join(wrapped_lines)


def create_realistic_stamp(
    text_top: str = "APPROVED",
    text_bottom: Optional[str] = None,
    inner_text: Optional[str] = None,
    shape: str = "circle",  # "circle" or "rectangle"
    size: Tuple[int, int] = (800, 800),  # final (width, height)
    color: str = "#C42828",  # hex or "r,g,b"
    border_thickness_ratio: float = 0.08,  # relative to min(width,height)
    font_path: Optional[str] = None,
    font_size: Optional[int] = None,  # base font size
    random_seed: Optional[int] = None,
    supersample: int = 3,  # supersampling factor
    rot_angle: float | None = None,
):
    """
    Generate a realistic-looking stamp PNG with transparent background.
    - Automatically adjusts font size to fit text
    - Fixes text cutoff issues
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    w, h = size
    scale = max(1, int(supersample))
    W, H = w * scale, h * scale

    if isinstance(color, str):
        if "," in color:
            color_rgb = tuple(int(x) for x in color.split(","))
        else:
            color_rgb = _hex_to_rgb(color)
    else:
        color_rgb = tuple(color)

    # big canvas (supersampled)
    stamp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shape_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d_shape = ImageDraw.Draw(shape_layer)

    min_side = min(W, H)
    border_w = max(2 * scale, int(min_side * border_thickness_ratio))

    jitter_x = random.randint(-int(min_side * 0.005), int(min_side * 0.005))
    jitter_y = random.randint(-int(min_side * 0.005), int(min_side * 0.005))

    # Draw the ring/rectangle onto shape_layer
    if shape.lower() == "circle":
        outer = [
            (border_w // 2 + jitter_x, border_w // 2 + jitter_y),
            (W - border_w // 2 + jitter_x, H - border_w // 2 + jitter_y),
        ]
        inner = [
            (border_w * 3 + jitter_x, border_w * 3 + jitter_y),
            (W - border_w * 3 + jitter_x, H - border_w * 3 + jitter_y),
        ]
        for i in range(border_w):
            off = random.randint(-scale, scale)
            d_shape.ellipse(
                [
                    (outer[0][0] + i + off, outer[0][1] + i + off),
                    (outer[1][0] - i + off, outer[1][1] - i + off),
                ],
                outline=color_rgb + (255,),
            )
        d_shape.ellipse(inner, outline=color_rgb + (220,), width=max(1, border_w // 6))
    else:
        pad = border_w // 2
        for i in range(border_w):
            off = random.randint(-scale, scale)
            rect = [
                pad + i + off + jitter_x,
                pad + i + off + jitter_y,
                W - (pad + i) + jitter_x,
                H - (pad + i) + jitter_y,
            ]
            d_shape.rounded_rectangle(
                rect, radius=max(6 * scale, border_w), outline=color_rgb + (255,)
            )

    # Blur the shape layer
    bleed_radius = max(1.0 * scale, scale * 0.9)
    shape_layer = shape_layer.filter(ImageFilter.GaussianBlur(radius=bleed_radius))
    stamp.alpha_composite(shape_layer, (0, 0))

    # Font loading helper
    def _try_load_ttf(desired_size):
        try:
            if font_path:
                return ImageFont.truetype(font_path, desired_size)
            else:
                return ImageFont.truetype("DejaVuSans-Bold.ttf", desired_size)
        except Exception:
            return ImageFont.load_default()

    # Calculate available space for inner text
    if inner_text:
        # Define text area boundaries
        if shape.lower() == "circle":
            # For circle: use area inside inner ring
            text_area_width = W - (border_w * 6)
            text_area_height = H - (border_w * 6)
        else:
            # For rectangle: use area inside borders with padding
            text_area_width = W - (border_w * 4)
            text_area_height = H - (border_w * 4)

        # Calculate initial font size
        if font_size:
            inner_font_size = int(font_size * 1.6 * scale)
        else:
            inner_font_size = int(min_side * 0.20)

        inner_font = _try_load_ttf(inner_font_size)

        # Wrap text to fit width
        inner_text = _wrap_text_to_fit(inner_text, inner_font, text_area_width * 0.95)

    # Small font for curved text
    if font_size:
        small_font_size = max(10 * scale, int(font_size * 0.6 * scale))
    else:
        small_font_size = max(10 * scale, int(min_side * 0.055))
    small_font = _try_load_ttf(small_font_size)

    d = ImageDraw.Draw(stamp)

    # Curved text (circle)
    if shape.lower() == "circle" and text_top:
        center = (W // 2 + jitter_x, H // 2 + jitter_y)
        radius = (min_side // 2) - border_w - int(min_side * 0.03)
        _draw_text_on_arc(
            stamp,
            text_top.upper(),
            center,
            radius,
            small_font,
            color_rgb + (255,),
            start_angle_deg=-90,
        )
        if text_bottom:
            _draw_text_on_arc(
                stamp,
                text_bottom.upper(),
                center,
                radius,
                small_font,
                color_rgb + (255,),
                start_angle_deg=90,
                inward=True,
            )

    # Inner/center text - FIXED VERTICAL POSITIONING
    if inner_text:
        centerx, centery = W // 2 + jitter_x, H // 2 + jitter_y
        lines = inner_text.split("\n")

        # Calculate total height and individual line metrics
        draw_tmp = ImageDraw.Draw(stamp)
        line_metrics = []
        total_h = 0

        for ln in lines:
            bbox = draw_tmp.textbbox((0, 0), ln, font=inner_font)
            # Use actual bbox for accurate height including descenders
            line_height = bbox[3] - bbox[1]
            line_metrics.append(
                {
                    "text": ln,
                    "bbox": bbox,
                    "width": bbox[2] - bbox[0],
                    "height": line_height,
                    "y_offset": -bbox[1],  # Offset to account for font baseline
                }
            )
            total_h += line_height

        # Start from top, centered vertically
        y = centery - total_h // 2

        for metric in line_metrics:
            ln = metric["text"]
            tw = metric["width"]
            th = metric["height"]
            y_off = metric["y_offset"]

            # Create image with extra padding to prevent cutoff
            padding = 30
            txt_img = Image.new(
                "RGBA", (tw + padding * 2, th + padding * 2), (0, 0, 0, 0)
            )
            td = ImageDraw.Draw(txt_img)

            # Draw text with proper baseline offset
            td.text(
                (padding, padding + y_off), ln, font=inner_font, fill=color_rgb + (255,)
            )

            angle = random.uniform(-1.0, 1.0)
            txt_img = txt_img.rotate(angle, resample=Image.BICUBIC, expand=True)

            paste_x = int(centerx - txt_img.width / 2)
            paste_y = int(y - padding)

            stamp.paste(txt_img, (paste_x, paste_y), txt_img)
            y += th

    # Add subtle overlay strokes
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    if shape.lower() == "circle":
        try:
            od.ellipse(
                [(border_w, border_w), (W - border_w, H - border_w)],
                outline=color_rgb + (180,),
                width=max(1, border_w // 6),
            )
        except Exception:
            pass
    else:
        try:
            od.rounded_rectangle(
                [border_w, border_w, W - border_w, H - border_w],
                radius=max(6 * scale, border_w),
                outline=color_rgb + (180,),
                width=max(1, border_w // 6),
            )
        except Exception:
            pass
    stamp.alpha_composite(overlay)

    # Add noise texture
    noise = _make_noise_image(
        (W, H), mean=0.78, std=0.18, blur=2 * scale, contrast=1.05
    )
    noise_biased = _bias_noise_towards_opaque(noise, min_val=210)

    orig_alpha = stamp.split()[-1]
    new_alpha = ImageChops.multiply(orig_alpha, noise_biased)
    a_arr = np.asarray(new_alpha).astype(np.float32)
    a_arr = np.clip(a_arr * 1.03, 0, 255).astype(np.uint8)
    new_alpha = Image.fromarray(a_arr, mode="L")
    stamp.putalpha(new_alpha)

    # Slight blur for ink bleed effect
    stamp = stamp.filter(ImageFilter.GaussianBlur(radius=0.4 * scale))

    # Add light speckle holes
    speck = _make_noise_image((W, H), mean=0.5, std=0.9, blur=0.6 * scale, contrast=1.6)
    speck_arr = np.asarray(speck)
    speck_mask = (speck_arr > 252).astype(np.uint8) * 255
    speck_img = Image.fromarray(speck_mask, mode="L")
    if speck_img.getbbox() is not None:
        alpha = stamp.split()[-1]
        alpha = ImageChops.subtract(alpha, speck_img)
        stamp.putalpha(alpha)

    # Random rotation
    rot_angle = rot_angle or random.uniform(-2.2, 2.2)
    stamp = stamp.rotate(rot_angle, resample=Image.Resampling.BICUBIC, expand=True)

    # Downsample to final size
    final = stamp.resize((w, h), resample=Image.Resampling.LANCZOS)

    # Final sharpening
    final = final.filter(ImageFilter.UnsharpMask(radius=0.6, percent=120, threshold=2))

    return final


def create_stamp_alt(text: str) -> PIL.Image.Image:
    coin = random.random() <= 0.5
    if coin:
        return create_realistic_stamp(
            "",
            text_bottom="",
            inner_text=text,
            shape="circle",
            size=(900, 900),
            color="#a81f1f",
            font_path=None,
            font_size=60,
            random_seed=42,
            supersample=3,
        )
    else:
        return create_realistic_stamp(
            text_top="",
            inner_text=text,
            shape="rectangle",
            size=(1100, 500),
            color="#1f7a1f",
            font_size=56,
            random_seed=7,
            supersample=3,
        )


def create_stamp(
    text: str, width: float, height: float, rot_angle: float | None
) -> PIL.Image.Image:
    coin = random.random() <= 0.5
    width = int(width)
    height = int(height)
    size_mult = 11  # previous default values were along 900/1000, but real sizes are around 100, which the text resizing cant handle
    if coin:
        return create_realistic_stamp(
            "",
            text_bottom="",
            inner_text=text,
            shape="circle",
            size=(width * size_mult, height * size_mult),
            color="#a81f1f",
            font_path=None,
            font_size=60,
            random_seed=42,
            supersample=3,
            rot_angle=rot_angle,
        )
    else:
        return create_realistic_stamp(
            text_top="",
            inner_text=text,
            shape="rectangle",
            size=(width * size_mult, height * size_mult),
            color="#1f7a1f",
            font_size=56,
            random_seed=7,
            supersample=3,
            rot_angle=rot_angle,
        )
