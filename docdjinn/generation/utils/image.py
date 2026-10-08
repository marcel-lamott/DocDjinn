import pathlib

import cv2
import numpy as np
from PIL import Image


def img_write_to_bytes(array, fmt: str = '.png') -> bytes:
    return cv2.imencode(fmt, array)[1].tobytes()


def img_read(input_file: pathlib.Path, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(input_file, np.uint8), flags)


def downscale_image(img, max_width):
    # Get current dimensions
    width, height = img.size

    # Check if resizing is needed
    if width <= max_width:
        return img

    # Calculate the new height to maintain aspect ratio
    new_height = int((max_width / width) * height)

    # Resize the image
    img_resized = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

    return img_resized


def downscale_and_compress_from_path(
        old_path: pathlib.Path,
        new_path: pathlib.Path,
        max_width: int = 500,
        quality: int = 80,
):
    # Open the image
    img = Image.open(old_path)
    img = downscale_image(img, max_width=max_width)
    img.save(new_path, format="JPEG", quality=quality)


def downscale_and_compress(
        img: Image.Image,
        save_to_path: pathlib.Path,
        max_width: int = 500,
        quality: int = 80,
):
    img = downscale_image(img, max_width=max_width)
    img.save(save_to_path, format="JPEG", quality=quality)
