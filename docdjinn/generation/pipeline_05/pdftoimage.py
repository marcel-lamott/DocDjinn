import pathlib
from pdf2image import convert_from_path
from PIL import Image

from docdjinn.generation.constants import PDF_DPI


def convert_from_path_singlepage(
    pdf_path: pathlib.Path, target_size: tuple[int, int] | None = None
) -> Image.Image:
    images = convert_from_path(pdf_path, dpi=PDF_DPI, size=target_size)
    assert len(images) == 1, "Multi-page document are not supported"
    img = images[0]
    return img
