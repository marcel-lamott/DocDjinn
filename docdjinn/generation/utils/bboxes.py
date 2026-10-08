from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pymupdf

from docdjinn.generation.models import OCRBox
from docdjinn.generation.models._bbox import LayoutBox


def is_in_rect(rect: dict, bbox: OCRBox, threshold: float, document_id: str | None = None):
    # Convert back PDF points to pixels
    r_x0 = rect["x"] - threshold
    r_y0 = rect["y"] - threshold
    r_x2 = r_x0 + rect["width"] + 2 * threshold
    r_y2 = r_y0 + rect["height"] + 2 * threshold

    left = bbox.x0 >= r_x0
    top = bbox.y0 >= r_y0
    right = bbox.x2 <= r_x2
    bottom = bbox.y2 <= r_y2

    # if document_id == '74577486-4e36-425e-b733-8a745ca840f1_0':
    #     print(f'{left=} {top=} {right=} {bottom=} {bbox.as_string} {rect=}')

    return left and top and right and bottom


def save_bboxes(
    bboxes: list[OCRBox],
    bbox_path: Path,
):
    bbox_path.parent.mkdir(exist_ok=True, parents=True)
    with bbox_path.open(mode="w", encoding="utf-8") as f:
        for i, box in enumerate(bboxes):
            line = box.as_string()
            if i < len(bboxes) - 1:
                line += "\n"
            f.write(line)


def read_syn_dataset_bbox_str(line: str) -> OCRBox:
    parts = line.split(",", 4)
    x0 = float(parts[0])
    y0 = float(parts[1])
    x2 = float(parts[2])
    y2 = float(parts[3])
    txt = parts[4]
    parts = txt.rsplit(",", 3)
    txt = parts[0]
    block_no = int(parts[1])
    line_no = int(parts[2])
    word_no = int(parts[3])
    return OCRBox(
        x0=x0,
        y0=y0,
        x2=x2,
        y2=y2,
        text=txt,
        block_no=block_no,
        line_no=line_no,
        word_no=word_no,
    )


def read_syn_dataset_bboxes(box_path) -> list[OCRBox]:
    """
    Reads bboxes from synthetic datasets
    """
    bboxes = []
    line: str
    for line in box_path.read_text(encoding="utf-8").splitlines():
        bboxes.append(read_syn_dataset_bbox_str(line))
    return bboxes


def draw_pdf_bboxes_on_pdf(pdf_path, outpath: Path):
    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc.pages()):
        for block in page.get_text("words"):
            x0, y0, x1, y1, txt = block[:5]
            # rect = pymupdf.Rect(block[:4])
            block = (round(x0), round(y0), round(x1), round(y1))
            rect = pymupdf.Rect(block)
            print(",".join([str(x) for x in block]))
            page.draw_rect(rect, color=(1, 0, 0))  # Red box

        doc.save(outpath)


def draw_bboxes_on_pdf(
    pdf_path: Path, outpath: Path, bboxes: list[OCRBox], color=(1, 0, 0)
):
    doc = pymupdf.open(pdf_path)
    for page_num, page in enumerate(doc.pages()):
        for bbox in bboxes:
            # rect = pymupdf.Rect(block[:4])
            block = (round(bbox.x0), round(bbox.y0), round(bbox.x2), round(bbox.y2))
            rect = pymupdf.Rect(block)
            page.draw_rect(rect, color=color)  # Red box

        doc.save(outpath)


def draw_bboxes_on_image(
    image, bboxes: list[OCRBox], color="red", width=3, show_text=True
) -> Image.Image:
    """
    Draws bounding boxes on a given Pillow image.

    :param image: Pillow Image object
    :param bboxes: List of bounding boxes [(x0, y0, x1, y1), ...]
    :param color: Color of the bounding box (default: red)
    :param width: Line width (default: 3)
    :return: Image with bounding boxes
    """
    draw = ImageDraw.Draw(image)

    bbox: OCRBox
    for bbox in bboxes:
        box = (bbox.x0, bbox.y0, bbox.x2, bbox.y2)
        draw.rectangle(box, outline=color, width=width)

        # font = ImageFont.truetype("sans-serif.ttf", 16)
        if show_text:
            font = ImageFont.load_default(32)
            draw.text(box, bbox.text, (255, 0, 255), font=font)  # type: ignore

    return image
