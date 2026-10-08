from pathlib import Path

import pymupdf

from docdjinn.generation.models._bbox import OCRBox
from docdjinn.generation.utils.bboxes import draw_bboxes_on_pdf


def draw_geos_on_pdf(geos: list[dict], pdf_in: Path, pdf_out: Path):
    bboxes = []
    for g in geos:
        x0 = float(g["rect"]["x"])
        y0 = float(g["rect"]["y"])
        x2 = x0 + float(g["rect"]["width"])
        y2 = y0 + float(g["rect"]["height"])
        b = OCRBox(
            x0=x0,
            y0=y0,
            x2=x2,
            y2=y2,
            text="",
            block_no=-1,
            line_no=-1,
            word_no=-1,
        )
        bboxes.append(b)

    draw_bboxes_on_pdf(
        pdf_path=pdf_in,
        outpath=pdf_out,
        bboxes=bboxes,
    )


def draw_geos_and_bboxes_on_pdf(
    geos: list[dict], bboxes_: list[OCRBox], pdf_in: Path, pdf_out: Path, verbose: bool
):
    bboxes = []
    for g in geos:
        x0 = float(g["rect"]["x"])
        y0 = float(g["rect"]["y"])
        x2 = x0 + float(g["rect"]["width"])
        y2 = y0 + float(g["rect"]["height"])
        b = OCRBox(
            x0=x0,
            y0=y0,
            x2=x2,
            y2=y2,
            text="",
            block_no=-1,
            line_no=-1,
            word_no=-1,
        )
        bboxes.append(b)

    doc = pymupdf.open(pdf_in)
    for page_num, page in enumerate(doc.pages()):
        # geos red
        color = (1, 0, 0)
        for bbox in bboxes:
            # rect = pymupdf.Rect(block[:4])
            block = (round(bbox.x0), round(bbox.y0), round(bbox.x2), round(bbox.y2))
            rect = pymupdf.Rect(block)
            page.draw_rect(rect, color=color)  # Red box

            if verbose:
                print(bbox)

        # bboxes green
        color = (0, 1, 0)
        for bbox in bboxes_:
            # rect = pymupdf.Rect(block[:4])
            block = (round(bbox.x0), round(bbox.y0), round(bbox.x2), round(bbox.y2))
            rect = pymupdf.Rect(block)
            page.draw_rect(rect, color=color)  # Red box

            if verbose:
                print(bbox)

        doc.save(pdf_out)
