"""
{
    "id": null,
    "tag": "div",
    "classes": "signature handwritten author1",
    "rect": {
      "x": 521.5546875,
      "y": 814.7109375,
      "width": 357.1640625,
      "height": 31.1953125
    },
    "visibility": "visible",
    "dataContent": null,
    "dataPlaceholder": null,
    "style": null,
    "text": "James Wellington",
    "selectorType": "handwriting"
  },
"""

import json
from pathlib import Path
from typing import Iterable

from docdjinn.generation.models._bbox import OCRBox


def read_visual_elements_from_geos(geo_path: Path) -> Iterable[dict]:
    data = json.loads(geo_path.read_text(encoding="utf-8"))
    for d in data:
        if "visual_element" in d["selectorTypes"]:
            yield d


def read_handwriting_elements_from_geos(geo_path: Path) -> Iterable[dict]:
    data = json.loads(geo_path.read_text(encoding="utf-8"))
    for d in data:
        if "handwriting" in d["selectorTypes"]:
            yield d


def read_layout_elements_from_geos(geo_path: Path) -> Iterable[dict]:
    data = json.loads(geo_path.read_text(encoding="utf-8"))
    for d in data:
        if "layout_element" in d["selectorTypes"]:
            yield d


def read_custom_elements_from_geos(geo_path: Path) -> Iterable[dict]:
    data = json.loads(geo_path.read_text(encoding="utf-8"))
    for d in data:
        if "custom" in d["selectorTypes"]:
            yield d


def rect_to_ocrbox(r: dict, text=None) -> OCRBox:
    return OCRBox(
        x0=r["x"],
        y0=r["y"],
        x2=r["x"] + r["width"],
        y2=r["y"] + r["height"],
        text=text,  # type: ignore
        block_no=-1,
        line_no=-1,
        word_no=-1,
    )
