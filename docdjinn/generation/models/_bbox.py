from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRBox:
    x0: float
    y0: float
    x2: float
    y2: float
    text: str
    block_no: int
    line_no: int
    word_no: int

    @property
    def key(self):
        return (self.block_no, self.line_no, self.word_no)

    def as_string(self) -> str:
        return f"{self.x0},{self.y0},{self.x2},{self.y2},{self.text},{self.block_no},{self.line_no},{self.word_no}"

    @property
    def width(self):
        return self.x2 - self.x0

    @property
    def height(self):
        return self.y2 - self.y0
    
    def unnormalize(self, width_px, height_px):
        return OCRBox(self.x0 * width_px, self.y0 * height_px, self.x2 * width_px, self.y2 * height_px, text=self.text, block_no=self.block_no, line_no=self.line_no, word_no=self.word_no)
    
    def scale(self, scale):
        return OCRBox(self.x0 * scale, self.y0 * scale, self.x2 * scale, self.y2 * scale, text=self.text, block_no=self.block_no, line_no=self.line_no, word_no=self.word_no)


@dataclass(frozen=True)
class LayoutBox:
    x0: float
    y0: float
    x2: float
    y2: float
    label: str

    @staticmethod
    def box_contains(outer: LayoutBox, inner: LayoutBox) -> bool:
        """Check if outer box fully contains inner box."""
        return (outer.x0 <= inner.x0 and outer.y0 <= inner.y0 and
                outer.x2 >= inner.x2 and outer.y2 >= inner.y2)

    @staticmethod
    def calculate_overlap_ratio(box1: LayoutBox, box2: LayoutBox) -> float:
        """
        Calculate the overlap ratio between two boxes.
        Returns the ratio of intersection area to the smaller box's area.
        """
        x_left = max(box1.x0, box2.x0)
        y_top = max(box1.y0, box2.y0)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (box1.x2 - box1.x0) * (box1.y2 - box1.y0)
        box2_area = (box2.x2 - box2.x0) * (box2.y2 - box2.y0)

        smaller_area = min(box1_area, box2_area)
        return intersection_area / smaller_area if smaller_area > 0 else 0.0

    @staticmethod
    def normalize_to_pdf(bbox: LayoutBox, width_pt: float, height_pt: float, dpi: float) -> LayoutBox:
        """Convert a bounding box from PDF points to normalized image coordinates."""
        # Convert PDF points to pixels
        scale = dpi / 72
        x_min_px = bbox.x0 * scale
        y_min_px = bbox.y0 * scale
        x_max_px = bbox.x2 * scale
        y_max_px = bbox.y2 * scale

        # Get image size in pixels
        img_w_px = width_pt * scale
        img_h_px = height_pt * scale

        # Normalize bounding box
        x_min_norm = x_min_px / img_w_px
        y_min_norm = y_min_px / img_h_px
        x_max_norm = x_max_px / img_w_px
        y_max_norm = y_max_px / img_h_px

        return LayoutBox(
            x0=x_min_norm, y0=y_min_norm,
            x2=x_max_norm, y2=y_max_norm,
            label=bbox.label,
        )
