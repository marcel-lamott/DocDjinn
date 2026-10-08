from __future__ import annotations

import json
from pathlib import Path


class MicrosoftOCRWord:
    def __init__(self, text: str, confidence: float, geo: list[int]):
        self.text = text
        self.confidence = confidence
        self.geo = geo


class MicrosoftOCR:
    def __init__(
            self,
            angle: float,
            width: int,
            height: int,
            words: list[MicrosoftOCRWord],
            lines: list[MicrosoftOCRWord] | None = None
    ):
        self.angle = angle
        self.width = width
        self.height = height
        self.words = words
        self.lines = lines if lines is not None else []

    @staticmethod
    def _geo_from_polygon(polygon_vals: list[float | int], scale: float) -> list[int]:
        # [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]
        # [   0,    1,    2,    3,    4,    5,    6,    7]
        x = [polygon_vals[0], polygon_vals[2], polygon_vals[4], polygon_vals[6]]
        y = [polygon_vals[1], polygon_vals[3], polygon_vals[5], polygon_vals[7]]
        left = int(round(min(x) * scale))
        top = int(round(min(y) * scale))
        right = int(round(max(x) * scale))
        bottom = int(round(max(y) * scale))

        width = right - left + 1
        height = bottom - top + 1
        return [left, top, width, height]

    @staticmethod
    def load_from_file(path: Path) -> MicrosoftOCR:
        data = json.loads(path.read_text(encoding="utf-8"))
        first_page = data["analyzeResult"]["pages"][0]
        return MicrosoftOCR(
            angle=first_page["angle"],
            width=first_page["width"],
            height=first_page["height"],
            words=[
                MicrosoftOCRWord(
                    text=word["content"],
                    confidence=word["confidence"],
                    geo=MicrosoftOCR._geo_from_polygon(word["polygon"], scale=1.0),
                )
                for word in first_page['words']
            ],
            lines=[
                MicrosoftOCRWord(
                    text=line["content"],
                    confidence=-1,
                    geo=MicrosoftOCR._geo_from_polygon(line["polygon"], scale=1.0),
                )
                for line in first_page['lines']
            ],
        )

    def save_to_file(self, path: Path) -> None:
        data = {
            'analyzeResult': {
                'pages': [
                    {
                        'angle': self.angle,
                        'width': self.width,
                        'height': self.height,
                        'words': [
                            {
                                'content': word.text,
                                'confidence': word.confidence,
                                'polygon': [
                                    word.geo[0],  # tl.x
                                    word.geo[1],  # tl.y
                                    word.geo[0] + word.geo[2] - 1,  # tr.x
                                    word.geo[1],  # tr.y
                                    word.geo[0] + word.geo[2] - 1,  # br.x
                                    word.geo[1] + word.geo[3] - 1,  # br.y
                                    word.geo[0],  # bl.x
                                    word.geo[1] + word.geo[3] - 1  # bl.y
                                ]
                                # Input:
                                # [tl.x, tl.y, w, h]
                                # [   0,    1, 2, 3]
                                # Output:
                                # [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]
                                # [   0,    1,    2,    3,    4,    5,    6,    7]
                            }
                            for word in self.words
                        ],
                        'lines': [
                            {
                                'content': line.text,
                                'confidence': -1,
                                'polygon': [
                                    line.geo[0],  # tl.x
                                    line.geo[1],  # tl.y
                                    line.geo[0] + line.geo[2] - 1,  # tr.x
                                    line.geo[1],  # tr.y
                                    line.geo[0] + line.geo[2] - 1,  # br.x
                                    line.geo[1] + line.geo[3] - 1,  # br.y
                                    line.geo[0],  # bl.x
                                    line.geo[1] + line.geo[3] - 1  # bl.y
                                ]
                                # Input:
                                # [tl.x, tl.y, w, h]
                                # [   0,    1, 2, 3]
                                # Output:
                                # [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]
                                # [   0,    1,    2,    3,    4,    5,    6,    7]
                            }
                            for line in self.lines
                        ]
                    }
                ]
            }
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# Example showing how to load MS OCR files
if __name__ == "__main__":
    base_dir = Path("data/temp/OCR/test-dataset")
    ocr_file = base_dir / "04276b91-eb12-4b47-80a6-666f6d09b6ce_1.jpg.0.MicrosoftOcrService.json"

    ocr = MicrosoftOCR.load_from_file(ocr_file)
    for word in ocr.words:
        print(f"{word.text:<20} | {word.confidence:>6.2f} | {word.geo}")
    for line in ocr.lines:
        print(f"{line.text:<20} | {line.confidence:>6.2f} | {line.geo}")

    print("=" * 128)
    print(" ".join(word.text for word in ocr.words))

    # PRECISION            |   0.99 | [1209, 68, 495, 69]
    # SAFETY               |   0.99 | [1735, 68, 331, 69]
    # ...
    # Seal                 |   0.99 | [2993, 2566, 122, 55]
    # ================================================================================================================================
    # PRECISION SAFETY INSPECTIONS FORKLIFT SAFETY INSPECTION REPORT Licensed Inspector . State Certified . ...

    # Check if loading and saving gives the same result
    tmp_ocr_file = Path("/tmp/test.0.MicrosoftOcrService.json")
    ocr.save_to_file(tmp_ocr_file)
    reloaded_ocr = MicrosoftOCR.load_from_file(tmp_ocr_file)
    for word in reloaded_ocr.words:
        print(f"{word.text:<20} | {word.confidence:>6.2f} | {word.geo}")
    for line in reloaded_ocr.lines:
        print(f"{line.text:<20} | {line.confidence:>6.2f} | {line.geo}")

    if not all([
        reloaded_ocr_word.__dict__ == ocr_word.__dict__
        for reloaded_ocr_word, ocr_word in zip(reloaded_ocr.words, ocr.words)
    ]):
        raise AssertionError('Saving and loading a file does not work!')

    if not all([
        reloaded_ocr_line.__dict__ == ocr_line.__dict__
        for reloaded_ocr_line, ocr_line in zip(reloaded_ocr.lines, ocr.lines)
    ]):
        raise AssertionError('Saving and loading a file does not work!')
