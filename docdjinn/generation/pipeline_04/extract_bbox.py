import pathlib
import re
from typing import Literal
import pymupdf

from docdjinn.generation.models import OCRBox


def extract_bboxes_from_pdf(
    pdf_path: pathlib.Path,
    level: Literal["word", "char"],
    round_coordinates: bool = False,
) -> list[OCRBox]:
    match level:
        # case 'span': bboxes = _extract_bboxes_from_pdf_words(pdf_path=pdf_path, level='blocks')
        case "word":
            bboxes = _extract_bboxes_from_pdf_words(pdf_path=pdf_path)
        case "char":
            bboxes = _extract_bboxes_from_pdf_chars(pdf_path=pdf_path)

    converted_bboxes = []
    for x0, y0, x2, y2, txt, block_no, line_no, word_no in bboxes:
        box = OCRBox(
            round(x0) if round_coordinates else x0,
            round(y0) if round_coordinates else y0,
            round(x2) if round_coordinates else x2,
            round(y2) if round_coordinates else y2,
            txt,
            block_no,
            line_no,
            word_no,
        )
        converted_bboxes.append(box)
    bboxes = converted_bboxes

    return bboxes


rtl_re = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u0700-\u074F\u0750-\u077F"
    r"\u0780-\u07BF\u07C0-\u07FF\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)


def contains_rtl(text):
    return bool(rtl_re.search(text))


def _extract_bboxes_from_pdf_words(
    pdf_path: pathlib.Path,
) -> list[tuple[float, float, float, float, str, int, int, int]]:
    doc = pymupdf.open(pdf_path)
    bboxes = []
    for page_num, page in enumerate(doc.pages()):
        for word_no, word in enumerate(page.get_text("words")):
            x0, y0, x2, y2, txt, block_no, line_no, word_no = word
            box = (x0, y0, x2, y2, txt, block_no, line_no, word_no)
            bboxes.append(box)
    doc.close()
    return bboxes


# Look behind
# Word level extraction extracts 'PCS-2024-SF-0087' as single word -> only space, new line, are delmiters
def _extract_bboxes_from_pdf_chars(
    pdf_path: pathlib.Path,
) -> list[tuple[float, float, float, float, str, int, int, int]]:
    doc = pymupdf.open(pdf_path)
    bboxes = []
    for page_num, page in enumerate(doc.pages()):
        # Skip images
        block_no = -1
        for b in page.get_text("rawdict")["blocks"]:
            if b["type"] == 0:
                block_no += 1
            else:
                continue
            for line_no, line in enumerate(b["lines"]):
                word_no = 0
                cur_word = ""
                for span in line["spans"]:
                    for char in span["chars"]:
                        x0, y0, x2, y2 = char["bbox"]
                        c = char["c"]

                        if (
                            (not c.isspace())
                            and len(cur_word.strip()) > 0
                            and cur_word[-1].isspace()
                        ):
                            word_no += 1
                        if not c.isspace():
                            box = (x0, y0, x2, y2, c, block_no, line_no, word_no)
                            bboxes.append(box)
                        cur_word += c

    doc.close()
    return bboxes


def _extract_text_from_pdf(pdf_path: pathlib.Path) -> list[str]:
    doc = pymupdf.open(pdf_path)
    texts = []
    for page_num, page in enumerate(doc.pages()):
        texts.append(page.get_text("text"))
    doc.close()
    return " ".join(texts)  # type: ignore


def _get_num_pages(pdf_path: pathlib.Path) -> list[str]:
    doc = pymupdf.open(pdf_path)
    pc = doc.page_count
    doc.close()
    return pc


def validate_char_bbox_word_mapping(
    char_bboxes: list[OCRBox], word_bboxes: list[OCRBox]
):
    from collections import defaultdict

    grouped = defaultdict(list)

    word_bbox_lookup = dict()
    for box in word_bboxes:
        word_bbox_lookup[box.key] = box.text

    for box in char_bboxes:
        grouped[box.key].append(box.text)

    for block_no, line_no, word_no in sorted(grouped):
        chars = grouped[(block_no, line_no, word_no)]
        word = "".join(chars).strip()

        # logfile = ENV.TEMP_DIR / 'bbox_debug.txt'
        # logfile.write_text('')
        # def log(s):
        #     with open(logfile, mode='a') as f:
        #         f.write(f'{s}\n')
        # print(s)
        if (
            block_no,
            line_no,
            word_no,
        ) not in word_bbox_lookup or word != word_bbox_lookup[
            (block_no, line_no, word_no)
        ]:
            # [log(b) for b in word_bboxes]
            # log("----------------")
            # [log(b) for b in char_bboxes]

            # print(f'{(block_no, line_no, word_no)}')
            # gt_word = word_bbox_lookup[(block_no, line_no, word_no)]
            # print(f'{word} != {gt_word} !!')
            # _print_block_types(pdf_path)
            # input()
            return False

    return True
