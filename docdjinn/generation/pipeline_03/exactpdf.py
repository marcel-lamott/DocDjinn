import json
import pathlib
import subprocess
import tempfile
import threading
from typing import Literal

from anyio import Semaphore
from docdjinn import ENV

# from docdjinn.generation.pipeline_03.css import (
#     prepare_html_for_size_measurement,
#     # prepare_html_for_rendering_first_pass,
# )
from docdjinn.generation.pipeline_03.extract_positions import (
    convert_to_bboxes,
    extract_positions_from_subprocess_result,
    prepare_html_for_position_extraction,
)
from docdjinn.generation.utils.bboxes import draw_bboxes_on_pdf


def convert_pdf_chromium_and_extract_positions(
    html: str,
    output_path: pathlib.Path,
    extract_positions_for_classes: list[str],
    semaphore: threading.Semaphore,
    timeout_s: int = 300,
):
    """Spawn external Chromium CLI to print-to-pdf but limited by a semaphore."""
    html = prepare_html_for_position_extraction(
        html=html, classes=extract_positions_for_classes
    )

    # write tmp html
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        tmp_file.write(html.encode("utf-8"))
        tmp_file_path = pathlib.Path(tmp_file.name)
    # <meta name="viewport" content="width=793, height=1124, initial-scale=1.0">
    cmd = [
        "chrome",
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        "--dump-dom",
        "--window-size=793,1124",  # dbg
        "--force-device-scale-factor=1",  # dbg
        f"--print-to-pdf={output_path}",
        f"file://{tmp_file_path.resolve()}",
    ]

    # Acquire semaphore so only a small number run concurrently
    result = None
    with semaphore:
        try:
            # Use subprocess.run with timeout
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Chromium PDF conversion failed (rc={result.returncode}): {result.stderr}"
                )
        finally:
            # Always attempt to unlink temp file
            try:
                tmp_file_path.unlink(missing_ok=True)
            except Exception:
                pass

    if result is not None:
        positions_by_class = extract_positions_from_subprocess_result(result=result)
        bboxes_by_class = convert_to_bboxes(positions_by_class)
        return bboxes_by_class
    else:
        return None


def convert_pdf_chromium(
    html: str,
    output_path: pathlib.Path,
    semaphore: threading.Semaphore,
    timeout_s: int = 300,
):
    """Spawn external Chromium CLI to print-to-pdf but limited by a semaphore."""

    # write tmp html
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        tmp_file.write(html.encode("utf-8"))
        tmp_file_path = pathlib.Path(tmp_file.name)

    cmd = [
        "chrome",
        "--headless",
        "--no-sandbox",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--print-to-pdf={output_path}",
        f"file://{tmp_file_path.resolve()}",
    ]

    # Acquire semaphore so only a small number run concurrently
    with semaphore:
        try:
            # Use subprocess.run with timeout
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Chromium PDF conversion failed (rc={result.returncode}): {result.stderr}"
                )
        finally:
            # Always attempt to unlink temp file
            try:
                tmp_file_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    path: pathlib.Path = pathlib.Path(
        "data/datasets/synthesized_datasets/docvqa-handwritten-sizes4/raw_html/7c232ba5-1d62-4f90-af6a-7d416b6568ef_0.html"
    )
    convert_pdf_chromium(
        html=path.read_text(encoding="utf-8"),
        output_path="test.pdf",
        semaphore=threading.Semaphore(1),
    )
