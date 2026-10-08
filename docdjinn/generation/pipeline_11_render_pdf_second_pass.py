import asyncio
from playwright.async_api import async_playwright
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pathlib
import re
import time
import tempfile
import os

from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from rich.progress import Progress

from docdjinn import ENV
from docdjinn.generation.constants import (
    BS_PARSER,
    HANDWRITING_CLASS_NAME,
    PIPELINE_03_RENDER_PDF__CHROMIUM_CONCURRENCY,
    PIPELINE_03_RENDER_PDF__MAX_WORKERS,
    PIPELINE_03_RENDER_PDF__PER_PDF_RENDER_MAX_RETRIES,
    PIPELINE_03_RENDER_PDF__PER_PDF_RENDER_TIMEOUT,
)
from docdjinn.generation.models import (
    DocLogKey,
    PipelineParameters,
    SyntheticDatasetFileStructure,
    SynDatasetDefinition,
)
from docdjinn.generation.models._log import SynDocumentLog
from docdjinn.generation.pipeline_03.css import (
    increase_handwriting_font_size,
    postprocess_handwriting,
    unmark_visual_elements,
)
from docdjinn.generation.utils.debug import draw_geos_on_pdf
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.status import get_progress_bar


def safe_count_pages(pdf_path: pathlib.Path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)


async def render_pdf_async(
    doclog: SynDocumentLog,
    dsfiles: SyntheticDatasetFileStructure,
    extract_geos_for_classes: list[str],
    semaphore: asyncio.Semaphore,
    max_retries=2,
    timeout_seconds=60,
):
    """
    Async version: Render PDF using Playwright with automatic size detection.
    """
    doc_id = doclog.document_id

    last_error = None
    for attempt in range(1, max_retries + 2):
        browser = None
        try:
            pdf_path = (
                dsfiles.pdf_without_handwriting_placeholder_directory / f"{doc_id}.pdf"
            )
            render_html_path = (
                dsfiles.render_html_second_pass_directory / f"{doc_id}.html"
            )
            html_path = dsfiles.render_html_directory / f"{doc_id}.html"
            html = html_path.read_text(encoding="utf-8")

            soup = BeautifulSoup(html, BS_PARSER)
            soup = postprocess_handwriting(soup)
            prep_html = soup.prettify()
            render_html_path.write_text(prep_html, encoding="utf-8")  # type: ignore

            # Acquire semaphore for Chromium concurrency control
            async with semaphore:
                try:
                    async with asyncio.timeout(timeout_seconds):
                        async with async_playwright() as p:
                            browser = await p.chromium.launch(headless=True)
                            page = await browser.new_page()

                            # Load HTML
                            await page.goto(
                                f"file://{render_html_path}",
                                wait_until="domcontentloaded",
                            )
                            await page.emulate_media(media="screen")

                            page_width_px = doclog.render_html_width
                            page_height_px = doclog.render_html_height

                            # Set viewport and wait for layout
                            await page.set_viewport_size(
                                {"width": page_width_px, "height": page_height_px}  # type: ignore
                            )
                            await page.wait_for_timeout(30)

                            # Generate PDF
                            page_width_inches = page_width_px / 96  # type: ignore
                            page_height_inches = page_height_px / 96  # type: ignore

                            await page.pdf(
                                path=str(pdf_path),
                                width=f"{page_width_inches}in",
                                height=f"{page_height_inches}in",
                                margin={
                                    "top": "0",
                                    "bottom": "0",
                                    "left": "0",
                                    "right": "0",
                                },
                                print_background=True,
                                display_header_footer=False,
                                prefer_css_page_size=False,
                                scale=1.0,
                            )

                            await browser.close()
                except asyncio.TimeoutError:
                    print(
                        f"PDF rendering timed out after {timeout_seconds}s for {doc_id}"
                    )
                    raise TimeoutError(
                        f"PDF rendering timed out after {timeout_seconds}s for {doc_id}"
                    )
                finally:
                    # Ensure browser closes even on timeout
                    if browser is not None:
                        try:
                            await browser.close()
                        except Exception:
                            pass

            pdf_num_pages = safe_count_pages(pdf_path)

            return {
                DocLogKey.document_id: doc_id,
                DocLogKey.pdf_num_pages: pdf_num_pages,
                DocLogKey.pdf_render_error: None,
            }

        except Exception as e:
            print(f"[yellow]Attempt {attempt} failed for {doc_id}: {e}")
            await asyncio.sleep(1)
            last_error = str(e)

    return {
        DocLogKey.document_id: doc_id,
        DocLogKey.pdf_num_pages: None,
        DocLogKey.pdf_render_error: last_error,
    }


async def process_batch_async(
    doclogs: list[SynDocumentLog],
    dsfiles,
    extract_geos_for_classes,
    chromium_concurrency,
    dsdef,
    progress,
    render_task,
    max_retries,
    timeout_seconds,
):
    """Process a batch of PDFs asynchronously."""
    semaphore = asyncio.Semaphore(chromium_concurrency)

    tasks = [
        render_pdf_async(
            doclog,
            dsfiles,
            extract_geos_for_classes,
            semaphore,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        for (doclog) in doclogs
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            dsdef.write_to_document_log(
                document_id=result[DocLogKey.document_id], vals=result
            )
            progress.update(render_task, advance=1)
            results.append(result)

            if result[DocLogKey.pdf_render_error]:
                print(
                    f"[red]PDF failed for {result[DocLogKey.document_id]}: {result[DocLogKey.pdf_render_error]}"
                )
            elif (
                result[DocLogKey.pdf_num_pages] and result[DocLogKey.pdf_num_pages] > 1
            ):
                print(
                    f"[yellow]Warning: {result[DocLogKey.document_id]} rendered to {result[DocLogKey.pdf_num_pages]} pages"
                )

        except Exception as e:
            print(f"[red]Unexpected error: {e}")
            progress.update(render_task, advance=1)

    return results


def pipeline_render_pdf_second_pass(params: PipelineParameters):
    """
    Render HTML documents to PDF using async Playwright with automatic size detection.
    Much faster than sync version!
    """
    log_pipeline_level()

    chromium_concurrency = PIPELINE_03_RENDER_PDF__CHROMIUM_CONCURRENCY
    max_retries = PIPELINE_03_RENDER_PDF__PER_PDF_RENDER_MAX_RETRIES
    timeout_seconds = PIPELINE_03_RENDER_PDF__PER_PDF_RENDER_TIMEOUT
    # extract_positions_for_classes = ["handwritten"]  # or whatever you need

    dsdef = params.dsdef
    dsfiles = dsdef.get_file_structure()
    extract_geos_for_classes = dsdef.valid_labels or []

    # Get valid documents that need PDF generation
    valid_doclogs = []
    total_samples = 0

    for doc in dsdef.get_document_logs():
        total_samples += 1
        pdf_path = (
            dsfiles.pdf_without_handwriting_placeholder_directory
            / f"{doc.document_id}.pdf"
        )
        if not pdf_path.exists():
            html_path = dsfiles.render_html_directory / f"{doc.document_id}.html"
            if html_path.exists():
                valid_doclogs.append(doc)

    total = len(valid_doclogs)
    print(
        f"{total} valid samples out of {total_samples} total samples need to be converted."
    )

    with get_progress_bar() as progress:
        render_task = progress.add_task("[red]Rendering PDFs Pass 2...", total=total)

        # Run async event loop
        results = asyncio.run(
            process_batch_async(
                valid_doclogs,
                dsfiles,
                extract_geos_for_classes,
                chromium_concurrency,
                dsdef,
                progress,
                render_task,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
            )
        )

    print(f"✅ Finished rendering {len(results)}/{total} PDFs.")

    # Summary stats
    successful = sum(1 for r in results if r[DocLogKey.pdf_num_pages] == 1)
    multi_page = sum(
        1
        for r in results
        if r[DocLogKey.pdf_num_pages] and r[DocLogKey.pdf_num_pages] > 1
    )
    failed = sum(1 for r in results if r[DocLogKey.pdf_render_error])

    print(
        f"📊 Summary: {successful} single-page, {multi_page} multi-page, {failed} failed"
    )

    return results
