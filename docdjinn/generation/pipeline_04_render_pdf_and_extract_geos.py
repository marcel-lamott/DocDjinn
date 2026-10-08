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
from docdjinn.generation.pipeline_03.css import (
    increase_handwriting_font_size,
    postprocess_handwriting,
    unmark_visual_elements,
)
from docdjinn.generation.utils.debug import draw_geos_on_pdf
from docdjinn.generation.utils.log import log_pipeline_level
from docdjinn.generation.utils.pdfjs import MEASURE_DIMENSIONS
from docdjinn.generation.utils.status import get_progress_bar


def safe_count_pages(pdf_path: pathlib.Path):
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)


def preprocess_html_for_pdf_alt(html_content: str) -> str:
    """
    Preprocess HTML to remove @page rules and add override styles.
    This is done before browser parsing, so we only need text manipulation.
    """
    # Remove all @page rules (handles both inline styles and style tags)
    html_content = re.sub(
        r"@page\s*\{[^}]*\}", "", html_content, flags=re.IGNORECASE | re.DOTALL
    )

    # Add override style to head with !important to ensure it wins
    override_style = """<style>
        @page {
            size: auto !important;
            margin: 0mm !important;
        }
    </style>
</head>"""

    html_content = re.sub(
        r"</head>", override_style, html_content, count=1, flags=re.IGNORECASE
    )

    return html_content


def preprocess_html_for_pdf(html_content: str) -> str:
    """
    Universal preprocessing for LLM-generated HTML.
    Removes @page rules and ensures content determines PDF size.
    Safe for: receipts, multi-column, tables, any layout.
    """
    # Step 1: Remove @page rules
    html_content = re.sub(
        r"@page\s*\{[^}]*\}", "", html_content, flags=re.IGNORECASE | re.DOTALL
    )

    # Step 2: Add override styles
    # Key: Don't force any widths - let content be measured as-is
    override_style = """<style>
        @page {
            size: auto !important;
            margin: 0mm !important;
        }
    </style>
</head>"""

    html_content = re.sub(
        r"</head>", override_style, html_content, count=1, flags=re.IGNORECASE
    )

    return html_content


async def render_pdf_async(
    doc_id,
    html,
    dsfiles: SyntheticDatasetFileStructure,
    extract_geos_for_classes: list[str],
    semaphore: asyncio.Semaphore,
    max_retries=2,
    timeout_seconds=60,  # Add timeout parameter
):
    """
    Async version: Render PDF using Playwright with automatic size detection.
    Also extracts element geometries for specified classes.
    """
    selectorMap = {
        "layout_element": '[class*="LE-"]',
        "handwriting": f".{HANDWRITING_CLASS_NAME}",
        "visual_element": "[data-placeholder]",
    }
    if any(extract_geos_for_classes):
        selectorMap["custom"] = ", ".join([f".{c}" for c in extract_geos_for_classes])

    last_error = None
    for attempt in range(1, max_retries + 2):
        browser = None
        try:
            pdf_path = dsfiles.pdf_initial_directory / f"{doc_id}.pdf"
            render_html_path = dsfiles.render_html_directory / f"{doc_id}.html"
            geometry_json_path = dsfiles.geometries_directory / f"{doc_id}.json"

            # Preprocess HTML (synchronous - fast)
            html = preprocess_html_for_pdf(html)
            soup = BeautifulSoup(html, BS_PARSER)

            soup = increase_handwriting_font_size(
                soup, dbg=doc_id == "1f100208-1fd8-4f60-b071-a51be9d7b495_2"
            )
            # soup = postprocess_handwriting(soup)
            soup = unmark_visual_elements(soup)

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

                            # Auto-detect content dimensions
                            dimensions = await page.evaluate(MEASURE_DIMENSIONS)

                            page_width_px = dimensions["width"]
                            page_height_px = dimensions["height"]

                            # Set viewport and wait for layout
                            await page.set_viewport_size(
                                {"width": page_width_px, "height": page_height_px}
                            )
                            await page.wait_for_timeout(30)

                            # Extract geometries
                            # class_selectors = ", ".join(
                            #     [f".{cls}" for cls in extract_positions_for_classes]
                            # )

                            geo_eval_str = f"""
                            () => {{
                                const data = [];

                                // Define individual selectors with labels
                                const selectorMap = {selectorMap};

                                const processedElements = new Map(); // Use Map to track matches

                                // First pass: collect all elements and their matching selectors
                                Object.entries(selectorMap).forEach(([label, selector]) => {{
                                    document.querySelectorAll(selector).forEach(el => {{
                                        if (!processedElements.has(el)) {{
                                            processedElements.set(el, []);
                                        }}
                                        processedElements.get(el).push(label);
                                    }});
                                }});

                                // Second pass: create geometry data for each unique element
                                processedElements.forEach((selectorTypes, el) => {{
                                    const rect = el.getBoundingClientRect();
                                    const computed = window.getComputedStyle(el);

                                    // Get text content (matches your Python logic)
                                    let text = '';
                                    if (el.tagName.toLowerCase() === 'input') {{
                                        text = (el.value || '').trim();
                                    }} else {{
                                        text = (el.innerText || el.textContent || '').trim();
                                    }}

                                    data.push({{
                                        id: el.id || null,
                                        tag: el.tagName.toLowerCase(),
                                        classes: el.className || null,
                                        rect: {{
                                            x: rect.x,
                                            y: rect.y,
                                            width: rect.width,
                                            height: rect.height
                                        }},
                                        visibility: computed.visibility,
                                        dataContent: el.getAttribute('data-content') || null,
                                        dataPlaceholder: el.getAttribute('data-placeholder') || null,
                                        style: el.getAttribute('style') || null,
                                        text: text,
                                        selectorTypes: selectorTypes  // Array of all matching selector types
                                    }});
                                }});

                                return data;
                            }}
                            """
                            # input(geo_eval_str)
                            geometries = await page.evaluate(geo_eval_str)

                            # Generate PDF
                            page_width_inches = page_width_px / 96
                            page_height_inches = page_height_px / 96

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
                    # os.unlink(temp_path)

            # Convert CSS pixels to PDF points
            scale = 72 / 96
            for g in geometries:
                g["rect"]["x"] *= scale
                g["rect"]["y"] *= scale
                g["rect"]["width"] *= scale
                g["rect"]["height"] *= scale

            # Save geometry JSON
            with open(geometry_json_path, "w") as f:
                json.dump(geometries, f, indent=2)

            # DEBUG
            draw_geos_on_pdf(
                geos=geometries,
                pdf_in=pdf_path,
                pdf_out=dsfiles.debug_pdf_geometries_directory / f"{doc_id}.pdf",
            )

            pdf_num_pages = safe_count_pages(pdf_path)

            return {
                DocLogKey.document_id: doc_id,
                DocLogKey.render_html_width: page_width_px,
                DocLogKey.render_html_height: page_height_px,
                DocLogKey.pdf_num_pages: pdf_num_pages,
                DocLogKey.pdf_render_error: None,
                DocLogKey.num_geometries_extracted: len(geometries),
            }

        except Exception as e:
            print(f"[yellow]Attempt {attempt} failed for {doc_id}: {e}")
            await asyncio.sleep(1)
            last_error = str(e)

    return {
        DocLogKey.document_id: doc_id,
        DocLogKey.render_html_width: None,
        DocLogKey.render_html_height: None,
        DocLogKey.pdf_num_pages: None,
        DocLogKey.pdf_render_error: last_error,
        DocLogKey.num_geometries_extracted: 0,
    }


async def process_batch_async(
    html_data,
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
            doc_id,
            html,
            dsfiles,
            extract_geos_for_classes,
            semaphore,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        for (doc_id, html) in html_data
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


def pipeline_render_pdf_and_extract_geos_parallel(params: PipelineParameters):
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
    html_data = []
    total_samples = 0

    for doc in dsdef.get_document_logs():
        total_samples += 1
        pdf_path = dsfiles.pdf_initial_directory / f"{doc.document_id}.pdf"
        valid_pdf = pdf_path.exists() and doc.pdf_num_pages == 1
        if not valid_pdf:
            html_path = dsfiles.raw_html_directory / f"{doc.document_id}.html"
            if html_path.exists():
                html = html_path.read_text(encoding="utf-8")
                html_data.append((doc.document_id, html))

    total = len(html_data)
    print(
        f"{total} valid samples out of {total_samples} total samples need to be converted."
    )

    with get_progress_bar() as progress:
        render_task = progress.add_task("[red]Rendering PDFs Pass 1...", total=total)

        # Run async event loop
        results = asyncio.run(
            process_batch_async(
                html_data,
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
