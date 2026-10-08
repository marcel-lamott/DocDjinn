import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from playwright.sync_api import sync_playwright


class HeadlessRenderer:
    """Keep a single Chromium browser open to measure HTML body size."""

    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        self.page = self.browser.new_page()

    def calc_size(self, html: str):
        """Return rendered width/height of <body>."""
        self.page.set_content(html)
        size = self.page.evaluate(
            """() => {
                const rect = document.body.getBoundingClientRect();
                return { width: rect.width, height: rect.height };
            }"""
        )
        return size["width"], size["height"]

    def close(self):
        self.page.close()
        self.browser.close()
        self.playwright.stop()
