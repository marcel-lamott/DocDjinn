from abc import ABC
from io import BytesIO
from urllib.parse import quote

from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait


class Render:
    DRIVER_PATH: str | None = None
    DEFAULT_BROWSER: str = 'chrome'

    def __init__(self, width, height, **kwargs):
        self._width = width
        self._height = height
        self._driver = None

        self._driver = self.initialize_driver()
        self._driver.set_window_position(0, 0)
        self._driver.set_window_size(self._width, self._height)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._driver.close()
        self._driver.quit()

    def __del__(self):
        if self._driver:
            self._driver.quit()

    def initialize_driver(self):
        raise NotImplementedError

    def _ensure_image_size(self, image: Image) -> Image:
        fill_color = (255, 255, 255)
        padded_image = Image.new('RGB', (self._width, self._height), fill_color)
        padded_image.paste(image, (0, 0))

        return padded_image

    def render_html(self, html: str) -> Image:
        encoded_html = quote(html)
        self._driver.get(f"data:text/html;charset=utf-8,{encoded_html}")

        # self._driver.execute_script("return document.readyState") == "complete"

        image = Image.open(BytesIO(self._driver.get_screenshot_as_png()))
        # image = self._ensure_image_size(image)

        return image

    @staticmethod
    def bbox_from_element(element):
        elem_loc = element.location
        elem_size = element.size
        return [elem_loc['x'], elem_loc['y'], elem_size['width'], elem_size['height']]


class DocumentRender(Render, ABC):
    def get_geometries(self, element_ids: list[str]) -> dict[str, list[int]]:
        def _query_element(element_id: str, timeout_sec: int = 5):
            try:
                return WebDriverWait(self._driver, timeout_sec).until(
                    EC.presence_of_element_located((By.ID, str(element_id)))
                )
            except Exception:
                return None

        geometries = {}
        for element_id in element_ids:
            element = _query_element(element_id)
            if element is not None:
                geometries[element_id] = self.bbox_from_element(element)

        return geometries
