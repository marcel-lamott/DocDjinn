import os
from selenium.webdriver import Chrome, ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions

from docdjinn.generation.pipeline_03.render.render import DocumentRender


class ChromeRender(DocumentRender):
    def initialize_driver(self):
        driver_options = ChromeOptions()
        driver_options.add_argument("--headless=new")
        driver_options.add_argument("--no-sandbox")
        driver_options.add_argument("--disable-dev-shm-usage")
        driver_options.add_argument("--disable-gpu")
        driver_options.add_argument("--hide-scrollbars")
        driver_options.add_argument("--force-device-scale-factor=1")
        # driver_service = ChromeService(executable_path=self.DRIVER_PATH, log_output=subprocess.STDOUT)
        driver_service = ChromeService(executable_path=os.getenv("CHROME_DRIVER_PATH"))
        return Chrome(service=driver_service, options=driver_options)
