import os
from pathlib import Path

import numpy as np
import requests

from docdjinn.generation.utils.image import img_read, img_write_to_bytes
from docdjinn.utils.ocr import MicrosoftOCR, MicrosoftOCRWord

OCR_ENGINE = 'microsoft_di'
OCR_PORT_ENV = os.getenv('DOCDJINN_OCR_PORT')
OCR_PORT = OCR_PORT_ENV or '8000'
OCR_URL = 'http://localhost:' + OCR_PORT
OCR_POSTFIX = '0.MicrosoftOcrService.json'


def get_ocr_cache_path(image_path: Path, postfix: str) -> Path:
    return image_path.parent / f'{image_path.name}.{postfix}'


def call_ocr_service_from_image(image: np.ndarray,
                                url: str = OCR_URL,
                                engine: str = OCR_ENGINE,
                                client_caching: bool = True,
                                image_path: Path | None = None) -> MicrosoftOCR:
    headers = {'accept': 'application/json'}

    cache_path = None
    if client_caching:
        cache_path = get_ocr_cache_path(image_path, OCR_POSTFIX)
        if cache_path.exists():
            return MicrosoftOCR.load_from_file(cache_path)

    encoded_image = img_write_to_bytes(image)
    files = {'image': encoded_image, 'type': 'image/png'}
    endpoint = f'{url}/v1/sync/ocr/{engine}'
    response = requests.post(url=endpoint, headers=headers, files=files)
    response.raise_for_status()

    data = response.json()
    first_page = data['ocr']['pages'][0]
    ocr = MicrosoftOCR(
        angle=first_page['angle'],
        width=first_page['imageWidth'],
        height=first_page['imageHeight'],
        words=[
            MicrosoftOCRWord(
                text=proto['text'],
                confidence=proto['confidence'],
                geo=proto['geo']
            )
            for proto in first_page['words']
        ],
        lines=[
            MicrosoftOCRWord(
                text=proto['text'],
                confidence=proto['confidence'],
                geo=proto['geo']
            )
            for proto in first_page['lines']
        ],
    )

    if client_caching and cache_path:
        ocr.save_to_file(cache_path)

    return ocr


def call_ocr_service_from_file(image_path: Path,
                               url: str = OCR_URL,
                               engine: str = OCR_ENGINE,
                               client_caching: bool = True) -> MicrosoftOCR:
    if client_caching:
        cache_path = get_ocr_cache_path(image_path, OCR_POSTFIX)
        if cache_path.exists():
            return MicrosoftOCR.load_from_file(cache_path)

    image = img_read(image_path)
    return call_ocr_service_from_image(image, url, engine, client_caching=client_caching, image_path=image_path)


if __name__ == '__main__':
    base_dir = Path("data/temp/OCR/test-dataset")
    image_file = base_dir / "04276b91-eb12-4b47-80a6-666f6d09b6ce_1.jpg"

    # client_caching: True will also write the OCR file next to the image.
    ocr: MicrosoftOCR = call_ocr_service_from_file(image_file, client_caching=True)
    # ocr.save_to_file(...)
    print(ocr.words)
