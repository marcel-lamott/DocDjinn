from PIL import Image
import fitz

from docdjinn.generation.constants import IMAGE_RENDER_EXT
from docdjinn.generation.models._file import SyntheticDatasetFileStructure
from docdjinn.generation.models._log import SynDocumentLog


def get_pdf_size_pt(docid: str, dsfiles: SyntheticDatasetFileStructure):
    pdf_path = dsfiles.final_pdf_directory / f"{docid}.pdf"
    doc = fitz.open(pdf_path)
    page = doc[0]
    width_pt, height_pt = page.rect.width, page.rect.height
    width_px = width_pt
    height_px = height_pt
    doc.close()
    return width_px, height_px

def get_image_size_px(docid: str, dsfiles: SyntheticDatasetFileStructure):
    # Take size from image -> the bboxes we have are extracted from Image
    image_path = dsfiles.img_directory / f"{docid}.{IMAGE_RENDER_EXT}"
    img = Image.open(image_path)
    width_px, height_px = img.size  # in pixels
    return width_px, height_px

def get_document_size_for_bbox_unnormalization(docid: str, dsfiles: SyntheticDatasetFileStructure):
    doclog = SynDocumentLog(document_id=docid, logdir=dsfiles.document_logs_directory)
    if doclog.ocr_required:
        # Take size from image -> the bboxes we have are extracted from Image
        return get_image_size_px(docid=docid, dsfiles=dsfiles)
    else:
        # Take size from PDF -> the bboxes we have are extracted from PDF
        return get_pdf_size_pt(docid, dsfiles)