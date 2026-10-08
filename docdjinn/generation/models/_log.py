import json
import pathlib


class PromptMsgResultLogKey:
    custom_id = "custom_id"
    id = "id"
    result_type = "result_type"
    error = "error"
    response = "response"
    usage_input_tokens = "usage_input_tokens"
    usage_output_tokens = "usage_output_tokens"


class MessageProcessingLogKey:
    custom_id = "custom_id"
    result_type = "result_type"
    num_documents_expected = "num_documents_expected"
    num_documents_found = "num_documents_found"
    document_ids = "document_ids"


class DocLogKey:
    document_id = "document_id"
    html_len = "html_len"

    raw_json_gt_found = "raw_json_gt_found"
    raw_json_gt_valid_json = "raw_json_gt_valid_json"

    raw_annotation_gt_found = "raw_annotation_gt_found"
    raw_annotation_gt_extraction_errors = "raw_annotation_gt_extraction_errors"

    raw_gt_or_annotation_annotations_count = "raw_gt_or_annotation_annotations_count"

    render_html_width = "render_html_width"
    render_html_height = "render_html_height"

    pdf_num_pages = "pdf_num_pages"
    pdf_render_error = "pdf_render_error"

    num_geometries_extracted = "num_geometries_extracted"

    num_word_bboxes = "num_word_bboxes"
    num_char_bboxes = "num_char_bboxes"
    can_map_chars_to_words = "can_map_chars_to_words"

    handwriting_num_elements = "handwriting_num_elements"
    handwriting_element_extraction_errors = "handwriting_element_extraction_errors"

    handwriting_generation_authorid_to_writerstyle = (
        "handwriting_generation_authorid_to_writerstyle"
    )

    handwriting_insertion_success = "handwriting_insertion_success"
    handwriting_images_were_generated = "handwriting_images_were_generated"
    handwriting_missing_images = "handwriting_missing_images"

    visual_elements_insertion_success = "visual_elements_insertion_success"
    visual_elements_were_generated = "visual_elements_were_generated"
    visual_elements_missing_images = "visual_elements_missing_images"
    visual_elements_num_elements = "visual_elements_num_elements"
    visual_elements_extraction_errors = "visual_elements_extraction_errors"

    visual_elements_generation_logs = "visual_elements_generation_logs"
    visual_elements_generation_errors = "visual_elements_generation_errors"

    layout_elements_num_elements = "layout_elements_num_elements"
    layout_elements_extraction_errors = "layout_elements_extraction_errors"

    layout_elements_generation_logs = "layout_elements_generation_logs"
    layout_elements_generation_errors = "layout_elements_generation_errors"

    ocr_required = "ocr_required"
    ocr_found = "ocr_found"
    ocr_num_bboxes_words = "ocr_num_bboxes_words"
    ocr_num_bboxes_lines = "ocr_num_bboxes_lines"
    ocr_error = "ocr_error"

    gt_verification_confirmed_keys = "gt_verification_confirmed_keys"
    gt_verification_similarities = "gt_verification_similarities"
    gt_verification_passed = "gt_verification_passed"
    gt_verification_skipped = "gt_verification_skipped"


class SynDocumentLog:
    def __init__(self, document_id: str, logdir: pathlib.Path):
        self.document_id = document_id
        logpath = logdir / f"{document_id}.json"
        self.log = json.loads(logpath.read_text(encoding="utf-8"))

    @property
    def raw_json_gt_found(self):
        return self.log.get(DocLogKey.raw_json_gt_found, False)

    @property
    def raw_json_gt_valid_json(self):
        return self.log.get(DocLogKey.raw_json_gt_valid_json, False)

    @property
    def raw_annotation_gt_found(self):
        return self.log.get(DocLogKey.raw_annotation_gt_found, False)

    @property
    def raw_annotation_gt_extraction_errors(self):
        return self.log.get(DocLogKey.raw_annotation_gt_extraction_errors, [-1])

    @property
    def gt_verification_passed(self):
        return self.log.get(DocLogKey.gt_verification_passed, False)

    @property
    def gt_verification_skipped(self):
        return self.log.get(DocLogKey.gt_verification_skipped, False)

    @property
    def pdf_num_pages(self):
        return self.log.get(DocLogKey.pdf_num_pages, -1)

    @property
    def num_word_bboxes(self):
        return self.log.get(DocLogKey.num_word_bboxes, -1)

    @property
    def num_char_bboxes(self):
        return self.log.get(DocLogKey.num_char_bboxes, -1)

    @property
    def can_map_chars_to_words(self):
        return self.log.get(DocLogKey.can_map_chars_to_words, False)

    @property
    def handwriting_num_elements(self):
        return self.log.get(DocLogKey.handwriting_num_elements, -1)

    @property
    def handwriting_element_extraction_errors(self):
        return self.log.get(DocLogKey.handwriting_element_extraction_errors, [-1])

    @property
    def handwriting_missing_images(self):
        return self.log.get(DocLogKey.handwriting_missing_images, [-1])

    @property
    def visual_elements_num_elements(self):
        return self.log.get(DocLogKey.visual_elements_num_elements, -1)

    @property
    def visual_elements_extraction_errors(self):
        return self.log.get(DocLogKey.visual_elements_extraction_errors, [-1])

    @property
    def layout_elements_num_elements(self):
        return self.log.get(DocLogKey.layout_elements_num_elements, -1)

    @property
    def layout_elements_extraction_errors(self):
        return self.log.get(DocLogKey.layout_elements_extraction_errors, [-1])

    @property
    def ocr_required(self):
        return self.log.get(DocLogKey.ocr_required, False)

    @property
    def ocr_found(self):
        return self.log.get(DocLogKey.ocr_found, False)

    @property
    def render_html_width(self) -> int | None:
        return self.log.get(DocLogKey.render_html_width, None)

    @property
    def render_html_height(self) -> int | None:
        return self.log.get(DocLogKey.render_html_height, None)

    @property
    def annotations_count(self) -> int:
        return self.log.get(DocLogKey.raw_gt_or_annotation_annotations_count, 0)
