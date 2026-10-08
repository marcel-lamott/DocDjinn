import json

import cv2
import fitz
import numpy as np
import textdistance as td
import tqdm
from PIL import Image as PILImageLoader
from torch.utils.data import Dataset

from docdjinn.generation.constants import IMAGE_RENDER_EXT
from docdjinn.generation.models import (
    SynDatasetDefinition,
    SyntheticDatasetFileStructure,
)
from docdjinn.generation.models._consts import DatasetTask
from docdjinn.generation.models._log import SynDocumentLog
from docdjinn.generation.utils.bboxes import read_syn_dataset_bboxes
from docdjinn.logging import get_logger

from ._data_types import (
    AnnotatedObject,
    AnnotatedObjectList,
    BoundingBox,
    BoundingBoxList,
    ClassificationAnnotation,
    DocumentContent,
    DocumentInstance,
    EntityLabelingAnnotation,
    ExtractiveQAAnnotation,
    ExtractiveQAPair,
    Image,
    Label,
    LabelList,
    LayoutAnalysisAnnotation,
)
from ._utilities import TaskType

logger = get_logger(__name__)


def _compute_anls(
    predictions: list[list[str]], gold_labels: list[list[str]], tau=0.5, rank=0
):
    res = []
    for i, (preds, golds) in enumerate(zip(predictions, gold_labels)):
        max_s = 0
        for pred in preds:
            for gold in golds:
                dis = td.levenshtein.distance(pred.lower(), gold.lower())
                max_len = max(len(pred), len(gold))
                if max_len == 0:
                    s = 0
                else:
                    nl = dis / max_len
                    s = 1 - nl if nl < tau else 0
                max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res) / len(res)


def _compute_iou(box1, box2):
    """Compute IoU between two bounding boxes in format [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _foreground_bbox_clip(
    image,
    bboxes,
    coords_are_inclusive=True,
    min_area=10,
    morph_kernel_size=3,
    debug=False,
    unnormalize=True,
) -> list:
    if image is None:
        raise ValueError("Image is None")

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    refined = []
    debug_vis = image.copy()

    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = box

        # Handle normalized input
        if unnormalize:
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H

        # Convert to ints
        x1, y1, x2, y2 = map(lambda v: int(round(v)), (x1, y1, x2, y2))

        if coords_are_inclusive:
            x2_slice, y2_slice = x2 + 1, y2 + 1
        else:
            x2_slice, y2_slice = x2, y2

        # Clip to image boundaries
        x1c, y1c = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
        x2c, y2c = max(0, min(W, x2_slice)), max(0, min(H, y2_slice))

        if x2c <= x1c or y2c <= y1c:
            refined.append([x1c, y1c, x2c, y2c])
            continue

        crop = gray[y1c:y2c, x1c:x2c]
        blur = cv2.GaussianBlur(crop, (5, 5), 0)

        mean_val = float(np.mean(blur))
        invert = mean_val > 127

        # Apply Otsu threshold
        if invert:
            _, mask = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ---- REMOVE HORIZONTAL LINES ----
        # Tune these values depending on your document scale
        # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mask.shape[1] // 8, 1))
        # detect_horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        # Subtract detected lines from the mask
        # mask = cv2.subtract(mask, detect_horizontal)

        # (Optional) Also remove very thin components (height < 3 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if h > 3:  # ignore 1–2 pixel tall components (likely lines)
                clean_mask[labels == i] = 255
        mask = clean_mask

        # plt.figure(figsize=(12, 12))
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # Morphological closing
        if morph_kernel_size and morph_kernel_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # plt.figure(figsize=(12, 12))
        # plt.imshow(mask, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # Remove small noise components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        keep_mask = np.zeros_like(mask, dtype=np.uint8)

        for label in range(1, n_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                keep_mask[labels == label] = 255

        # If no foreground remains, keep original box
        if np.count_nonzero(keep_mask) == 0:
            refined.append([x1c, y1c, x2c, y2c])
            continue

        # Find tight bounds
        ys, xs = np.where(keep_mask > 0)
        y_min_local, y_max_local = int(ys.min()), int(ys.max())
        x_min_local, x_max_local = int(xs.min()), int(xs.max())

        new_x1, new_y1 = x1c + x_min_local, y1c + y_min_local
        new_x2, new_y2 = x1c + x_max_local, y1c + y_max_local

        new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
        new_x2, new_y2 = min(W - 1, new_x2), min(H - 1, new_y2)

        refined.append([new_x1, new_y1, new_x2, new_y2])

        # --- Debug Visualization ---
        if debug:
            # Overlay mask in red channel
            overlay = debug_vis.copy()
            colored_mask = cv2.cvtColor(keep_mask, cv2.COLOR_GRAY2BGR)
            colored_mask = cv2.resize(colored_mask, (x2c - x1c, y2c - y1c))
            overlay[y1c:y2c, x1c:x2c, 2] = np.maximum(
                overlay[y1c:y2c, x1c:x2c, 2], colored_mask[:, :, 2]
            )

            debug_vis = cv2.addWeighted(debug_vis, 0.7, overlay, 0.3, 0)

            # Draw original bbox (yellow) and new bbox (green)
            cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.rectangle(debug_vis, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

            # Label with index
            cv2.putText(
                debug_vis,
                f"{i}",
                (x1, max(10, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

    if debug:
        return refined, debug_vis
    return refined


class SynthesizedDataset(Dataset):
    def __init__(
        self,
        dsdef: SynDatasetDefinition,
        task_type: TaskType,
        dataset_labels: list[str],
        resize_images: bool = False,
        clip_bboxes_to_foreground: bool = False,
    ):
        self.dataset_labels = dataset_labels
        self.data = self._load_your_synthesized_data(dsdef)
        self.task_type = task_type
        self.resize_images = resize_images
        self.clip_bboxes_to_foreground = clip_bboxes_to_foreground

        # remap dataset labels if cord
        if dsdef.name.startswith("cord"):
            self.dataset_labels = [x.replace(".", "_") for x in self.dataset_labels]
        if dsdef.name.startswith("publaynet"):
            self.dataset_labels = ["LE-" + x.upper() for x in self.dataset_labels]
        if dsdef.name.startswith("doclaynet") and task_type == TaskType.layout_analysis:
            self.dataset_labels = ["LE-" + x.upper() for x in self.dataset_labels]
        if dsdef.name.startswith("icdar2019"):
            self.dataset_labels = ["LE-" + x.upper() for x in self.dataset_labels]
        if dsdef.name.startswith("tobacco3482"):
            self.dataset_labels = [x.upper() for x in self.dataset_labels]
            self.dataset_labels[self.dataset_labels.index("NEWS")] = "NEWS_ARTICLE"
            self.dataset_labels[self.dataset_labels.index("ADVE")] = "ADVERTISEMENT"

    def _load_qa_gt(self, annotations: dict) -> dict:
        qa_annotations = []
        for i, a in enumerate(annotations):
            # if no answer is found we remove the sample
            if len(a["answer_bbox_indices"]) == 0:
                logger.warning(
                    f"No answer found for question id {i} in synthesized data. Skipping annotation."
                )
                continue

            qa_annotation = {
                "question_id": i,
                "question": a["question"],
                "answer_text": [a["answer"]],
                "answer_start_indices": [a["answer_bbox_indices"][0]],
                "answer_end_indices": [a["answer_bbox_indices"][-1]],
            }
            qa_annotations.append(qa_annotation)

        return {"qa_annotations": qa_annotations}

    def _load_kie_as_qa_gt(
        self, annotations: dict, dsdef: SynDatasetDefinition
    ) -> dict:
        assert dsdef.prompt_task == "json", (
            "Modelling KIE tasks as QA in dataloader not implemented for annotation-type KIE."
        )
        qa_annotations = []
        for i, a in enumerate(annotations["entities"]):
            # if no answer is found we remove the sample
            if len(a["bbox_indices"]) == 0:
                logger.warning(
                    f"No answer found for KIE (modelled as QA) question id {i} in synthesized data. Skipping sample."
                )
                continue

            qa_annotation = {
                "question_id": i,
                "question": a["key"],
                "answer_text": [a["value"]],
                "answer_start_indices": [a["bbox_indices"][0]],
                "answer_end_indices": [a["bbox_indices"][-1]],
            }
            qa_annotations.append(qa_annotation)

        return {"qa_annotations": qa_annotations}

    def _load_classification_gt(self, annotations: dict) -> dict:
        assert len(annotations) == 1
        return annotations  # is already in correct format: {"label": "FORM"}

    def _load_kie_as_qa_gt(
        self, annotations: dict, dsdef: SynDatasetDefinition
    ) -> dict:
        assert dsdef.prompt_task == "json", (
            "Modelling KIE tasks as QA in dataloader not implemented for annotation-type KIE."
        )
        qa_annotations = []
        for i, a in enumerate(annotations["entities"]):
            # if no answer is found we remove the sample
            if len(a["bbox_indices"]) == 0:
                logger.warning(
                    f"No answer found for KIE (modelled as QA) question id {i} in synthesized data. Skipping sample."
                )
                continue

            qa_annotation = {
                "question_id": i,
                "question": a["key"],
                "answer_text": [a["value"]],
                "answer_start_indices": [a["bbox_indices"][0]],
                "answer_end_indices": [a["bbox_indices"][-1]],
            }
            qa_annotations.append(qa_annotation)

        return {"qa_annotations": qa_annotations}

    def _load_kie_gt(self, annotations: dict) -> dict:
        return {"word_labels": annotations["word_labels"]}

    def _load_dla_gt(self, annotations: dict) -> dict:
        dla_annotations = []
        for i, a in enumerate(annotations):
            dla_annotation = {
                "label": a["label"],
                "bbox": [a["x0"], a["y0"], a["x2"], a["y2"]],  # already normalized
            }
            dla_annotations.append(dla_annotation)

        return {"annotations": dla_annotations}

    def _load_your_synthesized_data(self, dsdef: SynDatasetDefinition) -> list[dict]:
        dsfiles: SyntheticDatasetFileStructure = dsdef.get_file_structure()
        dslog_path = dsfiles.base_path / "dataset_log.json"
        dslog: dict = json.loads(dslog_path.read_text(encoding="utf-8"))
        valid_samples = dslog["valid_samples"]["items"]

        samples = list()
        for docid in tqdm.tqdm(
            valid_samples, desc="Loading synthesized dataset samples"
        ):
            doclog = SynDocumentLog(
                document_id=docid, logdir=dsfiles.document_logs_directory
            )

            annotations_path = dsfiles.gt_directory / f"{docid}.json"
            annotations = json.loads(annotations_path.read_text(encoding="utf-8"))

            sample_annotations = None
            match dsdef.task:
                case DatasetTask.QA.value:
                    sample_annotations = self._load_qa_gt(annotations=annotations)
                case DatasetTask.CLASSIFICATION.value:
                    sample_annotations = self._load_classification_gt(
                        annotations=annotations
                    )
                case DatasetTask.KIE.value:
                    if dsdef.dataloader_model_task_as == DatasetTask.QA.value:
                        sample_annotations = self._load_kie_as_qa_gt(
                            annotations=annotations, dsdef=dsdef
                        )
                    else:
                        sample_annotations = self._load_kie_gt(annotations=annotations)
                case DatasetTask.DLA.value:
                    sample_annotations = self._load_dla_gt(annotations=annotations)
                case _:
                    raise ValueError(f"Unknown synthetic dataset task: {dsdef.task}")

            # TODO: implement other tasks than QA

            word_bbox_path = dsfiles.get_final_normalized_bbox_path(
                level="word", doc_id=docid
            )
            word_bboxes_raw = read_syn_dataset_bboxes(word_bbox_path)
            seg_bbox_path = dsfiles.get_final_normalized_bbox_path(
                level="segment", doc_id=docid
            )
            seg_bboxes_raw = read_syn_dataset_bboxes(seg_bbox_path)

            words = [b.text for b in word_bboxes_raw]
            word_bboxes = [[b.x0, b.y0, b.x2, b.y2] for b in word_bboxes_raw]
            segment_level_bboxes = [[b.x0, b.y0, b.x2, b.y2] for b in seg_bboxes_raw]

            if len(word_bboxes) == 0:
                logger.warning(
                    f"No word bboxes found for document id {docid} in synthesized data. Skipping sample."
                )
                continue

            if doclog.ocr_required:
                image_file_path = dsfiles.img_directory / f"{docid}.{IMAGE_RENDER_EXT}"
            else:
                image_file_path = dsfiles.final_pdf_directory / f"{docid}.pdf"

            sample = {
                "sample_id": docid,
                "image_file_path": image_file_path,
                "words": words,
                "word_bboxes": word_bboxes,
                "segment_level_bboxes": segment_level_bboxes,
            }
            sample.update(sample_annotations)
            samples.append(sample)
        return samples

    def _prepare_annotations(self, sample, image) -> list:
        if self.task_type == TaskType.sequence_classification:
            assert self.dataset_labels is not None, "Dataset labels must be provided."
            return [
                ClassificationAnnotation(  # assuming label is present as category map label to whichever classification category is output for synthesized data
                    label=Label(
                        name=sample["label"],
                        value=self.dataset_labels.index(sample["label"]),
                    )
                )
            ]
        elif self.task_type == TaskType.token_classification:
            # for token classification we use bio tagging. so we need to make sure label indices
            # map back to riginal
            assert self.dataset_labels is not None, "Dataset labels must be provided."
            return [
                EntityLabelingAnnotation(
                    word_labels=LabelList.from_list(
                        [
                            Label(value=self.dataset_labels.index(label), name=label)
                            for label in sample[
                                "word_labels"
                            ]  # here we assume word_labels are provided in synthesized data
                        ]
                    )
                ),
            ]

        elif self.task_type == TaskType.extractive_qa:
            qa_pairs = []
            for i, qa_annotation in enumerate(sample["qa_annotations"]):
                qa_pair = ExtractiveQAPair(
                    id=qa_annotation["question_id"],  # unique id if available
                    question_text=qa_annotation["question"],  # question text
                    answer_start=qa_annotation[
                        "answer_start_indices"
                    ],  # start index answer in word tokens
                    answer_end=qa_annotation[
                        "answer_end_indices"
                    ],  # end index of answer in word tokens
                    answer_text=qa_annotation["answer_text"],  # actual answer text
                )
                qa_pairs.append(qa_pair)
            return [ExtractiveQAAnnotation(qa_pairs=qa_pairs)]

        elif self.task_type == TaskType.layout_analysis:
            assert self.dataset_labels is not None, "Dataset labels must be provided."
            annotated_objects = []
            for annotation in sample["annotations"]:
                label = annotation["label"]
                assert label in self.dataset_labels, (
                    f"Label {label} not in dataset labels. Found labels: {self.dataset_labels}"
                )
                bbox = BoundingBox(value=annotation["bbox"], normalized=True)

                annotated_object = AnnotatedObject(
                    label=Label(value=self.dataset_labels.index(label), name=label),
                    bbox=bbox,
                )
                annotated_objects.append(annotated_object)

            # convert to AnnotatedObjectList
            annotated_objects = AnnotatedObjectList.from_list(annotated_objects)

            if self.clip_bboxes_to_foreground:
                image = np.array(image)
                refined_bboxes = _foreground_bbox_clip(
                    image,
                    annotated_objects.bbox.value,
                    coords_are_inclusive=False,
                    min_area=10,
                    morph_kernel_size=3,
                    unnormalize=annotated_objects.bbox.normalized,
                )
                annotated_objects = annotated_objects.model_copy(
                    update={
                        "bbox": BoundingBoxList(value=refined_bboxes).normalize(
                            image.shape[1], image.shape[0]
                        )
                    }
                )

            return [
                LayoutAnalysisAnnotation(annotated_objects=annotated_objects),
            ]
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        image_file_path = str(sample["image_file_path"])
        if image_file_path.endswith(IMAGE_RENDER_EXT):
            image = PILImageLoader.open(image_file_path)
        elif image_file_path.endswith(".pdf"):
            doc = fitz.open(image_file_path)
            page = doc[0]
            mat = fitz.Matrix(1, 1)
            pix = page.get_pixmap(matrix=mat)
            image = PILImageLoader.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )
        else:
            raise ValueError(f"Unsupported image file format: {image_file_path}")

        image = Image(file_path=sample["image_file_path"], content=image)
        word_bboxes = sample["word_bboxes"]
        segment_level_bboxes = sample["segment_level_bboxes"]

        # remap segment level bboxes to word level if counts mismatch
        if len(word_bboxes) != len(segment_level_bboxes):
            remapped_segment_level_bboxes = []
            for word_bbox in word_bboxes:
                best_iou = 0.0
                best_segment_bbox = word_bbox  # fallback to word bbox if no good match

                for segment_bbox in segment_level_bboxes:
                    iou = _compute_iou(word_bbox, segment_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_segment_bbox = segment_bbox

                remapped_segment_level_bboxes.append(best_segment_bbox)
            segment_level_bboxes = remapped_segment_level_bboxes

        assert len(segment_level_bboxes) == len(word_bboxes) == len(sample["words"]), (
            f"Length mismatch after remapping for sample {sample['sample_id']}. "
            f"Words: {len(sample['words'])}, Word BBoxes: {len(word_bboxes)}, "
            f"Segment Level BBoxes: {len(segment_level_bboxes)}"
        )

        if self.resize_images:
            image = image.resize_with_aspect_ratio(1024)

        return DocumentInstance(
            sample_id=sample["sample_id"],
            image=image,
            content=DocumentContent(
                words=sample["words"],  # simple list of words
                word_bboxes=BoundingBoxList(value=word_bboxes, normalized=True),
                word_segment_level_bboxes=BoundingBoxList(
                    value=segment_level_bboxes, normalized=True
                ),
            ),
            annotations=self._prepare_annotations(sample, image.content),
        )


"""
hey man I checked your file and it was just a small mistake on read.
1. I also fixed some other mistakes on write
2. added metadata file copying for labels
3. added normalization to word bboxes
4. I noticed you use xywh format is that correct? if so it'd be better to just change it to x1y1x2y2 right here

Hey man i just wrote this.
i havent tested it for anything but it will give you the idea of what you need to do.
You will also have to add the synthesized dataset name for each in DATASET_CONFIG_MAP i guess for it to finally be loaded after being saved.
 After that it could be loaded like any other dataset and preprocessed as well.
 We need to do preprocessing in later step only because different training will result in different preprocssing
"""
