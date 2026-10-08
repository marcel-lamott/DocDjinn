from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import textdistance as td
from PIL import Image, ImageDraw, ImageFont
import textwrap
from docdjinn.data._core._data_types import (
    AnnotatedObjectList,
    BoundingBoxList,
    DatasetLabels,
    DocumentInstance,
    ExtractiveQAPair,
)
from docdjinn.logging import get_logger

logger = get_logger(__name__)

GT_BG_FILL=(0, 0, 0, 120)

def merge_bio_bboxes(
    words: List[str], bboxes: List[List[int]], labels: List[str]
) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    Merge BIO-style labeled bounding boxes into combined entity boxes and labels.

    Args:
        words: List of words in sequence.
        bboxes: List of bounding boxes [x1, y1, x2, y2] corresponding to each word.
        labels: BIO labels, e.g., ["B-ANSWER", "I-ANSWER", "O", "B-QUESTION", "I-QUESTION"].

    Returns:
        merged_words: List of concatenated entity strings.
        merged_bboxes: List of merged bounding boxes for each entity.
        merged_labels: List of entity types (e.g., ["ANSWER", "QUESTION"]).
    """
    merged_words = []
    merged_bboxes = []
    merged_labels = []

    current_words = []
    current_boxes = []
    current_label_type = None

    for word, bbox, label in zip(words, bboxes, labels):
        if label.startswith("B-"):
            # Finalize previous entity if any
            if current_words:
                x1 = min(b[0] for b in current_boxes)
                y1 = min(b[1] for b in current_boxes)
                x2 = max(b[2] for b in current_boxes)
                y2 = max(b[3] for b in current_boxes)
                merged_words.append(" ".join(current_words))
                merged_bboxes.append([x1, y1, x2, y2])
                merged_labels.append(current_label_type)

            # Start new entity
            current_label_type = label.split("-", 1)[1]
            current_words = [word]
            current_boxes = [bbox]

        elif label.startswith("I-") and current_label_type == label.split("-", 1)[1]:
            # Continue same entity
            current_words.append(word)
            current_boxes.append(bbox)

        else:
            # Finalize previous if we hit O or mismatch
            if current_words:
                x1 = min(b[0] for b in current_boxes)
                y1 = min(b[1] for b in current_boxes)
                x2 = max(b[2] for b in current_boxes)
                y2 = max(b[3] for b in current_boxes)
                merged_words.append(" ".join(current_words))
                merged_bboxes.append([x1, y1, x2, y2])
                merged_labels.append(current_label_type)
                current_words, current_boxes, current_label_type = [], [], None

            # If "O", skip (non-entity)
            continue

    # Finalize last entity
    if current_words:
        x1 = min(b[0] for b in current_boxes)
        y1 = min(b[1] for b in current_boxes)
        x2 = max(b[2] for b in current_boxes)
        y2 = max(b[3] for b in current_boxes)
        merged_words.append(" ".join(current_words))
        merged_bboxes.append([x1, y1, x2, y2])
        merged_labels.append(current_label_type)

    return merged_words, merged_bboxes, merged_labels


def _save_visualization(
    sample: DocumentInstance,
    dataset_name: str,
    output_dir: str,
    split: str,
    dataset_labels: DatasetLabels,
    visualize_gt_only: bool = True,
):
    """Save visualizations of document instance with bounding boxes and annotations."""

    # Create output directory
    sample_id = sample.sample_id.split("/")[-1]
    output_path = Path(output_dir) / dataset_name / split
    os.makedirs(output_path, exist_ok=True)

    # Extract annotations
    annotations = _extract_annotations(sample=sample)

    # Extract content
    words, word_bboxes, word_segment_level_bboxes = _extract_content_data(sample=sample)

    # Create filename suffix
    label_suffix = ""
    if "label" in annotations and annotations["label"] is not None:
        label_suffix = (
            f"_label={annotations['label'].name}" if annotations["label"] else ""
        )

    # # Save visualizations
    image = sample.image.content
    if not visualize_gt_only:
        if words is not None and word_bboxes is not None:
            _save_word_bbox_visualization(
                image=image,
                word_bboxes=word_bboxes,
                words=words,
                word_labels=annotations["word_labels"],
                output_path=output_path,
                sample_id=sample_id,
                label_suffix=label_suffix,
            )

        if words is not None and word_segment_level_bboxes is not None:
            _save_segment_bbox_visualization(
                image=image,
                segment_bboxes=word_segment_level_bboxes,
                words=words,
                word_labels=annotations["word_labels"],
                output_path=output_path,
                sample_id=sample_id,
                label_suffix=label_suffix,
            )
    else:
        if words is not None and word_bboxes is not None and annotations["word_labels"]:
            _save_word_labels_visualization(
                image=image,
                word_bboxes=word_bboxes,
                words=words,
                word_labels=annotations["word_labels"],
                output_path=output_path,
                sample_id=sample_id,
                label_suffix=label_suffix,
            )

    if annotations["qa_pairs"]:
        _save_qa_visualization(
            image=image,
            word_bboxes=word_bboxes,
            words=words,
            qa_pairs=annotations["qa_pairs"],
            output_path=output_path,
            sample_id=sample_id,
            label_suffix=label_suffix,
        )

    if annotations["annotated_objects"]:
        _save_layout_visualization(
            image=image,
            annotated_objects=annotations["annotated_objects"],
            image_size=sample.image.size,
            output_path=output_path,
            sample_id=sample_id,
            layout_labels=dataset_labels.layout,
            label_suffix=label_suffix,
        )


def _extract_annotations(sample: DocumentInstance) -> Dict[str, Any]:
    """Extract annotations from sample."""
    annotations = {
        "label": None,
        "word_labels": None,
        "qa_pairs": None,
        "annotated_objects": None,
    }

    for annotation in sample.annotations:
        if annotation._type == "classification":
            annotations["label"] = annotation.label
        elif annotation._type == "entity_labeling":
            annotations["word_labels"] = annotation.word_labels
        elif annotation._type == "extractive_qa":
            annotations["qa_pairs"] = annotation.qa_pairs
        elif annotation._type == "layout":
            annotations["annotated_objects"] = annotation.annotated_objects

    return annotations


def _extract_content_data(
    sample: DocumentInstance,
) -> tuple[list[str], BoundingBoxList, Optional[BoundingBoxList]]:
    """Extract content data from sample."""
    if sample.content is None:
        return None, None, None

    words, word_bboxes, word_segment_level_bboxes = (
        sample.content.words,
        sample.content.word_bboxes,
        sample.content.word_segment_level_bboxes,
    )

    # Unnormalize bounding boxes
    word_bboxes: BoundingBoxList = (
        _unnormalize_bboxes(word_bboxes, sample.image.size)
        if word_bboxes.normalized
        else word_bboxes
    )

    if word_segment_level_bboxes:
        word_segment_level_bboxes = (
            _unnormalize_bboxes(word_segment_level_bboxes, sample.image.size)
            if word_segment_level_bboxes.normalized
            else word_segment_level_bboxes
        )
    return (
        words,
        word_bboxes,
        word_segment_level_bboxes,
    )


def _unnormalize_bboxes(bbox_data, img_size):
    """Unnormalize bounding boxes from 0-1 to pixel coordinates."""
    if not bbox_data or not bbox_data.value:
        return None

    img_width, img_height = img_size
    unnormalized_bboxes = []

    for bbox in bbox_data.value:
        unnormalized_bboxes.append(
            [
                int(bbox[0] * img_width),
                int(bbox[1] * img_height),
                int(bbox[2] * img_width),
                int(bbox[3] * img_height),
            ]
        )

    return bbox_data.model_copy(
        update={"value": unnormalized_bboxes, "normalized": False}
    )


def _draw_bboxes_on_image(
    image: Image,
    bboxes_data,
    word_labels=None,
) -> Image:
    """Draw bounding boxes with warm colors and readable transparent labels."""
    if not bboxes_data or not getattr(bboxes_data, "value", None):
        return image.copy()

    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy, "RGBA")

    # Warm color palette (soft oranges, reds, and golds)
    warm_colors = [
        (255, 99, 71),  # tomato
        (255, 140, 0),  # dark orange
        (255, 165, 0),  # orange
        (255, 69, 0),  # red-orange
        (255, 215, 0),  # gold
        (255, 182, 80),  # light orange
    ]

    # Calculate font size based on image dimensions as a ratio
    img_width, img_height = image.size
    base_size = img_height
    font_size = max(12, int(base_size * 0.015))  # 2% of smaller dimension, minimum 12px

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    unique_labels = []
    if word_labels:
        unique_labels = list(
            set(word_labels if isinstance(word_labels, list) else word_labels.name)
        )

    label_to_color = {}
    for idx, label in enumerate(unique_labels):
        label_to_color[label] = warm_colors[idx % len(warm_colors)]

    for i, bbox in enumerate(bboxes_data.value):
        if len(bbox) < 4:
            continue

        # Assign color based on label, fallback to random if no label
        if word_labels and i < len(word_labels):
            current_label = (
                word_labels[i] if isinstance(word_labels, list) else word_labels.name[i]
            )
            color = label_to_color.get(current_label, random.choice(warm_colors))
        else:
            color = random.choice(warm_colors)

        # Draw bounding box
        try:
            draw.rectangle(bbox[:4], outline=color + (255,), width=2)
        except Exception as e:
            print(f"Error drawing bounding box {bbox}: {e}")
            continue

        # Prepare label text
        text = ""
        # if words and i < len(words):
        #     text = words[i]
        if word_labels and i < len(word_labels):
            text += f"{word_labels[i]}"

        if not text:
            continue

        # Compute text size (modern Pillow uses textbbox)
        try:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            text_w, text_h = font.getsize(text)

        # Place text slightly above bbox
        text_x = bbox[0]
        text_y = max(0, bbox[1] - text_h - 4)

        # Draw transparent black background behind text
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4],
            fill=GT_BG_FILL,
        )

        # Draw white text on top
        draw.text((text_x + 3, text_y + 2), text, fill=(255, 255, 255, 255), font=font)

    return img_copy


def _draw_qa_answers_on_image(image, word_bboxes, qa_pairs):
    if not word_bboxes or not word_bboxes.value:
        return image.copy()

    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy, "RGBA")

    warm_colors = [
        (255, 99, 71),
        (255, 140, 0),
        (255, 165, 0),
        (255, 69, 0),
        (255, 215, 0),
        (255, 182, 80),
    ]

    img_width, img_height = image.size
    font_size = max(12, int(img_height * 0.018))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    max_text_width = int(img_width * 0.6)

    for qa_idx, qa_pair in enumerate(qa_pairs):
        color = warm_colors[qa_idx % len(warm_colors)]

        question_text = getattr(qa_pair, "question_text", f"Q{qa_idx + 1}")
        answer_starts = getattr(qa_pair, "answer_start", [])
        answer_ends = getattr(qa_pair, "answer_end", [])

        for start, end in zip(answer_starts, answer_ends):
            if start == -1 or end == -1 or start >= len(word_bboxes.value):
                continue

            # Merge bounding boxes for full answer span
            boxes = word_bboxes.value[start : min(end + 1, len(word_bboxes.value))]
            if not boxes:
                continue

            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)

            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)

            # Create wrapped text using textwrap
            label_text = f"Q{qa_idx + 1}: {question_text}"
            # Approximate chars per line based on width and font metrics
            char_width = font.getlength("A") or font_size * 0.6
            max_chars = max_text_width // int(char_width)
            wrapped = textwrap.fill(label_text, width=max_chars)

            # Compute text block size
            text_bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]

            # Define a box above the answer span (or clamp to top of image)
            text_x = x1
            text_y = max(0, y1 - th - 8)

            # Background rectangle
            draw.rectangle(
                [text_x, text_y, text_x + tw + 8, text_y + th + 6],
                fill=GT_BG_FILL,
            )

            # Draw wrapped text directly
            draw.multiline_text(
                (text_x + 4, text_y + 3),
                wrapped,
                font=font,
                fill=(255, 255, 255, 255),
                spacing=4,
            )

    return img_copy


def _save_word_labels_visualization(
    image: Image,
    word_bboxes,
    words: List[str],
    word_labels: Optional[List[str]],
    output_path: Path,
    sample_id: str,
    label_suffix: str,
):
    """Save word-level bounding box visualization."""
    has_bio_tagging = False
    if word_labels and any(
        label.startswith("B-") or label.startswith("I-") for label in word_labels.name
    ):
        has_bio_tagging = True
    if has_bio_tagging:
        words, word_bboxes, word_labels = merge_bio_bboxes(
            words, word_bboxes.value, word_labels.name
        )
        word_bboxes = BoundingBoxList(value=word_bboxes, normalized=False)

    image_with_bboxes = _draw_bboxes_on_image(
        image,
        word_bboxes,
        word_labels if isinstance(word_labels, list) else word_labels.name,
    )
    bbox_path = output_path / f"{sample_id}{label_suffix}_word_bboxes.png"
    image_with_bboxes.save(bbox_path)
    logger.info(f"Saved word bbox visualization: {bbox_path}")


def _save_word_bbox_visualization(
    image: Image,
    word_bboxes,
    words: List[str],
    word_labels: Optional[List[str]],
    output_path: Path,
    sample_id: str,
    label_suffix: str,
):
    """Save word-level bounding box visualization."""
    image_with_bboxes = _draw_bboxes_on_image(
        image, word_bboxes, words, word_labels, "red"
    )
    bbox_path = output_path / f"{sample_id}{label_suffix}_word_bboxes.png"
    image_with_bboxes.save(bbox_path)
    logger.info(f"Saved word bbox visualization: {bbox_path}")


def _save_segment_bbox_visualization(
    image: Image,
    segment_bboxes,
    words: List[str],
    word_labels: Optional[List[str]],
    output_path: Path,
    sample_id: str,
    label_suffix: str,
):
    """Save segment-level bounding box visualization."""
    try:
        image_with_bboxes = _draw_bboxes_on_image(
            image, segment_bboxes, words, word_labels, "blue"
        )
    except:
        logger.error(
            f"Error drawing segment bounding boxes for sample {sample_id}. Skipping visualization."
        )
        return
    bbox_path = output_path / f"{sample_id}{label_suffix}_segment_bboxes.png"
    image_with_bboxes.save(bbox_path)
    logger.info(f"Saved segment bbox visualization: {bbox_path}")


def _save_qa_visualization(
    image: Image,
    word_bboxes,
    words: List[str],
    qa_pairs: List[ExtractiveQAPair],
    output_path: Path,
    sample_id: str,
    label_suffix: str,
):
    """Save QA answer visualization and text file."""
    # Save QA image
    image_with_qa = _draw_qa_answers_on_image(image, word_bboxes, qa_pairs)
    qa_image_path = output_path / f"{sample_id}{label_suffix}_qa_answers.png"
    image_with_qa.save(qa_image_path)
    logger.info(f"Saved QA answers visualization: {qa_image_path}")

    # qa_txt_path = output_path / f"{sample_id}{label_suffix}_qa.txt"
    # with open(qa_txt_path, "w", encoding="utf-8") as f:
    #     f.write(f"Document Index: {sample_id}\n\n")
    #     f.write("Document OCR:\n")
    #     f.write(",".join(words) + "\n\n")

    #     for i, qa_pair in enumerate(qa_pairs):
    #         f.write(f"Q{i + 1}: {qa_pair.question_text}\n")
    #         f.write(f"A{i + 1}: {qa_pair.answer_text}\n")

    #         answer_starts, answer_ends = qa_pair.answer_start, qa_pair.answer_end
    #         for idx, (start, end) in enumerate(zip(answer_starts, answer_ends)):
    #             f.write(f"Answer Span [{idx}]: ({start}, {end})\n")
    #             f.write(f"Extracted Answer: {' '.join(words[start : end + 1])}\n")
    #         f.write("\n")

    # logger.info(f"Saved QA info: {qa_txt_path}")


def _draw_layout_bboxes_on_image(
    image: Image,
    annotated_objects,
    image_size: tuple[int, int],
    layout_labels: List[str],
) -> Image:
    """
    Draw layout bounding boxes with warm colors (one color per label) and filled area.
    """

    # Unnormalize if needed
    bboxes = (
        _unnormalize_bboxes(annotated_objects.bbox, image_size)
        if annotated_objects.bbox.normalized
        else annotated_objects.bbox
    )

    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy, "RGBA")

    # Warm color palette
    warm_colors = [
        (255, 99, 71),  # tomato
        (255, 140, 0),  # dark orange
        (255, 165, 0),  # orange
        (255, 69, 0),  # red-orange
        (255, 215, 0),  # gold
        (255, 182, 80),  # light orange
    ]

    # Calculate font size based on image dimensions
    img_width, img_height = image.size
    base_size = img_height
    font_size = max(12, int(base_size * 0.015))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Map each layout label to a warm color
    unique_labels = list(set(layout_labels))
    label_to_color = {
        label: warm_colors[idx % len(warm_colors)]
        for idx, label in enumerate(unique_labels)
    }

    for bbox, label_idx in zip(bboxes.value, annotated_objects.label.value):
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox
        label_text = layout_labels[label_idx]
        color = label_to_color.get(label_text, random.choice(warm_colors))

        # Draw bounding box (outline only)
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)

        # Draw label text with transparent black background
        text_x, text_y = x1, max(0, y1 - font_size - 4)
        try:
            text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = font.getsize(label_text)

        # Background rectangle
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4],
            fill=GT_BG_FILL,
        )

        # Draw text
        draw.text(
            (text_x + 3, text_y + 2), label_text, fill=(255, 255, 255, 255), font=font
        )

    return img_copy


def _save_layout_visualization(
    image: Image,
    annotated_objects: AnnotatedObjectList,
    image_size: tuple[int, int],
    output_path: Path,
    sample_id: str,
    layout_labels: list[str],
    label_suffix: str,
):
    """Save layout annotation visualization."""
    # Placeholder function for layout visualization
    layout_image_path = output_path / f"{sample_id}{label_suffix}_layout.png"
    img_copy = _draw_layout_bboxes_on_image(
        image,
        annotated_objects,
        image_size=image_size,
        layout_labels=layout_labels,
    )
    img_copy.save(str(layout_image_path) + ".png")
    logger.info(f"Saved layout visualization (placeholder): {layout_image_path}.png")


def _anls_metric_str(
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


def _compute_qa_stats(split_reader, split_name):
    """Compute QA statistics for a given dataset split."""
    import tqdm

    total_questions = 0
    total_answers_found = 0
    all_extracted_answers = []
    all_gold_answers = []

    for sample in tqdm.tqdm(split_reader, f"Computing QA stats for {split_name}..."):
        # Extract annotations
        words = sample.content.words if sample.content else []
        annotations = _extract_annotations(sample=sample)
        for qa_pair in annotations["qa_pairs"]:
            total_questions += 1
            extracted_answers = []
            for ans_start, ans_end in zip(qa_pair.answer_start, qa_pair.answer_end):
                if ans_start != -1 and ans_end != -1:
                    extracted_answers.append(
                        " ".join(words[ans_start : ans_end + 1])
                        if ans_start != -1 and ans_end != -1
                        else ""
                    )
            if len(extracted_answers) > 0:
                total_answers_found += 1
            all_extracted_answers.append(extracted_answers)
            all_gold_answers.append(qa_pair.answer_text)

    logger.info(f"{split_name} - total_questions: {total_questions}")
    logger.info(f"{split_name} - total_answers_found: {total_answers_found}")

    if total_questions > 0:
        logger.info("Computing ANLS metric...")
        logger.info("First 10 extracted answers:\n%s", all_extracted_answers[:50])
        logger.info("First 10 gold answers:\n%s", all_gold_answers[:50])
        _, anls = _anls_metric_str(all_extracted_answers, all_gold_answers)
        logger.info(f"{split_name} - anls: {anls}")
