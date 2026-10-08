import json
from abc import ABC, abstractmethod
from pathlib import Path

import editdistance
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from docdjinn.utils.ocr import MicrosoftOCRWord, MicrosoftOCR


class OCRQualityMetric(ABC):
    def __init__(self, gt: list[MicrosoftOCRWord], pred: list[MicrosoftOCRWord]):
        self._gt = gt
        self._pred = pred

    @abstractmethod
    def compute(self) -> float:
        raise NotImplementedError


class WordCharacterError(OCRQualityMetric):
    def compute(self) -> float:
        # for gt_word, pred_word in zip(self._gt, self._pred):
        #     err = editdistance.eval(gt_word.text, pred_word.text) / len(gt_word.text)
        #     print(f'[CER]: {err:.2f} "{gt_word.text}"  |  "{pred_word.text}"')
        return np.mean(
            [
                editdistance.eval(gt_word.text, pred_word.text) / len(gt_word.text)
                for gt_word, pred_word in zip(self._gt, self._pred)
            ],
            dtype=float,
        )


class WordErrorRate(OCRQualityMetric):
    def compute(self) -> float:
        # err = editdistance.eval(self._gt, self._pred) / len(self._gt)
        # print(f'[WER]: {err:.2f} {len(self._gt)} GT words  |  {len(self._pred)} PRED words')
        # for gt, pred in zip(self._gt, self._pred):
        #     if  gt.text != pred.text:
        #         print(gt.text, pred.text)
        return sum(
            gt.text != pred.text for gt, pred in zip(self._gt, self._pred)
        ) / len(self._gt)


def get_ocr_for_all_files(
    base_dir: Path, ocr_listing_path: Path
) -> dict[str, MicrosoftOCR]:
    entries = [
        json.loads(line)
        for line in ocr_listing_path.read_text(encoding="utf-8").splitlines()
    ]
    print(f"Reading ({len(entries)}) ocr files ...")
    return {
        entry["image_path"]: MicrosoftOCR.load_from_file(
            base_dir / entry["ms_ocr_path"]
        )
        for entry in track(entries, description=f"Reading OCR files ...")
    }


def ocr_to_word(ocr: MicrosoftOCR) -> MicrosoftOCRWord:
    if len(ocr.words) == 1:
        return ocr.words[0]

    return MicrosoftOCRWord(
        text=" ".join(word.text for word in ocr.words),
        confidence=np.mean([word.confidence for word in ocr.words], dtype=float),
        geo=[0, 0, 0, 0],
    )


def visualize_stats(writer_stats: dict[str, dict]) -> None:
    sorted_stats = sorted(writer_stats.items(), key=lambda writer: writer[1]["cer"])

    def _show_table() -> None:
        console = Console()
        table = Table(
            title="OCR Error Rates by Writer", show_header=True, header_style="bold"
        )
        table.add_column("Writer ID", justify="left")
        table.add_column("WER", justify="right")
        table.add_column("CER", justify="right")
        for writer_id, stats in sorted_stats:
            table.add_row(writer_id, f"{stats['wer']:.1f}", f"{stats['cer']:.1f}")

        console.print(table)

    def _show_figure() -> None:
        writer_ids = [writer_id for writer_id, _ in sorted_stats]
        wer_values = [stats["wer"] for _, stats in sorted_stats]
        cer_values = [stats["cer"] for _, stats in sorted_stats]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(280, 24))

        x = range(len(writer_ids))
        width = 0.35

        ax1.bar([i - width / 2 for i in x], wer_values, width, label="WER", alpha=0.8)
        ax1.bar([i + width / 2 for i in x], cer_values, width, label="CER", alpha=0.8)
        ax1.set_xlabel("Writer ID")
        ax1.set_ylabel("Error Rate")
        ax1.set_title("WER vs CER by Writer")
        ax1.set_xticks(x)
        ax1.set_xticklabels(writer_ids, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2.scatter(cer_values, wer_values, alpha=0.6, s=100)
        ax2.set_xlabel("CER")
        ax2.set_ylabel("WER")
        ax2.set_title("WER vs CER Correlation")
        ax2.grid(True, alpha=0.3)

        max_val = max(max(cer_values), max(wer_values))
        ax2.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="WER = CER")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    _show_table()
    _show_figure()


def main() -> None:
    base_dir = Path("ANON")

    gt_manifest_path = base_dir / "generations/writer_style_manifest.json"
    gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))

    ocr_listing_path = base_dir / "generations.ocr.jsonl"
    pred_ocr_files = get_ocr_for_all_files(base_dir / "generations", ocr_listing_path)

    writer_stats = {}
    for writer in track(
        gt_manifest.get("writers", []), description="Processing writers..."
    ):
        writer_gt_samples = writer.get("samples", [])
        writer_gt_words = [
            MicrosoftOCRWord(text=sample["gt"], confidence=1.0, geo=[0, 0, 0, 0])
            for sample in writer_gt_samples
        ]
        # print(pred_ocr_files)
        writer_pred_words = [
            MicrosoftOCRWord(
                text=ocr_to_word(pred_ocr_files[sample["image"]]).text,
                confidence=1.0,
                geo=[0, 0, 0, 0],
            )
            for sample in writer_gt_samples
        ]
        wer_metric = WordErrorRate(gt=writer_gt_words, pred=writer_pred_words)
        cer_metric = WordCharacterError(gt=writer_gt_words, pred=writer_pred_words)
        writer_stats[writer["writer_id"]] = {
            "wer": wer_metric.compute(),
            "cer": cer_metric.compute(),
        }

    visualize_stats(writer_stats)


if __name__ == "__main__":
    main()
