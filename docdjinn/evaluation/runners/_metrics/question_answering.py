import copy
import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import due_evaluator
import torch
from due_evaluator.utils import property_scores_to_string
from ignite.engine import Engine
from ignite.metrics.metric import Metric

from docdjinn import ENV
from docdjinn.evaluation.model_pipeline._core._data_types import (
    QAModelOutput,
)
from docdjinn.logging import get_logger

logger = get_logger(__name__)

BASE_DATASETS_URI = Path(ENV.BASE_DATASETS_DIR / "due_benchmark" / "datasets")


@dataclass
class DueEvalConfig:
    reference_path: Path
    metric: str
    ignore_case: bool = True
    is_pwc: bool = False

    def split_reference_path(self, split: str) -> Path:
        split = "dev" if split == "validation" else split
        return Path(str(self.reference_path).format(split=split))


# extractive DUE configs expect final answers in format of sample id -> list of (question, answer) pairs
EX_DUE_DATASET_CONFIGS = {
    "ex_docvqa": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "DocVQA/aws_neurips_time/DocVQA/{split}/document.jsonl",
        metric="ANLS",
        ignore_case=True,
    ),
    "ex_docvqa_hw": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "DocVQA/aws_neurips_time/DocVQA/{split}/document_hw.jsonl",
        metric="ANLS",
        ignore_case=True,
    ),
    "ex_infographics": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "InfographicsVQA/aws_neurips_time/infographics_vqa/{split}/document.jsonl",
        metric="ANLS",
        ignore_case=True,
    ),
    "ex_klc": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "KleisterCharity/aws_neurips_time/kleister-charity/{split}/document.jsonl",
        metric="F1",
        ignore_case=True,
    ),
    "ex_deepform": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "DeepForm/aws_neurips_time/DeepForm/{split}/document.jsonl",
        metric="F1",
        ignore_case=True,
    ),
    "ex_pwc": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "PWC/aws_neurips_time/AxCell/{split}/document.jsonl",
        metric="GROUP-ANLS",
        ignore_case=True,
        is_pwc=True,
    ),
    "ex_wiki": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "WikiTableQuestions/aws_neurips_time/WikiTableQuestions/{split}/document.jsonl",
        metric="WTQ",
        ignore_case=False,
    ),
    "ex_tabfact": DueEvalConfig(
        reference_path=BASE_DATASETS_URI
        / "TabFact/aws_neurips_time/TabFact/{split}/document.jsonl",
        metric="F1",
        ignore_case=False,
    ),
}


class ExDueEvalMetric(Metric):
    def __init__(
        self,
        dataset_name: str,
        stage: str,
        device: str | torch.device = torch.device("cpu"),
    ) -> None:
        self._eval_config = copy.deepcopy(EX_DUE_DATASET_CONFIGS[dataset_name])
        self._eval_config.reference_path = self._eval_config.split_reference_path(stage)
        super().__init__(device=device)

    def reset(self) -> None:
        self._sample_level_qa_pairs = defaultdict(dict)

    def update(self, model_output: QAModelOutput) -> None:
        if self._eval_config.is_pwc:
            return self._update_pwc(model_output)
        else:
            return self._update_default(model_output)

    def _update_pwc(self, model_output: QAModelOutput) -> None:
        raise NotImplementedError("PWC evaluation not implemented yet.")

    def _update_default(self, model_output: QAModelOutput) -> None:
        for qa_pair in model_output.qa_pairs:
            if qa_pair.question not in self._sample_level_qa_pairs[qa_pair.sample_id]:
                self._sample_level_qa_pairs[qa_pair.sample_id][qa_pair.question] = (
                    qa_pair.answer
                )

    def _compute_default(self) -> float:
        reference = []
        answers = []
        logger.info(
            f"Preparing answers and reference for DueEvaluator based on reference file: {self._eval_config.reference_path}"
        )
        with open(self._eval_config.reference_path) as expected:
            for per_sample_reference in expected:
                per_sample_reference = json.loads(per_sample_reference)

                # for these datasets, we always have one value per key
                # also add assertion that this is true so we catch any issues early
                # for ann in per_sample_reference["annotations"]:
                #     assert len(ann["values"]) == 1, (
                #         f"For default compute, we expect one value per key. "
                #         f"Found {len(ann['values'])} values for key {ann['key']} in sample {per_sample_reference['name']}"
                #     )

                # for each line we extract the doc id (sample_id) which is in the line['name'] field
                # and get the list of (question, answer) pairs from our predictions
                if per_sample_reference["name"] not in self._sample_level_qa_pairs:
                    # logger.warning(
                    #     f"Sample ID {per_sample_reference['name']} not found in predictions. Skipping."
                    # )
                    continue
                predicted_key_values = self._sample_level_qa_pairs[
                    per_sample_reference["name"]
                ]

                per_sample_answers = []
                for ann in per_sample_reference["annotations"]:
                    per_sample_answers.append(
                        {
                            "key": ann["key"],
                            "values": [
                                {"value": predicted_key_values.get(ann["key"], "")}
                            ],  # if no answer found, return empty string
                        }
                    )
                answers.append(
                    {
                        "name": per_sample_reference["name"],
                        "annotations": per_sample_answers,
                    }
                )
                reference.append(per_sample_reference)

        # log info
        assert len(reference) == len(answers), (
            f"Number of samples in reference and answers should be the same. "
            f"Found {len(reference)} in reference and {len(answers)} in answers."
        )
        logger.info("Running DUE Evaluation on %d samples", len(reference))
        logger.info("First reference sample:")
        logger.info(json.dumps(reference[0], indent=2))
        logger.info("First answer sample:")
        logger.info(json.dumps(answers[0], indent=2))

        # load the eval reference file
        evaluator = due_evaluator.DueEvaluator(
            reference=reference,
            answers=answers,
            property_set=None,
            ignore_case=self._eval_config.ignore_case,
            metric=self._eval_config.metric,
        )
        scores = property_scores_to_string(
            [evaluator], "json", ["Precision", "Recall", "F1"]
        )
        scores = json.loads(scores)
        print("Scores", scores)
        return scores["ALL"]

    def compute(self) -> float:
        if self._eval_config.is_pwc:
            raise NotImplementedError("PWC evaluation not implemented yet.")
        else:
            return self._compute_default()

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[name + "/" + key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result


def load_extractive_qa_metrics(
    dataset_name: str, stage: str, device: str
) -> dict[str, Metric]:
    metrics = {
        "ex_due_eval": ExDueEvalMetric(
            dataset_name=dataset_name, stage=stage, device=device
        ),
    }

    if dataset_name == "ex_docvqa":
        metrics["ex_due_eval_hw"] = ExDueEvalMetric(
            dataset_name="ex_docvqa_hw", stage=stage, device=device
        )

    return metrics


# def test_step(engine, batch):
#     return QAModelOutput(
#         qa_pairs=[
#             QAPair(sample_id="rnbx0223_193", question="What is the Compound Annual Growth Rate (CAGR) for dividend payout?", answer="Artificial Intelligence"),
#             QAPair(sample_id="rnbx0223_193", question="What is the Compound Annual Growth Rate (CAGR) for net worth per share?", answer="Machine Learning"),
#             QAPair(sample_id="rnbx0223_193", question="What is the Compound Annual Growth Rate (CAGR) for net worth?", answer="A programming language"),
#             QAPair(sample_id="rnbx0223_193", question="What is the Compound Annual Growth Rate (CAGR) for total assets?", answer="A snake"),
#             QAPair(sample_id="rnbx0223_193", question="What is the dividend payout in 1996?", answer="A car"),
#             QAPair(sample_id="rnbx0223_193", question="What is the dividend payout in 2012?", answer="A bike"),
#             QAPair(sample_id="rnbx0223_193", question="What is the net worth in 1996 (Rs. Cr.)?", answer="A train"),
#             QAPair(sample_id="rnbx0223_193", question="What is the net worth in 2012 (Rs. Cr.)?", answer="A plane"),
#         ]
#     )

# engine = Engine(test_step)

# metrics = load_extractive_qa_metrics("ex_docvqa", device="cpu")
# for metric_name, metric in metrics.items():
#     logger.info(f"Attaching metric {metric_name} to engine")
#     metric.attach(
#         engine,
#         f"train/{metric_name}",
#         usage=EpochWise(),
#     )

# state = engine.run([None], max_epochs=1)
# print(state.metrics)
