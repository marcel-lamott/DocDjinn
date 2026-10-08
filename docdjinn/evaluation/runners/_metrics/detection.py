import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import ignite.distributed as idist
import torch
from ignite.engine import Engine
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from docdjinn.evaluation.model_pipeline._core._data_types import MMDetEvaluationOutput


def xyxy2xywh(bbox):
    return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]


@dataclass
class GroundTruthInstances:
    bboxes: torch.Tensor
    labels: torch.Tensor

    def detach(self):
        self.bboxes = self.bboxes.detach()
        self.labels = self.labels.detach()


@dataclass
class PredInstances:
    bboxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor

    def detach(self):
        self.bboxes = self.bboxes.detach()
        self.labels = self.labels.detach()
        self.scores = self.scores.detach()


@dataclass
class CocoCategory:
    id: int
    name: str
    supercategory: str


def _cocoeval_output_transform(model_output: MMDetEvaluationOutput):
    from mmdet.structures.bbox import scale_boxes

    assert isinstance(model_output, MMDetEvaluationOutput), (
        f"Expected {MMDetEvaluationOutput}, got {type(model_output)}"
    )
    image_ids: list[int] = []
    gt_instances: list[GroundTruthInstances] = []
    pred_instances: list[PredInstances] = []

    for batch_sample in model_output.det_data_samples:
        scale_factor = batch_sample.metainfo.get("scale_factor")
        if "gt_instances" in batch_sample:
            batch_sample.gt_instances.bboxes = scale_boxes(
                batch_sample.gt_instances.bboxes, [1 / s for s in scale_factor]
            )

        image_ids.append(batch_sample.metainfo["img_id"])
        gt_instances.append(
            GroundTruthInstances(
                bboxes=batch_sample.gt_instances["bboxes"],
                labels=batch_sample.gt_instances["labels"],
            )
        )
        pred_instances.append(
            PredInstances(
                bboxes=batch_sample.pred_instances["bboxes"],
                labels=batch_sample.pred_instances["labels"],
                scores=batch_sample.pred_instances["scores"],
            )
        )
    return image_ids, gt_instances, pred_instances


class COCOEvalMetric(Metric):
    """COCO-style evaluation metric for object detection.

    This class accumulates only essential data and performs COCO evaluation
    after gathering the results from all GPUs during `compute()`.

    Args:
        device: The device where internal storage will reside.
    """

    _state_dict_all_req_keys = ("_image_ids", "_gt_instances", "_pred_instances")
    _output_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
        "ARs",
        "ARm",
        "ARl",
    )

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._image_ids: list[int] = []
        self._gt_instances: list[GroundTruthInstances] = []
        self._pred_instances: list[PredInstances] = []
        self._result: dict[str, float] | None = None

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        image_ids, gt_instances, pred_instances = output
        self.update(image_ids, gt_instances, pred_instances)

    def _check_type(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstances],
        pred_instances: list[PredInstances],
    ) -> None:
        if not isinstance(image_ids, list) or not all(
            isinstance(i, int) for i in image_ids
        ):
            raise TypeError("image_ids must be a list of integers.")
        if not isinstance(gt_instances, list) or not all(
            isinstance(gt, GroundTruthInstances) for gt in gt_instances
        ):
            raise TypeError("gt_instances must be a list of GroundTruthInstances.")
        if not isinstance(pred_instances, list) or not all(
            isinstance(pred, PredInstances) for pred in pred_instances
        ):
            raise TypeError("pred_instances must be a list of PredInstances.")

    @reinit__is_reduced
    def update(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstances],
        pred_instances: list[PredInstances],
    ):
        self._check_type(image_ids, gt_instances, pred_instances)
        for x in gt_instances:
            x.detach()
        for x in pred_instances:
            x.detach()
        self._image_ids.extend(image_ids)
        self._gt_instances.extend(gt_instances)
        self._pred_instances.extend(pred_instances)

    def compute(self) -> float:
        if len(self._image_ids) == 0:
            raise ValueError("No data available for COCO evaluation.")

        if self._result is None:
            ws = idist.get_world_size()
            if ws > 1:
                self._image_ids = idist.all_gather(self._image_ids)
                self._gt_instances = idist.all_gather(self._gt_instances)
                self._pred_instances = idist.all_gather(self._pred_instances)

            if idist.get_rank() == 0:
                # Run compute_fn on zero rank only
                self._result = self._call_coco_eval(
                    self._image_ids, self._gt_instances, self._pred_instances
                )

            if ws > 1:
                # broadcast result to all processes
                self._result = idist.broadcast(self._result, src=0)

        return self._result

    def _call_coco_eval(
        self,
        image_ids: list[int],
        gt_instances: list[GroundTruthInstances],
        pred_instances: list[PredInstances],
    ) -> float:
        # Create COCO format ground truth annotations
        coco_gt_annotations = []
        for idx, gt in enumerate(gt_instances):
            image_id = image_ids[idx]
            for i, bbox in enumerate(gt.bboxes):
                bbox = xyxy2xywh(bbox.tolist())
                coco_gt_annotations.append(
                    {
                        "id": len(coco_gt_annotations) + 1,
                        "image_id": image_id,
                        "category_id": int(gt.labels[i])
                        + 1,  # COCO categories start at 1
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],  # width * height
                        "iscrowd": 0,
                    }
                )

        # Create COCO format predicted annotations
        coco_pred_annotations = []
        for idx, pred in enumerate(pred_instances):
            image_id = image_ids[idx]
            for i, bbox in enumerate(pred.bboxes):
                if len(bbox) == 0:  # Skip if there's no prediction
                    continue
                bbox = xyxy2xywh(bbox.tolist())
                coco_pred_annotations.append(
                    {
                        "image_id": image_id,
                        "category_id": int(pred.labels[i])
                        + 1,  # COCO categories start at 1
                        "bbox": bbox,
                        "score": float(pred.scores[i]),  # COCO needs score
                    }
                )

        # If there are no predictions, return zeros to prevent evaluation errors
        if len(coco_pred_annotations) == 0:
            return dict.fromkeys(self._output_keys, 0.0)

        # Initialize COCO ground truth and result annotations
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": image_id} for image_id in image_ids],
            "annotations": coco_gt_annotations,
            "categories": [{"id": 1, "name": "table"}],
            "info": {},
        }
        with open("coco_gt.json", "w") as f:
            json.dump(coco_gt.dataset, f)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(coco_pred_annotations)
        with open("coco_dt.json", "w") as f:
            json.dump(coco_dt.dataset, f)

        # Perform evaluation
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return {key: coco_eval.stats[i] for i, key in enumerate(self._output_keys)}

    def completed(self, engine: Engine, name: str) -> None:
        result = self.compute()
        if isinstance(result, Mapping):
            if name in result.keys():
                raise ValueError(
                    f"Argument name '{name}' is conflicting with mapping keys: {list(result.keys())}"
                )

            for key, value in result.items():
                engine.state.metrics[name + "_" + key] = value
        else:
            if isinstance(result, torch.Tensor):
                if len(result.size()) == 0:
                    result = result.item()
                elif "cpu" not in result.device.type:
                    result = result.cpu()

            engine.state.metrics[name] = result


def load_detection_metrics(device: str) -> dict[str, Metric]:
    return {
        "coco_eval": COCOEvalMetric(
            output_transform=_cocoeval_output_transform, device=device
        ),
    }
