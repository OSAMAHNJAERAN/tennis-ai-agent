"""Fixed-threshold box matching with explicit false-positive/negative counts.

This reports detection precision/recall at a named IoU threshold, not COCO AP.
Matching maximizes valid matches first, then total IoU. Duplicate detections and
wrong-location predictions remain false positives even on positive images.
"""

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class DetectionCounts:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...] = ()

    def to_dict(self) -> dict:
        tp, fp, fn = self.true_positives, self.false_positives, self.false_negatives
        return {
            **asdict(self),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mean_matched_iou": float(np.mean(self.matched_ious)) if self.matched_ious else None,
        }


def _boxes(values: Sequence[Sequence[float]]) -> np.ndarray:
    boxes = np.asarray(values, dtype=float)
    if boxes.size == 0 and boxes.shape in ((0,), (0, 4)):
        return np.empty((0, 4))
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("Boxes must be finite Nx4 xyxy coordinates")
    if np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Boxes must have positive width and height")
    return boxes


def match_detections(predictions, targets, *, iou_threshold: float = 0.5) -> DetectionCounts:
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    pred, gt = _boxes(predictions), _boxes(targets)
    if len(pred) == 0 or len(gt) == 0:
        return DetectionCounts(0, len(pred), len(gt))
    low = np.maximum(pred[:, None, :2], gt[None, :, :2])
    high = np.minimum(pred[:, None, 2:], gt[None, :, 2:])
    intersection = np.prod(np.maximum(0, high - low), axis=2)
    pred_area = np.prod(pred[:, 2:] - pred[:, :2], axis=1)
    gt_area = np.prod(gt[:, 2:] - gt[:, :2], axis=1)
    ious = intersection / (pred_area[:, None] + gt_area[None, :] - intersection)
    valid = ious >= iou_threshold
    # Losing one valid match must cost more than all possible IoU tie gains.
    reward = valid * (min(len(pred), len(gt)) + 1 + ious)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    matched = tuple(float(ious[r, c]) for r, c in zip(rows, columns) if valid[r, c])
    tp = len(matched)
    return DetectionCounts(tp, len(pred) - tp, len(gt) - tp, matched)


def aggregate_counts(rows: Sequence[DetectionCounts]) -> DetectionCounts:
    return DetectionCounts(
        sum(row.true_positives for row in rows),
        sum(row.false_positives for row in rows),
        sum(row.false_negatives for row in rows),
        tuple(value for row in rows for value in row.matched_ious),
    )
