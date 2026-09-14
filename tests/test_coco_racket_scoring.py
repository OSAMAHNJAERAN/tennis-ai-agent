"""Controlled object matches verify that count reporting follows COCO ignores."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/tools/vision_eval_runtime'))
pytest.importorskip('pycocotools')

from scripts.evaluate.benchmark_coco_tennis_rackets import score_predictions


def test_coco_racket_duplicates_crowds_misses_and_empty_results(tmp_path):
    annotations = tmp_path / 'gt.json'
    annotations.write_text(json.dumps({
        'images': [{'id': 1, 'width': 100, 'height': 100}, {'id': 2, 'width': 100, 'height': 100}],
        'categories': [{'id': 43, 'name': 'tennis racket'}],
        'annotations': [
            {'id': 1, 'image_id': 1, 'category_id': 43, 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0},
            {'id': 2, 'image_id': 1, 'category_id': 43, 'bbox': [50, 50, 40, 40], 'area': 1600, 'iscrowd': 1},
            {'id': 3, 'image_id': 2, 'category_id': 43, 'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0}]}))
    predictions = [{'image_id': 1, 'category_id': 43, 'bbox': box, 'score': score}
                   for box, score in [([0, 0, 10, 10], .9), ([0, 0, 10, 10], .8), ([55, 55, 10, 10], .7)]]
    result = score_predictions(annotations, predictions, [1, 2])
    json.dumps(result, allow_nan=False)
    counts = result['iou50_counts']
    assert (counts['tp'], counts['fp'], counts['fn'], counts['ignored_detections']) == (1, 1, 1, 1)
    assert counts['precision'] == counts['recall'] == counts['f1'] == .5
    empty = score_predictions(annotations, [], [1, 2])['iou50_counts']
    assert (empty['tp'], empty['fp'], empty['fn']) == (0, 0, 2)
