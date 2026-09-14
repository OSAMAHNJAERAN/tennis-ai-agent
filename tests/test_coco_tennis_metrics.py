import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'artifacts/tools/vision_eval_runtime'))
pytest.importorskip('pycocotools')

from scripts.evaluate.benchmark_coco_tennis_people import coco_metrics


def annotation_file(tmp_path, crowd=False):
    path = tmp_path / 'annotations.json'
    keypoints = [value for index in range(17) for value in (10 + index, 20 + index, 2)]
    path.write_text(json.dumps({
        'images': [{'id': 1, 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'person', 'keypoints': [str(i) for i in range(17)], 'skeleton': []}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 60, 80],
                         'area': 4800, 'iscrowd': int(crowd), 'num_keypoints': 17, 'keypoints': keypoints}]}))
    return path, keypoints


@pytest.mark.parametrize('task', ['bbox', 'keypoints'])
def test_empty_predictions_are_scored_as_misses(tmp_path, task):
    path, _ = annotation_file(tmp_path)
    metrics = coco_metrics(path, [], [1], task)
    assert metrics['AP'] == 0


@pytest.mark.parametrize('task', ['bbox', 'keypoints'])
def test_perfect_prediction_has_perfect_ap(tmp_path, task):
    path, keypoints = annotation_file(tmp_path)
    prediction = {'image_id': 1, 'category_id': 1, 'bbox': [5, 5, 60, 80],
                  'score': .9, 'keypoints': keypoints}
    assert coco_metrics(path, [prediction], [1], task)['AP'] == pytest.approx(1.)


def test_crowd_only_annotation_is_not_a_recall_target(tmp_path):
    path, _ = annotation_file(tmp_path, crowd=True)
    assert coco_metrics(path, [], [1], 'bbox')['AP'] is None
