"""Evaluate person boxes or body poses on frozen COCO tennis-context images.

Uses COCO crowd/ignore/OKS rules. This does not measure player identities.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'artifacts/tools/vision_eval_runtime'))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from scripts.data.acquire_coco_tennis_validation import digest


def coco_metrics(annotations, predictions, image_ids, task):
    ground_truth = COCO(str(annotations))
    if predictions:
        detected = ground_truth.loadRes(predictions)
    else:
        # COCO.loadRes indexes the first result; construct an empty dataset
        # explicitly so an entirely failed detector still receives a score.
        detected = COCO()
        detected.dataset = copy.deepcopy(ground_truth.dataset)
        detected.dataset['annotations'] = []
        detected.createIndex()
    evaluator = COCOeval(ground_truth, detected, task)
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [1]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
              'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
             if task == 'bbox' else
             ('AP', 'AP50', 'AP75', 'AP_medium', 'AP_large',
              'AR', 'AR50', 'AR75', 'AR_medium', 'AR_large'))
    return {name: float(value) if value >= 0 else None
            for name, value in zip(names, evaluator.stats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--task', choices=('bbox', 'keypoints'), required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.model.is_file() or args.imgsz <= 0:
        raise ValueError('Require fresh output, existing model and positive image size')
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('Completed validation acquisition required')
    ids = manifest['selected_image_ids']
    if ids != sorted(set(ids)) or [item['id'] for item in manifest['images']] != ids:
        raise ValueError('Manifest image IDs are missing, duplicated or reordered')
    for name, checksum in manifest['subset_annotations'].items():
        if digest(args.dataset / name) != checksum:
            raise ValueError('Subset annotations changed')
    for item in manifest['images']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Validation image changed')
    report = {'schema_version': '1.0', 'complete': False, 'qualification_evidence': False,
              'scope': 'TENNIS_CONTEXT_STILL_IMAGE_PERSON_OR_POSE_VALIDATION; NO_PLAYER_ROLE_OR_IDENTITY_SCORE',
              'dataset_manifest_sha256': digest(manifest_path),
              'checkpoint_sha256': digest(args.model), 'checkpoint': str(args.model.resolve()),
              'evaluator_sha256': digest(__file__), 'task': args.task,
              'configuration': {'imgsz': args.imgsz, 'confidence': .001, 'nms_iou': .7,
                                'max_detections': 300, 'classes': [0], 'augmentation': False},
              'operating_point_scope': 'LOW_THRESHOLD_AP_SWEEP_NOT_PIPELINE_CONFIDENCE_POINT',
              'pose_score': 'PERSON_BOX_CONFIDENCE; ALL_PREDICTED_JOINT_COORDINATES_RETAINED_FOR_OKS',
              'versions': {name: importlib.metadata.version(name) for name in
                           ('ultralytics', 'torch', 'numpy', 'pycocotools')},
              'counts': manifest['counts'], 'images': [], 'predictions': []}
    model = YOLO(str(args.model))
    if model.names[0] != 'person' or (args.task == 'keypoints' and model.task != 'pose'):
        raise ValueError('Checkpoint task/category mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for item in manifest['images']:
        frame = cv2.imread(str(args.dataset / item['path']))
        if frame is None or frame.shape[:2] != (item['height'], item['width']):
            raise ValueError('Image failed to decode or source dimensions changed')
        result = model.predict(frame, imgsz=args.imgsz, conf=.001, iou=.7,
                               max_det=300, classes=[0], augment=False, verbose=False,
                               device=0 if torch.cuda.is_available() else 'cpu')[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        poses = result.keypoints.data.cpu().numpy() if args.task == 'keypoints' else None
        if poses is not None and poses.shape != (len(boxes), 17, 3):
            raise ValueError('Expected 17 COCO joints with confidences per detected person')
        for index, (box, score) in enumerate(zip(boxes, scores)):
            if not np.isfinite(box).all() or not np.isfinite(score):
                raise ValueError('Nonfinite prediction')
            x1, y1, x2, y2 = map(float, box)
            record = {'image_id': item['id'], 'category_id': 1, 'score': float(score),
                      'bbox': [x1, y1, x2 - x1, y2 - y1]}
            if poses is not None:
                if not np.isfinite(poses[index]).all():
                    raise ValueError('Nonfinite pose prediction')
                record['keypoints'] = poses[index].reshape(-1).tolist()
            report['predictions'].append(record)
        report['images'].append({'image_id': item['id'], 'detections': len(boxes)})
        if len(report['images']) % 20 == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({'processed_images': len(report['images'])}), flush=True)
    report['inference_seconds_excluding_setup_and_scoring'] = time.perf_counter() - started
    annotations = args.dataset / ('instances.json' if args.task == 'bbox' else 'keypoints.json')
    report['metrics'] = coco_metrics(annotations, report['predictions'], ids, args.task)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    main()
