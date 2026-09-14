"""Compare frozen racket operating points on COCO tennis-context stills.

All images contain a publisher racket annotation. This is neither racket-absence
qualification nor a player-role, temporal-association or contact benchmark.
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
from src.tracking.racket_tracking import RacketTracking
from src.tracking.global_racket_tracking import GlobalRacketTracking
from src.utils.bbox_utils import BBox


def score_predictions(annotations, predictions, image_ids):
    gt = COCO(str(annotations))
    if predictions:
        detected = gt.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = copy.deepcopy(gt.dataset)
        detected.dataset['annotations'] = []
        detected.createIndex()
    evaluator = COCOeval(gt, detected, 'bbox')
    evaluator.params.imgIds = image_ids
    evaluator.params.catIds = [43]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')
    metrics = {name: float(value) if value >= 0 else None
               for name, value in zip(names, evaluator.stats)}
    per_image = []
    for row in evaluator.evalImgs:
        if row is None or row['aRng'] != evaluator.params.areaRng[0] or row['maxDet'] != 100:
            continue
        matches, ignored = row['dtMatches'][0], row['dtIgnore'][0]
        gt_matches, gt_ignored = row['gtMatches'][0], row['gtIgnore']
        per_image.append({'image_id': int(row['image_id']),
                          'tp': int(np.sum((matches > 0) & ~ignored)),
                          'fp': int(np.sum((matches == 0) & ~ignored)),
                          'fn': int(np.sum((gt_matches == 0) & ~gt_ignored)),
                          'ignored_detections': int(np.sum(ignored)),
                          'matched_detection_ids': [int(i) for i, m, ig in zip(row['dtIds'], matches, ignored) if m > 0 and not ig],
                          'false_detection_ids': [int(i) for i, m, ig in zip(row['dtIds'], matches, ignored) if m == 0 and not ig],
                          'missed_annotation_ids': [int(i) for i, m, ig in zip(row['gtIds'], gt_matches, gt_ignored) if m == 0 and not ig]})
    if sorted(r['image_id'] for r in per_image) != sorted(image_ids):
        raise ValueError('COCO evaluation did not produce exactly one all-area row per image')
    counts = {key: sum(r[key] for r in per_image) for key in ('tp', 'fp', 'fn', 'ignored_detections')}
    tp, fp, fn = (counts[key] for key in ('tp', 'fp', 'fn'))
    counts.update(precision=tp / (tp + fp) if tp + fp else 0,
                  recall=tp / (tp + fn) if tp + fn else 0,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {'confidence_limited_coco_metrics': metrics, 'iou50_counts': counts,
            'per_image_iou50': per_image}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/coco_tennis_val2017')
    parser.add_argument('--model', type=Path, default=ROOT / 'yolo11m.pt')
    parser.add_argument('--racket-model', type=Path, help='Separate racket-only candidate; person detector remains --model')
    parser.add_argument('--mode', choices=('full640', 'full1024', 'person_crops640'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--association', choices=('independent', 'global'), default='independent')
    args = parser.parse_args()
    if args.output.exists() or not args.model.is_file():
        raise ValueError('Require a fresh output and existing checkpoint')
    if args.association == 'global' and args.mode != 'person_crops640':
        raise ValueError('Global association requires person crops')
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ids = manifest['selected_image_ids']
    if not manifest['complete'] or manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('Require completed validation acquisition')
    if ids != sorted(set(ids)) or [r['id'] for r in manifest['images']] != ids:
        raise ValueError('Image IDs missing, duplicated or reordered')
    for name, checksum in manifest['subset_annotations'].items():
        if digest(args.dataset / name) != checksum:
            raise ValueError('Annotations changed')
    for item in manifest['images']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Image checksum changed')
    model = YOLO(str(args.model))
    if model.names[0] != 'person' or model.names[38] != 'tennis racket':
        raise ValueError('Unexpected detector classes')
    racket_path = args.racket_model or args.model
    if not racket_path.is_file():
        raise ValueError('Racket checkpoint does not exist')
    racket_model = YOLO(str(racket_path)) if args.racket_model else model
    if racket_model.names[38] != 'tennis racket':
        raise ValueError('Racket candidate must preserve COCO class 38')
    device = 0 if torch.cuda.is_available() else 'cpu'
    report = {'schema_version': '1.0', 'complete': False, 'mode': args.mode,
              'qualification_evidence': False, 'association': args.association,
              'global_adapter_sha256': digest(ROOT / 'src/tracking/global_racket_tracking.py') if args.association == 'global' else None,
              'scope': 'ALL_RACKET_POSITIVE_COCO_VALIDATION_STILLS; NO_TEMPORAL_OR_ROLE_SCORE',
              'limitations': ['Public COCO validation is not a blind qualification set.',
                              'All images have racket labels; absent-racket images are not evaluated.',
                              'All detected persons supply crops; this differs from active-player selection.',
                              'AP is truncated at confidence .25, not the standard low-threshold AP sweep.',
                              'Timing includes image decode and inference, excludes model setup and scoring.'],
              'configuration': {'confidence': .25, 'nms_iou': .7, 'max_det': 300,
                                'association_memory': 'RESET_EACH_IMAGE', 'coco_category': 43},
              'dataset_manifest_sha256': digest(manifest_path),
              'annotations_sha256': digest(args.dataset / 'instances.json'),
              'checkpoint_sha256': digest(args.model), 'racket_checkpoint_sha256': digest(racket_path),
              'racket_checkpoint': str(racket_path.resolve()), 'evaluator_sha256': digest(__file__),
              'adapter_sha256': digest(ROOT / 'src/tracking/racket_tracking.py'),
              'versions': {name: importlib.metadata.version(name) for name in
                           ('ultralytics', 'torch', 'numpy', 'pycocotools')},
              'images': [], 'predictions': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for item in manifest['images']:
        frame = cv2.imread(str(args.dataset / item['path']))
        if frame is None or frame.shape[:2] != (item['height'], item['width']):
            raise ValueError('Image geometry or decode mismatch')
        crop_mode = args.mode == 'person_crops640'
        result = (model if crop_mode else racket_model).predict(frame, imgsz=1024 if args.mode == 'full1024' else 640,
                               conf=.25, iou=.7, max_det=300, classes=[0 if crop_mode else 38],
                               augment=False, verbose=False, device=device)[0]
        boxes, scores = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
        persons = len(boxes) if crop_mode else None
        if crop_mode:
            players = {i: BBox(*map(float, box), confidence=float(score), class_id=0, track_id=i)
                       for i, (box, score) in enumerate(zip(boxes, scores))}
            adapter = (GlobalRacketTracking if args.association == 'global' else RacketTracking)(model=racket_model, device=device, confidence=.25, imgsz=640)
            observations = adapter.observe(frame, players, 0.)
            pairs = [(r['bbox_xyxy'], r['confidence']) for r in observations.values() if r['state'] == 'DETECTED']
        else:
            pairs = list(zip(boxes, scores))
        for box, score in pairs:
            if not np.isfinite(box).all() or not np.isfinite(score):
                raise ValueError('Nonfinite prediction')
            x1, y1, x2, y2 = map(float, box)
            report['predictions'].append({'image_id': item['id'], 'category_id': 43,
                                          'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(score)})
        report['images'].append({'image_id': item['id'], 'person_proposals': persons, 'rackets': len(pairs)})
        if crop_mode and args.association == 'global':
            report['images'][-1]['player_boxes'] = {str(i): [p.x1, p.y1, p.x2, p.y2] for i, p in players.items()}
            report['images'][-1]['observations'] = observations
            report['images'][-1]['raw_candidates'] = [{**r, 'sources': sorted(r['sources'])} for r in adapter.last_candidates]
        if len(report['images']) % 20 == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({'mode': args.mode, 'processed': len(report['images'])}), flush=True)
    report['decode_and_inference_seconds'] = time.perf_counter() - started
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    report.update(score_predictions(args.dataset / 'instances.json', report['predictions'], ids))
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['iou50_counts']), flush=True)


if __name__ == '__main__':
    main()
