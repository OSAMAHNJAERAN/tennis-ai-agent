"""Evaluate existing crop pose adapter using frozen detected person boxes.

All people are retained as targets; this isolates spatial crop behavior without
claiming the court-role selector or temporal tracking has been evaluated.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from scripts.evaluate.benchmark_coco_tennis_people import coco_metrics
from scripts.data.acquire_coco_tennis_validation import digest
from src.tracking.player_motion_tracking import PlayerMotionTracking, JOINT_NAMES
from src.utils.bbox_utils import BBox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    detections = json.loads(args.detections.read_text())
    if (not manifest['complete'] or manifest['split'] != 'VALIDATION_ONLY'
            or not detections['complete'] or detections['task'] != 'bbox'
            or detections['dataset_manifest_sha256'] != digest(manifest_path)):
        raise ValueError('Require completed box predictions on this exact validation dataset')
    if [item['image_id'] for item in detections['images']] != manifest['selected_image_ids']:
        raise ValueError('Detection image list differs from selected validation images')
    for name, checksum in manifest['subset_annotations'].items():
        if digest(args.dataset / name) != checksum:
            raise ValueError('Annotation checksum changed')
    for item in manifest['images']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Image checksum changed')
    grouped = defaultdict(list)
    for prediction in detections['predictions']:
        if prediction['category_id'] == 1 and prediction['score'] >= .25:
            grouped[prediction['image_id']].append(prediction)
    report = {'schema_version': '1.0', 'complete': False, 'qualification_evidence': False,
              'scope': 'DETECTED_PERSON_CROPS_WITH_EXISTING_POSE_ADAPTER; ALL_PERSON_GT; NO_ROLE_SELECTION_OR_TEMPORAL_TRACKING',
              'dataset_manifest_sha256': digest(manifest_path), 'detections_sha256': digest(args.detections),
              'checkpoint_sha256': digest(args.model), 'evaluator_sha256': digest(__file__),
              'adapter_sha256': digest(ROOT / 'src/tracking/player_motion_tracking.py'),
              'configuration': {'person_confidence_minimum': .25, 'crop_width_margin_fraction': .25,
                                'crop_height_margin_fraction': .15, 'pose_image_size': 640,
                                'pose_detection_confidence': .25, 'joint_confidence': .35,
                                'pose_to_person_minimum_iou': .25},
              'scoring': 'INSTANCE_SCORE_IS_SOURCE_PERSON_CONFIDENCE; MISSING_JOINTS_ZERO_XY; COCO_OKS_ALL_LABELED_JOINTS',
              'predictions': [], 'images': []}
    model = PlayerMotionTracking(str(args.model), device='cuda' if torch.cuda.is_available() else 'cpu')
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for item in manifest['images']:
        frame = cv2.imread(str(args.dataset / item['path']))
        if frame is None or frame.shape[:2] != (item['height'], item['width']):
            raise ValueError('Source image did not decode with expected geometry')
        people = grouped[item['id']]
        boxes = {}
        for index, person in enumerate(people):
            x, y, width, height = person['bbox']
            boxes[index] = BBox(x, y, x + width, y + height, confidence=person['score'], class_id=0)
        poses = model.observe(frame, boxes)
        emitted = 0
        for index, joints in poses.items():
            if joints is None or not any(joint['position_px'] is not None for joint in joints.values()):
                continue
            keypoints = []
            for name in JOINT_NAMES:
                joint = joints[name]
                point = joint['position_px']
                keypoints.extend([*point, joint['confidence']] if point is not None else [0., 0., 0.])
            report['predictions'].append({'image_id': item['id'], 'category_id': 1,
                                          'bbox': people[index]['bbox'], 'score': people[index]['score'],
                                          'keypoints': keypoints})
            emitted += 1
        report['images'].append({'image_id': item['id'], 'detected_crops': len(boxes), 'emitted_poses': emitted})
        if len(report['images']) % 20 == 0:
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps({'processed_images': len(report['images'])}), flush=True)
    report['pose_seconds_excluding_box_detection_setup_and_scoring'] = time.perf_counter() - started
    report['metrics'] = coco_metrics(args.dataset / 'keypoints.json', report['predictions'],
                                     manifest['selected_image_ids'], 'keypoints')
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['metrics']), flush=True)


if __name__ == '__main__':
    main()
