"""Check frozen nearest-owner association on cached COCO and match controls."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.audit_uvy_videos import digest
from scripts.evaluate.benchmark_coco_tennis_rackets import score_predictions
from src.evaluation.detection_metrics import aggregate_counts, match_detections
from src.tracking.global_racket_tracking import GlobalRacketTracking
from src.tracking.nearest_racket_ownership import NearestRacketOwnership
from src.utils.bbox_utils import BBox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'REUSED_RACKET_POSITIVE_DEVELOPMENT_CONTROLS; NO_OWNERSHIP_GROUND_TRUTH',
              'configuration': {'confidence': .25, 'pool_iou': .5, 'reset_memory_each_image': True,
                                'nearest_gap_tolerance_px': 1e-9},
              'code_hashes': {name: digest(ROOT/name) for name in (
                  'scripts/evaluate/replay_racket_nearest_controls.py', 'src/tracking/nearest_racket_ownership.py',
                  'src/tracking/global_racket_tracking.py', 'src/tracking/racket_tracking.py',
                  'scripts/evaluate/benchmark_coco_tennis_rackets.py', 'src/evaluation/detection_metrics.py')},
              'datasets': {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for kind, filename in (('coco', 'coco_racket_global_assignment640.json'),
                           ('video', 'racketvision_racket_global_assignment640.json')):
        path = ROOT/'outputs/vision_upgrade_audit'/filename
        source = json.loads(path.read_text())
        if not source['complete'] or source['association'] != 'global' or source['global_adapter_sha256'] != digest(ROOT/'src/tracking/global_racket_tracking.py'):
            raise ValueError('Incomplete or changed source')
        adapter_hash = source.get('adapter_sha256', source.get('racket_adapter_sha256'))
        if adapter_hash != digest(ROOT/'src/tracking/racket_tracking.py'):
            raise ValueError('Original racket association changed')
        if kind == 'coco':
            dataset = ROOT/'data/external/coco_tennis_val2017'
            if digest(dataset/'manifest.json') != source['dataset_manifest_sha256'] or digest(dataset/'instances.json') != source['annotations_sha256']:
                raise ValueError('COCO source changed')
            ids = json.loads((dataset/'manifest.json').read_text())['selected_image_ids']
            images = source['images']
            if [r['image_id'] for r in images] != ids:
                raise ValueError('Image coverage differs')
        else:
            images = source['per_image']
            annotation_path = ROOT/'data/external/racketvision_validation/tennis/info/val_coco.json'
            if digest(annotation_path) != source['annotation_sha256']:
                raise ValueError('Match annotations changed')
            labels = json.loads(annotation_path.read_text())
            category = next(r['id'] for r in labels['categories'] if r['name'] == 'tennis_racket')
            by_id = {r['id']: r['file_name'] for r in labels['images']}
            targets = {}
            for annotation in labels['annotations']:
                if annotation['category_id'] == category:
                    x, y, w, h = annotation['bbox']
                    targets.setdefault(by_id[annotation['image_id']], []).append([x, y, x+w, y+h])
            for image in images:
                expected = targets.get(image['image'], [])
                # Original evaluator multiplies and divides by image dimensions;
                # even equal dimensions introduce sub-picopixel roundoff.
                if len(image['targets']) != len(expected) or any(
                    not math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                    for original, cached in zip(expected, image['targets'], strict=True)
                    for a, b in zip(original, cached, strict=True)):
                    raise ValueError('Cached match targets differ from source')
            if len(images) != source['images'] or len({r['image'] for r in images}) != len(images):
                raise ValueError('Incomplete or duplicate match frame coverage')
        rows, predictions, counts, baseline_counts = [], [], [], []
        for item in images:
            players = {int(i): BBox(*box) if box else None for i, box in item['player_boxes'].items()}
            original = GlobalRacketTracking(model=object()).assign_candidates(players, item['raw_candidates'], 0)
            if json.loads(json.dumps(original)) != item['observations']:
                raise ValueError('Original assignments do not reproduce exactly')
            changed = NearestRacketOwnership(model=object()).assign_candidates(players, item['raw_candidates'], 0)
            present = [v for v in changed.values() if v['state'] == 'DETECTED']
            boxes = [v['bbox_xyxy'] for v in present]
            row = {'image': item.get('image', item.get('image_id')), 'observations': changed,
                   'predictions': boxes, 'baseline_assignments_exact': True}
            if kind == 'coco':
                for value in present:
                    x1, y1, x2, y2 = value['bbox_xyxy']
                    predictions.append({'image_id': item['image_id'], 'category_id': 43,
                                        'bbox': [x1, y1, x2-x1, y2-y1], 'score': value['confidence']})
            else:
                result = match_detections(boxes, item['targets'], iou_threshold=.5)
                counts.append(result)
                baseline_counts.append(match_detections(item['predictions'], item['targets'], iou_threshold=.5))
                row.update(targets=item['targets'], counts=result.to_dict(), baseline_counts=item['counts'])
            rows.append(row)
        result = {'source_sha256': digest(path), 'images': len(rows), 'per_image': rows,
                  'baseline_assignments_exact': True}
        if kind == 'coco':
            result.update(score_predictions(dataset/'instances.json', predictions, ids))
            result['baseline_metrics'] = source['iou50_counts']
        else:
            baseline_metrics = aggregate_counts(baseline_counts).to_dict()
            if json.loads(json.dumps(baseline_metrics)) != source['metrics']:
                raise ValueError('Baseline metrics do not reproduce')
            result.update(metrics=aggregate_counts(counts).to_dict(), baseline_metrics=baseline_metrics,
                          source_annotation_sha256=source['annotation_sha256'])
        report['datasets'][kind] = result
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        metric = result.get('iou50_counts', result.get('metrics'))
        print(json.dumps({'kind': kind, 'metrics': {k: v for k, v in metric.items() if k != 'matched_ious'}}), flush=True)
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
