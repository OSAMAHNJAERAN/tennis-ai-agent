"""Label-assisted localization ceilings before racket assignment; not accuracy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.acquire_coco_tennis_validation import digest
from src.evaluation.detection_metrics import match_detections
from src.tracking.global_racket_tracking import GlobalRacketTracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--kind', choices=('coco', 'video'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Require fresh output')
    report = json.loads(args.report.read_text())
    if report['association'] != 'global':
        raise ValueError('Require global assignment raw proposals')
    if report['global_adapter_sha256'] != digest(ROOT / 'src/tracking/global_racket_tracking.py'):
        raise ValueError('Pooling implementation changed since inference')
    adapter = GlobalRacketTracking(model=object())
    rows = []
    if args.kind == 'coco':
        if not report['complete']:
            raise ValueError('Incomplete report')
        annotation_path = ROOT / 'data/external/coco_tennis_val2017/instances.json'
        if digest(annotation_path) != report['annotations_sha256']:
            raise ValueError('Annotations changed')
        annotations = json.loads(annotation_path.read_text())['annotations']
        if any(r.get('iscrowd', 0) for r in annotations if r['category_id'] == 43):
            raise ValueError('Diagnostic assumes non-crowd racket labels')
        for image in report['images']:
            targets = []
            for ann in annotations:
                if ann['image_id'] == image['image_id'] and ann['category_id'] == 43:
                    x, y, w, h = ann['bbox']
                    targets.append([x, y, x + w, y + h])
            rows.append((image['image_id'], targets, image['raw_candidates']))
    else:
        if len(report['per_image']) != report['images']:
            raise ValueError('Incomplete video image coverage')
        rows = [(image['image'], image['targets'], image['raw_candidates']) for image in report['per_image']]
    results = []
    for identifier, targets, candidates in rows:
        raw = [r['bbox_xyxy'] for r in candidates]
        pooled = [r['bbox_xyxy'] for r in adapter.pool_candidates(candidates)]
        results.append({'image': identifier, 'targets': len(targets), 'raw_proposals': len(raw),
                        'pooled_proposals': len(pooled),
                        'raw_oracle_matches': match_detections(raw, targets, iou_threshold=.5).true_positives,
                        'pooled_oracle_matches': match_detections(pooled, targets, iou_threshold=.5).true_positives})
    totals = {key: sum(r[key] for r in results) for key in ('targets', 'raw_proposals', 'pooled_proposals', 'raw_oracle_matches', 'pooled_oracle_matches')}
    output = {'complete': True, 'qualification_evidence': False,
              'scope': 'LABEL_ASSISTED_MAXIMUM_CARDINALITY_LOCALIZATION_CEILING; NOT_DEPLOYABLE_RECALL_OR_OWNERSHIP',
              'source_sha256': digest(args.report), 'script_sha256': digest(__file__),
              'totals': totals, 'per_image': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
