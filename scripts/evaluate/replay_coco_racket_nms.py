"""Replay one fixed IoU-.5 deduplication hypothesis on saved racket outputs.

Exploratory validation-set ablation, not independent qualification. Does not
establish player ownership or temporal consistency after suppressing boxes.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.benchmark_coco_tennis_rackets import score_predictions
from scripts.data.acquire_coco_tennis_validation import digest
import torch
from torchvision.ops import nms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/coco_tennis_val2017')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be fresh')
    sources = [json.loads(path.read_text()) for path in args.reports]
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    ids = manifest['selected_image_ids']
    for report in sources:
        if not report['complete'] or report['dataset_manifest_sha256'] != digest(args.dataset / 'manifest.json'):
            raise ValueError('Incomplete report or different dataset')
        if report['annotations_sha256'] != digest(args.dataset / 'instances.json'):
            raise ValueError('Annotation mismatch')
        if [r['image_id'] for r in report['images']] != ids:
            raise ValueError('Image coverage mismatch')
    predictions = []
    for image_id in ids:
        candidates = [p for source in sources for p in source['predictions'] if p['image_id'] == image_id]
        if not candidates:
            continue
        boxes = torch.tensor([p['bbox'] for p in candidates], dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        selected = nms(boxes, torch.tensor([p['score'] for p in candidates]), .5).tolist()
        predictions.extend({key: candidates[index][key] for key in ('image_id', 'category_id', 'bbox', 'score')}
                           for index in selected)
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'EXPLORATORY_REUSED_VALIDATION_SET; NO_PLAYER_OWNERSHIP_OR_TEMPORAL_SCORE',
              'configuration': {'confidence': .25, 'nms_iou': .5},
              'source_reports': [{'path': str(p), 'sha256': digest(p)} for p in args.reports],
              'script_sha256': digest(__file__), 'predictions': predictions}
    report.update(score_predictions(args.dataset / 'instances.json', predictions, ids))
    report['complete'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['iou50_counts']))


if __name__ == '__main__':
    main()
