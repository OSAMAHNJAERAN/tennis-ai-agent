"""Check frozen IoU-.5 deduplication on previously measured video-frame crops."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from torchvision.ops import nms
from scripts.data.acquire_coco_tennis_validation import digest
from src.evaluation.detection_metrics import aggregate_counts, match_detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'artifacts/validation/vision_upgrade/racketvision_racket_yolo11m_player_crops.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output must be fresh')
    source = json.loads(args.source.read_text())
    if source['mode'] != 'player_crops' or source['confidence'] != .25 or source['match_iou'] != .5:
        raise ValueError('Unexpected source configuration')
    if len(source['per_image']) != source['images']:
        raise ValueError('Incomplete source coverage')
    rows, counts, original_counts = [], [], []
    for item in source['per_image']:
        boxes, scores = item['predictions'], item['confidences']
        if len(boxes) != len(scores):
            raise ValueError('Box-score mismatch')
        original_counts.append(match_detections(boxes, item['targets'], iou_threshold=.5))
        selected = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), .5).tolist() if boxes else []
        filtered = [boxes[index] for index in selected]
        count = match_detections(filtered, item['targets'], iou_threshold=.5)
        counts.append(count)
        rows.append({'image': item['image'], 'predictions': filtered,
                     'removed': len(boxes) - len(filtered), 'counts': count.to_dict()})
    original = aggregate_counts(original_counts).to_dict()
    for key in ('true_positives', 'false_positives', 'false_negatives'):
        if original[key] != source['metrics'][key]:
            raise ValueError('Original saved metric does not reproduce')
    report = {'complete': True, 'qualification_evidence': False,
              'scope': 'PREVIOUSLY_USED_SPARSE_RACKET_POSITIVE_VIDEO_FRAMES; NO_TEMPORAL_ASSOCIATION_SCORE',
              'source': str(args.source), 'source_sha256': digest(args.source),
              'script_sha256': digest(__file__), 'nms_iou': .5,
              'original_metrics_reproduced': True, 'metrics': aggregate_counts(counts).to_dict(),
              'removed_detections': sum(r['removed'] for r in rows), 'per_image': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'per_image'}))


if __name__ == '__main__':
    main()
