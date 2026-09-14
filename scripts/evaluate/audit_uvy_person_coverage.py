"""Measure proposal localization coverage, not active-player precision or tracking accuracy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ultralytics

from scripts.evaluate.audit_uvy_videos import read_gt
from src.detection.player_detector import PlayerDetector
from src.evaluation.detection_metrics import aggregate_counts, match_detections
from src.evaluation.tracking_metrics import detection_gap_diagnostics
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/uvy_tennis_videos')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'outputs/vision_upgrade_audit/uvy_player_tracking_baseline/report.json')
    parser.add_argument('--mode', choices=['tracked640', 'raw640', 'raw1024'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    if not manifest['complete'] or not baseline['complete'] or baseline['dataset_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Require previously verified matching dataset')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    weights = ROOT / 'yolo11m.pt'
    if baseline['checkpoint_hashes'][str(weights)] != digest(weights):
        raise ValueError('Person checkpoint changed')
    code = [Path(__file__), ROOT / 'src/detection/player_detector.py',
            ROOT / 'src/evaluation/detection_metrics.py', ROOT / 'src/evaluation/tracking_metrics.py',
            ROOT / 'scripts/evaluate/audit_uvy_videos.py', ROOT / 'src/utils/video_frame_sequence.py']
    report = {'complete': False, 'qualification_evidence': False, 'mode': args.mode,
              'scope': 'Oracle frame matching of person proposals to original class1 labels; NOT a deployable player selector',
              'limitations': ['Known original annotation omissions/misclassification retained',
                              'Unmatched people are not false positives: player GT is not complete all-person GT',
                              'Raw predictions have no temporal identity; coverage does not establish re-identification'],
              'dataset_manifest_sha256': digest(manifest_path), 'baseline_sha256': digest(args.baseline),
              'checkpoint_sha256': digest(weights), 'ultralytics_version': ultralytics.__version__,
              'code_hashes': {str(p): digest(p) for p in code}, 'sequences': {}}
    args.output.mkdir(parents=True)
    for sequence, info in manifest['sequences'].items():
        detector = PlayerDetector(str(weights), device='cuda' if torch.cuda.is_available() else 'cpu')
        started = time.perf_counter()
        predictions = []
        with VideoFrameSequence(str(args.dataset / info['video'])) as frames:
            if len(frames) != info['publisher_frames']:
                raise ValueError('Frame count changed')
            for frame in frames:
                if args.mode == 'tracked640':
                    boxes = detector.detect_and_track(frame, persist=True)
                    rows = [{'id': int(b.track_id), 'box': list(map(float, (b.x1, b.y1, b.x2, b.y2))),
                             'confidence': float(b.confidence)} for b in boxes]
                else:
                    with torch.no_grad():
                        result = detector.model.predict(frame, classes=[0], imgsz=int(args.mode[3:]),
                                                        conf=.25, verbose=False)[0]
                    rows = [{'id': i, 'box': list(map(float, box)), 'confidence': float(conf)}
                            for i, (box, conf) in enumerate(zip(result.boxes.xyxy.cpu().tolist(),
                                                               result.boxes.conf.cpu().tolist(), strict=True))]
                predictions.append(rows)
            gt = [[item for item in row if item['class'] == 1]
                  for row in read_gt(args.dataset / 'UVY' / sequence / 'gt/gt.txt', len(frames))]
            fps = frames.metadata.fps
        counts = [match_detections([r['box'] for r in pred], [r['box'] for r in truth], iou_threshold=.5)
                  for pred, truth in zip(predictions, gt, strict=True)]
        total = aggregate_counts(counts)
        gaps = detection_gap_diagnostics(gt, predictions)
        assert gaps['missed_annotated_frames'] == total.false_negatives
        entry = {'frames': len(predictions), 'fps': fps, 'seconds': time.perf_counter() - started,
                 'annotated_player_boxes': total.true_positives + total.false_negatives,
                 'matched_player_boxes': total.true_positives, 'missed_player_boxes': total.false_negatives,
                 'coverage_recall_iou_0_5': total.to_dict()['recall'],
                 'mean_matched_iou': total.to_dict()['mean_matched_iou'],
                 'person_proposals': sum(map(len, predictions)),
                 'unmatched_proposals_not_false_positives': total.false_positives,
                 'localization_gaps': gaps,
                 'resolved_predictor_settings': {k: getattr(detector.model.predictor.args, k)
                                                 for k in ['imgsz', 'conf', 'iou', 'max_det', 'tracker']},
                 'per_frame_counts': [{'matched': c.true_positives, 'missed': c.false_negatives,
                                       'unmatched_proposals': c.false_positives} for c in counts]}
        prediction_path = args.output / f'{sequence}_proposals.json'
        prediction_path.write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
        entry['predictions_file'] = prediction_path.name
        entry['predictions_sha256'] = digest(prediction_path)
        report['sequences'][sequence] = entry
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({k: v for k, v in entry.items() if k not in ['per_frame_counts', 'localization_gaps']}, allow_nan=False), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
