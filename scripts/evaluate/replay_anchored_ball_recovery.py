"""Compare fixed strong-anchor recovery against unchanged strong top-one output."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.evaluation.point_metrics import evaluate_points
from src.tracking.anchored_ball_recovery import recover_anchored_gaps
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.temporal_ball_tracker import BallObservation


def filtered_candidates(clip):
    gate = StationaryCandidateFilter(minimum_seconds=.25)
    output = []
    for index, items in enumerate(clip['predictions']):
        values = [BallObservation(item['x'], item['y'], item['confidence']) for item in items]
        kept, _ = gate.filter(values, index / clip['fps'], clip['fps'], clip['size'])
        output.append(kept)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strong-report', type=Path, required=True)
    parser.add_argument('--weak-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {name: digest(path) for name, path in {
        'strong_report_sha256': args.strong_report, 'weak_report_sha256': args.weak_report,
        'evaluator_sha256': __file__, 'recovery_sha256': ROOT / 'src/tracking/anchored_ball_recovery.py',
        'filter_sha256': ROOT / 'src/tracking/stationary_candidate_filter.py',
        'metric_sha256': ROOT / 'src/evaluation/point_metrics.py'}.items()}
    strong_source = json.loads(args.strong_report.read_text())
    weak_source = json.loads(args.weak_report.read_text())
    for key in ('dataset_manifest_sha256', 'checkpoint_sha256', 'ensemble_checkpoint_sha256', 'wasb_temporal_step'):
        if strong_source.get(key) != weak_source.get(key):
            raise ValueError(f'Different model/data lineage: {key}')
    if (strong_source['wasb_heatmap_threshold'] != .2 or weak_source['wasb_heatmap_threshold'] != .05
            or len(strong_source['clips']) != len(weak_source['clips'])):
        raise ValueError('This predeclared experiment requires .20/.05 aligned reports')
    rows, baseline_rows, clips = [], [], []
    seen = set()
    for strong_clip, weak_clip in zip(strong_source['clips'], weak_source['clips']):
        for key in ('clip', 'size', 'fps'):
            if strong_clip[key] != weak_clip[key]:
                raise ValueError(f'Clip alignment mismatch: {key}')
        if len(strong_clip['predictions']) != len(weak_clip['predictions']):
            raise ValueError('Different decoded candidate frame counts')
        labels, weak_labels = strong_clip['raw_labeled_rows'], weak_clip['raw_labeled_rows']
        if len(labels) != len(weak_labels):
            raise ValueError('Different label counts')
        for row, other in zip(labels, weak_labels):
            key = row['clip'], row['frame']
            if key in seen or any(row[k] != other[k] for k in ('clip', 'frame', 'target_xy', 'width', 'height')):
                raise ValueError('Duplicate or mismatched labels')
            seen.add(key)
        strong = [items[0] if items else None for items in filtered_candidates(strong_clip)]
        weak = filtered_candidates(weak_clip)
        recovered, audit = recover_anchored_gaps(strong, weak, strong_clip['fps'], strong_clip['size'])
        clip_rows = []
        for row in labels:
            index = row['frame']
            value, base = recovered[index], strong[index]
            clip_rows.append({**row, 'prediction_xy': [value.x_px, value.y_px] if value else None})
            baseline_rows.append({**row, 'prediction_xy': [base.x_px, base.y_px] if base else None})
        rows.extend(clip_rows)
        clips.append({'clip': strong_clip['clip'], 'metrics': evaluate_points(clip_rows),
                      'recovered_observations': audit})
    result = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'VALIDATION_SELECTION; OFFLINE_FUTURE_ANCHORS; REAL_CANDIDATES_ONLY',
              'strong_report': str(args.strong_report), 'weak_report': str(args.weak_report), **hashes,
              'settings': {'max_span_seconds': .2, 'max_deviation_reference_px': 6.,
                           'minimum_anchor_confidence': .5, 'minimum_speed_reference_px_s': 20.,
                           'maximum_speed_reference_px_s': 1200., 'stationary_minimum_seconds': .25},
              'baseline_metrics': evaluate_points(baseline_rows), 'metrics': evaluate_points(rows),
              'clips': clips, 'labeled_rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result['metrics'].items() if k != 'localization_errors_reference_px'}))
    print('Recovered real candidate observations:', sum(len(clip['recovered_observations']) for clip in clips))


if __name__ == '__main__':
    main()
