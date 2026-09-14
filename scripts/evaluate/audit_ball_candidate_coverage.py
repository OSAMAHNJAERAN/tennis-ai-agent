"""Ground-truth-assisted candidate coverage, never a deployable tracking score.

Separates missing detections from ranking mistakes on explicit sparse labels.
The oracle uses labels only in this offline diagnostic. It cannot infer absence.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.point_metrics import evaluate_points
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.temporal_ball_tracker import BallObservation


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def analyze_report(source, minimum_seconds=.25, tolerance_px=4.):
    rows, raw_rows, peak_rows, details = [], [], [], []
    counts, seen = Counter(), set()
    for clip in source['clips']:
        gate = StationaryCandidateFilter(minimum_seconds=minimum_seconds)
        filtered = []
        for index, values in enumerate(clip['predictions']):
            observations = [BallObservation(item['x'], item['y'], item['confidence']) for item in values]
            kept, _ = gate.filter(observations, index / clip['fps'], clip['fps'], clip['size'])
            filtered.append(kept)
        for row in clip['raw_labeled_rows']:
            key = clip['clip'], row['frame']
            if key in seen or row['clip'] != clip['clip'] or not 0 <= row['frame'] < len(filtered):
                raise ValueError('Duplicate, mismatched or out-of-range labeled frame')
            seen.add(key)
            if [row['width'], row['height']] != list(clip['size']):
                raise ValueError('Label and candidate coordinates use different frame sizes')
            candidates = clip['predictions'][row['frame']]
            kept = filtered[row['frame']]
            raw = [[item['x'], item['y']] for item in candidates]
            points = [[item.x_px, item.y_px] for item in kept]
            raw_rows.append({**row, 'prediction_xy': raw[0] if raw else None})
            rows.append({**row, 'prediction_xy': points[0] if points else None})
            peak = max(kept, key=lambda item: item.confidence, default=None)
            peak_rows.append({**row, 'prediction_xy': [peak.x_px, peak.y_px] if peak else None})
            target = row['target_xy']
            if target is None:
                counts['absent_frames'] += 1
                counts['absent_with_raw_candidates'] += bool(raw)
                counts['absent_with_filtered_candidates'] += bool(points)
                continue
            def error(point):
                return math.hypot((point[0] - target[0]) * 512 / row['width'],
                                  (point[1] - target[1]) * 288 / row['height'])
            raw_ranks = [rank for rank, point in enumerate(raw, 1) if error(point) <= tolerance_px]
            ranks = [rank for rank, point in enumerate(points, 1) if error(point) <= tolerance_px]
            counts['visible_frames'] += 1
            counts['raw_candidate_hit_frames'] += bool(raw_ranks)
            counts['filtered_candidate_hit_frames'] += bool(ranks)
            category = ('correct_top1' if 1 in ranks else 'correct_lower_rank' if ranks else
                        'correct_candidate_removed_by_filter' if raw_ranks else 'no_correct_raw_candidate')
            counts[category] += 1
            details.append({'clip': row['clip'], 'frame': row['frame'], 'category': category,
                            'raw_correct_ranks': raw_ranks, 'filtered_correct_ranks': ranks,
                            'nearest_raw_error_reference_px': min(map(error, raw), default=None)})
    visible = counts['visible_frames']
    return {'counts': dict(counts),
            'raw_candidate_oracle_recall_ceiling': counts['raw_candidate_hit_frames'] / visible if visible else None,
            'filtered_candidate_oracle_recall_ceiling': counts['filtered_candidate_hit_frames'] / visible if visible else None,
            'raw_top1_metrics': evaluate_points(raw_rows, tolerance_px),
            'filtered_mass_rank_metrics': evaluate_points(rows, tolerance_px),
            'filtered_peak_confidence_rank_metrics': evaluate_points(peak_rows, tolerance_px),
            'visible_frame_diagnostics': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--minimum-seconds', type=float, default=.25)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes = {name: digest(path) for name, path in {
        'source_report_sha256': args.report, 'evaluator_sha256': __file__,
        'filter_sha256': ROOT / 'src/tracking/stationary_candidate_filter.py',
        'point_metric_sha256': ROOT / 'src/evaluation/point_metrics.py'}.items()}
    source = json.loads(args.report.read_text())
    report = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'LABEL_ASSISTED_DIAGNOSTIC; ORACLE_IS_NOT_DEPLOYABLE_ACCURACY',
              'absence_scope': 'Oracle recall does not measure absence rejection or precision',
              'source_report': str(args.report), **hashes,
              'source_threshold': source.get('wasb_heatmap_threshold'),
              'source_temporal_step': source.get('wasb_temporal_step'),
              'minimum_stationary_seconds': args.minimum_seconds, 'tolerance_reference_px': 4.,
              **analyze_report(source, args.minimum_seconds)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('counts', 'raw_candidate_oracle_recall_ceiling',
                                                   'filtered_candidate_oracle_recall_ceiling')}))


if __name__ == '__main__':
    main()
