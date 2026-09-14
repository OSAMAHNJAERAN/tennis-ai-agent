"""Measure stationary-candidate rejection on saved full-frame candidate sequences."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.replay_wasb_thresholds import digest
from src.evaluation.point_metrics import evaluate_points
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.temporal_ball_tracker import BallObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--minimum-seconds', type=float, nargs='+', default=[.25, .5, 1.])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.report.read_text())
    results = []
    for duration in args.minimum_seconds:
        rows, clip_reports = [], []
        for clip in source['clips']:
            gate = StationaryCandidateFilter(minimum_seconds=duration)
            predictions, rejected_count = [], 0
            for index, candidates in enumerate(clip['predictions']):
                values = [BallObservation(item['x'], item['y'], item['confidence']) for item in candidates]
                kept, rejected = gate.filter(values, index / clip['fps'], clip['fps'], clip['size'])
                predictions.append([kept[0].x_px, kept[0].y_px] if kept else None)
                rejected_count += len(rejected)
            labeled = [{**row, 'prediction_xy': predictions[row['frame']]} for row in clip['raw_labeled_rows']]
            rows.extend(labeled)
            clip_reports.append({'clip': clip['clip'], 'metrics': evaluate_points(labeled),
                                 'rejected_candidates': rejected_count})
        metrics = {str(t): evaluate_points(rows, t) for t in (2, 4, 8)}
        result = {'minimum_seconds': duration, 'radius_reference_px': 1.5, 'window_seconds': 2.,
                  'minimum_presence': .35, 'metrics': metrics, 'clips': clip_reports, 'labeled_rows': rows}
        results.append(result)
        print(json.dumps({k: v for k, v in {'duration': duration, **metrics['4']}.items()
                          if k != 'localization_errors_reference_px'}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'schema_version': '1.0', 'source_report': str(args.report),
                                      'source_sha256': digest(args.report), 'script_sha256': digest(__file__),
                                      'filter_sha256': digest(ROOT / 'src/tracking/stationary_candidate_filter.py'),
                                      'qualification_evidence': False,
                                      'selection_scope': 'VALIDATION_POSTPROCESSING_EXPERIMENT; NOT_INDEPENDENT_TEST',
                                      'risk': 'May suppress a real stationary ball; does not recognize scoreboards',
                                      'results': results}, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
