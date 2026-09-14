"""Fixed-grid appearance-change rejection on explicit sparse validation labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.evaluation.point_metrics import evaluate_points
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.temporal_ball_tracker import BallObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[2., 4., 8., 12.])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < value <= 255 for value in args.thresholds):
        raise ValueError('Pixel threshold must lie in (0,255]')
    source = json.loads(args.report.read_text())
    results = []
    for threshold in [0., *args.thresholds]:
        rows, clip_reports = [], []
        for clip in source['clips']:
            gate = StationaryCandidateFilter(minimum_seconds=.25)
            selected, rejected = [], 0
            for index, batch in enumerate(clip['predictions']):
                candidates = [BallObservation(item['x'], item['y'], item['confidence']) for item in batch]
                kept, _ = gate.filter(candidates, index / clip['fps'], clip['fps'], clip['size'])
                allowed = {id(candidate) for candidate in kept}
                final = []
                for item, candidate in zip(batch, candidates):
                    if id(candidate) not in allowed:
                        continue
                    score = item['pixel_motion_score']
                    if score is None or score >= threshold:
                        final.append(candidate)
                    else:
                        rejected += 1
                selected.append([final[0].x_px, final[0].y_px] if final else None)
            labeled = [{**row, 'prediction_xy': selected[row['frame']]} for row in clip['raw_labeled_rows']]
            rows.extend(labeled)
            clip_reports.append({'clip': clip['clip'], 'metrics': evaluate_points(labeled),
                                 'appearance_rejected_candidates': rejected})
        metrics = evaluate_points(rows)
        results.append({'pixel_difference_threshold': threshold, 'metrics': metrics, 'clips': clip_reports,
                        'labeled_rows': rows})
        print(json.dumps({'threshold': threshold, **{key: value for key, value in metrics.items()
                                                    if key != 'localization_errors_reference_px'}}), flush=True)
    report = {'schema_version': '1.0', 'qualification_evidence': False,
              'selection_scope': 'FIXED_GRID_REUSED_VALIDATION; NOT_INDEPENDENT_TEST',
              'source_report': str(args.report), 'source_sha256': digest(args.report),
              'evaluator_sha256': digest(__file__),
              'filter_sha256': digest(ROOT / 'src/tracking/stationary_candidate_filter.py'),
              'metric_sha256': digest(ROOT / 'src/evaluation/point_metrics.py'),
              'limitations': 'Can reject a real stationary ball; camera/lighting/compression changes can pass false candidates',
              'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
