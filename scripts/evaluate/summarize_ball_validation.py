"""Expose per-clip and absence failures that aggregate ball F1 can conceal."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.replay_wasb_thresholds import digest
from src.evaluation.point_metrics import evaluate_points


def summarize(rows):
    grouped, seen = defaultdict(list), set()
    for row in rows:
        key = row['clip'], row['frame']
        if key in seen:
            raise ValueError('Duplicate labeled clip/frame would inflate the evidence')
        seen.add(key)
        grouped[row['clip']].append(row)
    per_clip = {clip: evaluate_points(values) for clip, values in grouped.items()}
    macro = {}
    for key in ('precision', 'recall', 'f1', 'absent_specificity'):
        values = [metrics[key] for metrics in per_clip.values() if metrics[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
    return {'pooled': evaluate_points(rows), 'per_clip': per_clip, 'macro_available_clip_mean': macro,
            'clip_count': len(grouped),
            'clips_passing_both_95_percent': sum(m['precision'] is not None and m['recall'] is not None
                                               and m['precision'] > .95 and m['recall'] > .95
                                               for m in per_clip.values()),
            'independence_scope': 'PUBLISHER_CLIP_IDS; SOURCE_BROADCAST_GROUPING_UNVERIFIED'}


def operating_point(report, stationary_seconds=None, pixel_threshold=None):
    """Preserve source model settings when combining derived replay reports."""
    seen, feature_settings, feature_code = set(), None, None
    while 'source_report' in report:
        if report.get('feature_scope', '').startswith('CAUSAL_PIXEL_DIFFERENCE'):
            feature_settings = report['configuration']
            feature_code = report['feature_code_sha256']
        path = Path(report['source_report'])
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError('Cycle in replay provenance')
        seen.add(resolved)
        if digest(path) != report['source_sha256']:
            raise ValueError('Underlying prediction report changed after replay')
        report = json.loads(path.read_text())
    return {key: report.get(key) for key in ('backend', 'checkpoint_sha256', 'ensemble_checkpoint_sha256',
                                            'ensemble_second_weight', 'wasb_heatmap_threshold',
                                            'wasb_temporal_step')} | {
        'stationary_minimum_seconds': stationary_seconds, 'pixel_difference_threshold': pixel_threshold,
        'pixel_feature_configuration': feature_settings, 'pixel_feature_code_sha256': feature_code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--minimum-seconds', type=float, default=.25)
    parser.add_argument('--pixel-threshold', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, operating_points = [], []
    for path in args.reports:
        report = json.loads(path.read_text())
        if 'clips' in report:
            rows.extend(row for clip in report['clips'] for row in clip['raw_labeled_rows'])
            operating_points.append(operating_point(report))
        else:
            field, value = ('pixel_difference_threshold', args.pixel_threshold) if args.pixel_threshold is not None else ('minimum_seconds', args.minimum_seconds)
            selected = [item for item in report['results'] if item.get(field) == value]
            if len(selected) != 1:
                raise ValueError('Exactly one requested filter operating point is required')
            rows.extend(selected[0]['labeled_rows'])
            operating_points.append(operating_point(report, args.minimum_seconds, args.pixel_threshold))
    result = {'schema_version': '1.0', 'qualification_evidence': False,
              'source_reports': {str(path): digest(path) for path in args.reports},
              'source_operating_points': operating_points,
              'same_operating_point_across_sources': all(point == operating_points[0] for point in operating_points),
              'pooled_scope': 'DIAGNOSTIC_ONLY_IF_OPERATING_POINTS_OR_FEATURE_CODE_DIFFER; DO_NOT_CLAIM_ONE_FIXED_IMPLEMENTATION',
              'evaluator_sha256': digest(__file__),
              'metric_sha256': digest(ROOT / 'src/evaluation/point_metrics.py'), **summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'per_clip'
                      and key != 'pooled'}), flush=True)
    print(json.dumps({key: value for key, value in result['pooled'].items()
                      if key != 'localization_errors_reference_px'}), flush=True)


if __name__ == '__main__':
    main()
