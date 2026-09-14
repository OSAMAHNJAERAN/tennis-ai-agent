"""Verify the native-resolution control, then report paired detection changes."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.audit_ball_candidate_coverage import digest


def outcome(row):
    target, prediction = row['target_xy'], row['prediction_xy']
    if target is None:
        return 'correct_absence' if prediction is None else 'absent_false_detection'
    if prediction is None:
        return 'visible_abstention'
    error = math.hypot((target[0] - prediction[0]) * 512 / row['width'],
                       (target[1] - prediction[1]) * 288 / row['height'])
    return 'correct_ball' if error <= 4 else 'wrong_location'


def keyed_rows(rows):
    mapped = {(row['clip'], row['frame']): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError('Duplicate label in resolution comparison')
    return mapped


def compare(native, variant):
    first, second = keyed_rows(native), keyed_rows(variant)
    if first.keys() != second.keys():
        raise ValueError('Resolution runs use different explicit labels')
    changes = []
    for key, row in first.items():
        other = second[key]
        target, scaled_target = row['target_xy'], other['target_xy']
        if (target is None) != (scaled_target is None):
            raise ValueError('Target visibility changed across resolutions')
        if target is not None and any(abs(target[i] / row[dimension] - scaled_target[i] / other[dimension]) > 1e-9
                                      for i, dimension in enumerate(('width', 'height'))):
            raise ValueError('Target coordinates were not consistently scaled')
        before, after = outcome(row), outcome(other)
        if before != after:
            changes.append({'clip': key[0], 'frame': key[1], 'before': before, 'after': after})
    return {'new_correct_visible_frames': sum(change['after'] == 'correct_ball' for change in changes),
            'lost_correct_visible_frames': sum(change['before'] == 'correct_ball' for change in changes),
            'new_correct_absent_frames': sum(change['after'] == 'correct_absence' for change in changes),
            'lost_correct_absent_frames': sum(change['before'] == 'correct_absence' for change in changes),
            'changes': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution-report', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.resolution_report.read_text())
    baseline = json.loads(args.baseline_report.read_text())
    if [result['height'] for result in report['results']] != report['configuration']['heights']:
        raise ValueError('Resolution benchmark is not complete')
    native = next(result for result in report['results'] if result['height'] == 1080)
    native_rows = [row for clip in native['clips'] for row in clip['labeled_rows']]
    existing = next(result for result in baseline['results'] if result['pixel_difference_threshold'] == 12.)
    control = compare(existing['labeled_rows'], native_rows)
    if control['changes']:
        raise ValueError('Native control changed labeled outcome categories')
    first, second = keyed_rows(existing['labeled_rows']), keyed_rows(native_rows)
    for key, row in first.items():
        if row['prediction_xy'] is not None and any(abs(a - b) > .001 for a, b in zip(row['prediction_xy'], second[key]['prediction_xy'])):
            raise ValueError('Native control prediction differs by more than .001 source pixel')
    comparisons = []
    for result in report['results']:
        rows = [row for clip in result['clips'] for row in clip['labeled_rows']]
        comparisons.append({'height': result['height'], 'metrics': result['pooled'],
                            **compare(native_rows, rows)})
    output = {'schema_version': '1.0', 'qualification_evidence': False,
              'resolution_report_sha256': digest(args.resolution_report), 'baseline_report_sha256': digest(args.baseline_report),
              'evaluator_sha256': digest(__file__), 'native_control_matching_labeled_frames': len(native_rows),
              'native_control_source_coordinate_tolerance_px': .001,
              'scope': 'PAIRED_CONTROLLED_RESOLUTION_STRESS; SAME_LABELS_NOT_NEW_INDEPENDENT_EVIDENCE',
              'comparisons': comparisons}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    for value in comparisons:
        print(json.dumps({key: item for key, item in value.items() if key not in ('changes', 'metrics')}
                         | {key: value['metrics'][key] for key in ('precision', 'recall', 'f1')}))


if __name__ == '__main__':
    main()
