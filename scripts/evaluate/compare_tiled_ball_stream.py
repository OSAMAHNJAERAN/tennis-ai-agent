"""Check continuous/sparse agreement and paired tiled-filter outcomes."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.compare_ball_resolution import compare, keyed_rows


def verify_sparse_control(sparse_rows, continuous_rows, tolerance=.001):
    changes = compare(sparse_rows, continuous_rows)
    if changes['changes']:
        raise ValueError('Sparse and continuous raw outcomes differ')
    first, second = keyed_rows(sparse_rows), keyed_rows(continuous_rows)
    maximum = 0.
    for key, row in first.items():
        other = second[key]
        if (row['width'], row['height']) != (other['width'], other['height']):
            raise ValueError('Control source dimensions differ')
        if row['prediction_xy'] is not None:
            maximum = max(maximum, *(abs(a - b) for a, b in zip(row['prediction_xy'], other['prediction_xy'])))
    if maximum > tolerance:
        raise ValueError('Sparse and continuous raw coordinates differ')
    return {'matching_labels': len(first), 'maximum_coordinate_difference_px': maximum,
            'coordinate_tolerance_px': tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--sparse', type=Path, required=True)
    parser.add_argument('--resolution-baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    stream, sparse, baseline = [json.loads(path.read_text()) for path in
                                (args.stream, args.sparse, args.resolution_baseline)]
    if not stream['complete']:
        raise ValueError('Continuous benchmark is incomplete')
    for report in (sparse, baseline):
        if any(report[key] != stream[key] for key in ('dataset_manifest_sha256', 'checkpoint_sha256')):
            raise ValueError('Dataset or checkpoint differs across reports')
    raw = [row for clip in stream['clips'] for row in clip['labeled_rows']['raw']]
    control = verify_sparse_control(sparse['selected_labeled_rows']['combined_peak_top1'], raw)
    native = next(result for result in baseline['results'] if result['height'] == 1080)
    reference = [row for clip in native['clips'] for row in clip['labeled_rows']]
    variants = {name: [row for clip in stream['clips'] for row in clip['labeled_rows'][name]]
                for name in ('raw', 'stationary', 'pixel_motion')}
    frames = sum(clip['frames'] for clip in stream['clips'])
    result = {'schema_version': '1.0', 'qualification_evidence': False,
              'source_reports': {str(path): digest(path) for path in (args.stream, args.sparse, args.resolution_baseline)},
              'evaluator_sha256': digest(__file__), 'raw_sparse_control': control,
              'decoded_frames': frames, 'ball_only_frames_per_second': frames / stream['seconds'],
              'versus_filtered_full_frame': {name: compare(reference, rows) for name, rows in variants.items()},
              'filter_effect_versus_raw_tiles': {name: compare(raw, rows) for name, rows in variants.items() if name != 'raw'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('versus_filtered_full_frame', 'filter_effect_versus_raw_tiles')}))
    for name, value in result['versus_filtered_full_frame'].items():
        print(json.dumps({'variant': name, **{key: item for key, item in value.items() if key != 'changes'}}))


if __name__ == '__main__':
    main()
