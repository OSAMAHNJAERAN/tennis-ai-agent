"""Compare two model heatmap caches on identical validation labels, without inference."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.evaluate.replay_wasb_thresholds import digest
from src.detection.wasb_ball_detector import WASBBallDetector
from src.evaluation.point_metrics import evaluate_points


def aligned_rows(first, second):
    """Require identical clip order, frame identities, dimensions and ground truth."""
    if len(first['clips']) != len(second['clips']):
        raise ValueError('Different clip counts')
    pairs = []
    for left, right in zip(first['clips'], second['clips']):
        if left['clip'] != right['clip']:
            raise ValueError('Different clip ordering')
        a, b = left['raw_labeled_rows'], right['raw_labeled_rows']
        if len(a) != len(b):
            raise ValueError('Different label counts')
        for row_a, row_b in zip(a, b):
            # Predictions may differ; every field describing the example must agree.
            example_a = {k: v for k, v in row_a.items() if k != 'prediction_xy'}
            example_b = {k: v for k, v in row_b.items() if k != 'prediction_xy'}
            if example_a != example_b:
                raise ValueError('Different frame identity or ground truth')
        pairs.append((left['clip'], a))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--second-weights', type=float, nargs='+', default=[.25, .5, .75])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.1, .15, .2, .25, .35, .5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < n < 1 for n in args.second_weights + args.thresholds):
        raise ValueError('Weights and thresholds must lie in (0,1)')
    reports = [json.loads(path.read_text()) for path in (args.first, args.second)]
    pairs = aligned_rows(*reports)
    spec = importlib.util.spec_from_file_location(
        'wasb_ensemble_geometry', ROOT / 'artifacts/research/WASB-SBDT/src/utils/image.py')
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    decoder = WASBBallDetector.__new__(WASBBallDetector)
    decoder.geometry = geometry
    cache_hashes, inputs = {}, []
    for clip, rows in pairs:
        paths = [Path(report['labeled_heatmap_cache']) / (clip + '.npz') for report in reports]
        for path in paths:
            cache_hashes[str(path)] = digest(path)
        with np.load(paths[0], allow_pickle=False) as first, np.load(paths[1], allow_pickle=False) as second:
            for row in rows:
                a, b = first[str(row['frame'])], second[str(row['frame'])]
                if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Incompatible heatmaps')
                inputs.append((row, a.copy(), b.copy()))
    results = []
    for weight in args.second_weights:
        combined = [(row, (1 - weight) * a + weight * b) for row, a, b in inputs]
        for threshold in args.thresholds:
            decoder.threshold = threshold
            rows = []
            for row, heatmap in combined:
                candidates = decoder.decode_heatmaps([heatmap], (row['height'], row['width']))[0]
                prediction = [candidates[0].x_px, candidates[0].y_px] if candidates else None
                rows.append({**row, 'prediction_xy': prediction})
            metrics = {str(t): evaluate_points(rows, t) for t in (2, 4, 8)}
            results.append({'second_weight': weight, 'threshold': threshold,
                            'metrics': metrics, 'labeled_rows': rows})
            print(json.dumps({'second_weight': weight, 'threshold': threshold,
                              **{k: v for k, v in metrics['4'].items()
                                 if k != 'localization_errors_reference_px'}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'schema_version': '1.0', 'qualification_evidence': False,
        'selection_scope': 'VALIDATION_ONLY_ENSEMBLE_SELECTION; REUSED_SOURCE_LABELS',
        'annotated_frames': len(inputs),
        'timing_scope': 'CPU cached replay; deployed ensemble requires two model inferences',
        'source_reports': {str(path): digest(path) for path in (args.first, args.second)},
        'cache_sha256': cache_hashes, 'script_sha256': digest(__file__), 'results': results,
    }, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
