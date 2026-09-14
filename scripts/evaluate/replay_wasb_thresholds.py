"""CPU-only threshold analysis using explicitly labeled validation heatmaps."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.detection.wasb_ball_detector import WASBBallDetector
from src.evaluation.point_metrics import evaluate_points


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.05, .1, .15, .2, .25, .35, .5])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(not 0 < value < 1 for value in args.thresholds):
        raise ValueError('Thresholds must lie in (0,1)')
    original = json.loads(args.report.read_text())
    if not original.get('labeled_heatmap_cache'):
        raise ValueError('Report contains no labeled heatmap cache')
    source = ROOT / 'artifacts/research/WASB-SBDT/src/utils/image.py'
    spec = importlib.util.spec_from_file_location('wasb_geometry_replay', source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    decoder = WASBBallDetector.__new__(WASBBallDetector)
    decoder.geometry = geometry
    results = []
    caches = {}
    for threshold in args.thresholds:
        decoder.threshold = threshold
        rows = []
        for clip in original['clips']:
            path = Path(original['labeled_heatmap_cache']) / (clip['clip'] + '.npz')
            caches[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as heatmaps:
                for row in clip['raw_labeled_rows']:
                    heatmap = heatmaps[str(row['frame'])]
                    candidates = decoder.decode_heatmaps([heatmap], (row['height'], row['width']))[0]
                    prediction = [candidates[0].x_px, candidates[0].y_px] if candidates else None
                    rows.append({**row, 'prediction_xy': prediction})
        metrics = {str(t): evaluate_points(rows, t) for t in (2, 4, 8)}
        results.append({'threshold': threshold, 'metrics': metrics, 'labeled_rows': rows})
        print(json.dumps({'threshold': threshold, **{k: v for k, v in metrics['4'].items()
                                                   if k != 'localization_errors_reference_px'}}), flush=True)
    output = {'schema_version': '1.0', 'source_report': str(args.report), 'source_sha256': digest(args.report),
              'cache_sha256': caches, 'qualification_evidence': False,
              'selection_scope': 'VALIDATION_ONLY_THRESHOLD_ANALYSIS; REUSED_SOURCE_LABELS',
              'results': results}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
