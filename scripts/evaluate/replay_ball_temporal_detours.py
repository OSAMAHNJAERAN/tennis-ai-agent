"""Frozen offline rejection-only pilot on saved, hash-verified ball streams."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.evaluate.compare_ball_resolution import compare
from scripts.evaluate.summarize_ball_validation import summarize


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reject_detours(points, fps, size):
    """Return a simultaneous rejection mask and evidence, never new coordinates."""
    if not math.isfinite(fps) or fps <= 0 or len(size) != 2:
        raise ValueError('Invalid frame geometry or rate')
    if any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Invalid frame geometry')
    scale = np.array([512 / size[0], 288 / size[1]])
    values = np.full((len(points), 2), np.nan)
    for index, point in enumerate(points):
        if point is not None:
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError('Non-finite selected point')
            values[index] = np.asarray(point) * scale
    rejected, intervals = set(), []
    maximum_frames = math.floor(.10 * fps + 1e-9)
    for start in range(2, len(points) - 2):
        for length in range(1, maximum_frames + 1):
            end = start + length
            if end + 1 >= len(points):
                break
            indices = np.array([start - 2, start - 1, end, end + 1])
            anchors = values[indices]
            if not np.isfinite(anchors).all():
                continue
            if min(np.linalg.norm(anchors[1] - anchors[0]),
                   np.linalg.norm(anchors[3] - anchors[2])) < 2:
                continue
            times = (indices - start) / fps
            design = np.column_stack([times, np.ones(4)])
            fit = np.linalg.lstsq(design, anchors, rcond=None)[0]
            error = np.linalg.norm(design @ fit - anchors, axis=1)
            if error.max() > 4:
                continue
            interior_indices = np.arange(start, end)
            available = np.isfinite(values[start:end]).all(axis=1)
            interior_indices = interior_indices[available]
            if not len(interior_indices):
                continue
            interior_design = np.column_stack([(interior_indices - start) / fps,
                                               np.ones(len(interior_indices))])
            residuals = np.linalg.norm(values[interior_indices] - interior_design @ fit, axis=1)
            if residuals.min() < 20:
                continue
            rejected.update(map(int, interior_indices))
            intervals.append({'start': start, 'end_exclusive': end,
                              'anchor_frames': indices.tolist(),
                              'maximum_anchor_error_reference_px': float(error.max()),
                              'minimum_detour_error_reference_px': float(residuals.min()),
                              'rejected_frames': interior_indices.tolist()})
    return rejected, intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    residual = json.loads(args.residual_report.read_text())
    source_path = Path(residual['source_report'])
    if digest(source_path) != residual['source_sha256'] or not residual['complete']:
        raise ValueError('Incomplete or changed source report')
    source = json.loads(source_path.read_text())
    if not source['complete'] or source['dataset_manifest_sha256'] != residual['dataset_manifest_sha256']:
        raise ValueError('Source dataset or completion mismatch')
    if residual['configuration']['local_residual_maximum'] != 12:
        raise ValueError('Expected guarded residual input')
    selected = next(r for r in residual['results'] if r['threshold'] == .98)
    labeled = {(r['clip'], r['frame']): r for r in selected['labeled_rows']}
    if len(labeled) != len(selected['labeled_rows']):
        raise ValueError('Duplicate residual label')
    patch_clips = {c['clip']: c for c in residual['clips']}
    result_clips, rows = [], []
    for clip in source['clips']:
        patch = patch_clips[clip['clip']]
        raw = clip['predictions']['pixel_motion']
        if not (len(raw) == len(patch['scores']) == len(patch['local_residuals']) == clip['frames']):
            raise ValueError('Frame stream length mismatch')
        points = [None if score is not None and score >= .98 and value is not None and value <= 12
                  else point for point, score, value in zip(raw, patch['scores'], patch['local_residuals'])]
        rejected, intervals = reject_detours(points, clip['fps'], clip['size'])
        for original_row in clip['labeled_rows']['pixel_motion']:
            row = labeled[(clip['clip'], original_row['frame'])]
            frame = row['frame']
            if row['prediction_xy'] != points[frame]:
                raise ValueError('Reconstructed stream differs from saved residual scoring')
            rows.append({**row, 'prediction_xy': None if frame in rejected else points[frame]})
        result_clips.append({'clip': clip['clip'], 'frames': len(points), 'fps': clip['fps'],
                             'rejected_frames': sorted(rejected), 'intervals': intervals})
    if len(rows) != len(labeled):
        raise ValueError('Labels omitted')
    result = {'complete': True, 'qualification_evidence': False,
              'scope': 'OFFLINE_REJECTION_ONLY_REUSED_DEVELOPMENT_NO_INTERPOLATION',
              'source_reports': {str(args.residual_report.resolve()): digest(args.residual_report),
                                 str(source_path): digest(source_path)},
              'code_hashes': {name: digest(ROOT / name) for name in (
                  'scripts/evaluate/replay_ball_temporal_detours.py',
                  'scripts/evaluate/compare_ball_resolution.py',
                  'scripts/evaluate/summarize_ball_validation.py', 'src/evaluation/point_metrics.py')},
              'configuration': {'maximum_interval_seconds': .1, 'anchor_count_each_side': 2,
                                'maximum_anchor_error_reference_px': 4,
                                'minimum_anchor_pair_motion_reference_px': 2,
                                'minimum_detour_error_reference_px': 20, 'reference_size': [512, 288]},
              'baseline': summarize(selected['labeled_rows']), **summarize(rows),
              'paired': compare(selected['labeled_rows'], rows), 'labeled_rows': rows,
              'clips': result_clips}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'pooled': {k: v for k, v in result['pooled'].items()
                               if k != 'localization_errors_reference_px'}, 'paired': result['paired']}))


if __name__ == '__main__':
    main()
