"""Frozen 0/90/180/270-degree court-model probe; not independent accuracy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector

LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def undo_rotation(points, turns, width, height):
    x, y = np.asarray(points, dtype=float).T
    if turns == 0:
        return np.column_stack((x, y))
    if turns == 1:
        return np.column_stack((width - 1 - y, x))
    if turns == 2:
        return np.column_stack((width - 1 - x, height - 1 - y))
    if turns == 3:
        return np.column_stack((y, height - 1 - x))
    raise ValueError('Quarter turns must be 0..3')


def line_contrast(image, points):
    """Sample predicted lines against nearby parallel strips; never ground truth."""
    height, width = image.shape[:2]
    scale = 960 / width
    gray = cv2.cvtColor(cv2.resize(image, (960, round(height * scale))), cv2.COLOR_BGR2GRAY)
    points = np.asarray(points, dtype=float) * scale
    rows = []
    for first, second in LINES:
        a, b = points[first], points[second]
        distance = np.linalg.norm(b - a)
        if distance < 10:
            rows.append({'supported': False, 'reason': 'SHORT_SEGMENT'})
            continue
        normal = np.array([-(b - a)[1], (b - a)[0]]) / distance
        samples = a[None, :] + np.linspace(.05, .95, 80)[:, None] * (b - a)[None, :]
        locations = samples[:, None, :] + np.array([-6, -5, -1, 0, 1, 5, 6])[None, :, None] * normal
        valid = (locations[:, :, 0] >= 0).all(axis=1) & (locations[:, :, 0] < gray.shape[1] - 1).all(axis=1) & \
                (locations[:, :, 1] >= 0).all(axis=1) & (locations[:, :, 1] < gray.shape[0] - 1).all(axis=1)
        if valid.sum() < 40:
            rows.append({'supported': False, 'reason': 'OUT_OF_FRAME'})
            continue
        loc = locations[valid].astype(np.float32)
        values = cv2.remap(gray, loc[:, :, 0], loc[:, :, 1], cv2.INTER_LINEAR).astype(float)
        center = values[:, 2:5].max(axis=1)
        background = np.median(values[:, [0, 1, 5, 6]], axis=1)
        contrast = center - background
        rows.append({'supported': True, 'mean_contrast': float(contrast.mean()),
                     'fraction_contrast_over_15': float((contrast > 15).mean())})
    return {'lines': rows,
            'mean_supported_fraction': float(np.mean([row.get('fraction_contrast_over_15', 0.) for row in rows])),
            'scope': 'IMAGE_LINE_CONTRAST_DIAGNOSTIC; CROWD_EDGES_CAN_SCORE; NOT_COURT_VALIDITY_OR_GT'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/court/geoaug_research/keypoints_model_geoaug.pth'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Verify coordinate inversion independently using known marked pixels.
    marker = np.zeros((7, 11), np.uint8)
    marker[2, 8] = 1
    for turns in range(4):
        y, x = np.argwhere(np.rot90(marker, turns) == 1)[0]
        np.testing.assert_array_equal(undo_rotation([[x, y]], turns, 11, 7), [[8, 2]])
    source = Path('data/external/uvy_tennis_videos')
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Complete video acquisition required')
    args.output.mkdir(parents=True)
    detector = CourtKeypointDetector(str(args.checkpoint))
    report = {'complete': False, 'qualification_evidence': False,
              'protocol': 'All four quarter rotations on every first UVY frame; no threshold tuning or label-driven view selection',
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sequences': {}}
    for sequence, info in manifest['sequences'].items():
        video = source / info['video']
        expected = next(item['sha256'] for item in manifest['files'] if item['path'] == info['video'])
        if hashlib.sha256(video.read_bytes()).hexdigest() != expected:
            raise ValueError('Source video changed')
        cap = cv2.VideoCapture(str(video))
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError('Source frame cannot decode')
        height, width = image.shape[:2]
        panels, variants = [], []
        for turns in range(4):
            rotated = np.ascontiguousarray(np.rot90(image, turns))
            points = undo_rotation(detector.predict(rotated), turns, width, height)
            calibration = calibrate_court(points, TennisCourtGeometry.get_canonical_keypoints())
            contrast = line_contrast(image, points)
            variants.append({'quarter_turns': turns, 'keypoints': points.tolist(),
                             'calibration': calibration.to_dict(), 'line_contrast': contrast})
            canvas = cv2.resize(image, (640, 360))
            scaled = points * np.array([640 / width, 360 / height])
            for a, b in LINES:
                cv2.line(canvas, tuple(np.rint(scaled[a]).astype(int)), tuple(np.rint(scaled[b]).astype(int)), (0, 0, 255), 1)
            header = np.full((32, 640, 3), 24, np.uint8)
            cv2.putText(header, f'{turns * 90}deg input / fit={calibration.is_valid} / contrast={contrast["mean_supported_fraction"]:.3f}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
            panels.append(np.vstack((header, canvas)))
        review = args.output / f'{sequence}_rotations.jpg'
        if not cv2.imwrite(str(review), np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))):
            raise RuntimeError('Orientation review write failed')
        report['sequences'][sequence] = {'source_sha256': expected, 'variants': variants, 'review': str(review)}
        print(json.dumps({'sequence': sequence, 'variants': [{'turns': item['quarter_turns'],
                           'fit_valid': item['calibration']['is_valid'],
                           'line_contrast': item['line_contrast']['mean_supported_fraction']} for item in variants]}), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
