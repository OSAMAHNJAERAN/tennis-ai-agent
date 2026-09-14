"""Render explicit court-coordinate hypotheses; never declare calibration GT."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.court.court_geometry import TennisCourtGeometry

LINES = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/caltennis_diagnostic')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise ValueError('Complete acquisition required')
    for row in manifest['files']:
        if digest(args.dataset / row['path']) != row['sha256']:
            raise ValueError('Dataset file changed')
    args.output.mkdir(parents=True)
    court = TennisCourtGeometry.get_canonical_keypoints().astype(float)
    # Explicit hypothesis: publisher X runs along length; Y along width; Z up;
    # origin at doubles baseline corner. Verify with publisher before GT use.
    world = np.column_stack((court[:, 1], court[:, 0], np.zeros(len(court))))
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'HYPOTHESIS_ONLY: corner origin; X length, Y width, Z up; image transform unresolved',
              'manifest_sha256': digest(manifest_path), 'script_sha256': digest(__file__),
              'court_geometry_sha256': digest(ROOT / 'src/court/court_geometry.py'),
              'world_points': world.tolist(), 'sequences': []}
    for index, sequence in enumerate(manifest['sequences']):
        calibration = json.loads((args.dataset / sequence['calibration']).read_text(encoding='utf-8'))
        K, R, t = [np.asarray(calibration[k], dtype=float) for k in ('K', 'R_w2c', 't_w2c')]
        if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or not all(np.isfinite(a).all() for a in (K, R, t)):
            raise ValueError('Invalid calibration matrices')
        if calibration['camera_model'] != 'pinhole' or calibration['dist_coeffs'] is not None:
            raise ValueError('Diagnostic only supports supplied undistorted pinhole model')
        camera = world @ R.T + t
        if np.any(camera[:, 2] <= 0):
            raise ValueError('Hypothesis projects court behind camera; inspect before rendering')
        homogeneous = camera @ K.T
        points = homogeneous[:, :2] / homogeneous[:, 2:]
        # Cross-check matrix convention against OpenCV's independent projection.
        projected, _ = cv2.projectPoints(world, cv2.Rodrigues(R)[0], t, K, None)
        np.testing.assert_allclose(points, projected.reshape(-1, 2), atol=1e-8)
        capture = cv2.VideoCapture(str(args.dataset / sequence['video']))
        ok, frame = capture.read()
        capture.release()
        if not ok:
            raise ValueError('Cannot decode initial image')
        height, width = frame.shape[:2]
        calibration_size = [calibration['image_shape']['width'], calibration['image_shape']['height']]
        variants = {'identity_pixels': points,
                    'anisotropic_resize_hypothesis': points * [width / calibration_size[0], height / calibration_size[1]]}
        image = cv2.resize(frame, (960, round(height * 960 / width)))
        for name, color in [('identity_pixels', (255, 255, 0)), ('anisotropic_resize_hypothesis', (255, 0, 255))]:
            coords = variants[name] * (960 / width)
            for a, b in LINES:
                success, start, end = cv2.clipLine((0, 0, image.shape[1], image.shape[0]),
                                                  tuple(np.rint(coords[a]).astype(int)), tuple(np.rint(coords[b]).astype(int)))
                if success:
                    cv2.line(image, start, end, color, 1, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (960, 45), (20, 20, 20), -1)
        cv2.putText(image, f'clip{index+1}: HYPOTHESIS - corner origin, X length, Y width', (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 'cyan: unchanged calibration pixels / magenta: scale height1080 to1088', (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        image_path = args.output / f'clip{index+1}_projection_hypotheses.jpg'
        if not cv2.imwrite(str(image_path), image):
            raise IOError('Cannot save diagnostic')
        report['sequences'].append({'video': sequence['video'], 'video_size': [width, height],
                                    'calibration_size': calibration_size, 'positive_camera_depth': True,
                                    'image_hypotheses': {name: coords.tolist() for name, coords in variants.items()},
                                    'image': image_path.name})
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
