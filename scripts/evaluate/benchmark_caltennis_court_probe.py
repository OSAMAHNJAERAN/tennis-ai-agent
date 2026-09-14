"""Fixed-frame court checkpoint probe on ground-level recordings; no GT scores."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_caltennis_projection import LINES
from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector


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
            raise ValueError('Dataset changed')
    weights = {'baseline': ROOT / 'models/keypoints_model.pth',
               'geoaug': ROOT / 'artifacts/models/court/geoaug_research/keypoints_model_geoaug.pth'}
    models = {name: CourtKeypointDetector(str(path)) for name, path in weights.items()}
    report = {'complete': False, 'qualification_evidence': False,
              'protocol': 'Both frozen checkpoints on first/middle/last decoded frame of every acquired clip; native video representation; no GT or threshold selection',
              'manifest_sha256': digest(manifest_path), 'checkpoint_hashes': {name: digest(path) for name, path in weights.items()},
              'code_hashes': {str(path): digest(path) for path in [Path(__file__), ROOT / 'src/court/court_keypoint_detector.py',
                               ROOT / 'src/court/calibration.py', ROOT / 'src/court/court_geometry.py']}, 'clips': []}
    args.output.mkdir(parents=True)
    canonical = TennisCourtGeometry.get_canonical_keypoints()
    for clip, sequence in enumerate(manifest['sequences'], 1):
        capture = cv2.VideoCapture(str(args.dataset / sequence['video']))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rows = []
        for index in sorted({0, count // 2, count - 1}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise ValueError('Frame seek failed')
            height, width = frame.shape[:2]
            panels, predictions = [], {}
            for name, model in models.items():
                points = model.predict(frame)
                calibration = calibrate_court(points, canonical)
                predictions[name] = {'keypoints': points.tolist(), 'calibration': calibration.to_dict()}
                image = cv2.resize(frame, (640, round(height * 640 / width)))
                coords = points * (640 / width)
                for a, b in LINES:
                    p, q = tuple(np.rint(coords[a]).astype(int)), tuple(np.rint(coords[b]).astype(int))
                    ok_line, p, q = cv2.clipLine((0, 0, image.shape[1], image.shape[0]), p, q)
                    if ok_line:
                        cv2.line(image, p, q, (0, 0, 255), 1, cv2.LINE_AA)
                image = cv2.copyMakeBorder(image, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                cv2.putText(image, f'clip{clip} frame{index+1} {name} fit={calibration.is_valid} (not GT)',
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(image)
            path = args.output / f'clip{clip}_frame{index+1}_checkpoints.jpg'
            if not cv2.imwrite(str(path), np.hstack(panels)):
                raise IOError('Preview write failed')
            rows.append({'frame_index': index, 'predictions': predictions, 'image': path.name})
        capture.release()
        report['clips'].append({'video': sequence['video'], 'frame_count': count, 'samples': rows})
        print(json.dumps({'clip': clip, 'fit_counts': {name: sum(row['predictions'][name]['calibration']['is_valid'] for row in rows) for name in weights}}), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
