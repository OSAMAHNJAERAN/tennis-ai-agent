"""Image-supported line proposals, without landmark identity or court claims."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def line_evidence(image, segment):
    """Contrast on both sides of a bright ridge; support fractions stay explicit."""
    a, b = np.asarray(segment, dtype=float).reshape(2, 2)
    length = np.linalg.norm(b - a)
    if length < 30:
        return None
    tangent = (b - a) / length
    normal = np.array([-tangent[1], tangent[0]])
    # LSD gives edges: sample both candidate ridge centers3px away.
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    base = a + np.linspace(.05, .95, 80)[:, None] * (b - a)
    for shift in [-3, -1.5, 0, 1.5, 3]:
        centers = base + normal * shift
        locations = centers[:, None, :] + normal * np.array([-8, -6, -1, 0, 1, 6, 8])[None, :, None]
        valid = ((locations[:, :, 0] >= 0) & (locations[:, :, 0] < image.shape[1]-1) &
                 (locations[:, :, 1] >= 0) & (locations[:, :, 1] < image.shape[0]-1)).all(axis=1)
        if valid.sum() < 40:
            continue
        locations = locations[valid].astype(np.float32)
        values = cv2.remap(gray, locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        saturation = cv2.remap(hsv[:, :, 1], locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        ridge = values[:, 2:5].max(axis=1)
        sides = np.maximum(values[:, :2].mean(axis=1), values[:, 5:].mean(axis=1))
        side_difference = np.abs(values[:, :2].mean(axis=1) - values[:, 5:].mean(axis=1))
        bright = (ridge - sides > 10) & (ridge > 100)
        white = saturation[:, 2:5].min(axis=1) < 110
        consistent_sides = side_difference < 45
        support = bright & white & consistent_sides
        results.append({'support_fraction': float(support.mean()), 'bright_fraction': float(bright.mean()),
                        'white_fraction': float(white.mean()), 'similar_sides_fraction': float(consistent_sides.mean()),
                        'length': float(length), 'center_shift': shift,
                        'segment': np.r_[a + normal * shift, b + normal * shift].tolist()})
    return max(results, key=lambda row: row['support_fraction']) if results else None


def propose(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    rows = []
    if lines is not None:
        for line in lines[:, 0]:
            row = line_evidence(image, line)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row['length'] * row['support_fraction'], reverse=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {'complete': False, 'qualification_evidence': False,
              'protocol': 'First frame of all3 UVY and all3 CalTennis; width960; LSD length>=30; bright low-saturation ridge with similar adjacent sides; threshold0.5 support fixed before run',
              'scope': 'Line proposals only; white advertising, clothing, fences and nets can pass; no semantic court assignment or homography',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'frames': []}
    for dataset_name in ['uvy_tennis_videos', 'caltennis_diagnostic']:
        dataset = ROOT / 'data/external' / dataset_name
        manifest_path = dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not manifest['complete']:
            raise ValueError('Completed acquisition required')
        files = {row['path']: row for row in manifest['files']}
        sequences = list(manifest['sequences'].values()) if isinstance(manifest['sequences'], dict) else manifest['sequences']
        for index, sequence in enumerate(sequences, 1):
            path = dataset / sequence['video']
            if hashlib.sha256(path.read_bytes()).hexdigest() != files[sequence['video']]['sha256']:
                raise ValueError('Video changed')
            capture = cv2.VideoCapture(str(path))
            ok, image = capture.read()
            capture.release()
            if not ok:
                raise ValueError('First frame unavailable')
            native_size = list(image.shape[:2][::-1])
            image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))
            proposals = propose(image)
            selected = [row for row in proposals if row['support_fraction'] >= .5]
            preview = image.copy()
            for i, row in enumerate(selected):
                x1, y1, x2, y2 = np.rint(row['segment']).astype(int)
                cv2.line(preview, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview, str(i), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
            name = f'{dataset_name}_clip{index}'
            cv2.imwrite(str(args.output / f'{name}.jpg'), preview)
            report['frames'].append({'dataset': dataset_name, 'video': sequence['video'], 'native_size': native_size,
                                     'proposal_size': list(image.shape[:2][::-1]), 'proposals': proposals, 'selected': selected,
                                     'image': name + '.jpg'})
            print(json.dumps({'frame': name, 'candidate_segments': len(proposals), 'supported_segments': len(selected)}), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
