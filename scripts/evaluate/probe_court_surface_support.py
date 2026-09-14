"""Color/smoothness surface proposal paired with saved line evidence; diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def surface_candidates(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (9, 9))
    variance = np.maximum(0, cv2.boxFilter(gray * gray, -1, (9, 9)) - mean * mean)
    base = (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 50) & (variance <= 400)
    candidates = []
    for hue in range(0, 180, 15):
        distance = np.abs(hsv[:, :, 0].astype(float) - hue)
        mask = (base & (np.minimum(distance, 180 - distance) <= 15)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < image.shape[0] * image.shape[1] * .025 or centers[label][1] < image.shape[0] * .3:
                continue
            component = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(np.vstack(contours))
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            if solidity < .6:
                continue
            candidates.append({'hue_center': hue, 'area': int(area), 'solidity': float(solidity),
                               'centroid': centers[label].tolist(), 'hull': hull[:, 0].tolist(), 'mask': component})
    candidates.sort(key=lambda row: row['area'], reverse=True)
    return candidates


def supported_fraction(mask, segment):
    a, b = np.asarray(segment, dtype=float).reshape(2, 2)
    direction = b - a
    length = np.linalg.norm(direction)
    if length < 1:
        return 0.
    normal = np.array([-direction[1], direction[0]]) / length
    # Both near-side samples must be on the proposed surface; no hull filling.
    samples = a + np.linspace(.05, .95, 80)[:, None] * direction
    locations = samples[:, None, :] + normal * np.array([-5, 5])[None, :, None]
    values = cv2.remap(mask, locations[:, :, 0].astype(np.float32), locations[:, :, 1].astype(np.float32), cv2.INTER_NEAREST)
    return float((values.min(axis=1) > 0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lines', type=Path, default=ROOT / 'outputs/vision_upgrade_audit/court_line_segments/report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.lines.read_text(encoding='utf-8'))
    if not original['complete']:
        raise ValueError('Completed line probe required')
    report = {'complete': False, 'qualification_evidence': False, 'line_report_sha256': hashlib.sha256(args.lines.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Largest connected chromatic smooth region, not semantic court segmentation; grayscale courts/occlusion/distractor surfaces may fail',
              'settings': 'HSV saturation>=35,value>=50,local9x9 gray variance<=400,hue radius15 centers every15; closing15;area>=2.5%,centroid y>=30%,solidity>=0.6; selected line both-side surface fraction>=0.5',
              'frames': []}
    args.output.mkdir(parents=True)
    for row in original['frames']:
        video = ROOT / 'data/external' / row['dataset'] / row['video']
        capture = cv2.VideoCapture(str(video))
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise ValueError('Image unavailable')
        image = cv2.resize(image, tuple(row['proposal_size']))
        candidates = surface_candidates(image)
        selected = []
        preview = image.copy()
        if candidates:
            mask = candidates[0]['mask']
            preview[mask > 0] = (preview[mask > 0].astype(float) * .7 + np.array([50, 200, 50]) * .3).astype(np.uint8)
            for line in row['selected']:
                support = supported_fraction(mask, line['segment'])
                if support >= .5:
                    selected.append({**line, 'surface_support_fraction': support})
            for i, line in enumerate(selected):
                x1, y1, x2, y2 = np.rint(line['segment']).astype(int)
                cv2.line(preview, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview, str(i), ((x1+x2)//2, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
        name = row['image']
        cv2.imwrite(str(args.output / name), preview)
        report['frames'].append({'dataset': row['dataset'], 'video': row['video'], 'image': name,
                                 'candidates': [{k: v for k, v in candidate.items() if k != 'mask'} for candidate in candidates],
                                 'selected': selected, 'line_count_before': len(row['selected'])})
        print(json.dumps({'image': name, 'surface_candidates': len(candidates), 'lines_before': len(row['selected']), 'lines_after': len(selected)}), flush=True)
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
