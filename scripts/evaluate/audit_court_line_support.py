"""Compare frozen court predictions to independently extracted line segments."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_caltennis_projection import LINES
from scripts.evaluate.probe_court_line_segments import propose


def support(points, segments, width, height):
    """Fraction of visible interior samples near a detected bright line, per line."""
    rows = []
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    for a, b in LINES:
        if points[a] is None or points[b] is None:
            rows.append({'supported_fraction': 0., 'visible_samples': 0, 'reason': 'MISSING_ENDPOINT'})
            continue
        a, b = np.asarray(points[a]), np.asarray(points[b])
        positions = a + np.linspace(.05, .95, 100)[:, None] * (b - a)
        visible = (positions[:, 0] >= 0) & (positions[:, 0] < width) & (positions[:, 1] >= 0) & (positions[:, 1] < height)
        positions = positions[visible]
        if len(positions) < 10 or not len(segments):
            rows.append({'supported_fraction': 0., 'visible_samples': len(positions), 'reason': 'NO_COMPARABLE_SEGMENTS'})
            continue
        starts, vectors = segments[:, 0], segments[:, 1] - segments[:, 0]
        parameters = ((positions[:, None] - starts) * vectors).sum(axis=2) / np.maximum((vectors * vectors).sum(axis=1), 1e-8)
        nearest = starts + np.clip(parameters, 0, 1)[:, :, None] * vectors
        distances = np.linalg.norm(positions[:, None] - nearest, axis=2).min(axis=1)
        rows.append({'supported_fraction': float((distances <= 3).mean()), 'visible_samples': len(positions)})
    considered = [row for row in rows if row['visible_samples'] >= 10]
    value = float(np.mean([row['supported_fraction'] for row in considered])) if considered else 0.
    return {'lines': rows, 'mean_visible_line_support': value, 'considered_lines': len(considered),
            'diagnostic_gate_pass': len(considered) >= 6 and value >= .5}


def main():
    root = ROOT / 'outputs/vision_upgrade_audit'
    output = root / 'court_line_support_comparison.json'
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for name in ['uvy_player_tracking_baseline', 'uvy_player_tracking_geoaug']:
        source = json.loads((root / name / 'report.json').read_text())
        manifest = json.loads((ROOT / 'data/external/uvy_tennis_videos/manifest.json').read_text())
        for sequence, row in source['sequences'].items():
            cases.append((name, sequence, ROOT / 'data/external/uvy_tennis_videos' / manifest['sequences'][sequence]['video'], row['keypoints']))
    source = json.loads((root / 'caltennis_court_probe/report.json').read_text())
    for clip, row in enumerate(source['clips'], 1):
        for name, prediction in row['samples'][0]['predictions'].items():
            cases.append(('caltennis_' + name, str(clip), ROOT / 'data/external/caltennis_diagnostic' / row['video'], prediction['keypoints']))
    source = json.loads((root / 'court_heatmap_broadcast_control_verified/report.json').read_text())
    for row in source['frames']:
        cases.append(('broadcast_heatmap_control', row['video'], ROOT / 'data/external/racketvision_validation' / row['video'], row['points']))
    cache, rows = {}, []
    for model, identity, video, points in cases:
        if str(video) not in cache:
            capture = cv2.VideoCapture(str(video))
            ok, image = capture.read()
            capture.release()
            if not ok:
                raise ValueError('Frame unavailable')
            scale = 960 / image.shape[1]
            image = cv2.resize(image, (960, round(image.shape[0] * scale)))
            segments = [row['segment'] for row in propose(image) if row['support_fraction'] >= .5]
            cache[str(video)] = (scale, image.shape[0], segments)
        scale, height, segments = cache[str(video)]
        mapped = [None if point is None else np.asarray(point) * scale for point in points]
        result = support(mapped, segments, 960, height)
        rows.append({'model': model, 'video': str(video), 'identity': identity, **result})
        print(json.dumps({'model': model, 'identity': identity, 'support': result['mean_visible_line_support'], 'pass': result['diagnostic_gate_pass']}), flush=True)
    output.write_text(json.dumps({'complete': True, 'qualification_evidence': False,
                                 'protocol': 'Fixed distance3 atwidth960, mean across visible lines>=0.5 with>=6 lines; no threshold tuning; original line proposals without surface filter',
                                 'scope': 'Diagnostic image evidence only; mismatched lines and off-plane edges may still pass; not an independently labeled accuracy test', 'cases': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
