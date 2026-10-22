"""Inspect saved ball evidence without changing results or running a model."""
import json
import math
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/product/analyses/match_1bb12f2aa52a438ebce8a9895fa2bd4f'


def main():
    data = json.loads((RUN / 'detections.json').read_text(encoding='utf-8'))
    result = json.loads((RUN / 'result.json').read_text(encoding='utf-8'))
    points = result['trajectories']['ballTrajectory']
    dimensions = data['metadata']
    jumps = []
    for previous, current in zip(points, points[1:]):
        coordinates = [previous['x_px'], previous['y_px'], current['x_px'], current['y_px']]
        if any(value is None for value in coordinates):
            continue
        displacement = math.dist(coordinates[:2], coordinates[2:])
        if displacement > .2 * math.hypot(dimensions['width'], dimensions['height']):
            jumps.append({'frame': current['frame_index'], 'displacement_px': displacement,
                          'previous': previous, 'current': current})
    target = ROOT / 'artifacts/validation/product_execution/ball_render_original_review'
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv'
    chosen = {0, 30, 300, 309, 310, 311, 312, 330}
    capture = cv2.VideoCapture(str(source))
    assert capture.isOpened()
    saved = []
    for index in range(max(chosen) + 1):
        ok, frame = capture.read()
        assert ok and frame is not None, index
        if index in chosen:
            path = target / f'original_frame_{index:04d}.png'
            assert cv2.imwrite(str(path), frame)
            saved.append({'decoder_frame_index': index, 'source_frame_index': index + 390,
                          'relative_timestamp_seconds': index / 60, 'path': str(path.relative_to(ROOT))})
    capture.release()
    (target / 'frame_manifest.json').write_text(json.dumps({
        'source': str(source.relative_to(ROOT)), 'fps': 60, 'frame_index_basis': 'SEQUENTIAL_DECODE',
        'source_input_sha256': '79b556aa969ca4ee28eb1ddcfe31d4f2a83dd8bd31443f9fb6e4898323c51a21',
        'frames': saved, 'status': 'DEVELOPMENT_DIAGNOSTIC; NOT_MODEL_ACCURACY'}, indent=2), encoding='utf-8')
    report = {'analysis_id': RUN.name, 'frames': len(points),
              'diagnostic_discontinuity_threshold': '.2 * image diagonal; exploratory display inspection only',
              'large_adjacent_image_displacements': jumps,
              'missing_points_with_coordinates': [point['frame_index'] for point in points
                                                  if point['state'] in ('MISSING', 'OCCLUDED')
                                                  and point['x_px'] is not None and point['y_px'] is not None],
              'render_gap_clear_behavior': 'Current Phase6 clears trail when either coordinate is unavailable',
              'status': 'NO_DETECTOR_OR_RESULT_CHANGE; NOT_MODEL_ACCURACY'}
    destination = target.parent / 'ball_render_discontinuities.json'
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(destination), 'exact_original_frames_saved': len(saved),
                      'adjacent_large_displacements': len(jumps),
                      'missing_points_with_coordinates': report['missing_points_with_coordinates']}))


if __name__ == '__main__':
    main()
