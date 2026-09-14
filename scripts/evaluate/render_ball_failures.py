"""Render explicit validation misses and false detections from cached predictions."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.2)
    parser.add_argument('--clip', help='Optionally inspect only one clip from a video benchmark')
    parser.add_argument('--dataset', type=Path, default=Path('data/external/racketvision_validation'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    rows = ([row for clip in report['clips'] for row in clip['raw_labeled_rows']]
            if 'clips' in report else
            next(item['labeled_rows'] for item in report['results'] if item['threshold'] == args.threshold))
    if args.clip:
        rows = [row for row in rows if row['clip'] == args.clip]
        if not rows:
            raise ValueError('Requested clip has no scored labels')
    failures = []
    for row in rows:
        target, prediction = row['target_xy'], row['prediction_xy']
        if target is None and prediction is not None:
            failures.append({**row, 'failure': 'FALSE_VISIBLE', 'error_reference_px': None})
        elif target is not None and prediction is None:
            failures.append({**row, 'failure': 'MISSED_VISIBLE', 'error_reference_px': None})
        elif target is not None:
            error = math.hypot((target[0] - prediction[0]) * 512 / row['width'],
                               (target[1] - prediction[1]) * 288 / row['height'])
            if error > 4:
                failures.append({**row, 'failure': 'WRONG_LOCATION', 'error_reference_px': error})
    args.output.mkdir(parents=True)
    (args.output / 'failures.json').write_text(json.dumps(failures, indent=2), encoding='utf-8')
    canvas = np.full((((len(failures) + 3) // 4) * 235, 4 * 360, 3), 24, dtype=np.uint8)
    for index, row in enumerate(failures):
        with VideoFrameSequence(str(args.dataset / ('tennis/videos/' + row['clip'] + '.mp4'))) as frames:
            frame = frames[row['frame']].copy()
        points = [point for point in (row['target_xy'], row['prediction_xy']) if point is not None]
        # A shared crop includes both marks; far false detections remain visible.
        left = max(0, int(min(point[0] for point in points) - 120))
        right = min(frame.shape[1], int(max(point[0] for point in points) + 120))
        top = max(0, int(min(point[1] for point in points) - 80))
        bottom = min(frame.shape[0], int(max(point[1] for point in points) + 80))
        for point, color in ((row['target_xy'], (60, 245, 60)), (row['prediction_xy'], (50, 50, 255))):
            if point is not None:
                cv2.drawMarker(frame, tuple(map(round, point)), color, cv2.MARKER_CROSS, 18, 2)
        crop = frame[top:bottom, left:right]
        image = cv2.resize(crop, (360, 190))
        y, x = index // 4 * 235, index % 4 * 360
        canvas[y:y + 190, x:x + 360] = image
        cv2.putText(canvas, f"{row['clip']}  f{row['frame']}", (x + 7, y + 208), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)
        cv2.putText(canvas, row['failure'], (x + 7, y + 227), cv2.FONT_HERSHEY_SIMPLEX, .43, (80, 180, 255), 1)
    if failures:
        cv2.imwrite(str(args.output / 'failure_contact_sheet.jpg'), canvas)
    print(f'{len(failures)} annotated failure frames; green=publisher label, red=prediction', flush=True)


if __name__ == '__main__':
    main()
