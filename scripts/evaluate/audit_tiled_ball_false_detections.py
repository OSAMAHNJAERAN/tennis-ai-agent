"""Review every false labeled tiled-ball selection in its actual temporal context."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.data.acquire_coco_tennis_validation import digest
from scripts.evaluate.compare_ball_resolution import outcome
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--similarity-loss-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not report['complete'] or digest(manifest_path) != report['dataset_manifest_sha256']:
        raise ValueError('Incomplete report or dataset mismatch')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Source data changed')
    review_keys = None
    if args.similarity_loss_report:
        loss_report = json.loads(args.similarity_loss_report.read_text())
        if not loss_report['complete'] or loss_report['source_sha256'] != digest(args.report):
            raise ValueError('Loss report does not match this detector report')
        selected = next(item for item in loss_report['results'] if item['threshold'] == .98)
        review_keys = {(row['clip'],row['frame']) for row in selected['paired']['changes']
                       if row['before'] == 'correct_ball' and row['after'] != 'correct_ball'}
    args.output.mkdir(parents=True)
    failures = []
    for clip in report['clips']:
        points = clip['predictions']['pixel_motion']
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip["clip"]}.mp4')) as frames:
            fps = frames.metadata.fps
            for row in clip['labeled_rows']['pixel_motion']:
                category = outcome(row)
                if review_keys is not None:
                    if (clip['clip'], row['frame']) not in review_keys:
                        continue
                elif category not in ('wrong_location', 'absent_false_detection'):
                    continue
                index = row['frame']
                point = row['prediction_xy']
                x, y = map(round, point)
                current = frames[index].copy()
                height, width = current.shape[:2]
                canvas = np.full((800, 1280, 3), 24, dtype=np.uint8)
                cv2.drawMarker(current, (x, y), (40, 40, 255), cv2.MARKER_CROSS, 24, 2)
                if row['target_xy'] is not None:
                    cv2.drawMarker(current, tuple(map(round, row['target_xy'])),
                                   (50, 245, 50), cv2.MARKER_CROSS, 24, 2)
                canvas[45:585, :960] = cv2.resize(current, (960, 540))
                lag = max(1, round(.1 * fps))
                times = sorted(set([max(0, index - lag), index, min(len(frames) - 1, index + lag)]))
                for column, frame_index in enumerate(times):
                    source = frames[frame_index]
                    patch = source[max(0, y - 30):min(height, y + 31), max(0, x - 30):min(width, x + 31)].copy()
                    left = column * 250 + 20
                    canvas[600:775, left:left + 175] = cv2.resize(patch, (175, 175), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(canvas, f'Frame {frame_index}', (left, 795), cv2.FONT_HERSHEY_SIMPLEX, .5, (235,235,235), 1)
                cv2.putText(canvas, f'{clip["clip"]} frame {index}: {category}', (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (235,235,235), 1)
                cv2.putText(canvas, 'Red: prediction', (980, 80), cv2.FONT_HERSHEY_SIMPLEX, .6, (60,60,255), 1)
                cv2.putText(canvas, 'Green: label', (980, 110), cv2.FONT_HERSHEY_SIMPLEX, .6, (50,245,50), 1)
                cv2.putText(canvas, 'Fixed source patch', (980, 150), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,235,235), 1)
                cv2.putText(canvas, 'at three real times', (980, 180), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,235,235), 1)
                nearby = []
                for offset in range(-3, 4):
                    other = index + offset
                    if offset and 0 <= other < len(points) and points[other] is not None:
                        displacement = math.hypot((point[0] - points[other][0]) * 512 / width,
                                                  (point[1] - points[other][1]) * 288 / height)
                        nearby.append({'offset_frames': offset, 'distance_reference_px': displacement,
                                       'reference_image_speed_px_s': displacement * fps / abs(offset)})
                path = args.output / f'{clip["clip"]}_{index:05}.jpg'
                if not cv2.imwrite(str(path), canvas):
                    raise RuntimeError('Image write failed')
                failures.append({**row, 'outcome': category, 'nearby_selected_points': nearby,
                                 'image': str(path), 'context_frames': times})
    result = {'qualification_evidence': False, 'source_report_sha256': digest(args.report),
              'renderer_sha256': digest(__file__), 'scope': 'ALL_LABELED_FALSE_SELECTIONS_NO_INDEPENDENT_RELABELING',
              'reviewed_selections': failures,
              'similarity_loss_report_sha256': digest(args.similarity_loss_report) if args.similarity_loss_report else None}
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'reviewed_selections': len(failures),
                      'absent_false_detections': sum(row['outcome'] == 'absent_false_detection' for row in failures)}))


if __name__ == '__main__':
    main()
