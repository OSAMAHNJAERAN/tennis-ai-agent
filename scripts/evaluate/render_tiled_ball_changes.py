"""Render all changed labeled outcomes between full-frame and tiled filtering."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.compare_ball_resolution import compare, keyed_rows
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--resolution-baseline', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.stream.read_text())
    baseline = json.loads(args.resolution_baseline.read_text())
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not report['complete'] or any(source['dataset_manifest_sha256'] != digest(manifest_path)
                                     for source in (report, baseline)):
        raise ValueError('Incomplete report or dataset mismatch')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset checksum changed')
    native = next(result for result in baseline['results'] if result['height'] == 1080)
    before = [row for clip in native['clips'] for row in clip['labeled_rows']]
    after = [row for clip in report['clips'] for row in clip['labeled_rows']['pixel_motion']]
    changes = compare(before, after)['changes']
    old, new = keyed_rows(before), keyed_rows(after)
    args.output.mkdir(parents=True)
    for change in changes:
        key = change['clip'], change['frame']
        first, second = old[key], new[key]
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{key[0]}.mp4')) as frames:
            current = frames[key[1]].copy()
        height, width = current.shape[:2]
        point = second['target_xy'] or second['prediction_xy'] or first['prediction_xy']
        x, y = map(round, point)
        patch = current[max(0, y - 45):min(height, y + 46), max(0, x - 45):min(width, x + 46)].copy()
        for position, color in ((first['prediction_xy'], (70, 70, 255)),
                                 (second['prediction_xy'], (255, 220, 40)),
                                 (second['target_xy'], (50, 245, 50))):
            if position is not None:
                cv2.drawMarker(current, tuple(map(round, position)), color, cv2.MARKER_CROSS, 30, 2)
        canvas = np.full((650, 1280, 3), 24, dtype=np.uint8)
        canvas[65:605, :960] = cv2.resize(current, (960, 540))
        canvas[120:400, 980:1260] = cv2.resize(patch, (280, 280), interpolation=cv2.INTER_NEAREST)
        lines = [(f'{key[0]} frame {key[1]}: {change["before"]} -> {change["after"]}', (15, 25)),
                 ('Green: publisher label | Red: full-frame filtered | Cyan: tiled filtered', (15, 52)),
                 ('Raw source patch', (980, 430)),
                 ('No enhancement', (980, 455)),
                 ('Absent label' if second['target_xy'] is None else 'Label-centered patch', (980, 480))]
        for text, position in lines:
            cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, .58, (235, 235, 235), 1)
        path = args.output / f'{key[0]}_frame_{key[1]:05}.jpg'
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError('Could not write review image')
        change.update({'image': str(path), 'before_xy': first['prediction_xy'],
                       'after_xy': second['prediction_xy'], 'target_xy': second['target_xy']})
    (args.output / 'review.json').write_text(json.dumps({
        'stream_report_sha256': digest(args.stream), 'baseline_report_sha256': digest(args.resolution_baseline),
        'renderer_sha256': digest(__file__), 'changes': changes}, indent=2), encoding='utf-8')
    print(f'Rendered all {len(changes)} changed labeled outcomes')


if __name__ == '__main__':
    main()
