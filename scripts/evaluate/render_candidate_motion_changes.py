"""Review changed labeled decisions beside the actual past/current pixel patches."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=12.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.report.read_text())
    before = next(row for row in source['results'] if row['pixel_difference_threshold'] == 0)['labeled_rows']
    after = next(row for row in source['results'] if row['pixel_difference_threshold'] == args.threshold)['labeled_rows']
    if len(before) != len(after):
        raise ValueError('Comparison rows do not align')
    args.output.mkdir(parents=True)
    reviewed = []
    for old, new in zip(before, after):
        if any(old[key] != new[key] for key in ('clip', 'frame', 'target_xy')):
            raise ValueError('Comparison label mismatch')
        if old['prediction_xy'] == new['prediction_xy']:
            continue
        video = args.dataset / ('tennis/videos/' + old['clip'] + '.mp4')
        with VideoFrameSequence(str(video)) as frames:
            current = frames[old['frame']].copy()
            past_index = max(0, old['frame'] - int(np.ceil(.1 * frames.metadata.fps - 1e-9)))
            past = frames[past_index].copy()
        point = old['prediction_xy']
        x, y = map(round, point)
        x0, x1 = max(0, x - 35), min(current.shape[1], x + 36)
        y0, y1 = max(0, y - 35), min(current.shape[0], y + 36)
        patches = [image[y0:y1, x0:x1].copy() for image in (past, current)]
        for location, color in ((old['target_xy'], (50, 245, 50)), (old['prediction_xy'], (60, 60, 255)),
                                (new['prediction_xy'], (255, 210, 50))):
            if location is not None:
                cv2.drawMarker(current, tuple(map(round, location)), color, cv2.MARKER_CROSS, 22, 2)
        canvas = np.full((620, 1280, 3), 24, dtype=np.uint8)
        canvas[40:580, :960] = cv2.resize(current, (960, 540))
        for index, patch in enumerate(patches):
            top = 40 + index * 270
            canvas[top:top + 230, 990:1220] = cv2.resize(patch, (230, 230), interpolation=cv2.INTER_NEAREST)
            label = f'Frame {past_index if index == 0 else old["frame"]}'
            cv2.putText(canvas, label, (990, top + 252), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1)
        cv2.putText(canvas, f'{old["clip"]} f{old["frame"]} | green: label, red: before, cyan: after',
                    (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1)
        cv2.putText(canvas, 'Actual candidate patch over time; magnification is nearest-neighbor',
                    (15, 605), cv2.FONT_HERSHEY_SIMPLEX, .6, (235, 235, 235), 1)
        path = args.output / f'{old["clip"]}_frame_{old["frame"]:05}.jpg'
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError('Review image write failed')
        reviewed.append({'clip': old['clip'], 'frame': old['frame'], 'past_frame': past_index,
                         'target_xy': old['target_xy'], 'before_xy': old['prediction_xy'],
                         'after_xy': new['prediction_xy'], 'image': str(path)})
    (args.output / 'review.json').write_text(json.dumps({'source_report_sha256': digest(args.report),
                                                       'renderer_sha256': digest(__file__), 'changes': reviewed}, indent=2))
    print(f'Rendered {len(reviewed)} changed labeled decisions')


if __name__ == '__main__':
    main()
