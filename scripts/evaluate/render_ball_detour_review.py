"""Render every rejected frame, including unlabeled ones, for diagnostic review."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.replay_ball_temporal_detours import digest
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    sources = []
    for name, sha in report['source_reports'].items():
        if digest(name) != sha:
            raise ValueError('Source report changed')
        sources.append(json.loads(Path(name).read_text()))
    source = next(item for item in sources if 'checkpoint_sha256' in item)
    manifest_path = args.dataset / 'manifest.json'
    if digest(manifest_path) != source['dataset_manifest_sha256']:
        raise ValueError('Dataset mismatch')
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset file changed')
    args.output.mkdir(parents=True)
    clips = {item['clip']: item for item in source['clips']}
    labels = {(r['clip'], r['frame']): r for r in report['labeled_rows']}
    panels, reviews = [], []
    for clip in report['clips']:
        original = clips[clip['clip']]
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip["clip"]}.mp4')) as frames:
            if len(frames) != clip['frames'] or frames.metadata.fps != clip['fps']:
                raise ValueError('Frame metadata mismatch')
            for index in clip['rejected_frames']:
                point = original['predictions']['pixel_motion'][index]
                x, y = map(round, point)
                row = labels.get((clip['clip'], index))
                label = 'UNLABELED' if row is None else ('PUBLISHER ABSENT' if row['target_xy'] is None else 'PUBLISHER VISIBLE')
                panel = np.full((300, 1100, 3), 25, np.uint8)
                current = frames[index].copy()
                h, w = current.shape[:2]
                cv2.drawMarker(current, (x, y), (20, 50, 255), cv2.MARKER_CROSS, 24, 2)
                if row is not None and row['target_xy'] is not None:
                    cv2.drawMarker(current, tuple(map(round, row['target_xy'])), (30, 240, 30), cv2.MARKER_CROSS, 24, 2)
                panel[35:294, :460] = cv2.resize(current, (460, 259))
                contexts = [max(0, index - 2), index, min(len(frames) - 1, index + 2)]
                for column, other in enumerate(contexts):
                    frame = frames[other]
                    patch = frame[max(0, y - 25):min(h, y + 26), max(0, x - 25):min(w, x + 26)]
                    left = 475 + column * 205
                    panel[60:260, left:left + 200] = cv2.resize(patch, (200, 200), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel, f'Frame {other}', (left, 285), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
                cv2.putText(panel, f'{clip["clip"]} / {index} / {label} | red: rejected, green: label',
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1)
                panels.append(panel)
                reviews.append({'clip': clip['clip'], 'frame': index, 'publisher_label': row,
                                'rejected_point_xy': point, 'context_frames': contexts,
                                'board': f'board_{(len(panels)-1)//4:02}.jpg', 'row_on_board': (len(panels)-1) % 4})
    for start in range(0, len(panels), 4):
        if not cv2.imwrite(str(args.output / f'board_{start//4:02}.jpg'), np.concatenate(panels[start:start+4]),
                           [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError('Review image write failed')
    result = {'complete': True, 'qualification_evidence': False, 'source_report_sha256': digest(args.report),
              'renderer_sha256': digest(__file__), 'scope': 'ALL_REJECTED_FRAMES_NO_INDEPENDENT_RELABELING',
              'reviewed_frames': reviews}
    (args.output / 'review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rendered_rejections': len(reviews)}))


if __name__ == '__main__':
    main()
