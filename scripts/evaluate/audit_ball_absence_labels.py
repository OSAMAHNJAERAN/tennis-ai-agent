"""Review all remaining absent-frame predictions without relabeling metrics."""

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
    if not report['complete']:
        raise ValueError('Incomplete replay')
    parent_reports = []
    for name, checksum in report['source_reports'].items():
        if digest(name) != checksum:
            raise ValueError('Parent report changed')
        parent_reports.append(json.loads(Path(name).read_text()))
    source = next(item for item in parent_reports if 'checkpoint_sha256' in item)
    manifest_path = args.dataset / 'manifest.json'
    if digest(manifest_path) != source['dataset_manifest_sha256']:
        raise ValueError('Dataset mismatch')
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Source file changed')
    rows = [row for row in report['labeled_rows']
            if row['target_xy'] is None and row['prediction_xy'] is not None]
    if len(rows) != report['pooled']['absent_false_detections']:
        raise ValueError('Selected errors disagree with reported counts')
    keys = {(row['clip'], row['frame']) for row in rows}
    if len(keys) != len(rows):
        raise ValueError('Duplicate reviewed label')
    args.output.mkdir(parents=True)
    panels, reviews = [], []
    for clip in sorted({row['clip'] for row in rows}):
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip}.mp4')) as frames:
            for row in [item for item in rows if item['clip'] == clip]:
                index = row['frame']
                x, y = map(round, row['prediction_xy'])
                frame = frames[index].copy()
                h, w = frame.shape[:2]
                if [w, h] != [row['width'], row['height']]:
                    raise ValueError('Label frame geometry mismatch')
                panel = np.full((330, 1200, 3), 25, np.uint8)
                cv2.drawMarker(frame, (x, y), (20, 50, 255), cv2.MARKER_CROSS, 24, 2)
                panel[40:321, :500] = cv2.resize(frame, (500, 281))
                lag = max(1, round(.1 * frames.metadata.fps))
                contexts = [max(0, index - lag), index, min(len(frames)-1, index + lag)]
                for column, other in enumerate(contexts):
                    original = frames[other]
                    patch = original[max(0, y-30):min(h, y+31), max(0, x-30):min(w, x+31)]
                    left = 515 + column * 225
                    panel[65:285, left:left+220] = cv2.resize(patch, (220, 220), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel, f'Frame {other}', (left, 313), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,235,235), 1)
                cv2.putText(panel, f'{clip} / frame {index} / publisher ABSENT / red: selected observation',
                            (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .6, (235,235,235), 1)
                panels.append(panel)
                reviews.append({**row, 'context_frames': contexts, 'fps': frames.metadata.fps,
                                'board': f'board_{(len(panels)-1)//4:02}.jpg',
                                'row_on_board': (len(panels)-1)%4, 'review_judgment': 'NOT_YET_REVIEWED'})
    images = {}
    for start in range(0, len(panels), 4):
        path = args.output / f'board_{start//4:02}.jpg'
        if not cv2.imwrite(str(path), np.concatenate(panels[start:start+4]), [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise RuntimeError('Image write failed')
        images[path.name] = digest(path)
    result = {'complete': True, 'qualification_evidence': False,
              'scope': 'ALL_REMAINING_ABSENT_FRAME_PREDICTIONS_NO_RELABELING',
              'source_report': str(args.report.resolve()), 'source_sha256': digest(args.report),
              'dataset_manifest_sha256': digest(manifest_path), 'renderer_sha256': digest(__file__),
              'image_hashes': images, 'reviews': reviews}
    (args.output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'rendered_errors': len(reviews)}))


if __name__ == '__main__':
    main()
