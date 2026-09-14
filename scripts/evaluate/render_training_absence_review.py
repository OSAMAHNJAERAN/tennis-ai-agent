"""Render a frozen training-negative review selection without changing labels."""

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
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())
    cache_path = Path(selection['cache_manifest'])
    if not selection['complete'] or digest(cache_path) != selection['cache_manifest_sha256']:
        raise ValueError('Changed selection source')
    cache = json.loads(cache_path.read_text())
    if cache['split'] != 'TRAINING_ONLY':
        raise ValueError('Training source required')
    train = {'_'.join(c) for c in cache['selected_clips'][:16]}
    reserved = {'_'.join(c) for c in cache['selected_clips'][16:]}
    if train & reserved or train != set(selection['training_clips']) or reserved != set(selection['excluded_internal_selection_clips']):
        raise ValueError('Training split mismatch')
    expected = [{'clip': r['clip'], 'frame': r['frame'], 'width': r['width'], 'height': r['height'], 'candidate': c}
                for r in cache['rows'] if r['clip'] in train and r['target_xy'] is None
                for c in r['candidates'] if c['confidence'] >= selection['frozen_minimum_confidence']]
    if expected != selection['selected_candidates']:
        raise ValueError('Frozen review subset changed')
    dataset = Path(selection['dataset'])
    manifest_path = dataset / 'manifest.json'
    if digest(manifest_path) != selection['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        if digest(dataset / item['path']) != item['sha256']:
            raise ValueError('Source file changed')
    args.output.mkdir(parents=True)
    panels, reviews = [], []
    for clip in sorted({r['clip'] for r in expected}):
        with VideoFrameSequence(str(dataset / f'tennis/videos/{clip}.mp4')) as frames:
            for row in [r for r in expected if r['clip'] == clip]:
                index = row['frame']
                candidate = row['candidate']
                x, y = round(candidate['x']), round(candidate['y'])
                current = frames[index].copy()
                h, w = current.shape[:2]
                if [w, h] != [row['width'], row['height']]:
                    raise ValueError('Frame geometry mismatch')
                panel = np.full((330, 1200, 3), 25, np.uint8)
                cv2.drawMarker(current, (x, y), (20, 50, 255), cv2.MARKER_CROSS, 24, 2)
                panel[40:321, :500] = cv2.resize(current, (500, 281))
                lag = max(1, round(.1 * frames.metadata.fps))
                contexts = [max(0, index-lag), index, min(len(frames)-1, index+lag)]
                for column, other in enumerate(contexts):
                    frame = frames[other]
                    patch = frame[max(0,y-30):min(h,y+31), max(0,x-30):min(w,x+31)]
                    left = 515 + column*225
                    panel[65:285, left:left+220] = cv2.resize(patch, (220,220), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel, f'Frame {other}', (left,313), cv2.FONT_HERSHEY_SIMPLEX, .55, (235,235,235), 1)
                cv2.putText(panel, f'{clip} / {index} / training target ABSENT / confidence {candidate["confidence"]:.3f}',
                            (8,27), cv2.FONT_HERSHEY_SIMPLEX, .6, (235,235,235), 1)
                panels.append(panel)
                reviews.append({**row, 'context_frames': contexts, 'board': f'board_{(len(panels)-1)//4:02}.jpg',
                                'row_on_board': (len(panels)-1)%4})
    images = {}
    for start in range(0,len(panels),4):
        path = args.output / f'board_{start//4:02}.jpg'
        if not cv2.imwrite(str(path), np.concatenate(panels[start:start+4]), [cv2.IMWRITE_JPEG_QUALITY,90]):
            raise RuntimeError('Image write failed')
        images[path.name] = digest(path)
    result = {'complete': True, 'qualification_evidence': False,
              'selection_sha256': digest(args.selection), 'renderer_sha256': digest(__file__),
              'scope': 'TRAINING_ONLY_REVIEW_NO_LABEL_CHANGES', 'image_hashes': images, 'reviews': reviews}
    (args.output / 'render.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'rendered_training_candidates': len(reviews)}))


if __name__ == '__main__':
    main()
