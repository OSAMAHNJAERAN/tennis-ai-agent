"""Verify UVY original-video alignment and render publisher player annotations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gt(path, count):
    frames, seen = [[] for _ in range(count)], set()
    with Path(path).open(newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            values = np.asarray(row, dtype=float)
            if values.shape != (9,) or not np.isfinite(values).all():
                raise ValueError('Invalid MOT annotation row')
            frame, identity, category = (int(values[i]) for i in (0, 1, 7))
            if any(values[i] != int(values[i]) for i in (0, 1, 7)) or not 1 <= frame <= count:
                raise ValueError('Invalid annotation frame/ID/class')
            if identity < 0 or category not in range(1, 8) or min(values[4:6]) <= 0 or (frame, identity) in seen:
                raise ValueError('Invalid box or duplicate frame identity')
            if values[6] != 1 or not 0 <= values[8] <= 1:
                raise ValueError('Unsupported mark/visibility semantics')
            seen.add((frame, identity))
            x, y, width, height = values[2:6]
            frames[frame - 1].append({'id': identity, 'class': category,
                                      'box': [x, y, x + width, y + height], 'visibility': float(values[8])})
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/external/uvy_tennis_videos'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete'] or set(manifest['sequences']) != {'tennis_V01', 'tennis_V02', 'tennis_V03'}:
        raise ValueError('Complete frozen three-sequence acquisition required')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Acquired file integrity mismatch')
    args.output.mkdir(parents=True)
    report = {'complete': False, 'qualification_evidence': False, 'dataset_manifest_sha256': digest(manifest_path),
              'script_sha256': digest(__file__), 'alignment_criterion': 'Exact frame count/dimensions/FPS and MAE<=3 BGR levels on five fixed publisher JPEGs',
              'alignment_limit': 'Five image comparisons do not prove every intermediate frame is pixel-identical; source-video decode is complete',
              'sequences': {}, 'visual_annotation_review_complete': False}
    for sequence, info in manifest['sequences'].items():
        count = info['publisher_frames']
        folder = args.dataset / 'UVY' / sequence
        labels = (folder / 'gt/labels.txt').read_text().splitlines()
        if labels != ['player', 'sports ball', 'goal', 'referee', 'goalkeeper', 'pcamera', 'person playing']:
            raise ValueError('Unexpected publisher class mapping')
        gt = read_gt(folder / 'gt/gt.txt', count)
        track_frames = {}
        for index, frame in enumerate(gt):
            for item in frame:
                if item['class'] == 1:
                    track_frames.setdefault(item['id'], []).append(index)
        review_indices = {0, count // 2, count - 1}
        for indices in track_frames.values():
            review_indices.update([indices[0], indices[-1]])
        review_indices.update(index for index, frame in enumerate(gt) if not any(item['class'] == 1 for item in frame))
        cap = cv2.VideoCapture(str(args.dataset / info['video']))
        if not cap.isOpened():
            raise ValueError('Original video cannot be decoded')
        fps = cap.get(cv2.CAP_PROP_FPS)
        comparisons, review_images, decoded, actual_size = [], [], 0, None
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if decoded >= count:
                raise ValueError('Original video exceeds annotated image sequence')
            height, width = image.shape[:2]
            if [width, height] != info['size']:
                raise ValueError('Original video geometry differs from publisher images')
            actual_size = [width, height]
            one_based = decoded + 1
            if one_based in info['alignment_image_ids_one_based']:
                reference = cv2.imread(str(folder / f'img1/{one_based:06}.jpg'))
                if reference is None or reference.shape != image.shape:
                    raise ValueError('Missing or mis-sized alignment image')
                error = np.abs(reference.astype(float) - image.astype(float))
                comparisons.append({'frame_one_based': one_based, 'mae_bgr': float(error.mean()),
                                     'p99_absolute_difference': float(np.percentile(error, 99))})
            if decoded in review_indices:
                canvas = image.copy()
                for item in gt[decoded]:
                    color = (255, 230, 30) if item['class'] == 1 else (80, 140, 230)
                    x1, y1, x2, y2 = [round(value) for value in item['box']]
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(canvas, f"{labels[item['class']-1]} #{item['id']}", (max(0, x1), max(12, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
                destination = args.output / f'{sequence}_{one_based:06}.jpg'
                if not cv2.imwrite(str(destination), canvas):
                    raise RuntimeError('Annotation review image write failed')
                review_images.append(str(destination))
            decoded += 1
        cap.release()
        passes = decoded == count and abs(fps - info['publisher_fps']) < 1e-6 and len(comparisons) == 5 and all(item['mae_bgr'] <= 3 for item in comparisons)
        report['sequences'][sequence] = {'decoded_frames': decoded, 'fps': fps, 'size': actual_size,
                                         'alignment_checks_pass': passes, 'sampled_comparisons': comparisons,
                                         'player_tracks': {identity: {'first_frame_one_based': min(indices)+1,
                                                                      'last_frame_one_based': max(indices)+1,
                                                                      'annotated_frames': len(indices)}
                                                           for identity, indices in track_frames.items()},
                                         'player_boxes': sum(len(indices) for indices in track_frames.values()),
                                         'review_images': review_images}
        print(json.dumps({'sequence': sequence, 'alignment_checks_pass': passes, 'decoded_frames': decoded,
                          'fps': fps, 'comparisons': comparisons}), flush=True)
    report['complete'] = True
    report['alignment_checks_pass'] = all(item['alignment_checks_pass'] for item in report['sequences'].values())
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
