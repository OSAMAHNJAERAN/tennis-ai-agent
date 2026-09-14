"""Prepare disjoint source-ID training/selection images for a racket-only pilot.

Preserves COCO class 38 in an 80-class head. Any trained checkpoint is exclusively
a racket candidate: missing person annotations prohibit its use as a person model.
"""
import hashlib
import json
from pathlib import Path
import random

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'data/external/racketvision_racket_training'
    output = ROOT / 'data/external/racketvision_racket_pilot01'
    if output.exists():
        raise ValueError('Require a fresh output directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest['complete'] or digest(source / 'train_coco.json') != manifest['annotation_sha256']:
        raise ValueError('Incomplete acquisition or annotation checksum mismatch')
    coco = json.loads((source / 'train_coco.json').read_text())
    images = {r['id']: r for r in coco['images']}
    annotations = {}
    for ann in coco['annotations']:
        if ann['category_id'] == manifest['racket_category_id']:
            annotations.setdefault(ann['image_id'], []).append(ann)
    sources = {}
    for item in manifest['source_manifests']:
        path = ROOT / item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Training source manifest changed')
        data = json.loads(path.read_text())
        sources[path.parent.name] = {r['path']: r['sha256'] for r in data['files']}
    keys = sorted(manifest['clips'])
    if len(keys) != 40 or len({manifest['clips'][k]['match'] for k in keys}) != 40:
        raise ValueError('Expected forty distinct publisher source IDs')
    random.Random(17).shuffle(keys)
    split_keys = {'train': sorted(keys[:32]), 'val': sorted(keys[32:])}
    output.mkdir(parents=True)
    records, clipped = [], 0
    for split, clips in split_keys.items():
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
        for key in clips:
            clip = manifest['clips'][key]
            dataset = ROOT / 'data/external' / clip['dataset']
            relative_video = f'tennis/videos/{key}.mp4'
            video = dataset / relative_video
            if digest(video) != sources[clip['dataset']][relative_video]:
                raise ValueError('Video checksum mismatch')
            required = {int(Path(images[i]['file_name']).stem): i for i in clip['positive_image_ids']}
            if len(required) != len(clip['positive_image_ids']) or not required:
                raise ValueError('Missing or duplicate labeled frame indices')
            capture = cv2.VideoCapture(str(video))
            index = 0
            try:
                while required:
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f'Truncated video before labeled frame: {key}')
                    if index in required:
                        image_id = required.pop(index)
                        info = images[image_id]
                        width, height = info['width'], info['height']
                        lines = []
                        for ann in annotations[image_id]:
                            x, y, w, h = ann['bbox']
                            if w <= 0 or h <= 0:
                                raise ValueError('Degenerate annotation')
                            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
                            if x2 <= x1 or y2 <= y1:
                                raise ValueError('Fully outside annotation')
                            clipped += int((x1, y1, x2, y2) != (x, y, x+w, y+h))
                            lines.append(f'38 {(x1+x2)/(2*width):.9f} {(y1+y2)/(2*height):.9f} {(x2-x1)/width:.9f} {(y2-y1)/height:.9f}')
                        stem = f'{key}_{index:05d}'
                        image_path = output / 'images' / split / f'{stem}.jpg'
                        label_path = output / 'labels' / split / f'{stem}.txt'
                        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise ValueError('Image export failed')
                        label_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
                        records.append({'split': split, 'source_clip': key, 'frame_index': index,
                                        'publisher_image_id': image_id, 'annotation_size': [width, height],
                                        'native_size': [frame.shape[1], frame.shape[0]], 'rackets': len(lines),
                                        'image': str(image_path.relative_to(output)), 'image_sha256': digest(image_path),
                                        'label': str(label_path.relative_to(output)), 'label_sha256': digest(label_path)})
                    index += 1
            finally:
                capture.release()
            print(json.dumps({'split': split, 'clip': key, 'exported_images': len(records)}), flush=True)
    model = YOLO(str(ROOT / 'yolo11m.pt'))
    if model.names[38] != 'tennis racket' or len(model.names) != 80:
        raise ValueError('Expected original COCO head')
    (output / 'data.yaml').write_text(yaml.safe_dump({'path': output.as_posix(), 'train': 'images/train',
                                                     'val': 'images/val', 'names': model.names}), encoding='utf-8')
    report = {'complete': True, 'qualification_evidence': False, 'seed': 17,
              'scope': 'RACKET_ONLY_TRAINING; OTHER_CLASSES_UNLABELED; DO_NOT_USE_CANDIDATE_FOR_PERSON_DETECTION',
              'source_manifest_sha256': digest(source / 'manifest.json'), 'script_sha256': digest(Path(__file__)),
              'split_clips': split_keys, 'split_independence': 'PUBLISHER_MATCH_IDS_DISJOINT; ORIGINAL_BROADCAST_INDEPENDENCE_UNVERIFIED',
              'clipped_boxes': clipped, 'records': records,
              'counts': {split: {'images': sum(r['split'] == split for r in records),
                                 'rackets': sum(r['rackets'] for r in records if r['split'] == split)} for split in split_keys}}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
