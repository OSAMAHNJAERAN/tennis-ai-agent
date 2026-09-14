"""Prepare local positive racket contexts from sparse publisher frame labels."""
import hashlib
import json
from pathlib import Path
import random

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_labels(boxes, bounds):
    left, top, right, bottom = bounds
    result = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        result.append([(x1+x2-2*left)/(2*(right-left)), (y1+y2-2*top)/(2*(bottom-top)),
                       (x2-x1)/(right-left), (y2-y1)/(bottom-top)])
    return result


def main():
    source = ROOT / 'data/external/racketvision_racket_pilot01'
    output = ROOT / 'data/external/racketvision_racket_crop_pilot01'
    if output.exists():
        raise ValueError('Require fresh crop output')
    manifest = json.loads((source / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Incomplete source exports')
    rng, records = random.Random(17), []
    for split in ('train', 'val'):
        (output / 'images' / split).mkdir(parents=True)
        (output / 'labels' / split).mkdir(parents=True)
    for item in manifest['records']:
        image_path, label_path = source / item['image'], source / item['label']
        if digest(image_path) != item['image_sha256'] or digest(label_path) != item['label_sha256']:
            raise ValueError('Source image or label changed')
        frame = cv2.imread(str(image_path)); height, width = frame.shape[:2]
        boxes = []
        for line in label_path.read_text().splitlines():
            category, x, y, w, h = map(float, line.split())
            if category != 38:
                raise ValueError('Unexpected source category')
            boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            side = int(min(max(256, 8*max(x2-x1, y2-y1)), .6*min(width, height)))
            center_x = (x1+x2)/2 + rng.uniform(-.25, .25)*side
            center_y = (y1+y2)/2 + rng.uniform(-.25, .25)*side
            left = max(0, min(width-side, round(center_x-side/2)))
            top = max(0, min(height-side, round(center_y-side/2)))
            bounds = [left, top, left+side, top+side]
            labels = crop_labels(boxes, bounds)
            if not labels:
                raise ValueError('Positive anchor lost during cropping')
            stem = image_path.stem + f'_r{index}'
            image_out = output / 'images' / item['split'] / f'{stem}.jpg'
            label_out = output / 'labels' / item['split'] / f'{stem}.txt'
            if not cv2.imwrite(str(image_out), frame[top:top+side,left:left+side], [cv2.IMWRITE_JPEG_QUALITY,95]):
                raise ValueError('Crop export failed')
            label_out.write_text('\n'.join('38 '+' '.join(f'{v:.9f}' for v in r) for r in labels)+'\n', encoding='utf-8')
            records.append({'split': item['split'], 'source_clip': item['source_clip'], 'frame_index': item['frame_index'],
                            'source_image': item['image'], 'bounds_xyxy': bounds, 'rackets': len(labels),
                            'image': str(image_out.relative_to(output)), 'image_sha256': digest(image_out),
                            'label': str(label_out.relative_to(output)), 'label_sha256': digest(label_out)})
    config = yaml.safe_load((source / 'data.yaml').read_text()); config['path'] = output.as_posix()
    (output / 'data.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    report = {'complete': True, 'qualification_evidence': False, 'seed': 17,
              'scope': 'GROUND_TRUTH_ANCHORED_POSITIVE_CONTEXTS; SELECTION_BIAS; ABSENCE_AND_OWNERSHIP_UNMEASURED',
              'source_manifest_sha256': digest(source / 'manifest.json'), 'script_sha256': digest(Path(__file__)),
              'split_clips': manifest['split_clips'], 'records': records,
              'counts': {split: {'images': sum(r['split']==split for r in records),
                                 'rackets': sum(r['rackets'] for r in records if r['split']==split)} for split in ('train','val')}}
    (output / 'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()
