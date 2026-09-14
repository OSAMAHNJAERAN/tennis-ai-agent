"""Change half the training crops to larger apparent rackets; preserve selection."""
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.data.prepare_racket_crop_pilot import crop_labels, digest


def anchor_crop(box, width, height, factor, offset_x, offset_y):
    x1, y1, x2, y2 = box
    side = min(min(width, height), max(16, round(factor * max(x2-x1, y2-y1))))
    center_x, center_y = (x1+x2)/2+offset_x*side, (y1+y2)/2+offset_y*side
    left = max(0, min(width-side, round(center_x-side/2)))
    top = max(0, min(height-side, round(center_y-side/2)))
    return [left, top, left+side, top+side]


def main():
    previous = ROOT / 'data/external/racketvision_racket_crop_pilot01'
    frames = ROOT / 'data/external/racketvision_racket_pilot01'
    output = ROOT / 'data/external/racketvision_racket_scale_pilot03'
    if output.exists():
        raise ValueError('Require a fresh output directory')
    manifest = json.loads((previous/'manifest.json').read_text())
    frame_manifest = json.loads((frames/'manifest.json').read_text())
    if not manifest['complete'] or not frame_manifest['complete']:
        raise ValueError('Incomplete input acquisition')
    source_images = {r['image']: r for r in frame_manifest['records']}
    rng = random.Random(17)
    records, train_index, changed = [], 0, 0
    for item in manifest['records']:
        for field in ('image', 'label'):
            if digest(previous/item[field]) != item[field+'_sha256']:
                raise ValueError('Previous crop checksum mismatch')
            (output/item[field]).parent.mkdir(parents=True, exist_ok=True)
        record = dict(item)
        change = item['split'] == 'train' and train_index % 2 == 0
        if item['split'] == 'train':
            train_index += 1
        if change:
            source = source_images[item['source_image']]
            if digest(frames/source['image']) != source['image_sha256'] or digest(frames/source['label']) != source['label_sha256']:
                raise ValueError('Original image or labels changed')
            frame = cv2.imread(str(frames/source['image']))
            height, width = frame.shape[:2]
            boxes = []
            for line in (frames/source['label']).read_text().splitlines():
                cls, x, y, w, h = map(float, line.split())
                if cls != 38:
                    raise ValueError('Unexpected source class')
                boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
            index = int(Path(item['image']).stem.rsplit('_r',1)[1])
            factor = rng.uniform(1.5, 4.)
            bounds = anchor_crop(boxes[index], width, height, factor, rng.uniform(-.1,.1), rng.uniform(-.1,.1))
            x1,y1,x2,y2 = bounds
            anchor = boxes[index]
            if not (x1 <= anchor[0] <= anchor[2] <= x2 and y1 <= anchor[1] <= anchor[3] <= y2):
                raise ValueError('Anchor was truncated')
            labels = crop_labels(boxes,bounds)
            if not labels:
                raise ValueError('Anchor label disappeared')
            if not cv2.imwrite(str(output/item['image']), frame[y1:y2,x1:x2], [cv2.IMWRITE_JPEG_QUALITY,95]):
                raise ValueError('Crop export failed')
            (output/item['label']).write_text('\n'.join('38 '+' '.join(f'{v:.9f}' for v in row) for row in labels)+'\n',encoding='utf-8')
            record.update(bounds_xyxy=bounds,rackets=len(labels),scale_factor=factor)
            changed += 1
        else:
            for field in ('image','label'):
                shutil.copyfile(previous/item[field],output/item[field])
        for field in ('image','label'):
            record[field+'_sha256'] = digest(output/item[field])
        if item['split'] == 'val' and any(record[f+'_sha256'] != item[f+'_sha256'] for f in ('image','label')):
            raise ValueError('Internal selection must be byte-identical')
        records.append(record)
    config = yaml.safe_load((previous/'data.yaml').read_text())
    config['path'] = output.as_posix()
    (output/'data.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    report = {'complete':True,'qualification_evidence':False,'seed':17,
              'scope':'SCALE_ABLATION_OF_POSITIVE_CROPS; NO_NEW_APPEARANCES; NOT_INDEPENDENT_QUALIFICATION',
              'previous_manifest_sha256':digest(previous/'manifest.json'), 'script_sha256':digest(Path(__file__)),
              'split_clips':manifest['split_clips'],'changed_training_crops':changed,
              'internal_selection_byte_identical':True,'records':records,
              'counts':{split:{'images':sum(r['split']==split for r in records),
                              'rackets':sum(r['rackets'] for r in records if r['split']==split)} for split in ('train','val')}}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'changed_training_crops':changed,'counts':report['counts']}))


if __name__ == '__main__':
    main()
