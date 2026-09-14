"""Acquire verified racket boxes for the forty already downloaded training clips."""
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
REVISION = '85157ca21faa2abca96d837dd2b963738029bcc8'
SIZE = 18068541
SHA256 = '0fd3d365b7d39486f3485b025bcc9ed2a7c4c217be930bb3d96c1d37889c9e74'


def main():
    output = ROOT / 'data/external/racketvision_racket_training'
    path = output / 'train_coco.json'
    output.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/datasets/linfeng302/RacketVision/resolve/{REVISION}/tennis/info/train_coco.json'
    if not path.exists():
        response = requests.get(url, stream=True, timeout=(15, 60))
        response.raise_for_status()
        parts, size = [], 0
        for chunk in response.iter_content(1024 * 1024):
            size += len(chunk)
            if size > SIZE:
                raise ValueError('Publisher annotation exceeds verified size')
            parts.append(chunk)
        payload = b''.join(parts)
        if len(payload) != SIZE or hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('Publisher LFS checksum or size mismatch')
        path.write_bytes(payload)
    if path.stat().st_size != SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Existing annotation changed')
    coco = json.loads(path.read_text())
    category = next(r['id'] for r in coco['categories'] if r['name'] == 'tennis_racket')
    annotated = {r['image_id'] for r in coco['annotations'] if r['category_id'] == category}
    clips = {}
    manifests = []
    for name in ('racketvision_training20', 'racketvision_training_additional20'):
        manifest_path = ROOT / 'data/external' / name / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest['split'] != 'TRAINING_ONLY' or manifest['revision'] != REVISION:
            raise ValueError('Training source split or revision mismatch')
        manifests.append({'path': str(manifest_path.relative_to(ROOT)),
                          'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        for match, rally in manifest['selected_clips']:
            key = f'{match}_{rally}'
            if key in clips:
                raise ValueError('Duplicate source clip')
            clips[key] = {'dataset': name, 'match': match, 'rally': rally, 'positive_image_ids': []}
    for image in coco['images']:
        parts = image['file_name'].split('/')
        key = f'{parts[0]}_{parts[2]}'
        if key in clips and image['id'] in annotated:
            clips[key]['positive_image_ids'].append(image['id'])
    manifest = {'complete': True, 'split': 'TRAINING_SOURCE_ONLY', 'revision': REVISION,
                'annotation_url': url, 'annotation_sha256': SHA256, 'annotation_bytes': SIZE,
                'integrity': 'PINNED_PUBLISHER_LFS_SHA256_AND_SIZE',
                'source_manifests': manifests, 'racket_category_id': category,
                'annotation_scope': 'ONLY_EXPLICIT_RACKET_POSITIVE_FRAMES; OTHER_FRAMES_UNLABELED',
                'clips': clips}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'clips': len(clips), 'positive_frames': sum(len(r['positive_image_ids']) for r in clips.values()),
                      'positive_clips': sum(bool(r['positive_image_ids']) for r in clips.values()),
                      'categories': coco['categories']}))


if __name__ == '__main__':
    main()
