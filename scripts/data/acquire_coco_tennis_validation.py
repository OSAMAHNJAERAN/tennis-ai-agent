"""Acquire publisher COCO validation labels/images selected by tennis-racket GT.

This is still-image tennis-context validation, not temporal player identity GT.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import zipfile


ARCHIVE_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
MEMBERS = ('annotations/instances_val2017.json', 'annotations/person_keypoints_val2017.json')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def select_tennis_context(instances, poses):
    names = {item['name']: item['id'] for item in instances['categories']}
    if names.get('tennis racket') != 43 or names.get('person') != 1:
        raise ValueError('Expected official sparse COCO category IDs: racket 43, person 1')
    ids = sorted({item['image_id'] for item in instances['annotations'] if item['category_id'] == names['tennis racket']})
    if not ids:
        raise ValueError('No tennis-racket annotated images')
    selected = set(ids)
    images = {item['id']: item for item in instances['images']}
    if len(images) != len(instances['images']) or not selected <= images.keys():
        raise ValueError('Missing or duplicate image metadata')
    pose_images = {item['id']: item for item in poses['images']}
    for index in selected & pose_images.keys():
        if any(images[index][key] != pose_images[index][key] for key in ('width', 'height', 'file_name')):
            raise ValueError('Instance and pose image metadata differ')
    def filtered(source):
        return {**source, 'images': [images[index] for index in ids],
                'annotations': [item for item in source['annotations'] if item['image_id'] in selected]}
    return ids, filtered(instances), filtered(poses)


def download(url, destination, maximum_bytes):
    if destination.exists():
        raise FileExistsError(destination)
    request = urllib.request.Request(url, headers={'User-Agent': 'TennisVision-Validation/1.0'})
    partial = destination.with_suffix(destination.suffix + '.partial')
    if partial.exists():
        raise FileExistsError(partial)
    size = 0
    with urllib.request.urlopen(request, timeout=90) as response, partial.open('xb') as stream:
        if response.headers.get('Content-Length') and int(response.headers['Content-Length']) > maximum_bytes:
            raise ValueError('Published download exceeds remaining byte budget')
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > maximum_bytes:
                raise ValueError('Download exceeds remaining byte budget')
            stream.write(block)
    partial.rename(destination)
    return size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path, help='Optional existing official archive; read only')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; incomplete acquisitions are preserved')
    if shutil.disk_usage(args.output.resolve().anchor).free < 1024 ** 3:
        raise ValueError('At least 1 GiB free space required for bounded acquisition')
    args.output.mkdir(parents=True)
    archive = args.archive or args.output / 'annotations_trainval2017.zip'
    if args.archive is None:
        print('Downloading official annotation archive (300 MiB limit)', flush=True)
        download(ARCHIVE_URL, archive, 300 * 1024 ** 2)
    source_dir = args.output / 'source_annotations'
    source_dir.mkdir()
    sources = []
    with zipfile.ZipFile(archive) as bundle:
        for member in MEMBERS:
            info = bundle.getinfo(member)
            if info.file_size > 150 * 1024 ** 2:
                raise ValueError('Unexpected annotation member size')
            path = source_dir / Path(member).name
            with bundle.open(info) as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            sources.append(path)
    instances, poses = [json.loads(path.read_text()) for path in sources]
    ids, selected_instances, selected_poses = select_tennis_context(instances, poses)
    for name, value in (('instances.json', selected_instances), ('keypoints.json', selected_poses)):
        (args.output / name).write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')
    people = [item for item in selected_instances['annotations'] if item['category_id'] == 1]
    eligible = [item for item in selected_poses['annotations'] if not item.get('iscrowd', 0) and item.get('num_keypoints', 0) > 0]
    manifest = {'schema_version': '1.0', 'complete': False, 'split': 'VALIDATION_ONLY',
                'scope': 'COCO_2017_TENNIS_CONTEXT_STILL_IMAGES_NOT_TEMPORAL_PLAYER_IDENTITIES',
                'selection': 'All val2017 image IDs containing publisher tennis racket category 43; sorted before model inference',
                'source_archive_url': ARCHIVE_URL, 'source_archive_sha256': digest(archive),
                'integrity_scope': 'Locally recorded SHA256; no independently published archive checksum verified',
                'source_annotations': {path.name: digest(path) for path in sources},
                'selection_script_sha256': digest(__file__), 'selected_image_ids': ids,
                'counts': {'images': len(ids), 'person_instances': len(people),
                           'crowd_person_instances': sum(item.get('iscrowd', 0) for item in people),
                           'eligible_pose_instances': len(eligible),
                           'images_with_eligible_pose': len({item['image_id'] for item in eligible}),
                           'eligible_joint_visibility': dict(Counter(value for item in eligible for value in item['keypoints'][2::3]))},
                'subset_annotations': {name: digest(args.output / name) for name in ('instances.json', 'keypoints.json')},
                'licenses': instances.get('licenses', []),
                'pretraining_scope': 'Public COCO validation; independent annotations do not establish untouched model-selection holdout',
                'images': []}
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest['counts']), flush=True)
    image_dir = args.output / 'images'
    image_dir.mkdir()
    total_bytes = 0
    for item in selected_instances['images']:
        name = item['file_name']
        url = item['coco_url']
        parsed = urllib.parse.urlsplit(url)
        if (Path(name).name != name or parsed.hostname != 'images.cocodataset.org'
                or parsed.scheme not in ('http', 'https') or parsed.path != '/val2017/' + name):
            raise ValueError('Unexpected publisher image path or host')
        path = image_dir / name
        size = download(url, path, min(10 * 1024 ** 2, 200 * 1024 ** 2 - total_bytes))
        total_bytes += size
        manifest['images'].append({'id': item['id'], 'path': 'images/' + name, 'source_url': url,
                                   'sha256': digest(path), 'bytes': size, 'width': item['width'],
                                   'height': item['height'], 'license': item.get('license')})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        if len(manifest['images']) % 20 == 0:
            print(json.dumps({'downloaded_images': len(manifest['images']), 'bytes': total_bytes}), flush=True)
    manifest.update({'complete': True, 'image_bytes': total_bytes})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, **manifest['counts'], 'image_bytes': total_bytes}), flush=True)


if __name__ == '__main__':
    main()
