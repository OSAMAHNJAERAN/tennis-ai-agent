"""Acquire original UVY tennis videos, labels and five frozen alignment images."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import requests

from scripts.data.acquire_uvy_tennis import bounded_range, decode_member


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path('artifacts/validation/vision_upgrade/uvy_tennis_archive_index.json'))
    parser.add_argument('--metadata', type=Path, default=Path('data/external/uvy_tennis_metadata'))
    parser.add_argument('--publisher-record', type=Path, default=Path('data/external/uvy_tennis/publisher_record.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.index.read_text())
    record = json.loads(args.publisher_record.read_text())
    cached = json.loads((args.metadata / 'manifest.json').read_text())
    archive = next(item for item in record['files'] if item['key'] == 'UVY.zip')
    if record['id'] != 21303900 or record['metadata']['license']['id'] != 'cc-by-4.0' or not cached['complete']:
        raise ValueError('Require the reviewed publisher record and completed metadata acquisition')
    if archive['size'] != 3274165269 or archive['checksum'] != 'md5:f99594a1bd9f627ebe21219db86317a6':
        raise ValueError('Publisher archive changed')
    url = 'https://zenodo.org/records/21303900/files/UVY.zip?download=1'
    if index['archive_url'] != url or index['archive_size'] != archive['size']:
        raise ValueError('Archive index differs from publisher record')
    indexed = {item['name']: item for item in index['files']}
    if len(indexed) != len(index['files']):
        raise ValueError('Duplicate indexed member')
    for item in cached['files']:
        path = args.metadata / item['path']
        if digest(path) != item['sha256'] or zlib.crc32(path.read_bytes()) != indexed[item['path']]['crc32']:
            raise ValueError('Cached metadata integrity mismatch')
    args.output.mkdir(parents=True)
    manifest = {'complete': False, 'qualification_evidence': False,
                'scope': 'ALL_THREE_ORIGINAL_TENNIS_VIDEOS_PLUS_GT_AND_FIVE_ALIGNMENT_IMAGES_PER_SEQUENCE',
                'alignment_verified': False, 'license': 'CC-BY-4.0',
                'source_record': 'https://zenodo.org/records/21303900',
                'whole_archive_checksum_verified': False, 'publisher_archive_md5': index['publisher_archive_md5'],
                'provenance_hashes': {str(path): digest(path) for path in (args.index, args.publisher_record,
                                                                        args.metadata / 'manifest.json', Path(__file__),
                                                                        ROOT / 'scripts/data/acquire_uvy_tennis.py')},
                'sequences': {}, 'files': [], 'ranges': []}
    manifest_path = args.output / 'manifest.json'
    session = requests.Session()
    for sequence in ('tennis_V01', 'tennis_V02', 'tennis_V03'):
        prefix = f'UVY/{sequence}/'
        for suffix in ('gt/gt.txt', 'gt/labels.txt', 'video_info.txt'):
            name = prefix + suffix
            content = (args.metadata / name).read_bytes()
            destination = args.output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            manifest['files'].append({'path': name, 'bytes': len(content), 'sha256': digest(destination),
                                      'publisher_zip_crc32': indexed[name]['crc32']})
        info = dict(line.split('=', 1) for line in (args.metadata / (prefix + 'video_info.txt')).read_text().splitlines() if '=' in line)
        count = int(info['numFrames'])
        sample_ids = sorted({1 + round((count - 1) * fraction) for fraction in (0, .25, .5, .75, 1)})
        videos = [item for name, item in indexed.items() if name.startswith(prefix) and name.endswith('.mp4')]
        if len(videos) != 1:
            raise ValueError('Expected one original video per sequence')
        selected = videos + [indexed[prefix + f'img1/{frame:06}.jpg'] for frame in sample_ids]
        manifest['sequences'][sequence] = {'publisher_frames': count, 'publisher_fps': float(info['FPS']),
                                           'size': [int(info['imWidth']), int(info['imHeight'])],
                                           'video': videos[0]['name'], 'alignment_image_ids_one_based': sample_ids,
                                           'title': info['YT_video_title']}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        for item in selected:
            start = item['offset']
            end = start + 30 + len(item['name'].encode()) + item['compressed_bytes'] + 4096
            if sum(row['bytes'] for row in manifest['ranges']) + end - start + 1 > 20_000_000:
                raise ValueError('Video acquisition exceeds 20 MB transfer budget')
            block = bounded_range(session, url, start, end, index['archive_size'])
            content = decode_member(block, start, item)
            destination = args.output / item['name']
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            manifest['ranges'].append({'start': start, 'end': end, 'bytes': len(block),
                                       'sha256': hashlib.sha256(block).hexdigest()})
            manifest['files'].append({'path': item['name'], 'bytes': len(content), 'sha256': digest(destination),
                                      'publisher_zip_crc32': item['crc32']})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(json.dumps({'acquired': item['name'], 'bytes': len(content)}), flush=True)
    manifest['complete'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
