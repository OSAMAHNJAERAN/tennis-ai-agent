"""Acquire a small source-grouped court pilot using verified ZIP member ranges."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.data.inspect_court_training_archive import BoundedHTTPFile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'artifacts/validation/vision_upgrade/court_heatmap_training_source'
    output = ROOT / 'data/external/court_heatmap_pilot'
    source_manifest = json.loads((source / 'annotation_manifest.json').read_text(encoding='utf-8'))
    if not source_manifest['complete']:
        raise ValueError('Verified annotations required')
    for row in source_manifest['files']:
        if digest(source / row['name']) != row['sha256']:
            raise ValueError('Annotations changed')
    train = json.loads((source / 'data_train.json').read_text(encoding='utf-8'))
    val = json.loads((source / 'data_val.json').read_text(encoding='utf-8'))
    groups = {}
    for row in train:
        groups.setdefault(row['id'].rsplit('_', 1)[0], []).append(row)
    group_ids = sorted(groups)
    random.Random(17).shuffle(group_ids)
    # Two temporally spread frames per selected source. No prediction-based selection.
    selected = []
    for partition, chosen in [('train', group_ids[:64]), ('selection', group_ids[64:80])]:
        for group in chosen:
            rows = sorted(groups[group], key=lambda row: int(row['id'].rsplit('_', 1)[1]))
            for index in sorted({0, len(rows) - 1}):
                selected.append({**rows[index], 'partition': partition, 'source_video_id': group})
    for row in val:
        group = row['id'].rsplit('_', 1)[0]
        if group not in groups:
            selected.append({**row, 'partition': 'external_source_check', 'source_video_id': group})
    index = json.loads((source / 'archive_index.json').read_text(encoding='utf-8'))
    entries = {item['name']: item for item in index['entries']}
    probe = json.loads((source / 'range_probe.json').read_text(encoding='utf-8'))
    requests = []
    for row in selected:
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}_[0-9]+', row['id']):
            raise ValueError('Unexpected image ID')
        item = entries[f"data/images/{row['id']}.png"]
        if item['bytes'] > 5_000_000 or item['compression'] != 8:
            raise ValueError('Unexpected member format/size')
        length = 30 + len(item['name'].encode()) + 1024 + item['compressed_bytes']
        requests.append((row, item, length))
    total = sum(request[2] for request in requests)
    if total > 250_000_000:
        raise ValueError('Pilot exceeds250MB network budget')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    manifest = {'complete': False, 'qualification_evidence': False, 'selection_seed': 17,
                'selection': 'Shuffle sorted publisher training source IDs;64 training/16 internal-selection groups; earliest/latest training image per group; all8 validation-only-source images external',
                'pretraining_caveat': 'Author checkpoint was trained on publisher train; internal selection was therefore seen in pretraining. Source-independent external check is only8 images; actual checkpoint lineage not independently verified',
                'source_manifest_sha256': digest(source / 'annotation_manifest.json'), 'archive_index_sha256': digest(source / 'archive_index.json'),
                'whole_archive_checksum_verified': False, 'planned_range_bytes': total, 'samples': selected, 'files': []}
    path = output / 'manifest.json'
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['complete'] or previous['samples'] != selected or previous['archive_index_sha256'] != manifest['archive_index_sha256']:
            raise ValueError('Resume requires incomplete unchanged selection')
        attempt = previous.get('attempt', 1) + 1
        backup = output / f'manifest_attempt{attempt-1:02}.json'
        if backup.exists():
            raise FileExistsError(backup)
        backup.write_bytes(path.read_bytes())
        manifest['attempt'] = attempt
        manifest['resume_note'] = 'Previous process ended on HTTP503; revalidate cached decoded bytes by publisher CRC and dimensions; do not repeat completed transfers'
    else:
        manifest['attempt'] = 1
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def acquire(request):
        row, item, length = request
        destination = output / 'images' / f"{row['id']}.png"
        if destination.exists():
            data = destination.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if len(data) != item['bytes'] or zlib.crc32(data) & 0xffffffff != item['crc32'] or image is None or image.shape != (720, 1280, 3):
                raise ValueError('Existing cached image fails publisher checks')
            return {'path': str(destination.relative_to(output)).replace('\\', '/'), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'crc32_verified': True, 'range_bytes': 0, 'reused_verified_image': True}
        with BoundedHTTPFile(probe['url'], probe['params'], index['archive_bytes'], budget=length) as remote:
            remote.seek(item['header_offset'])
            block = remote.read(length)
        signature, version, flags, method, mod_time, mod_date, crc, compressed, uncompressed, name_len, extra_len = struct.unpack('<4s5H3I2H', block[:30])
        if signature != b'PK\x03\x04' or flags & 1 or method != 8 or extra_len > 1024:
            raise ValueError('Unsupported member header')
        if block[30:30 + name_len].decode('utf-8') != item['name']:
            raise ValueError('ZIP member name mismatch')
        start = 30 + name_len + extra_len
        compressed_data = block[start:start + item['compressed_bytes']]
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(compressed_data, item['bytes'] + 1)
        if (len(data) != item['bytes'] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or zlib.crc32(data) & 0xffffffff != item['crc32']):
            raise ValueError('ZIP size/CRC verification failed')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (720, 1280, 3):
            raise ValueError('Unexpected image geometry')
        destination = output / 'images' / f"{row['id']}.png"
        destination.write_bytes(data)
        return {'path': str(destination.relative_to(output)).replace('\\', '/'), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'crc32_verified': True, 'range_bytes': length}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(acquire, request) for request in requests]
        for future in concurrent.futures.as_completed(futures):
            manifest['files'].append(future.result())
            path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            if len(manifest['files']) % 20 == 0:
                print(json.dumps({'completed': len(manifest['files']), 'total': len(selected)}), flush=True)
    manifest['files'].sort(key=lambda item: item['path'])
    manifest['complete'] = True
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'samples': len(selected), 'range_bytes': total}), flush=True)


if __name__ == '__main__':
    main()
