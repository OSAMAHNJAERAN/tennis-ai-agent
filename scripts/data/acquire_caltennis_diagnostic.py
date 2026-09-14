"""Acquire the three smallest indexed CalTennis videos and camera/time metadata.

Size-based selection is frozen before model inference. Research-only CC BY-NC
source; calibration provenance and coordinate conventions need further review.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import requests

REVISION = '6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d'
TREE_SHA256 = '63be664549e351af452a2e954a55c6adf46597a0e54f7fc6e080306929e07da1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-audit', type=Path, default=Path('artifacts/validation/vision_upgrade/caltennis_source_audit'))
    parser.add_argument('--output', type=Path, default=Path('data/external/caltennis_diagnostic'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tree_data = (args.source_audit / 'tree.json').read_bytes()
    if digest(tree_data) != TREE_SHA256:
        raise ValueError('Reviewed pinned tree changed')
    entries = {row['path']: row for row in json.loads(tree_data) if row['type'] == 'file'}
    videos = sorted([row for row in entries.values() if row['path'].lower().endswith(('.mp4', '.mov'))],
                    key=lambda row: (row['size'], row['path']))[:3]
    rows, paths = [], []
    for video in videos:
        path = PurePosixPath(video['path'])
        row = {'video': str(path), 'timestamps': str(path.with_name(path.stem + '_timestamps.npy')),
               'calibration': f'camera_calibration/{path.parent}/{path.stem}/calib.json'}
        rows.append(row)
        paths.extend(row.values())
    total = sum(entries[path]['size'] for path in paths)
    if total > 100_000_000:
        raise ValueError('Acquisition exceeds fixed 100MB limit')
    for path in paths:
        p = PurePosixPath(path)
        if p.is_absolute() or any(part in ('..', '.') for part in p.parts) or '\\' in path or ':' in path:
            raise ValueError('Unsafe source path')
    args.output.mkdir(parents=True)
    manifest = {'complete': False, 'qualification_evidence': False, 'source': 'https://huggingface.co/datasets/demalenk/caltennis',
                'revision': REVISION, 'tree_sha256': TREE_SHA256, 'license': 'CC-BY-NC-4.0; research diagnostic use only',
                'selection': 'Three smallest video byte sizes in reviewed tree, before inference; two sessions, not independent qualification split',
                'expected_bytes': total, 'sequences': rows, 'files': [],
                'limitations': ['Calibration coordinate convention and annotation method not yet verified',
                                'Multi-view consistency is not independent 3D ground truth',
                                'No ball, player-role, pose or speed annotations established']}
    target_manifest = args.output / 'manifest.json'
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name in ['README.md', 'metadata_mini.jsonl', 'tree.json']:
        data = (args.source_audit / name).read_bytes()
        (args.output / name).write_bytes(data)
        manifest['files'].append({'path': name, 'bytes': len(data), 'sha256': digest(data), 'kind': 'source_documentation'})
    for path in paths:
        entry = entries[path]
        destination = args.output / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/demalenk/caltennis/resolve/{REVISION}/{path}'
        received = 0
        sha256 = hashlib.sha256()
        git_hash = hashlib.sha1(b'blob ' + str(entry['size']).encode() + b'\0')
        with requests.get(url, timeout=(20, 30), stream=True) as response:
            response.raise_for_status()
            with destination.open('xb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > entry['size']:
                        raise ValueError('File exceeds publisher size')
                    handle.write(chunk)
                    sha256.update(chunk)
                    git_hash.update(chunk)
        expected = entry.get('lfs', {}).get('oid', entry['oid'])
        actual = sha256.hexdigest() if 'lfs' in entry else git_hash.hexdigest()
        if received != entry['size'] or actual != expected:
            raise ValueError('Publisher file size/hash mismatch')
        manifest['files'].append({'path': path, 'bytes': received, 'sha256': sha256.hexdigest(),
                                  'publisher_hash_verified': True, 'source_url': url})
        target_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'path': path, 'bytes': received, 'verified': True}), flush=True)
    manifest['complete'] = True
    target_manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
