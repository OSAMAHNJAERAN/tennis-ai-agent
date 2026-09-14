"""Acquire the three UVY tennis sequences using verified bounded ZIP ranges.

Only reviewed public archive members are decoded. Member CRCs and local SHA-256
are checked/recorded; the whole-archive publisher MD5 is NOT verified by ranges.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import time
import zlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import HTTPError as Urllib3HTTPError


def safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError('Unsafe archive path')
    return bool(re.fullmatch(
        r'UVY/tennis_V0[123]/(?:img1/\d{6}\.jpg|gt/(?:gt|labels)\.txt|video_info\.txt|vid_[A-Za-z0-9_-]+\.mp4)', name))


def decode_member(block, block_start, item):
    offset = item['offset'] - block_start
    if offset < 0 or offset + 30 > len(block):
        raise ValueError('Member header outside downloaded range')
    header = struct.unpack_from('<4s5H3L2H', block, offset)
    signature, _, flags, compression, _, _, _, _, _, name_size, extra_size = header
    if signature != b'PK\x03\x04' or flags & 1 or compression != item['compression']:
        raise ValueError('Unsupported or inconsistent ZIP member header')
    name_start = offset + 30
    name = block[name_start:name_start + name_size].decode('utf-8' if flags & 2048 else 'cp437')
    if name != item['name'] or not safe_member(name):
        raise ValueError('Member name differs from reviewed index')
    start = name_start + name_size + extra_size
    end = start + item['compressed_bytes']
    if end > len(block) or not 0 <= item['bytes'] <= 12_000_000:
        raise ValueError('Member exceeds bounded extraction limits')
    compressed = block[start:end]
    if compression == 0:
        decoded = compressed
    elif compression == 8:
        decoder = zlib.decompressobj(-15)
        decoded = decoder.decompress(compressed, item['bytes'] + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Invalid or oversized deflate member')
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(decoded) != item['bytes'] or zlib.crc32(decoded) != item['crc32']:
        raise ValueError('Member size/CRC mismatch')
    return decoded


def bounded_range(session, url, start, end, total):
    size = end - start + 1
    if not 0 < size <= 120_000_000:
        raise ValueError('Range exceeds download budget')
    data = bytearray()
    # Small independent ranges survive servers that truncate long responses.
    while len(data) < size:
        first = start + len(data)
        last = min(end, first + 1024 * 1024 - 1)
        expected = last - first + 1
        for attempt in range(3):
            try:
                with session.get(url, headers={'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'},
                                 stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{total}':
                        raise ValueError('Server did not honor exact byte range')
                    chunk = response.raw.read(expected + 1)
                    if len(chunk) != expected:
                        raise IOError('Incomplete bounded archive chunk')
                data.extend(chunk)
                break
            except (requests.RequestException, Urllib3HTTPError, IOError) as error:
                if attempt == 2:
                    raise
                print(json.dumps({'range_retry': first, 'attempt': attempt + 1, 'error': type(error).__name__}), flush=True)
                time.sleep(attempt + 1)
        if len(data) % (8 * 1024 * 1024) == 0 or len(data) == size:
            print(json.dumps({'range_start': start, 'received_bytes': len(data), 'expected_bytes': size}), flush=True)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path('artifacts/validation/vision_upgrade/uvy_tennis_archive_index.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads(args.index.read_text())
    if index['archive_size'] != 3274165269 or index['publisher_archive_md5'] != 'f99594a1bd9f627ebe21219db86317a6':
        raise ValueError('Archive differs from reviewed public metadata')
    if index['archive_url'] != 'https://zenodo.org/records/21303900/files/UVY.zip?download=1':
        raise ValueError('Unexpected archive URL')
    groups = index['groups']
    if set(groups) != {'tennis_V01', 'tennis_V02', 'tennis_V03'}:
        raise ValueError('Acquire the complete frozen tennis subset')
    total_bytes = sum(group['range_end_exclusive'] - group['range_start'] for group in groups.values())
    if total_bytes > 250_000_000:
        raise ValueError('Tennis subset exceeds 250 MB transfer budget')
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET'])))
    metadata_response = session.get('https://zenodo.org/api/records/21303900', timeout=30)
    metadata_response.raise_for_status()
    metadata = metadata_response.json()
    archive = next(item for item in metadata['files'] if item['key'] == 'UVY.zip')
    if metadata['metadata']['license']['id'] != 'cc-by-4.0' or archive['size'] != index['archive_size'] or archive['checksum'] != 'md5:' + index['publisher_archive_md5']:
        raise ValueError('Publisher license or archive metadata changed')
    args.output.mkdir(parents=True)
    (args.output / 'publisher_record.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    manifest = {'complete': False, 'qualification_evidence': False,
                'source_record': index['record'], 'license': 'CC-BY-4.0',
                'selection': 'ALL_THREE_PUBLISHER_TENNIS_SEQUENCES_BEFORE_INFERENCE',
                'split': 'EXTERNAL_VALIDATION_CANDIDATE; ANNOTATION_AUDIT_PENDING',
                'archive_size': index['archive_size'], 'publisher_archive_md5': index['publisher_archive_md5'],
                'whole_archive_checksum_verified': False,
                'integrity_scope': 'EXACT_CONTENT_RANGES; MEMBER_CRC32; LOCAL_SHA256; NOT_WHOLE_ARCHIVE_MD5',
                'index_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
                'publisher_record_sha256': hashlib.sha256((args.output / 'publisher_record.json').read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'annotation_source': 'Publisher YOLO-World proposals with manual CVAT review/correction; quality audit pending',
                'files': [], 'ranges': []}
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    names = set()
    for group_name in sorted(groups):
        group = groups[group_name]
        start, end = group['range_start'], group['range_end_exclusive'] - 1
        block = bounded_range(session, index['archive_url'], start, end, index['archive_size'])
        manifest['ranges'].append({'start': start, 'end': end, 'bytes': len(block),
                                   'sha256': hashlib.sha256(block).hexdigest()})
        selected = [item for item in index['files'] if item['name'].startswith(f'UVY/{group_name}/')
                    and safe_member(item['name'])]
        for item in selected:
            if item['name'] in names:
                raise ValueError('Duplicate output member')
            names.add(item['name'])
            decoded = decode_member(block, start, item)
            relative = PurePosixPath(item['name'])
            destination = args.output.joinpath(*relative.parts)
            if not destination.resolve().is_relative_to(args.output.resolve()):
                raise ValueError('Extraction target escapes output directory')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(decoded)
            manifest['files'].append({'path': relative.as_posix(), 'bytes': len(decoded),
                                      'sha256': hashlib.sha256(decoded).hexdigest(), 'publisher_zip_crc32': item['crc32']})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'sequence': group_name, 'saved_members': len(selected), 'download_bytes': len(block)}), flush=True)
    manifest['complete'] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
