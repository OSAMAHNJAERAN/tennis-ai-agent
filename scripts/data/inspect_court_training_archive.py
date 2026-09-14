"""Bounded HTTP range inspection of the author's ZIP64 court dataset."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import requests


class BoundedHTTPFile(io.RawIOBase):
    def __init__(self, url, params, size, budget=10_000_000):
        self.url, self.params, self.size, self.budget = url, params, size, budget
        self.position = 0
        self.received = 0
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if whence not in (0, 1, 2) or not 0 <= target <= self.size:
            raise ValueError('Invalid archive seek')
        self.position = target
        return target

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count == 0:
            return b''
        if count > 5_000_000 or self.received + count > self.budget:
            raise ValueError('Archive read exceeds fixed network budget')
        start, end = self.position, self.position + count - 1
        with self.session.get(self.url, params=self.params, headers={'Range': f'bytes={start}-{end}'},
                              stream=True, timeout=(20, 30)) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server returned different range or archive size')
            data = response.raw.read(count + 1)
        self.received += len(data)
        if len(data) != count:
            raise IOError('Truncated archive range')
        self.position += count
        return data

    def close(self):
        self.session.close()
        super().close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('artifacts/validation/vision_upgrade/court_heatmap_training_source'))
    args = parser.parse_args()
    probe = json.loads((args.source / 'range_probe.json').read_text(encoding='utf-8'))
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', probe['range'])
    if not match:
        raise ValueError('Invalid reviewed range probe')
    size = int(match[3])
    reader = BoundedHTTPFile(probe['url'], probe['params'], size)
    with reader, zipfile.ZipFile(reader) as archive:
        entries = [{'name': item.filename, 'bytes': item.file_size, 'compressed_bytes': item.compress_size,
                    'header_offset': item.header_offset, 'crc32': item.CRC, 'compression': item.compress_type}
                   for item in archive.infolist()]
        (args.source / 'archive_index.json').write_text(json.dumps({'archive_bytes': size, 'entries': entries}, indent=2), encoding='utf-8')
        selected = [item for item in archive.infolist() if item.filename.rsplit('/', 1)[-1] in ('data_train.json', 'data_val.json')]
        if len(selected) != 2:
            raise ValueError('Expected exactly two author annotation files')
        manifest = {'complete': False, 'archive_bytes': size, 'whole_archive_checksum_verified': False,
                    'source_probe_sha256': hashlib.sha256((args.source / 'range_probe.json').read_bytes()).hexdigest(), 'files': []}
        for item in selected:
            if item.file_size > 5_000_000 or item.flag_bits & 1:
                raise ValueError('Oversized or encrypted annotation')
            data = archive.read(item)
            name = item.filename.rsplit('/', 1)[-1]
            (args.source / name).write_bytes(data)
            rows = json.loads(data)
            manifest['files'].append({'name': name, 'archive_member': item.filename, 'bytes': len(data),
                                      'sha256': hashlib.sha256(data).hexdigest(), 'crc32_verified': True, 'rows': len(rows)})
            print(json.dumps({'name': name, 'rows': len(rows), 'first_two': rows[:2]}), flush=True)
        manifest['complete'] = True
        manifest['range_bytes_received'] = reader.received
        (args.source / 'annotation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps({'entries': len(entries), 'range_bytes_received': reader.received, 'first_entries': entries[:4]}), flush=True)


if __name__ == '__main__':
    main()
