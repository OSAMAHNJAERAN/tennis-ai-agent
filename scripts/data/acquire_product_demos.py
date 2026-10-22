"""Acquire three pinned display-licensed diagnostic demos; preserve source timing.

This is product demo acquisition, not an untouched qualification set. Original
WebM bytes are retained; MP4 is a lossless VP9 remux for the existing uploader.
Decoded-frame hashes and per-frame timestamp deltas must agree before use.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    {'id': 'djokovic_2012', 'title': 'Djokovic point · 2012 US Open', 'author': 'JC',
     'sourcePage': 'https://commons.wikimedia.org/wiki/File:2012_US_Open_Djokovic_point.webm',
     'sourceUrl': 'https://upload.wikimedia.org/wikipedia/commons/5/5e/2012_US_Open_Djokovic_point.webm',
     'sourceSha1': '7fb5ece2e30a0fa7ab5b0124c9bb3cb000cae6bc', 'sourceBytes': 9089830,
     'license': 'CC BY-SA 2.0', 'licenseUrl': 'https://creativecommons.org/licenses/by-sa/2.0/',
     'players': ['Novak Djokovic', None], 'sourceSession': '2012_US_Open_Djokovic_point'},
    {'id': 'schiavone_2012', 'title': 'Schiavone rally · 2012 US Open', 'author': 'JC',
     'sourcePage': 'https://commons.wikimedia.org/wiki/File:2012_US_Open_Schiavone_rally.webm',
     'sourceUrl': 'https://upload.wikimedia.org/wikipedia/commons/3/38/2012_US_Open_Schiavone_rally.webm',
     'sourceSha1': '173a6157d5ffae70add9ef687c443953c11c8c06', 'sourceBytes': 11937043,
     'license': 'CC BY-SA 2.0', 'licenseUrl': 'https://creativecommons.org/licenses/by-sa/2.0/',
     'players': ['Francesca Schiavone', None], 'sourceSession': '2012_US_Open_Schiavone_rally'},
    {'id': 'dimitrov_tiafoe_2024', 'title': 'Dimitrov & Tiafoe · 2024 US Open', 'author': 'Oleg Yunakov',
     'sourcePage': 'https://commons.wikimedia.org/wiki/File:2024_US_Open._Grigor_Dimitrov_vs._Frances_Tiafoe.webm',
     'sourceUrl': 'https://upload.wikimedia.org/wikipedia/commons/1/1a/2024_US_Open._Grigor_Dimitrov_vs._Frances_Tiafoe.webm',
     'sourceSha1': '1b8b8b0ea37bbdac4b0d36b05738cc7dee603880', 'sourceBytes': 75005479,
     'license': 'CC BY-SA 4.0', 'licenseUrl': 'https://creativecommons.org/licenses/by-sa/4.0/',
     'players': ['Grigor Dimitrov', 'Frances Tiafoe'], 'sourceSession': '2024_US_Open_Dimitrov_Tiafoe'},
]


def digest(path, algorithm='sha256'):
    with path.open('rb') as source:
        return hashlib.file_digest(source, algorithm).hexdigest()


def run(args):
    return subprocess.check_output(args, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def probe(path, ffprobe):
    data = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'frame=pts_time:stream=width,height,avg_frame_rate,r_frame_rate,time_base,start_time,duration:format=duration',
                          '-of', 'json', str(path)]))
    pts = [float(frame['pts_time']) for frame in data['frames']]
    if not 2 <= len(pts) <= 5000 or not 0 < pts[-1] - pts[0] < 90:
        raise ValueError('Demo input exceeded its declared short-clip scope.')
    return {'stream': data['streams'][0], 'frames': len(pts), 'pts': pts,
            'formatDuration': data['format'].get('duration')}


def frame_hashes(path, ffmpeg):
    lines = run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-an',
                 '-fps_mode', 'passthrough', '-f', 'framemd5', '-']).splitlines()
    return [line.split(',')[-1].strip() for line in lines if line and not line.startswith('#')]


def main():
    ffmpeg = os.getenv('TENNIS_FFMPEG') or shutil.which('ffmpeg')
    ffprobe = os.getenv('TENNIS_FFPROBE') or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('Configured FFmpeg/ffprobe required.')
    directory = ROOT / 'artifacts/demos/sources'
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'data/demos/sources.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if manifest_path.is_file():
        previous = {item['id']: item for item in json.loads(manifest_path.read_text(encoding='utf-8')).get('sources', [])}
    records = []
    session = requests.Session()
    session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (source provenance verification)'
    for candidate in SOURCES:
        original = directory / (candidate['id'] + '.webm')
        if not original.exists():
            print('Acquiring licensed original:', candidate['id'], flush=True)
            temporary = directory / (candidate['id'] + '.webm.part')
            with session.get(candidate['sourceUrl'], stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                size = 0
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > candidate['sourceBytes']:
                            raise ValueError('Publisher media differs from pinned size.')
                        output.write(chunk)
            if temporary.stat().st_size != candidate['sourceBytes'] or digest(temporary, 'sha1') != candidate['sourceSha1']:
                raise ValueError('Publisher media differs from pinned source revision.')
            temporary.replace(original)
        if original.stat().st_size != candidate['sourceBytes'] or digest(original, 'sha1') != candidate['sourceSha1']:
            raise ValueError('Existing original differs from the pinned publisher file.')
        converted = directory / (candidate['id'] + '.mp4')
        if not converted.exists():
            run([ffmpeg, '-nostdin', '-v', 'error', '-copyts', '-start_at_zero', '-i', str(original),
                 '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart', str(converted)])
        source, remux = probe(original, ffprobe), probe(converted, ffprobe)
        if source['frames'] != remux['frames'] or frame_hashes(original, ffmpeg) != frame_hashes(converted, ffmpeg):
            raise ValueError('Remux changed decoded demo frames.')
        offset = remux['pts'][0] - source['pts'][0]
        mismatch = max(abs((out - inp) - offset) for inp, out in zip(source['pts'], remux['pts']))
        if mismatch > 1e-6:
            raise ValueError('Remux changed the source timestamp cadence.')
        record = {**candidate, 'scientificSplit': 'DIAGNOSTIC_PRODUCT_DEMO', 'qualificationEvidence': False,
                  'sourcePath': original.relative_to(ROOT).as_posix(), 'sourceSha256': digest(original),
                  'inputPath': converted.relative_to(ROOT).as_posix(), 'inputSha256': digest(converted),
                  'inputBytes': converted.stat().st_size, 'permissionCheckedAt': '2026-09-30',
                  'acquiredAt': datetime.now(timezone.utc).isoformat(),
                  'derivativeLicense': candidate['license'],
                  'modifications': 'Lossless video-only MP4 remux; audio omitted. Analysis adds tracks, court markers and pose/racket overlays.',
                  'sourceVideo': {key: value for key, value in source.items() if key != 'pts'},
                  'inputVideo': {key: value for key, value in remux.items() if key != 'pts'},
                  'timingVerification': {'decodedFramesIdentical': True, 'constantTimestampOffsetSeconds': offset,
                                         'maximumCadenceDifferenceSeconds': mismatch, 'trim': 'NONE'},
                  'analysisId': None, 'status': 'ACQUIRED_NOT_ANALYZED'}
        retained = previous.get(candidate['id'])
        if retained and retained.get('sourceSha256') == record['sourceSha256']:
            # Acquisition is repeatable without detaching a curated analysis or
            # replacing its explicitly documented excerpt with the full remux.
            selected = ROOT / retained.get('inputPath', '')
            if retained.get('analysisId') or retained.get('excerpt'):
                if not selected.is_file() or digest(selected) != retained.get('inputSha256'):
                    raise ValueError('Retained demo input is missing or changed; preserve its analysis record for review.')
                record = {**record, **retained}
        records.append(record)
        manifest_path.write_text(json.dumps({'schemaVersion': 1, 'purpose': 'Three distinct licensed product demos; no qualification claim.',
                                              'sources': records}, indent=2), encoding='utf-8')
        print(candidate['id'], source['frames'], 'frames; exact decoded-frame/timing remux verified', flush=True)


if __name__ == '__main__':
    main()
