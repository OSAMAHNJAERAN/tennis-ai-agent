"""Select original-timed diagnostic examples and queue their own real analyses.

Only the explicitly licensed Commons manifest is eligible. No private uploads
or qualification media are published. Re-running preserves existing job IDs.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.api.jobs import JobStore, Settings

MANIFEST = ROOT / 'data/demos/sources.json'
ALLOWED = {'djokovic_2012', 'schiavone_2012', 'dimitrov_tiafoe_2024'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(args):
    return subprocess.check_output(args, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)


def frame_metadata(path, ffprobe):
    value = json.loads(run([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                           'frame=pts:stream=time_base,avg_frame_rate,width,height', '-of', 'json', str(path)]))
    return value['streams'][0], [int(frame['pts']) for frame in value['frames']]


def frame_hashes(path, ffmpeg):
    lines = run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:v:0', '-an',
                 '-fps_mode', 'passthrough', '-f', 'framemd5', '-']).splitlines()
    return [line.split(',')[-1].strip() for line in lines if line and not line.startswith('#')]


def save(manifest):
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(MANIFEST)


def prepare(manifest):
    ffmpeg = os.getenv('TENNIS_FFMPEG') or shutil.which('ffmpeg')
    ffprobe = os.getenv('TENNIS_FFPROBE') or shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('Configured FFmpeg and ffprobe are required.')
    for record in manifest['sources']:
        if record['id'] not in ALLOWED or record.get('qualificationEvidence') is not False:
            raise ValueError('Only explicit diagnostic demo sources may be prepared.')
        original = ROOT / record['sourcePath']
        if digest(original) != record['sourceSha256']:
            raise ValueError('Licensed source hash changed.')
        if record.get('analysisId'):
            continue
        if record['id'] == 'dimitrov_tiafoe_2024':
            # Visual review: serve and exchanges; exclude camera pan into stands.
            # Select decoded frames, preserving every original millisecond PTS.
            start, end = 390, 990
            selected = original.with_name('dimitrov_tiafoe_2024_rally.mkv')
            source_stream, source_pts = frame_metadata(original, ffprobe)
            if not selected.exists():
                run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(original), '-map', '0:v:0', '-an',
                     '-vf', f'select=between(n\\,{start}\\,{end - 1}),setpts=PTS-STARTPTS',
                     '-fps_mode', 'passthrough', '-c:v', 'libx264', '-qp', '0', '-preset', 'fast', str(selected)])
            input_stream, input_pts = frame_metadata(selected, ffprobe)
            source_base, input_base = Fraction(source_stream['time_base']), Fraction(input_stream['time_base'])
            expected = [(pts - source_pts[start]) * source_base for pts in source_pts[start:end]]
            actual = [pts * input_base for pts in input_pts]
            if len(actual) != end - start or actual != expected:
                raise ValueError('Excerpt changed decoded frame count or presentation cadence.')
            if frame_hashes(selected, ffmpeg) != frame_hashes(original, ffmpeg)[start:end]:
                raise ValueError('Lossless excerpt changed decoded frame pixels.')
            record['excerpt'] = {'startSeconds': float(source_pts[start] * source_base),
                                 'endSeconds': float(source_pts[end] * source_base),
                                 'startFrame': start, 'endFrameExclusive': end,
                                 'sourceTimeBase': source_stream['time_base'],
                                 'firstSourcePts': str(source_pts[start]), 'lastSourcePts': str(source_pts[end - 1])}
            record['inputVideo'] = {'stream': input_stream, 'frames': len(actual)}
            record['modifications'] = 'Ten-second excerpt (source frames 390–989), lossless video-only H.264 encoding; audio omitted. Analysis adds tracks, court markers and pose/racket overlays.'
            record['timingVerification'] = {'decodedFramesIdentical': True, 'maximumCadenceDifferenceSeconds': 0,
                                           'constantTimestampOffsetSeconds': -float(source_pts[start] * source_base),
                                           'trim': 'EXPLICIT_DECODED_FRAME_RANGE'}
        else:
            selected = original
            record['inputVideo'] = record['sourceVideo']
            record['modifications'] = 'Original WebM supplied to analysis without trimming or changing cadence. Analysis playback omits audio and adds tracks, court markers and pose/racket overlays.'
            record['timingVerification'] = {'decodedFramesIdentical': True, 'maximumCadenceDifferenceSeconds': 0,
                                           'constantTimestampOffsetSeconds': 0, 'trim': 'NONE'}
        record.update(inputPath=selected.relative_to(ROOT).as_posix(), inputSha256=digest(selected),
                      inputBytes=selected.stat().st_size, status='PREPARED_NOT_ANALYZED')
        save(manifest)
        print(record['id'], 'prepared with original frame cadence', flush=True)


def queue(manifest):
    settings = Settings.from_env()
    store = JobStore(settings.root)
    for record in manifest['sources']:
        if record.get('analysisId'):
            job = store.get(record['analysisId'])
            if not job:
                raise ValueError('Curated job record disappeared; do not silently create a replacement.')
            print(record['id'], job['id'], job['status'], flush=True)
            continue
        selected = (ROOT / record['inputPath']).resolve()
        if not selected.is_relative_to(ROOT / 'artifacts/demos/sources') or digest(selected) != record['inputSha256']:
            raise ValueError('Selected demo input is outside the curated source directory or changed.')
        analysis_id = store.new_id()
        upload, output = store.upload_dir(analysis_id), store.output_dir(analysis_id)
        upload.mkdir(parents=True, exist_ok=False)
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(selected, upload / ('original' + selected.suffix))
        attribution = {key: record[key] for key in ('id', 'title', 'author', 'sourcePage', 'license', 'licenseUrl',
                                                  'derivativeLicense', 'modifications', 'sourceSha256', 'inputSha256')}
        if record.get('excerpt'):
            attribution['excerpt'] = record['excerpt']
        (output / 'media_source.json').write_text(json.dumps(attribution, indent=2), encoding='utf-8')
        # Commit provenance/manifest before the worker can claim the new job.
        record.update(analysisId=analysis_id, status='QUEUED')
        save(manifest)
        store.create(analysis_id, selected.name, selected.suffix, selected.stat().st_size, settings.config)
        print(record['id'], analysis_id, 'queued', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', action='store_true', help='Queue each prepared source once.')
    parser.add_argument('--publish-completed', action='store_true', help='Publish only completed matching saved jobs.')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if args.publish_completed:
        store = JobStore(Settings.from_env().root)
        for record in manifest['sources']:
            job = store.get(record.get('analysisId', ''))
            if job and job['status'] == 'COMPLETED':
                result = json.loads((store.output_dir(job['id']) / 'result.json').read_text(encoding='utf-8'))
                if result['summary']['id'] != job['id'] or result.get('mediaSource', {}).get('inputSha256') != record['inputSha256']:
                    raise ValueError('Completed demo identity/source provenance mismatch.')
                record['status'] = 'COMPLETED'
        save(manifest)
    else:
        prepare(manifest)
        if args.queue:
            queue(manifest)


if __name__ == '__main__':
    main()
