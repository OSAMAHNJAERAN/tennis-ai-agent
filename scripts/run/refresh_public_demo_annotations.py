"""Re-render only the three curated, court-unavailable diagnostic demos.

No models run and no analysis JSON, identities, source timing or job state changes.
Prepare first, review the staged videos, then explicitly publish. Prior served
videos remain recoverable beside the receipt. This is not accuracy evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.api.jobs import Settings
from src.api.media_source import ANALYSIS_ID, validate_media_source
from src.visualization.video_annotator import VideoAnnotator

ALLOWED = {'djokovic_2012', 'schiavone_2012', 'dimitrov_tiafoe_2024'}
TARGET = ROOT / 'artifacts/validation/product_execution/public_demo_presentation_refresh_2026-10-01'
MANIFEST = ROOT / 'data/demos/sources.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def binary(name):
    value = os.getenv('TENNIS_' + name.upper()) or shutil.which(name)
    if not value:
        raise RuntimeError(f'{name} is required.')
    return value


def probe(path):
    output = subprocess.check_output([binary('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pts:stream=width,height,avg_frame_rate,time_base,codec_name',
        '-of', 'json', str(path)], text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    value = json.loads(output)
    stream = value['streams'][0]
    base = Fraction(stream['time_base'])
    return stream, [int(frame['pts']) * base for frame in value['frames']]


def saved_job(storage, analysis_id):
    connection = sqlite3.connect((storage / 'analyses.sqlite3').as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute('SELECT * FROM analyses WHERE id=?', (analysis_id,)).fetchone()
        if not row or row['status'] != 'COMPLETED':
            raise ValueError('Only an existing completed curated job may be rendered.')
        return dict(row)
    finally:
        connection.close()


def validate_evidence(detections, result):
    count = result['summary']['frames']
    frames = detections['frames']
    if not isinstance(count, int) or count <= 0 or len(frames) != count:
        raise ValueError('Detection/result frame counts disagree.')
    capabilities = result['summary']['capabilities']
    if any(capabilities.get(key) is True for key in
           ('court', 'players', 'pose', 'racket', 'events', 'lineCalls', 'scoring')):
        raise ValueError('This narrow renderer cannot omit valid optional evidence.')
    if result.get('events') or result.get('lineCalls'):
        raise ValueError('This renderer does not handle observed events.')
    published_frames = result.get('detections', {}).get('frames')
    if not isinstance(published_frames, list) or len(published_frames) != count:
        raise ValueError('Raw detections and published evidence disagree.')
    width, height = detections['metadata']['width'], detections['metadata']['height']
    for index, (frame, published) in enumerate(zip(frames, published_frames)):
        if (frame['frame_index'] != index or frame.get('court_registration_valid') is not False
                or frame.get('player_1') is not None or frame.get('player_2') is not None):
            raise ValueError('This renderer requires explicit unavailable court/players on every frame.')
        if (published.get('frame_index') != index or published.get('court_registration_valid') is not False
                or published.get('player_1') is not None or published.get('player_2') is not None):
            raise ValueError('Raw detections and published evidence disagree.')
        raw_position = frame['ball'].get('position_px')
        # The normal result contract uses the exported trajectory's 2-decimal
        # image coordinates. Preserve that public evidence, not a new estimate.
        expected_position = [round(value, 2) for value in raw_position] if raw_position is not None else None
        ball = published['ball']
        if ball['state'] != frame['ball']['state'] or ball.get('position_px') != expected_position:
            raise ValueError('Raw detections and published evidence disagree.')
        position = ball.get('position_px')
        if position is not None and (not isinstance(position, list) or len(position) != 2
                or any(isinstance(number, bool) or not isinstance(number, (int, float))
                       or not math.isfinite(number) for number in position)):
            raise ValueError('Invalid saved ball coordinates.')
        if position is not None and not (0 <= position[0] < width and 0 <= position[1] < height):
            raise ValueError('Saved ball is outside the source image.')
        if ball['state'] not in {'DETECTED', 'INTERPOLATED', 'PREDICTED', 'MISSING', 'OCCLUDED'}:
            raise ValueError('Unknown saved ball state.')
        if ball['state'] in {'MISSING', 'OCCLUDED'} and position is not None:
            raise ValueError('Unavailable ball must not have invented image coordinates.')
    return published_frames


def annotate(image, ball, trail):
    position = ball['position_px']
    if position is None:
        trail.clear()
    else:
        trail.append(tuple(position))
        del trail[:-25]
        image = VideoAnnotator.draw_ball(image, tuple(position), ball['state'])
    image = VideoAnnotator.draw_ball_trajectory(image, trail, max_trail=25)
    cv2.rectangle(image, (0, 0), (image.shape[1], 90), (20, 20, 20), -1)
    cv2.line(image, (0, 90), (image.shape[1], 90), (0, 255, 255), 2)
    for label, origin, scale, color, weight in (
            ('Tennis analysis', (30, 38), .8, (240, 240, 240), 2),
            ('Shot and rally analysis unavailable', (30, 72), .65, (0, 210, 255), 1),
            ('Court positions unavailable', (30, 115), .6, (0, 180, 255), 2)):
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)
    return image


def encode(source, staged, frames, stream, log_path):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError('Cannot decode the existing normalized source.')
    command = [binary('ffmpeg'), '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s:v', f"{stream['width']}x{stream['height']}", '-r', stream['avg_frame_rate'], '-i', 'pipe:0',
        '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(staged)]
    started = time.perf_counter()
    with log_path.open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            trail = []
            for item in frames:
                ok, image = capture.read()
                if not ok or image is None or image.shape[:2] != (stream['height'], stream['width']):
                    raise ValueError('Source ended early or changed image dimensions.')
                process.stdin.write(annotate(image, item['ball'], trail).tobytes())
            if capture.read()[0]:
                raise ValueError('Source has more frames than the saved evidence.')
        finally:
            capture.release()
            process.stdin.close()
            return_code = process.wait(timeout=120)
        if return_code:
            raise RuntimeError(f'Encoding failed; inspect {log_path}.')
    return time.perf_counter() - started


def verify_video(source_stream, source_pts, staged):
    stream, timestamps = probe(staged)
    if (len(timestamps) != len(source_pts) or timestamps != source_pts
            or Fraction(stream['avg_frame_rate']) != Fraction(source_stream['avg_frame_rate'])
            or (stream['width'], stream['height']) != (source_stream['width'], source_stream['height'])
            or stream['codec_name'] != 'h264'):
        raise ValueError('Staged video changed frame alignment, timing, dimensions or browser codec.')
    return {'decodedFrames': len(timestamps), 'fpsRational': stream['avg_frame_rate'],
            'allDecodedPtsExactlyMatchSource': True, 'width': stream['width'], 'height': stream['height'],
            'codec': stream['codec_name'], 'bytes': staged.stat().st_size, 'sha256': digest(staged)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Publish previously staged and reviewed videos.')
    args = parser.parse_args()
    manifest = read(MANIFEST)
    if {entry['id'] for entry in manifest['sources']} != ALLOWED or len(manifest['sources']) != 3:
        raise ValueError('This one-time refresh is restricted to the three existing curated demos.')
    storage = Settings.from_env().root
    TARGET.mkdir(parents=True, exist_ok=True)
    receipt_path = TARGET / 'receipt.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['rendererSha256'] != digest(Path(__file__)) or receipt['manifestSha256'] != digest(MANIFEST):
            raise ValueError('Refresh code or curated manifest changed; inspect the existing receipt.')
    else:
        receipt = {'purpose': 'PRESENTATION_ONLY; SAVED_DETECTIONS_UNCHANGED; NOT_MODEL_ACCURACY',
                   'preparedAt': datetime.now(timezone.utc).isoformat(), 'rendererSha256': digest(Path(__file__)),
                   'manifestSha256': digest(MANIFEST), 'modelInferenceRun': False, 'demos': []}
        save(receipt_path, receipt)
    if args.publish and {item['id'] for item in receipt['demos']} != ALLOWED:
        raise ValueError('Prepare and review all three staged videos before publication.')
    for entry in manifest['sources']:
        analysis_id = entry['analysisId']
        if not ANALYSIS_ID.fullmatch(analysis_id) or entry['status'] != 'COMPLETED':
            raise ValueError('Invalid curated completed analysis identity.')
        job = saved_job(storage, analysis_id)
        directory = storage / 'analyses' / analysis_id
        result = read(directory / 'result.json')
        if (result['summary']['id'] != analysis_id or result.get('mediaSource') != validate_media_source(entry)
                or validate_media_source(read(directory / 'media_source.json')) != validate_media_source(entry)):
            raise ValueError('Curated source attribution or saved result identity mismatch.')
        frames = validate_evidence(read(directory / 'detections.json'), result)
        original = storage / 'uploads' / analysis_id / ('original' + job['extension'])
        if digest(original) != entry['inputSha256'] or digest(ROOT / entry['sourcePath']) != entry['sourceSha256']:
            raise ValueError('Original source hashes changed.')
        protected = {path.name: digest(path) for path in directory.iterdir()
                     if path.is_file() and path.name != 'annotated_web.mp4'}
        source = directory / 'source.mp4'
        source_stream, source_pts = probe(source)
        if len(source_pts) != len(frames) or not math.isclose(float(Fraction(source_stream['avg_frame_rate'])),
                result['summary']['fps'], rel_tol=0, abs_tol=1e-9):
            raise ValueError('Source media and saved measurements disagree.')
        timing_rate = Fraction(read(directory / 'media_metadata.json')['normalization']['fpsRational'])
        if timing_rate != Fraction(source_stream['avg_frame_rate']):
            raise ValueError('Saved timing provenance and source playback rate disagree.')
        folder = TARGET / analysis_id
        folder.mkdir(exist_ok=True)
        staged = folder / 'refreshed_annotated_web.mp4'
        previous = folder / 'previous_annotated_web.mp4'
        record = next((item for item in receipt['demos'] if item['analysisId'] == analysis_id), None)
        if record is None:
            if args.publish:
                raise ValueError('Prepare and review all staged videos before publication.')
            if previous.exists() or staged.exists():
                raise ValueError('Unrecorded refresh artifacts exist; inspect before retrying.')
            shutil.copy2(directory / 'annotated_web.mp4', previous)
            previous_hash = digest(previous)
            elapsed = encode(source, staged, frames, source_stream, folder / 'encoding.log')
            if protected != {path.name: digest(path) for path in directory.iterdir()
                             if path.is_file() and path.name != 'annotated_web.mp4'}:
                raise ValueError('Historical inputs changed while rendering.')
            record = {'id': entry['id'], 'analysisId': analysis_id, 'protectedFileHashes': protected,
                      'jobSnapshot': job, 'inputSha256': entry['inputSha256'], 'sourceSha256': entry['sourceSha256'],
                      'previousSha256': previous_hash, 'staged': verify_video(source_stream, source_pts, staged),
                      'renderSeconds': elapsed, 'published': False}
            receipt['demos'].append(record)
            save(receipt_path, receipt)
        if protected != record['protectedFileHashes'] or job != record['jobSnapshot']:
            raise ValueError('Historical evidence or completed job state changed.')
        if digest(previous) != record['previousSha256'] or verify_video(source_stream, source_pts, staged) != record['staged']:
            raise ValueError('Preserved or staged video changed.')
        served = directory / 'annotated_web.mp4'
        expected = record['staged']['sha256'] if record['published'] else record['previousSha256']
        if digest(served) != expected:
            # A stopped command may have atomically published before saving its receipt.
            if args.publish and digest(served) == record['staged']['sha256']:
                record['published'] = True
            else:
                raise ValueError('Served video changed outside this refresh.')
        if args.publish and not record['published']:
            pending = directory / 'presentation_refresh_pending.mp4'
            if pending.exists():
                raise ValueError('Unrecorded pending publication exists; inspect before retrying.')
            shutil.copyfile(staged, pending)
            if digest(pending) != record['staged']['sha256']:
                raise ValueError('Publication copy changed.')
            pending.replace(served)
            record['published'] = True
            record['publishedAt'] = datetime.now(timezone.utc).isoformat()
        if args.publish:
            if digest(served) != record['staged']['sha256']:
                raise ValueError('Published video hash mismatch.')
            save(receipt_path, receipt)
        print(json.dumps({'id': entry['id'], 'analysisId': analysis_id, 'published': record['published'],
                          'frames': len(frames), 'fps': source_stream['avg_frame_rate']}), flush=True)
    if args.publish:
        receipt['allPublishedAt'] = datetime.now(timezone.utc).isoformat()
        save(receipt_path, receipt)
    print(receipt_path.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
