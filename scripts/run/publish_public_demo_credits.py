"""Publish hash-bound credits for the three already reviewed refreshed videos.

Preserve the original acquisition metadata, results, videos and job records.
Only new sibling derivative-credit files and a separate receipt are written.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.api.jobs import Settings
from src.api.media_source import ANALYSIS_ID, validate_media_source

MANIFEST = ROOT / 'data/demos/sources.json'
REFRESH = ROOT / 'artifacts/validation/product_execution/public_demo_presentation_refresh_2026-10-01/receipt.json'
RENDERER = ROOT / 'scripts/run/refresh_public_demo_annotations.py'
CHANGES = {
    'djokovic_2012': 'Original WebM supplied to analysis without trimming or changing cadence. '
                     'Annotated playback omits audio and adds ball image tracking and notices for unavailable measurements.',
    'schiavone_2012': 'Original WebM supplied to analysis without trimming or changing cadence. '
                      'Annotated playback omits audio and adds ball image tracking and notices for unavailable measurements.',
    'dimitrov_tiafoe_2024': 'Ten-second excerpt (source frames 390–989), lossless video-only H.264 encoding; audio omitted. '
                          'Annotated playback adds ball image tracking and notices for unavailable measurements.'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def publish_exclusive(path, value, stage_directory):
    """Expose only complete bytes, retaining staging evidence on interruption.

    Hard-link creation atomically refuses an existing destination on Windows
    and POSIX. Staging and publication stay on the workspace filesystem.
    """
    payload = json.dumps(value, indent=2, allow_nan=False).encode('utf-8')
    staged = stage_directory / f'{path.name}.{uuid.uuid4().hex}.staged'
    with staged.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    assert staged.read_bytes() == payload and read(staged) == value
    os.link(staged, path)
    assert path.read_bytes() == payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'artifacts/validation/product_execution/public_demo_derivative_credit_2026-10-04.json')
    args = parser.parse_args()
    output = args.report.resolve()
    storage = Settings.from_env().root.resolve()
    receipt_root = (ROOT / 'artifacts/validation/product_execution').resolve()
    if (output.suffix.lower() != '.json' or not output.is_relative_to(ROOT) or not receipt_root.is_relative_to(ROOT)
            or not output.is_relative_to(receipt_root) or not storage.is_relative_to(ROOT)
            or output.is_relative_to((storage / 'analyses').resolve())):
        raise ValueError('Publication and receipts must remain inside this workspace.')
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve the existing publication receipt.')
    refresh = read(REFRESH)
    assert refresh['manifestSha256'] == digest(MANIFEST)
    assert refresh['rendererSha256'] == digest(RENDERER)
    assert refresh['allPublishedAt'] and refresh['modelInferenceRun'] is False
    entries = [item for item in read(MANIFEST)['sources'] if item['status'] == 'COMPLETED']
    assert {item['id'] for item in entries} == set(CHANGES) and len(entries) == 3
    frozen = {path: digest(path) for path in (MANIFEST, REFRESH, RENDERER, Path(__file__))}
    planned = []
    directory_snapshots = {}
    with sqlite3.connect((storage / 'analyses.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        for entry in entries:
            analysis_id = entry['analysisId']
            assert ANALYSIS_ID.fullmatch(analysis_id)
            directory = (storage / 'analyses' / analysis_id).resolve()
            assert directory.parent == (storage / 'analyses').resolve()
            record = next(item for item in refresh['demos'] if item['analysisId'] == analysis_id)
            assert record['id'] == entry['id'] and record['published'] is True
            job = connection.execute('SELECT * FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            assert job is not None and dict(job) == record['jobSnapshot']
            old_hashes = {path.name: digest(path) for path in directory.iterdir()
                          if path.is_file() and path.name != 'annotated_web.mp4'}
            assert old_hashes == record['protectedFileHashes']
            directory_snapshots[directory] = old_hashes
            video = directory / 'annotated_web.mp4'
            assert digest(video) == record['staged']['sha256']
            source = validate_media_source(entry)
            result = read(directory / 'result.json')
            assert source == validate_media_source(read(directory / 'media_source.json'))
            assert source == validate_media_source(result['mediaSource'])
            assert result['summary']['id'] == analysis_id
            assert not any(result['summary']['capabilities'].get(key) is True for key in
                           ('court', 'players', 'pose', 'racket', 'events', 'lineCalls', 'scoring'))
            sidecar = directory.parent / f'{analysis_id}.annotated_web_credit.json'
            assert sidecar.resolve().parent == directory.parent and sidecar.resolve().is_relative_to(ROOT)
            assert not sidecar.is_symlink() and (not sidecar.exists() or sidecar.is_file())
            assert sidecar != output
            metadata = {'schemaVersion': 1, 'analysisId': analysis_id,
                        'sourceSha256': source['sourceSha256'], 'inputSha256': source['inputSha256'],
                        'annotatedSha256': record['staged']['sha256'], 'modifications': CHANGES[entry['id']]}
            if sidecar.exists():
                assert read(sidecar) == metadata, 'Preserve and inspect any different existing derivative credit.'
            frozen.update({directory / name: expected for name, expected in old_hashes.items()})
            frozen[video] = record['staged']['sha256']
            planned.append((sidecar, metadata, source['modifications']))
        stage_directory = (output.parent / 'public_demo_derivative_credit_staging').resolve()
        assert stage_directory.is_relative_to(receipt_root) and stage_directory.is_relative_to(ROOT)
        stage_directory.mkdir(parents=True, exist_ok=True)
        for sidecar, metadata, _ in planned:
            if not sidecar.exists():
                publish_exclusive(sidecar, metadata, stage_directory)
        assert all(digest(path) == expected for path, expected in frozen.items())
        for directory, expected in directory_snapshots.items():
            assert {path.name: digest(path) for path in directory.iterdir()
                    if path.is_file() and path.name != 'annotated_web.mp4'} == expected
        for sidecar, metadata, _ in planned:
            assert not sidecar.is_symlink() and read(sidecar) == metadata
            job = connection.execute('SELECT * FROM analyses WHERE id=?', (metadata['analysisId'],)).fetchone()
            record = next(item for item in refresh['demos'] if item['analysisId'] == metadata['analysisId'])
            assert dict(job) == record['jobSnapshot']
    receipt = {'publishedAt': datetime.now(timezone.utc).isoformat(),
               'purpose': 'DERIVATIVE_MODIFICATION_CREDIT; NOT_MODEL_ACCURACY',
               'modelInferenceRun': False, 'savedMeasurementsChanged': False,
               'originalMetadataAndVideosUnchanged': True, 'jobSnapshotsUnchanged': True,
               'protectedHashes': {path.relative_to(ROOT).as_posix(): value for path, value in frozen.items()},
               'demos': [{'analysisId': metadata['analysisId'],
                          'sidecar': sidecar.relative_to(ROOT).as_posix(), 'sidecarSha256': digest(sidecar),
                          'annotatedSha256': metadata['annotatedSha256'],
                          'historicalModifications': previous, 'presentedModifications': metadata['modifications']}
                         for sidecar, metadata, previous in planned]}
    publish_exclusive(output, receipt, stage_directory)
    print(f'Published {len(planned)} derivative credits; {len(frozen)} original files and all three job snapshots unchanged.')
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
