"""Verify curated identities, licenses, and original PTS through the real API.

This verifies artifact integrity and timing provenance, not detector accuracy.
"""
from collections import Counter
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from html import escape
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re
from urllib.parse import urlsplit, parse_qs

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.api.media_source import ATTRIBUTION_FIELDS, validate_media_source
from src.api.media_revision import presented_media_source
from src.api.jobs import Settings


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
        default=ROOT / 'artifacts/validation/product_execution/public_demos_integrity.json',
        help='Save a separate receipt when verifying refreshed presentations.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/demos/sources.json').read_text(encoding='utf-8'))
    ffprobe = os.getenv('TENNIS_FFPROBE') or shutil.which('ffprobe')
    if not ffprobe:
        raise RuntimeError('FFprobe is required for independent original-PTS comparison.')
    records = []
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=60) as client:
        response = client.get('/api/demos')
        response.raise_for_status()
        catalog = response.json()['demos']
        expected_ids = {record['id'] for record in manifest['sources'] if record['status'] == 'COMPLETED'}
        assert {demo['id'] for demo in catalog} == expected_ids
        assert len({demo['analysisId'] for demo in catalog}) == len(catalog)
        for record in manifest['sources']:
            if record['status'] != 'COMPLETED':
                continue
            demo = next(item for item in catalog if item['id'] == record['id'])
            analysis_id = record['analysisId']
            assert demo['analysisId'] == analysis_id
            preview = urlsplit(demo['previewUrl'])
            assert preview.path == f'/api/analyses/{analysis_id}/media/annotated' and not preview.scheme and not preview.netloc
            revision = parse_qs(preview.query)
            assert set(revision) == {'v'} and len(revision['v']) == 1 and re.fullmatch(r'[a-f0-9]{1,16}-[a-f0-9]{1,16}', revision['v'][0])
            base = '/api/analyses/' + analysis_id
            result = client.get(base + '/result').raise_for_status().json()
            assert result['summary']['id'] == analysis_id
            directory = Settings.from_env().root / 'analyses' / analysis_id
            historical_source = validate_media_source(record)
            current_source = presented_media_source(historical_source, directory, analysis_id)
            assert result['mediaSource'] == current_source
            assert demo['attribution'] == {key: current_source[key] for key in ATTRIBUTION_FIELDS}
            assert digest(ROOT / record['sourcePath']) == record['sourceSha256']
            assert digest(ROOT / record['inputPath']) == record['inputSha256']
            mapping_response = client.get(base + '/download/source-timing')
            mapping_response.raise_for_status()
            assert mapping_response.headers['content-type'].startswith('application/x-ndjson')
            mapping = [json.loads(line) for line in mapping_response.text.splitlines() if line.strip()]
            assert len(mapping) == result['summary']['frames']
            probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'frame=pts:stream=time_base', '-of', 'json', str(ROOT / record['inputPath'])],
                text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0))
            source_base = Fraction(probe['streams'][0]['time_base'])
            timestamps = [int(frame['pts']) * source_base for frame in probe['frames']]
            seen = []
            for frame_index, mapped in enumerate(mapping):
                assert mapped['frame_index'] == frame_index
                source_index = mapped['source_frame_index']
                assert isinstance(source_index, int) and 0 <= source_index < len(timestamps)
                assert int(mapped['source_pts']) * Fraction(mapped['source_time_base']) == timestamps[source_index]
                assert abs(mapped['timestamp_seconds'] - frame_index / result['summary']['fps']) < 1e-6
                seen.append(source_index)
            assert seen == sorted(seen)
            timing = result['mediaMetadata']['timeMapping']
            assert timing['uniqueMappedSourceFrames'] == len(set(seen))
            assert timing['duplicatedOutputFrames'] == len(seen) - len(set(seen))
            html = client.get(result['exports']['readableReport']).raise_for_status().text
            assert analysis_id in html and record['author'] in html and record['license'] in html
            assert 'share-alike' in html.lower()
            assert escape(current_source['modifications'], quote=True) in html
            json_export = client.get(base + '/download/report').raise_for_status().json()
            assert json_export == result
            csv_export = client.get(base + '/download/csv').raise_for_status().text
            csv_rows = list(csv.DictReader(io.StringIO(csv_export)))
            assert len(csv_rows) == result['summary']['frames']
            assert {row['media_modifications'] for row in csv_rows} == {current_source['modifications']}
            frames = result['detections']['frames']
            records.append({'id': record['id'], 'analysisId': analysis_id,
                'sourceSha256': record['sourceSha256'], 'inputSha256': record['inputSha256'],
                'frames': len(mapping), 'fps': result['summary']['fps'],
                'originalDecodedFrames': len(timestamps), 'unmappedOriginalFrames': len(timestamps) - len(set(seen)),
                'sourcePtsExactlyMatched': True, 'timeMapping': timing,
                'capabilities': result['summary']['capabilities'],
                'ballStates': dict(Counter(frame['ball']['state'] for frame in frames)),
                'warnings': result.get('warnings'), 'reportCreditsMatched': True,
                'catalogResultHtmlJsonCsvCreditsMatched': True,
                'currentModifications': current_source['modifications'],
                'historicalSourceUnchanged': validate_media_source(json.loads((directory / 'media_source.json').read_text(encoding='utf-8'))) == historical_source,
                'qualificationEvidence': False})
    output = args.report.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Keep verification receipts within this workspace.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'checkedAt': datetime.now(timezone.utc).isoformat(),
        'verification': 'CURATED_IDENTITY_LICENSE_EXPORT_AND_ORIGINAL_PTS; NOT_MODEL_ACCURACY',
        'demos': records}, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Verified {len(records)} curated analyses against their original decoded PTS and source credits.')
    print(output.relative_to(ROOT))


if __name__ == '__main__':
    main()
