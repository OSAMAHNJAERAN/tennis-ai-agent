"""Acquire bounded public publisher metadata, without viewing CalTennis frames."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


REPO = 'demalenk/caltennis'
ROOT = Path(__file__).resolve().parents[2]


def acquire(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Choose a fresh metadata directory inside this workspace')
    def read(url, limit):
        request = urllib.request.Request(url, headers={'User-Agent': 'TennisAcademicMetadataAudit/1.0'})
        with urllib.request.urlopen(request, timeout=40) as response:
            content = response.read(limit + 1)
        if len(content) > limit:
            raise ValueError('Publisher metadata exceeded the declared transfer limit')
        return content
    publisher = read(f'https://huggingface.co/api/datasets/{REPO}', 4 * 1024 ** 2)
    metadata = json.loads(publisher)
    revision = metadata.get('sha', '')
    if (not re.fullmatch('[0-9a-f]{40}', revision) or metadata.get('gated') not in (False, None)
            or metadata.get('cardData', {}).get('license') != 'cc-by-nc-4.0'):
        raise ValueError('Public publisher revision/license changed or access is gated')
    output.mkdir(parents=True)
    (output / 'publisher_metadata.json').write_bytes(publisher)
    receipt = dict(schema_version=1, status='ACQUIRED_PUBLISHER_METADATA_ONLY', dataset=REPO,
                   revision=revision, license='CC BY-NC 4.0', use_scope='PRIVATE_NONCOMMERCIAL_ACADEMIC_REFERENCE_DISCOVERY',
                   created_utc=datetime.now(timezone.utc).isoformat(), frames_viewed=0, model_inference=False,
                   production_qualification=False, metadata_files=[], actual_media_acquired=0)
    names = {entry['rfilename'] for entry in metadata.get('siblings', [])}
    for name in ('README.md', 'metadata_mini.jsonl', 'metadata_mid.jsonl'):
        if name not in names:
            raise ValueError('Expected publisher index is unavailable; no substitution')
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}'
        content = read(url, 5 * 1024 ** 2)
        (output / name).write_bytes(content)
        receipt['metadata_files'].append(dict(path=name, source_url=url, bytes=len(content),
                                              sha256=hashlib.sha256(content).hexdigest()))
    rows = [json.loads(line) for line in (output / 'metadata_mini.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    sessions = sorted({row['session_id'] for row in rows})
    if not sessions:
        raise ValueError('Publisher mini index contains no original sessions')
    receipt['mini_index_rows'] = len(rows)
    receipt['mini_original_session_groups'] = sessions
    receipt['publisher_metadata_sha256'] = hashlib.sha256(publisher).hexdigest()
    receipt['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt['limits'] = ['Metadata only; source angles and native timing not yet verified',
                         'Multi-view consistency and supplied calibration are not independent human ground contacts',
                         'No tennis ball speed reference verified', 'No raw media or public example acquired']
    (output / 'acquisition_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire(args.output), indent=2))
