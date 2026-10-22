"""Bounded publisher text inventory; never fetch archives, media or targets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen


ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
MAX_BYTES = 2 * 1024 * 1024
REPO = 'rozumden/fmo-deblurring-benchmark'


def fetch(url, name):
    request = Request(url, headers={'User-Agent': 'TennisResearchMetadata/1.0',
                                   'Accept': 'application/json,text/plain,text/html'})
    started = datetime.now(timezone.utc).isoformat()
    with urlopen(request, timeout=20) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {'application/json', 'text/plain', 'text/html'}:
            raise ValueError('Refusing non-text response: ' + content_type)
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError('Publisher text exceeds 2 MiB bound')
        status, final_url = response.status, response.url
    text = body.decode('utf-8')
    if content_type == 'application/json':
        json.loads(text)
    with (ART / name).open('xb') as stream:
        stream.write(body)
    return {'requested_url': url, 'final_url': final_url, 'status': status,
            'acquired_utc': started, 'content_type': content_type, 'bytes': len(body),
            'sha256': hashlib.sha256(body).hexdigest(), 'path': (ART / name).relative_to(ROOT).as_posix()}


def main():
    if set(path.name for path in ART.iterdir()) != {'acquire_text.py'}:
        raise RuntimeError('Preserve existing inventory attempts; use a new directory')
    records, failures = [], []
    try:
        record = fetch('https://api.github.com/repos/' + REPO + '/commits/master', 'benchmark_commit.json')
        records.append(record)
        commit = json.loads((ART / 'benchmark_commit.json').read_text(encoding='utf-8'))['sha']
        if re.fullmatch('[0-9a-f]{40}', commit) is None:
            raise ValueError('Invalid publisher Git commit')
        urls = [('https://cmp.felk.cvut.cz/fmo/', 'project_page.html'),
            ('https://api.github.com/repos/' + REPO + '/git/trees/' + commit + '?recursive=1', 'benchmark_tree.json')]
        urls.extend(('https://raw.githubusercontent.com/' + REPO + '/' + commit + '/' + name, name)
                    for name in ('README.md', 'LICENSE', 'download_datasets.sh'))
        for url, name in urls:
            try:
                records.append(fetch(url, name))
            except Exception as error:
                failures.append({'url': url, 'error_type': type(error).__name__, 'error': str(error)})
    except Exception as error:
        commit = None
        failures.append({'stage': 'commit_inventory', 'error_type': type(error).__name__, 'error': str(error)})
    receipt = {'schema_version': 1, 'status': 'TEXT_METADATA_ACQUIRED' if not failures else 'PARTIAL_OR_ACCESS_FAILURE',
        'publisher_repository': REPO, 'pinned_commit': commit, 'records': records, 'failures': failures,
        'limits': {'max_response_bytes': MAX_BYTES, 'raw_media_archives_or_annotations_acquired': False,
                   'test_frames_or_targets_exposed': False, 'code_downloaded_but_never_executed': True,
                   'models_training_or_ram_gpu_probe': False, 'data_license_or_metric_speed_reference_verified': False}}
    with (ART / 'acquisition_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'commit': commit, 'text_files': len(records),
                      'bytes': sum(value['bytes'] for value in records), 'failures': failures}, indent=2))


if __name__ == '__main__':
    main()
