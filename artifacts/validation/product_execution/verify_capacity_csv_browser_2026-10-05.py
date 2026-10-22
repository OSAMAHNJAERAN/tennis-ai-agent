"""Validate the actual browser CSV against the original serializer and live analysis."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
IDENTITY = 'match_9de7cb2d09eb46a0af0e34c3b0e6b818'
download = Path('C:/Users/ac-98/Downloads') / f'{IDENTITY}_csv.csv'
owned = HERE / 'capacity_browser_downloaded_csv_2026-10-05.csv'
if owned.exists():
    assert owned.read_bytes() == download.read_bytes()
else:
    shutil.copyfile(download, owned)
with urllib.request.urlopen(f'http://127.0.0.1:8000/api/analyses/{IDENTITY}/result', timeout=30) as response:
    result = json.load(response)
spec = importlib.util.spec_from_file_location('dated_csv_baseline', HERE / 'reporting_before_streamed_csv_capacity_2026-10-05.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
expected = legacy.telemetry_csv(result).encode('utf-8')
actual = owned.read_bytes()
assert actual == expected
rows = list(csv.DictReader(io.StringIO(actual.decode('utf-8'))))
assert len(rows) == result['summary']['frames'] == 469
assert [int(row['frame']) for row in rows] == list(range(469))
assert all(row['analysis_id'] == IDENTITY and row['media_author'] == 'JC' and row['media_license'] == 'CC BY-SA 2.0' for row in rows)
response_hash = hashlib.sha256()
chunks = 0
with urllib.request.urlopen(f'http://127.0.0.1:8000/api/analyses/{IDENTITY}/download/csv', timeout=30) as response:
    headers = dict(response.headers)
    while block := response.read(65536):
        response_hash.update(block)
        chunks += 1
assert response_hash.hexdigest() == hashlib.sha256(actual).hexdigest()
assert headers.get('Transfer-Encoding') == 'chunked' and 'Content-Length' not in headers
report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'scope': 'Actual browser-downloaded saved example CSV plus live HTTP framing and original-serializer equivalence; software evidence only.',
          'analysisId': IDENTITY, 'browserDownload': str(download), 'ownedCopy': owned.relative_to(ROOT).as_posix(),
          'bytes': len(actual), 'sha256': hashlib.sha256(actual).hexdigest(), 'rows': len(rows),
          'legacySerializerSha256': hashlib.sha256((HERE / 'reporting_before_streamed_csv_capacity_2026-10-05.py').read_bytes()).hexdigest(),
          'firstFrame': rows[0]['frame'], 'lastFrame': rows[-1]['frame'],
          'transferEncoding': headers.get('Transfer-Encoding'), 'contentLengthPresent': 'Content-Length' in headers,
          'readChunks': chunks, 'qualificationEvidence': False}
with (HERE / 'capacity_browser_csv_verified_2026-10-05.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2)
print(json.dumps(report))
manifest = json.loads((ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json').read_text())
print(json.dumps({'manifestKeys': list(manifest), 'verification': manifest.get('verification'),
                  'restorationStatus': manifest.get('restorationStatus')}))
