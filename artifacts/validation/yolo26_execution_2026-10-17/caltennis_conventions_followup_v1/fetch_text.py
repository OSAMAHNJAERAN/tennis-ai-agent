"""Bounded primary text capture only. Downloaded source is never executed."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = Path(__file__).resolve().parent
requests_path = BASE/sys.argv[1]
requests = json.loads(requests_path.read_text(encoding='utf-8'))
destination = BASE/requests_path.stem
destination.mkdir(exist_ok=False)
MAX_BYTES = 524288
ALLOWED = {'api.github.com', 'raw.githubusercontent.com', 'huggingface.co', 'arxiv.org', 'ilonadem.github.io'}
assert len(requests) <= 12
records = []
for item in requests:
    url = item['url']
    parsed = urllib.parse.urlparse(url)
    assert parsed.scheme == 'https' and parsed.hostname in ALLOWED
    assert not parsed.path.lower().endswith(('.mp4', '.mov', '.avi', '.jpg', '.png', '.jpeg', '.npy', '.npz', '.zip', '.tar', '.gz', '.pdf'))
    name = item['name']
    assert Path(name).name == name
    record = {'requested_url': url, 'checked_utc': datetime.now(timezone.utc).isoformat(), 'name': name,
              'max_response_bytes': MAX_BYTES}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'CalTennis-private-academic-conventions-audit',
                                                      'Accept': 'application/json,text/plain,text/html'})
        with urllib.request.urlopen(request, timeout=25) as response:
            record.update(status=response.status, final_url=response.url,
                          content_type=response.headers.get('Content-Type'), etag=response.headers.get('ETag'))
            assert urllib.parse.urlparse(response.url).hostname in ALLOWED
            content = response.read(MAX_BYTES+1)
            assert len(content) <= MAX_BYTES, 'Bounded text response exceeded cap'
            content.decode('utf-8')
            (destination/name).write_bytes(content)
            record.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    except Exception as error:
        record.update(error=type(error).__name__+': '+str(error), status=getattr(error, 'code', record.get('status')))
    records.append(record)
receipt = {'scope': 'BOUNDED_PRIMARY_METADATA_AND_TEXT_ONLY; NO_MEDIA_TESTFRAMES_SOURCE_EXECUTION_OR_INFERENCE',
           'finished_utc': datetime.now(timezone.utc).isoformat(), 'requests': records,
           'fetch_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'frames_acquired': 0, 'frames_viewed': 0, 'downloaded_source_executed': False,
           'qualification_evidence': False}
(destination/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
