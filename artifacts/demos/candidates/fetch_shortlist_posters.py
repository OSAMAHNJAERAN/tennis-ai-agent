"""Fetch only already-pinned Commons poster URLs, without new API queries."""
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT / 'camera_metadata_shortlist.json').read_text(encoding='utf-8'))
session = requests.Session()
session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (licensed camera suitability review)'
session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])))
for number, record in enumerate(records, 1):
    target = ROOT / 'posters' / (str(number) + '.jpg')
    if target.exists():
        continue
    url = record['imageinfo'].get('thumburl')
    if not url or not url.startswith(('https://upload.wikimedia.org/', 'https://thumb.wikimedia.org/')):
        continue
    response = session.get(url, timeout=(15, 60))
    response.raise_for_status()
    if len(response.content) >= 2_000_000 or not response.headers.get('content-type', '').startswith('image/'):
        raise ValueError('Unexpected poster payload')
    target.write_bytes(response.content)
    print(number, response.status_code, len(response.content), flush=True)
