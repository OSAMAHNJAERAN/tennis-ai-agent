"""Acquire one camera-review source from a pinned Commons metadata record."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
TITLE = 'File:Auto curation for sports highlights for Wimbledon and US Open 2017.webm'
records = json.loads((ROOT / 'camera_metadata_shortlist.json').read_text(encoding='utf-8'))
info = next(x['imageinfo'] for x in records if x['title'] == TITLE)
metadata = {k: v['value'] for k, v in info.get('extmetadata', {}).items()}
if metadata.get('LicenseShortName') != 'CC BY 3.0' or not info['url'].startswith('https://upload.wikimedia.org/wikipedia/commons/') or not 0 < info['size'] < 30_000_000:
    raise ValueError('Pinned source is outside the declared license, publisher, or size bounds')
session = requests.Session()
session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (licensed camera suitability review)'
session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])))
target = ROOT / 'ibm_tennis_highlights_2017.webm'
def digest(path, algorithm):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()
if not target.exists():
    temporary = target.with_suffix('.webm.part')
    with session.get(info['url'], stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        size = 0
        with temporary.open('wb') as stream:
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                if size > info['size']:
                    raise ValueError('Downloaded source exceeded the pinned byte size')
                stream.write(chunk)
    if temporary.stat().st_size != info['size'] or digest(temporary, 'sha1') != info['sha1']:
        raise ValueError('Downloaded source differs from publisher revision')
    temporary.replace(target)
if target.stat().st_size != info['size'] or digest(target, 'sha1') != info['sha1']:
    raise ValueError('Retained source differs from publisher revision')
record = {'id': 'ibm_tennis_highlights_2017', 'fileTitle': TITLE, 'sourcePage': info['descriptionurl'], 'sourceUrl': info['url'], 'sourceBytes': info['size'], 'sourceSha1': info['sha1'], 'sourceSha256': digest(target, 'sha256'), 'publishedRevisionTimestamp': info['timestamp'], 'author': 'IBM Research', 'license': 'CC BY 3.0', 'licenseUrl': metadata['LicenseUrl'], 'attribution': 'IBM Research / CC BY 3.0', 'shareAlikeRequired': False, 'publicDisplayLicenseAvailable': None, 'licenseVerificationStatus': 'EXTERNAL_SOURCE_LICENSE_REVIEW_PENDING', 'modifications': 'None; publisher original retained for camera review.', 'sourcePath': str(target), 'publisherMetadata': metadata, 'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'ACQUIRED_CAMERA_REVIEW_PENDING', 'qualificationEvidence': False, 'cameraSuitability': None}
manifest = ROOT / 'baseline_source_candidates.json'
if manifest.exists():
    old = next((x for x in json.loads(manifest.read_text(encoding='utf-8')).get('candidates', []) if x.get('id') == record['id']), None)
    if old and old.get('sourceSha256') == record['sourceSha256']:
        for key in ['cameraReview', 'cameraSuitability', 'status', 'publicDisplayLicenseAvailable', 'licenseVerificationStatus']:
            if key in old:
                record[key] = old[key]
manifest.write_text(json.dumps({'schemaVersion': 1, 'purpose': 'Camera suitability review only; no inference, queue, or public manifest change.', 'candidates': [record]}, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in record.items() if k != 'publisherMetadata'}, ensure_ascii=True))
