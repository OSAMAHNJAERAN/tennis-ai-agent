"""Acquire only the two requested Commons originals into a separate review area."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
FILES = [
    ('alcaraz_wu_2026_06', 'Carlos_Alcaraz_vs._Wu_Yibing_(2026_US_Open)_-_06.webm'),
    ('sabalenka_zheng_2024_20', 'Aryna_Sabalenka_vs._Qinwen_Zheng_in_a_quarterfinals_of_the_2024_US_Open_-_20.webm'),
]

def digest(path, algorithm):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    session = requests.Session()
    session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (licensed camera suitability review)'
    prior_path = ROOT / 'candidate_sources.json'
    prior_records = json.loads(prior_path.read_text(encoding='utf-8')).get('candidates', []) if prior_path.exists() else []
    records = []
    for identity, filename in FILES:
        response = session.get('https://commons.wikimedia.org/w/api.php', params={
            'action': 'query', 'format': 'json', 'titles': 'File:' + filename,
            'prop': 'imageinfo', 'iiprop': 'url|size|sha1|timestamp|extmetadata',
        }, timeout=(15, 60))
        response.raise_for_status()
        page = next(iter(response.json()['query']['pages'].values()))
        info = page['imageinfo'][0]
        if not info['url'].startswith('https://upload.wikimedia.org/wikipedia/commons/'):
            raise ValueError('Publisher file URL is outside Wikimedia Commons.')
        if not 0 < info['size'] <= 150_000_000:
            raise ValueError('Candidate is outside the declared acquisition byte bound.')
        metadata = {key: value['value'] for key, value in info.get('extmetadata', {}).items()}
        if metadata.get('LicenseShortName') != 'CC BY-SA 4.0':
            raise ValueError('Candidate license differs from the requested source license.')
        record = {'id': identity, 'fileTitle': page['title'], 'sourcePage': info['descriptionurl'],
                  'sourceUrl': info['url'], 'sourceBytes': info['size'], 'sourceSha1': info['sha1'],
                  'publishedRevisionTimestamp': info['timestamp'], 'width': info['width'], 'height': info['height'],
                  'author': 'Oleg Yunakov', 'license': metadata['LicenseShortName'],
                  'licenseUrl': metadata['LicenseUrl'], 'attribution': 'Oleg Yunakov / CC BY-SA 4.0',
                  'checkedAt': datetime.now(timezone.utc).isoformat(), 'publisherMetadata': metadata,
                  'qualificationEvidence': False, 'status': 'METADATA_REVIEW_ONLY'}
        target = ROOT / (identity + '.webm')
        if args.download:
            if not target.exists():
                temporary = target.with_suffix('.webm.part')
                with session.get(info['url'], stream=True, timeout=(15, 90)) as source:
                    source.raise_for_status()
                    size = 0
                    with temporary.open('wb') as output:
                        for chunk in source.iter_content(1024 * 1024):
                            size += len(chunk)
                            if size > info['size']:
                                raise ValueError('Candidate exceeded the publisher-pinned byte size.')
                            output.write(chunk)
                if temporary.stat().st_size != info['size'] or digest(temporary, 'sha1') != info['sha1']:
                    raise ValueError('Downloaded candidate does not match the current publisher SHA-1/byte size.')
                temporary.replace(target)
            if target.stat().st_size != info['size'] or digest(target, 'sha1') != info['sha1']:
                raise ValueError('Retained candidate does not match the publisher file revision.')
            record.update(sourcePath=str(target), sourceSha256=digest(target, 'sha256'),
                          downloadedBytes=target.stat().st_size, status='ACQUIRED_NOT_ANALYZED')
        old = next((x for x in prior_records if x.get('id') == identity), None)
        if old and record.get('sourceSha256') and old.get('sourceSha256') == record['sourceSha256']:
            for key in ['cameraReview', 'cameraSuitability', 'status', 'modifications', 'publicDisplayLicenseAvailable']:
                if key in old:
                    record[key] = old[key]
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != 'publisherMetadata'}), flush=True)
    (ROOT / 'candidate_sources.json').write_text(json.dumps({'schemaVersion': 1, 'purpose': 'Camera suitability review only; no queue or public manifest change.', 'candidates': records}, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
