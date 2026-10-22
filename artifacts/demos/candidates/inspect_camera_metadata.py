"""Read pinned primary metadata and poster images; no source-video downloads."""
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
FILES = [
    '2018 Davis Cup Americas Zone - Uruguay vs Mexico - 03.webm',
    '2018 Davis Cup Americas Zone - Uruguay vs Mexico - 04.webm',
    '2018 Davis Cup Americas Zone - Uruguay vs Mexico - 05.webm',
    '2018 Davis Cup Americas Zone - Uruguay vs Mexico - 06.webm',
    '2018 Davis Cup Americas Zone - Uruguay vs Mexico - 07.webm',
    'Anna Kalinskaya Qualifying for Wimbledon 2019.webm',
    'Anna Kalinskaya Wimbledon Qualifying 2019.webm',
    'Auto curation for sports highlights for Wimbledon and US Open 2017.webm',
]
session = requests.Session()
session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (licensed camera suitability review)'
records = []
for title in FILES:
    response = session.get('https://commons.wikimedia.org/w/api.php', params={
        'action': 'query', 'format': 'json', 'titles': 'File:' + title,
        'prop': 'imageinfo', 'iiprop': 'url|size|sha1|timestamp|extmetadata', 'iiurlwidth': 640,
    }, timeout=(15, 60))
    response.raise_for_status()
    page = next(iter(response.json()['query']['pages'].values()))
    info = page.get('imageinfo', [{}])[0]
    metadata = {k: v['value'] for k, v in info.get('extmetadata', {}).items()}
    record = {'title': page['title'], 'checkedAt': datetime.now(timezone.utc).isoformat(), 'imageinfo': info}
    records.append(record)
    print(json.dumps({'title': page['title'], 'bytes': info.get('size'), 'width': info.get('width'), 'height': info.get('height'), 'license': metadata.get('LicenseShortName'), 'author': metadata.get('Artist'), 'sourcePage': info.get('descriptionurl'), 'posterUrl': info.get('thumburl')}, ensure_ascii=True), flush=True)
    poster_url = info.get('thumburl')
    if poster_url and poster_url.startswith(('https://upload.wikimedia.org/', 'https://thumb.wikimedia.org/')):
        target = ROOT / 'posters' / (str(len(records)) + '.jpg')
        target.parent.mkdir(exist_ok=True)
        poster = session.get(poster_url, timeout=(15, 60))
        if poster.ok and len(poster.content) < 2_000_000:
            target.write_bytes(poster.content)
            record['posterPath'] = str(target)
(ROOT / 'camera_metadata_shortlist.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
