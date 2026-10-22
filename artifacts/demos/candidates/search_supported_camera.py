"""Read public Commons records to shortlist camera-review candidates; download none."""
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
session = requests.Session()
session.headers['User-Agent'] = 'TennisAcademicProductDemo/1.0 (licensed camera suitability review)'
records = []
for query in ['tennis filetype:video -"table tennis" -"football tennis"', 'intitle:"US Open" filetype:video', 'intitle:Wimbledon filetype:video']:
    result = session.get('https://commons.wikimedia.org/w/api.php', params={
        'action': 'query', 'format': 'json', 'list': 'search', 'srnamespace': 6,
        'srsearch': query, 'srlimit': 50,
    }, timeout=(15, 60))
    result.raise_for_status()
    records.append({'query': query, 'results': result.json()})
out = ROOT / 'commons_camera_search.json'
out.write_text(json.dumps(records, indent=2), encoding='utf-8')
for record in records:
    print(record['query'])
    for row in record['results'].get('query', {}).get('search', []):
        print(json.dumps(row['title'], ensure_ascii=True))
