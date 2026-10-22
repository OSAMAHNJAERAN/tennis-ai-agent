"""Bounded publisher-verified indoor DEV CSV acquisition; never opens TEST."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
RESERVATION = ROOT/'data/yolo26/nkust_reference_reservation.json'
TARGET = ROOT/'data/external/nkust_landing_reference_indoor_development_v4'
URL = 'https://data.mendeley.com/public-files/datasets/75m8vz7jr2/files/9991b6fc-1eea-476e-83ac-a0af8ef424e5/file_downloaded'
EXPECTED = '6ea7649134718c692f31aacf55095e1797aec673b2905a7e1215aae21ba3b53d'
SIZE = 1411

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

reservation = json.loads(RESERVATION.read_text(encoding='utf-8'))
assert reservation['publisher_license'] == 'CC-BY-4.0'
assert any(row['group_id'] == 'nkust-2023-03-02-indoor-session'
           and row['partition'] == 'reserved_reference_development' for row in reservation['groups'])
inventory = ART.parent/'nkust_indoor_inventory_v1/root_files_v4.json'
assert sha(inventory) == '6be70b218b5ae16079523a9dddb8d1223f7fd0c0ea49e2946e7bc267822328b9'
record, = [row for row in json.loads(inventory.read_text(encoding='utf-8'))
           if row['filename'] == 'Indoor Field - Straight Shot.csv']
assert record['content_details']['download_url'] == URL
assert record['content_details']['sha256_hash'] == EXPECTED and record['size'] == SIZE
TARGET.mkdir(exist_ok=True)
destination = TARGET/'Indoor Field - Straight Shot.csv'
assert not destination.exists(), 'Preserve existing acquisition'
receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'STARTED',
    'dataset_url': reservation['source'], 'source_doi': reservation['source_doi'],
    'publisher_license': reservation['publisher_license'], 'group_id': 'nkust-2023-03-02-indoor-session',
    'partition': 'reserved_reference_development',
    'reservation_path': str(RESERVATION.relative_to(ROOT)), 'reservation_sha256': sha(RESERVATION),
    'file_id': '9991b6fc-1eea-476e-83ac-a0af8ef424e5', 'publisher_filename': destination.name,
    'publisher_sha256': EXPECTED, 'publisher_bytes': SIZE, 'url': URL,
    'metadata_evidence': {'path': str(inventory.relative_to(ROOT)), 'sha256': sha(inventory)},
    'test_files_accessed': False, 'model_or_training_executed': False, 'accuracy_computed': False}
try:
    request = urllib.request.Request(URL, headers={'User-Agent': 'TennisAcademicDataAcquisition/1.0'})
    with urllib.request.urlopen(request, timeout=40) as response:
        receipt['http_status'] = response.status
        receipt['final_url'] = response.geturl()
        assert response.status == 200
        raw = response.read(SIZE+1)
    assert len(raw) == SIZE, 'Publisher file size differs'
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Publisher SHA256 differs'
    with destination.open('xb') as stream:
        stream.write(raw)
    receipt.update(status='ACQUIRED_HASH_VERIFIED_DEVELOPMENT_ONLY',
        path=str(destination.relative_to(ROOT)), bytes=len(raw), sha256=sha(destination),
        target_values_inspected=False, clip_to_csv_join_established=False)
except BaseException as error:
    receipt.update(status='FAILED_NO_VERIFIED_ACQUISITION', error=type(error).__name__+': '+str(error))
    raise
finally:
    receipt['runner_sha256'] = sha(Path(__file__))
    (ART/'csv_acquisition_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('status', 'publisher_filename', 'publisher_bytes')}))
