"""Bounded official indoor DEV pair acquisition from retained file ancestry."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
INVENTORY = ART.parent/'nkust_indoor_inventory_v1'
TARGET = ROOT/'data/external/nkust_landing_reference_indoor_development_v4'
RESERVATION = ROOT/'data/yolo26/nkust_reference_reservation.json'
EXPECTED = {
    'IST01.mp4': ('indoor_straight_top_files_v4.json', 'Top-View', 6430268,
        '1c457c85355f7b29af814f33f18fcd5ae617ad93317a165081dae3276f119b9b'),
    'ISS01.mp4': ('indoor_straight_side_files_v4.json', 'Side-View', 6755013,
        'ebf721e1502143a5df207383ddd4649bc86ef7cf252d650f9f0e7e9f93a940fe'),
}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def ref(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}

reservation = json.loads(RESERVATION.read_text(encoding='utf-8'))
assert reservation['publisher_license'] == 'CC-BY-4.0'
assert any(row['group_id'] == 'nkust-2023-03-02-indoor-session'
           and row['partition'] == 'reserved_reference_development' for row in reservation['groups'])
folders_path = INVENTORY/'folders_v4.json'
folders = {row['id']: row for row in json.loads(folders_path.read_text(encoding='utf-8'))}
selected = []
for filename, (inventory_name, view, size, digest) in EXPECTED.items():
    inventory_path = INVENTORY/inventory_name
    row, = [item for item in json.loads(inventory_path.read_text(encoding='utf-8'))
            if item['filename'] == filename]
    details = row['content_details']
    assert row['size'] == details['size'] == size < 20*1024**2
    assert details['sha256_hash'] == digest and details['content_type'] == 'video/mp4'
    ancestry, folder = [], folders[row['folder_id']]
    while True:
        ancestry.append(folder['name'])
        if 'parent_id' not in folder:
            break
        folder = folders[folder['parent_id']]
    assert ancestry == [view, 'Straight Shot', 'Indoor Field']
    url = details['download_url']
    assert url == 'https://data.mendeley.com/public-files/datasets/75m8vz7jr2/files/'+row['id']+'/file_downloaded'
    selected.append({'file_id': row['id'], 'filename': filename, 'publisher_bytes': size,
        'publisher_sha256': digest, 'download_url': url, 'ancestry': ancestry,
        'inventory': ref(inventory_path)})
selection = {'selected_before_download': selected, 'folders': ref(folders_path),
    'reservation': ref(RESERVATION), 'pairing_basis': 'Publisher matching category/view sequence number only; no exact synchronization or CSV join inferred'}
with (ART/'media_selection.json').open('x', encoding='utf-8') as stream:
    json.dump(selection, stream, indent=2)
TARGET.mkdir(exist_ok=True)
assert TARGET.resolve().is_relative_to(ROOT/'data/external')
receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'STARTED',
    'dataset_url': reservation['source'], 'publisher_license': reservation['publisher_license'],
    'group_id': 'nkust-2023-03-02-indoor-session', 'partition': 'reserved_reference_development',
    'attribution': 'Lai, Kalin Guanlun; Huang, Hsu-Chun; Lin, Shang-Yi; Lin, Wei-Ting; Lin, Kawuu Weicheng (2024), Tennis Shot Side-View and Top-View Data Set for Player Analysis in Tennis, Mendeley Data V4, DOI 10.17632/75m8vz7jr2.4.',
    'selection': ref(ART/'media_selection.json'), 'files': [], 'test_files_accessed': False,
    'frames_rendered': False, 'model_or_training_executed': False, 'accuracy_computed': False}
try:
    for row in selected:
        destination = TARGET/row['filename']
        temporary = destination.with_suffix('.download.part')
        assert not destination.exists() and not temporary.exists(), 'Preserve existing source bytes'
        request = urllib.request.Request(row['download_url'],
            headers={'User-Agent': 'TennisAcademicDataAcquisition/1.0'})
        count, digest = 0, hashlib.sha256()
        with urllib.request.urlopen(request, timeout=40) as response, temporary.open('xb') as stream:
            assert response.status == 200
            final_url = response.geturl()
            while block := response.read(256*1024):
                count += len(block)
                assert count <= row['publisher_bytes'], 'Unexpected oversized response'
                stream.write(block)
                digest.update(block)
        assert count == row['publisher_bytes'] and digest.hexdigest() == row['publisher_sha256']
        temporary.rename(destination)
        receipt['files'].append({**row, 'path': str(destination.relative_to(ROOT)),
            'bytes': count, 'sha256': sha(destination), 'http_status': 200, 'final_url': final_url})
        print(json.dumps({'filename': row['filename'], 'bytes': count, 'hash_matches_publisher': True}), flush=True)
    for row in selected:
        assert sha(ROOT/row['inventory']['path']) == row['inventory']['sha256']
    assert sha(folders_path) == selection['folders']['sha256']
    assert sha(RESERVATION) == selection['reservation']['sha256']
    receipt['status'] = 'ACQUIRED_PUBLISHER_HASH_VERIFIED_INDOOR_DEVELOPMENT_PAIR'
except BaseException as error:
    receipt.update(status='FAILED_OR_PARTIAL_ACQUISITION_RETAIN_DIAGNOSTICS',
        error=type(error).__name__+': '+str(error))
    raise
finally:
    receipt['runner_sha256'] = sha(Path(__file__))
    (ART/'media_acquisition_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'verified_files': len(receipt['files'])}), flush=True)
