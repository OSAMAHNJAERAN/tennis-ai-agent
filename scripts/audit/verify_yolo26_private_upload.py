"""Exercise real loopback upload HTTP; never start inference or qualify vision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/validation/yolo26_execution_2026-10-08/private_upload_http.json'
BASE = 'http://127.0.0.1:8002'
SOURCES = [ROOT / f'data/external/racketvision_training20/tennis/videos/match{n}_000.mp4' for n in (172, 175)]


def request(session, method, path, expected, **kwargs):
    response = session.request(method, BASE + path, timeout=30, **kwargs)
    if response.status_code != expected:
        raise AssertionError(f'{method} {path}: {response.status_code}, expected {expected}: {response.text[:400]}')
    return response.json()


def login(username):
    session = requests.Session()
    credentials = {'username': username, 'password': 'LocalOnly-Test-2026'}
    result = session.post(BASE + '/api/auth/login', json=credentials, timeout=30)
    if result.status_code == 401:
        request(session, 'POST', '/api/auth/register', 201, json=credentials)
    elif result.status_code != 200:
        raise AssertionError(f'Fixture login failed: {result.status_code}')
    return session


def main():
    started = time.time()
    health = request(requests.Session(), 'GET', '/api/health', 200)
    owner = login('local_browser_test')
    other = login('local_upload_other')
    batch = request(owner, 'POST', '/api/batches', 201, json={
        'mode': 'independent', 'files': [{'filename': p.name, 'size_bytes': p.stat().st_size, 'clip_index': i}
                                       for i, p in enumerate(SOURCES)]})
    receipt = {'kind': 'ACTUAL_LOOPBACK_HTTP_UPLOAD', 'started_unix': started,
               'worker_enabled': False, 'vision_qualification': False,
               'browser_file_chooser_tested': False, 'browser_file_chooser_limitation':
               'Chrome extension requires Allow access to file URLs; permission was not changed.',
               'server_health': health, 'batch_id': batch['id'], 'files': [], 'checks': []}
    for path, upload in zip(SOURCES, batch['uploads']):
        upload_id, offset, counts = upload['id'], 0, 0
        with path.open('rb') as source:
            while chunk := source.read(524288):
                current = request(owner, 'PATCH', f'/api/uploads/{upload_id}', 200,
                                  data=chunk, headers={'Content-Type': 'application/octet-stream', 'Upload-Offset': str(offset)})
                assert current['offset'] == offset + len(chunk)
                counts += 1
                offset = current['offset']
                if counts == 1:
                    request(owner, 'PATCH', f'/api/uploads/{upload_id}', 409, data=b'x',
                            headers={'Content-Type': 'application/octet-stream', 'Upload-Offset': '0'})
                    owner.close()
                    owner = login('local_browser_test')
                    resumed = request(owner, 'GET', f'/api/uploads/{upload_id}', 200)
                    assert resumed['offset'] == offset
                    request(other, 'GET', f'/api/uploads/{upload_id}', 404)
        job = request(owner, 'POST', f'/api/uploads/{upload_id}/complete', 202, json={})
        repeated = request(owner, 'POST', f'/api/uploads/{upload_id}/complete', 202, json={})
        assert repeated['id'] == job['id'] and job['status'] == 'UPLOADED'
        stored = request(owner, 'GET', f'/api/uploads/{upload_id}', 200)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert stored['checksum'] == digest and stored['offset'] == path.stat().st_size
        request(other, 'GET', f"/api/analyses/{job['id']}", 404)
        receipt['files'].append({'source': str(path), 'sha256': digest, 'bytes': offset, 'chunks': counts,
                                 'resumed_from_committed_offset': True, 'upload_id': upload_id, 'job_id': job['id']})
    status = request(owner, 'GET', f"/api/batches/{batch['id']}", 200)
    assert len(status['uploads']) == 2 and status['unified_report'] is None
    request(other, 'GET', f"/api/batches/{batch['id']}", 404)
    original_id = receipt['files'][0]['job_id']
    cancelled = request(owner, 'POST', f'/api/analyses/{original_id}/cancel', 200, json={})
    assert cancelled['status'] == 'FAILED' and cancelled['cancelled'] is True
    retry = request(owner, 'POST', f'/api/analyses/{original_id}/retry', 202, json={})
    assert retry['id'] != original_id and retry['retry_of'] == original_id and retry['status'] == 'UPLOADED'
    unchanged = request(owner, 'GET', f'/api/analyses/{original_id}', 200)
    assert unchanged['status'] == 'FAILED'
    receipt.update({'checks': ['actual_bytes_and_server_sha256', 'multiple_chunks', 'connection_relogin_resume',
                               'stale_offset_conflict', 'completion_idempotency', 'cross_account_404',
                               'independent_batch_no_invented_unified_report', 'cancel_and_new_id_retry_preserves_original'],
                    'retry_job_id': retry['id'], 'batch_status': status, 'elapsed_seconds': time.time() - started,
                    'status': 'PASS', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    OUT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'receipt': str(OUT), 'files': len(receipt['files']),
                      'bytes': sum(item['bytes'] for item in receipt['files']), 'elapsed_seconds': receipt['elapsed_seconds']}))


if __name__ == '__main__':
    main()
