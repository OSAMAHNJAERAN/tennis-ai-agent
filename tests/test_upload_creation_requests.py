"""Durable initial upload/batch acceptance without media inference."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import os
import sqlite3
import subprocess
import sys
from threading import Barrier

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.creation_requests import CreationRequests
from src.api.jobs import Settings

HEADER = {'Idempotency-Key': 'creation-request-test-0001'}
FILE = {'filename': 'first.mp4', 'size_bytes': 32}
BATCH = {'mode': 'same_match', 'files': [{**FILE, 'clip_index': 1},
    {'filename': 'second.mp4', 'size_bytes': 32, 'clip_index': 0}],
    'metadata': {'tournament': 'Owner selected match'}}


@pytest.fixture
def service(tmp_path):
    settings = Settings(tmp_path/'storage', tmp_path/'config', max_upload_bytes=64,
                        max_owner_pending_jobs=2, max_owner_storage_bytes=256)
    app = create_app(settings, start_worker=False)
    with TestClient(app) as client:
        yield client, app, settings


@pytest.mark.parametrize('route,payload', [('/api/uploads', FILE), ('/api/batches', BATCH)])
@pytest.mark.parametrize('headers', [{}, {'Idempotency-Key': 'short'}, {'Idempotency-Key': 'x'*129},
    [('Idempotency-Key', 'creation-request-test-0001'), ('Idempotency-Key', 'creation-request-test-0001')]])
def test_creation_requires_one_explicit_bounded_key(service, route, payload, headers):
    client, app, _ = service
    assert client.post(route, headers=headers, json=payload).status_code == 422
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM upload_creation_requests').fetchone()[0] == 0


def test_batch_metadata_sessions_and_ack_commit_together_with_original_order(service):
    client, app, settings = service
    response = client.post('/api/batches', headers=HEADER, json=BATCH)
    assert response.status_code == 201, response.text
    value = response.json()
    assert [s['filename'] for s in value['uploads']] == ['first.mp4', 'second.mp4']
    assert [s['clip_index'] for s in value['uploads']] == [1, 0]
    with app.state.store.connect() as conn:
        matches = conn.execute('SELECT metadata FROM matches').fetchall()
        assert len(matches) == 2
        assert all(json.loads(row['metadata'])['tournament'] == 'Owner selected match' for row in matches)
    # Replaying through new service settings must not validate new admission or
    # allocate new drafts. Return current session state in its original order.
    first = value['uploads'][0]['id']
    with app.state.store.connect() as conn:
        conn.execute("UPDATE upload_sessions SET offset=13,status='CANCELLED' WHERE id=?", (first,))
    with TestClient(create_app(replace(settings, max_upload_bytes=12, max_owner_pending_jobs=1,
                                      max_owner_storage_bytes=1), start_worker=False)) as restarted:
        repeated = restarted.post('/api/batches', headers=HEADER, json=BATCH)
    assert repeated.status_code == 201
    assert repeated.json()['id'] == value['id']
    assert [s['id'] for s in repeated.json()['uploads']] == [s['id'] for s in value['uploads']]
    assert repeated.json()['uploads'][0]['status'] == 'CANCELLED'
    assert repeated.json()['uploads'][0]['offset'] == 13
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == 2
        assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == 2


def test_same_key_rejects_changed_metadata_files_mode_and_kind_without_mutation(service):
    client, app, _ = service
    assert client.post('/api/batches', headers=HEADER, json=BATCH).status_code == 201
    for changed in [{**BATCH, 'metadata': {'tournament': 'Different'}},
                    {**BATCH, 'mode': 'independent'}, {**BATCH, 'files': BATCH['files'][::-1]}]:
        assert client.post('/api/batches', headers=HEADER, json=changed).status_code == 409
    assert client.post('/api/uploads', headers=HEADER, json=FILE).status_code == 409
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == 2


def test_session_replay_precedes_consumed_draft_validation(service):
    client, app, _ = service
    draft = client.post('/api/matches', json={}).json()['id']
    payload = {**FILE, 'match_id': draft}
    first = client.post('/api/uploads', headers=HEADER, json=payload)
    assert first.status_code == 201
    session = first.json()
    ident = session['id']
    assert client.patch('/api/uploads/'+ident, headers={'Upload-Offset': '0'},
                        content=b'\0\0\0\x18ftypisom'+b'x'*20).status_code == 200
    accepted = client.post('/api/uploads/'+ident+'/complete')
    assert accepted.status_code == 202
    again = client.post('/api/uploads', headers=HEADER, json=payload)
    assert again.status_code == 201 and again.json()['id'] == ident
    assert again.json()['analysis_id'] == accepted.json()['id']
    assert len(app.state.store.list()) == 1


def test_atomic_creation_rollback_has_no_batch_session_draft_or_ack(service, monkeypatch):
    client, app, _ = service
    save = CreationRequests.save

    def reject(*args):
        save(*args)
        raise OSError('simulated failure before transaction commit')

    monkeypatch.setattr(CreationRequests, 'save', reject)
    with pytest.raises(OSError):
        client.post('/api/batches', headers=HEADER, json=BATCH)
    with app.state.store.connect() as conn:
        for table in ('upload_sessions', 'upload_batches', 'matches', 'upload_creation_requests'):
            assert conn.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] == 0
    monkeypatch.setattr(CreationRequests, 'save', save)
    assert client.post('/api/batches', headers=HEADER, json=BATCH).status_code == 201


def test_parallel_same_key_creation_uses_one_batch_and_one_quota_reservation_set(service):
    _, app, _ = service
    barrier = Barrier(2)

    def send():
        with TestClient(app) as client:
            barrier.wait(timeout=5)
            response = client.post('/api/batches', headers=HEADER, json=BATCH)
            assert response.status_code == 201, response.text
            return response.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        work = [pool.submit(send) for _ in range(2)]
        first, second = [future.result(timeout=20) for future in work]
    assert first == second
    with TestClient(app) as client:
        usage = client.get('/api/usage').json()
    assert usage['pending_jobs'] == 2 and usage['committed_bytes'] == 64


def test_owner_isolation_and_cross_owner_profile_metadata_are_enforced(tmp_path):
    app = create_app(Settings(tmp_path/'storage', tmp_path/'config', require_auth=True), start_worker=False)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        for client, name in [(alice, 'alice'), (bob, 'bob')]:
            assert client.post('/api/auth/register', json={'username': name, 'password': 'creation-test-password-123'}).status_code == 201
        profile = alice.post('/api/players', json={'name': 'Private player'}).json()['id']
        malicious = {**BATCH, 'metadata': {'participants': [{'profile_id': profile}, {}]}}
        assert bob.post('/api/batches', headers=HEADER, json=malicious).status_code == 422
        assert guest.post('/api/batches', headers=HEADER, json=BATCH).status_code == 401
        one = alice.post('/api/batches', headers=HEADER, json=BATCH)
        two = bob.post('/api/batches', headers=HEADER, json=BATCH)
        assert one.status_code == two.status_code == 201
        assert one.json()['id'] != two.json()['id']
        assert bob.get('/api/batches/'+one.json()['id']).status_code == 404


@pytest.mark.parametrize('failure', ['before', 'after'])
@pytest.mark.parametrize('kind', ['upload', 'batch'])
def test_actual_creation_process_exit_is_all_or_none_and_replay_does_not_duplicate(service, tmp_path, failure, kind):
    client, app, settings = service
    record_path = tmp_path/('creation_'+kind+'_'+failure+'.json')
    route, payload = ('/api/uploads', FILE) if kind == 'upload' else ('/api/batches', BATCH)
    request_path = tmp_path/'creation_request.json'
    request_path.write_text(json.dumps({'key': HEADER['Idempotency-Key'], 'payload': payload}), encoding='utf-8')
    child = Path(__file__).parent/'helpers/upload_creation_crash_child.py'
    result = subprocess.run([sys.executable, '-B', str(child), str(settings.root), kind, failure, str(record_path), str(request_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == (88 if failure == 'before' else 89), result.stderr
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['pid'] != os.getpid() and record['forbidden_import_attempts'] == record['forbidden_loaded'] == []
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == (0 if failure == 'before' else 1 if kind == 'upload' else 2)
    with TestClient(create_app(settings, start_worker=False)) as restarted:
        one = restarted.post(route, headers=HEADER, json=payload)
        two = restarted.post(route, headers=HEADER, json=payload)
    assert one.status_code == two.status_code == 201 and one.json() == two.json()
    if failure == 'after':
        assert one.json()['id'] == record['descriptor']['id']
    with app.state.store.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM upload_creation_requests').fetchone()[0] == 1
        assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == (1 if kind == 'upload' else 2)
        assert conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0] == (0 if kind == 'upload' else 2)
