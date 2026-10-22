"""Real owner HTTP retry identity and pre/post-queue crash convergence."""
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from threading import Barrier

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import Settings

VIDEO = b'\0\0\0\x18ftypisom' + b'x'*20
KEY = 'retry-intent-test-000001'
HEADERS = {'Idempotency-Key': KEY}


@pytest.fixture
def service(tmp_path):
    settings = Settings(tmp_path/'storage', tmp_path/'unused.yaml', max_upload_bytes=64,
                        max_owner_pending_jobs=1, max_owner_storage_bytes=1024)
    app = create_app(settings, start_worker=False, demos_manifest=tmp_path/'no-demo.json')
    with TestClient(app) as client:
        first = client.post('/api/analyses?filename=match.mp4', content=VIDEO)
        assert first.status_code == 202
        source = first.json()['id']
        app.state.store.finish(source, 'synthetic failure')
        yield client, app, settings, source


def retry(client, source, headers=None):
    return client.post('/api/analyses/'+source+'/retry', headers=HEADERS if headers is None else headers)


@pytest.mark.parametrize('headers', [{}, {'Idempotency-Key': ''}, {'Idempotency-Key': 'short'},
    {'Idempotency-Key': 'x'*129}, {'Idempotency-Key': 'spaces are invalid'},
    {'Idempotency-Key': '../escape-not-an-intent'},
    [('Idempotency-Key', KEY), ('Idempotency-Key', KEY)]])
def test_retry_requires_one_bounded_explicit_intent_before_admission(service, headers):
    client, app, _, source = service
    assert retry(client, source, headers).status_code == 422
    assert len(app.state.store.list()) == 1
    assert client.get('/api/usage').json()['pending_jobs'] == 0


def test_lost_ack_recovers_same_child_and_metadata_without_repeating_copy(service, monkeypatch):
    client, app, settings, source = service
    jobs = app.state.store
    with jobs.connect() as conn:
        conn.execute('UPDATE matches SET notes=? WHERE analysis_id=?', ('retained original notes', source))
    create = jobs.create

    def commit_then_fail(*args, **kwargs):
        create(*args, **kwargs)
        raise OSError('injected acknowledgement loss after actual queue commit')

    monkeypatch.setattr(jobs, 'create', commit_then_fail)
    assert retry(client, source).status_code == 507
    children = [job for job in jobs.list() if job['retry_of'] == source]
    assert len(children) == 1
    child = children[0]['id']
    assert (jobs.upload_dir(child)/'original.mp4').read_bytes() == VIDEO
    with jobs.connect() as conn:
        assert conn.execute('SELECT notes FROM matches WHERE analysis_id=?', (child,)).fetchone()[0] == 'retained original notes'
        conn.execute('UPDATE matches SET notes=? WHERE analysis_id=?', ('child owner edit after acceptance', child))
    # Accepted acknowledgement precedes source availability and current storage
    # admission, and never recopies or overwrites edited child metadata.
    (jobs.upload_dir(source)/'original.mp4').unlink()
    restarted = create_app(replace(settings, max_owner_storage_bytes=1), start_worker=False)
    monkeypatch.setattr(shutil, 'copyfile', lambda *a: pytest.fail('accepted retry recopied input'))
    with TestClient(restarted) as restored:
        acknowledged = retry(restored, source)
        assert acknowledged.status_code == 202 and acknowledged.json()['id'] == child
        assert retry(restored, source).json()['id'] == child
    with jobs.connect() as conn:
        assert conn.execute('SELECT notes FROM matches WHERE analysis_id=?', (child,)).fetchone()[0] == 'child owner edit after acceptance'
    assert len(jobs.list()) == 2


def test_failure_before_queue_reuses_saved_child_and_rechecks_released_capacity(service, monkeypatch):
    client, app, _, source = service
    jobs = app.state.store
    create = jobs.create

    def fail_before_commit(*args, **kwargs):
        raise OSError('injected failure before queue insert')

    monkeypatch.setattr(jobs, 'create', fail_before_commit)
    assert retry(client, source).status_code == 507
    with jobs.connect() as conn:
        saved = conn.execute('SELECT * FROM analysis_retry_intents').fetchone()
        child = saved['analysis_id']
        assert conn.execute('SELECT active FROM resource_reservations WHERE analysis_id=?', (child,)).fetchone()[0] == 0
    assert client.get('/api/usage').json()['pending_jobs'] == 0
    # Another admitted request occupies the slot. Reactivating this stable
    # identity must fail atomically, without silently taking a second slot.
    incoming = client.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': 'other.mp4', 'size_bytes': 32}).json()
    monkeypatch.setattr(jobs, 'create', create)
    assert retry(client, source).status_code == 429
    assert client.delete('/api/uploads/'+incoming['id']).status_code == 200
    result = retry(client, source)
    assert result.status_code == 202 and result.json()['id'] == child
    assert len(jobs.list()) == 2


def test_prequeue_crash_with_partial_input_reuses_identity_after_server_restart(service):
    client, app, settings, source = service
    from src.api.retry_intents import RetryIntents
    jobs = app.state.store
    intents = RetryIntents(jobs, app.state.quotas)
    with intents.lock('local', KEY):
        child = intents.admit('local', KEY, source, len(VIDEO))
    directory = jobs.upload_dir(child)
    directory.mkdir(parents=True)
    (directory/'upload.part').write_bytes(VIDEO[:13])
    with TestClient(create_app(settings, start_worker=False)) as restarted:
        response = retry(restarted, source)
        assert response.status_code == 202 and response.json()['id'] == child
    assert (directory/'original.mp4').read_bytes() == VIDEO
    assert not (directory/'upload.part').exists()
    assert len(jobs.list()) == 2


def test_concurrent_http_requests_with_same_intent_create_one_child(service):
    _, app, _, source = service
    barrier = Barrier(2)

    def submit():
        with TestClient(app) as client:
            barrier.wait(timeout=5)
            response = retry(client, source)
            assert response.status_code == 202, response.text
            return response.json()['id']

    with ThreadPoolExecutor(max_workers=2) as pool:
        work = [pool.submit(submit) for _ in range(2)]
        ids = [future.result(timeout=30) for future in work]
    assert ids[0] == ids[1]
    assert len(app.state.store.list()) == 2


def test_intent_is_bound_to_source_and_failed_child_requires_new_intent(service):
    client, app, _, source = service
    jobs = app.state.store
    accepted = retry(client, source)
    assert accepted.status_code == 202
    child = accepted.json()['id']
    jobs.finish(child, 'synthetic child failure')
    assert retry(client, source).json()['id'] == child
    assert retry(client, child).status_code == 409
    fresh = retry(client, child, {'Idempotency-Key': 'a-new-explicit-intent-0002'})
    assert fresh.status_code == 202 and fresh.json()['retry_of'] == child
    assert len(jobs.list()) == 3


def test_owner_authorization_precedes_key_validation_and_same_key_is_owner_scoped(tmp_path):
    settings = Settings(tmp_path/'storage', tmp_path/'unused', require_auth=True)
    app = create_app(settings, start_worker=False)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        for client, name in [(alice, 'alice'), (bob, 'bob')]:
            assert client.post('/api/auth/register', json={
                'username': name, 'password': 'retry-intent-password-123'}).status_code == 201
        first = alice.post('/api/analyses?filename=a.mp4', content=VIDEO).json()['id']
        second = bob.post('/api/analyses?filename=b.mp4', content=VIDEO).json()['id']
        for ident in (first, second):
            app.state.store.finish(ident, 'synthetic failure')
        assert retry(bob, first, {}).status_code == 404
        assert retry(bob, first).status_code == 404
        assert retry(guest, first).status_code == 401
        one = retry(alice, first)
        two = retry(bob, second)
        assert one.status_code == two.status_code == 202
        assert one.json()['id'] != two.json()['id']
        assert bob.get('/api/analyses/'+one.json()['id']).status_code == 404


@pytest.mark.parametrize('failure', ['before', 'after'])
def test_actual_process_exit_converges_same_retry_identity_and_one_queue_row(service, tmp_path, failure):
    import hashlib
    import os
    client, app, settings, source = service
    record_path = tmp_path/('retry_child_'+failure+'.json')
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).parent/'helpers/retry_intent_crash_child.py'),
        str(settings.root), source, failure, str(record_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == (85 if failure == 'before' else 86), result.stderr
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['pid'] != os.getpid() and record['case'] == failure
    assert record['forbidden_import_attempts'] == record['forbidden_loaded'] == []
    assert record['queued_before_exit'] is (failure == 'after')
    assert record['original_sha256'] == hashlib.sha256(VIDEO).hexdigest()
    child = record['selected']['analysis_id']
    # The dead child's OS-backed request lock must be recoverable. Repeating
    # the exact persisted browser intent yields that same child after restart.
    with TestClient(create_app(settings, start_worker=False)) as restarted:
        first = retry(restarted, source)
        second = retry(restarted, source)
        assert first.status_code == second.status_code == 202
        assert first.json()['id'] == second.json()['id'] == child
        assert restarted.get('/api/usage').json()['pending_jobs'] == 1
    children = [job for job in app.state.store.list() if job['retry_of'] == source]
    assert [job['id'] for job in children] == [child]
    assert (app.state.store.upload_dir(child)/'original.mp4').read_bytes() == VIDEO
