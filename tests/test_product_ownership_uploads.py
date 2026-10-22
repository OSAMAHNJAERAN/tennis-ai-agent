"""Software account/isolation and durable transfer evidence, without CV claims."""
from uuid import uuid4
import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import JobCancelled, JobStore, Settings
from tests.test_product_api import write_artifacts
from tests.test_public_demos import curated_job

VIDEO = b'\0\0\0\x18ftypisom' + b'x' * 20
PASSWORD = 'test-password-only-123'


@pytest.fixture
def accounts(tmp_path):
    settings = Settings(tmp_path / 'storage', tmp_path / 'config.yaml', max_upload_bytes=64, require_auth=True)
    manifest = tmp_path / 'demos.json'
    manifest.write_text(json.dumps({'schemaVersion': 1, 'sources': []}))
    app = create_app(settings, start_worker=False, demos_manifest=manifest)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        first = alice.post('/api/auth/register', json={'username': 'alice', 'password': PASSWORD})
        second = bob.post('/api/auth/register', json={'username': 'bob', 'password': PASSWORD})
        assert first.status_code == second.status_code == 201
        yield alice, bob, guest, app.state.store, settings, manifest


def saved_analysis(alice, store):
    response = alice.post('/api/analyses?filename=private.mp4', content=VIDEO)
    assert response.status_code == 202
    job = response.json()
    output = store.output_dir(job['id'])
    output.mkdir(parents=True)
    artifact_job = write_artifacts(output)
    artifact_job.update(id=job['id'], filename=job['filename'])
    from src.api.results import build_result
    saved = build_result(output, artifact_job)
    (output / 'result.json').write_text(json.dumps(saved))
    (output / 'source.mp4').write_bytes(VIDEO)
    (output / 'annotated_web.mp4').write_bytes(VIDEO)
    (output / 'source_frame_mapping.jsonl').write_text('{"frame_index":0}\n')
    store.finish(job['id'])
    return job


def test_session_restart_password_storage_revocation_and_csrf(accounts):
    alice, _, guest, store, settings, manifest = accounts
    session = alice.get('/api/auth/session').json()
    assert session['required'] and session['authenticated'] and session['user']['username'] == 'alice'
    with store.connect() as conn:
        row = dict(conn.execute("SELECT * FROM accounts WHERE username='alice'").fetchone())
        tokens = [dict(row) for row in conn.execute('SELECT * FROM auth_sessions')]
    assert PASSWORD not in str(row) and len(row['password_hash']) == 128
    assert all(len(token['token_hash']) == 64 for token in tokens)
    with TestClient(create_app(settings, start_worker=False, demos_manifest=manifest)) as restored:
        restored.cookies.update(alice.cookies)
        assert restored.get('/api/auth/session').json() == session
        assert restored.post('/api/players', json={'name': 'blocked'}, headers={'Origin': 'https://evil.example'}).status_code == 403
        assert restored.post('/api/auth/logout').status_code == 200
    assert alice.get('/api/analyses').status_code == 401
    assert guest.post('/api/auth/login', json={'username': 'alice', 'password': PASSWORD + 'wrong'}).status_code == 401
    assert guest.post('/api/auth/login', json={'username': 'ALICE', 'password': PASSWORD}).status_code == 200


@pytest.mark.parametrize('suffix', ['', '/result', '/media/source', '/media/annotated',
                                      '/download/original', '/download/annotated', '/download/report',
                                      '/download/readable-report', '/download/csv', '/download/source-timing'])
def test_every_private_analysis_media_and_export_fails_for_nonowner(accounts, suffix):
    alice, bob, guest, store, _, _ = accounts
    job = saved_analysis(alice, store)
    endpoint = f'/api/analyses/{job["id"]}{suffix}'
    assert alice.get(endpoint).status_code == 200
    assert bob.get(endpoint, headers={'Range': 'bytes=0-5'}).status_code == 404
    assert guest.get(endpoint).status_code == 401
    assert bob.get('/api/analyses').json() == []
    assert bob.get('/api/runs').json() == []
    assert alice.get(endpoint).headers['cache-control'] == 'private, no-store'


def test_match_profile_draft_commands_and_legacy_isolation(accounts):
    alice, bob, _, store, _, _ = accounts
    legacy = store.create(store.new_id(), 'legacy.mp4', '.mp4', 12, Path('unused'))
    profile = alice.post('/api/players', json={'name': 'Private player'}).json()
    match = alice.post('/api/matches', json={'source': 'manual', 'initial_server': 1}).json()
    draft = alice.post('/api/matches', json={}).json()
    assert bob.get('/api/players').json() == [] and bob.get('/api/matches').json() == []
    assert alice.get(f'/api/analyses/{legacy["id"]}').status_code == 404
    assert bob.get(f'/api/players/{profile["id"]}').status_code == 404
    assert bob.get(f'/api/matches/{match["id"]}').status_code == 404
    assert bob.put(f'/api/matches/{match["id"]}/notes', json={'expected_revision': 0, 'notes': 'stolen'}).status_code == 404
    assert bob.patch(f'/api/matches/{match["id"]}/metadata', json={'expected_revision': 0, 'metadata': {}}).status_code == 404
    assert bob.post(f'/api/matches/{match["id"]}/score-commands', json={'command_id': 'private-point', 'expected_revision': 0, 'action': 'point', 'winner': 1}).status_code == 404
    assert bob.post('/api/matches', json={'metadata': {'participants': [{'profile_id': profile['id']}, {}]}}).status_code == 422
    assert bob.post('/api/analyses', params={'filename': 'stolen.mp4', 'match_id': draft['id']}, content=VIDEO).status_code == 404
    assert bob.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': 'stolen.mp4', 'size_bytes': len(VIDEO), 'match_id': draft['id']}).status_code == 404
    job = alice.post('/api/analyses?filename=private.mp4', content=VIDEO).json()
    assert bob.get(f'/api/matches/by-analysis/{job["id"]}').status_code == 404
    assert bob.post(f'/api/analyses/{job["id"]}/cancel').status_code == 404
    assert bob.post(f'/api/analyses/{job["id"]}/retry').status_code == 404
    assert store.get(job['id'])['status'] == 'UPLOADED'


def test_resumable_upload_offset_crash_surplus_completion_and_restart(accounts):
    alice, _, _, store, settings, manifest = accounts
    response = alice.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': '../../match.mp4', 'size_bytes': len(VIDEO)})
    assert response.status_code == 201
    session = response.json()
    assert session['filename'] == 'match.mp4' and session['analysis_id'] is None
    url = '/api/uploads/' + session['id']
    assert alice.post(url + '/complete').status_code == 409
    first = alice.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO[:16])
    assert first.json()['offset'] == 16
    # Simulates persisted file bytes whose database transaction did not commit.
    partial = store.root / 'resumable' / session['id'] / 'upload.part'
    with partial.open('ab') as handle:
        handle.write(b'UNCOMMITTED')
    with TestClient(create_app(settings, start_worker=False, demos_manifest=manifest)) as restarted:
        restarted.cookies.update(alice.cookies)
        assert restarted.get(url).json()['offset'] == 16
        assert restarted.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO[:16]).status_code == 409
        finished = restarted.patch(url, headers={'Upload-Offset': '16'}, content=VIDEO[16:])
        assert finished.json()['offset'] == len(VIDEO)
        queued = restarted.post(url + '/complete')
        assert queued.status_code == 202 and queued.json()['status'] == 'UPLOADED'
        duplicate = restarted.post(url + '/complete')
        assert duplicate.status_code == 202 and duplicate.json()['id'] == queued.json()['id']
        saved = restarted.get(url).json()
        assert saved['analysis_id'] == queued.json()['id']
        assert saved['checksum'] == hashlib.sha256(VIDEO).hexdigest()
    assert (store.upload_dir(queued.json()['id']) / 'original.mp4').read_bytes() == VIDEO
    assert len(alice.get('/api/analyses').json()) == 1


def test_upload_and_batch_access_controls_cancellation_and_declared_limits(accounts):
    alice, bob, guest, store, _, _ = accounts
    batch = alice.post('/api/batches', headers={'Idempotency-Key': uuid4().hex}, json={'files': [{'filename': 'a.mp4', 'size_bytes': len(VIDEO)},
                                                     {'filename': 'b.mp4', 'size_bytes': len(VIDEO)}]}).json()
    upload = batch['uploads'][0]
    url = '/api/uploads/' + upload['id']
    assert bob.get('/api/batches/' + batch['id']).status_code == 404
    assert guest.get(url).status_code == 401
    assert bob.get(url).status_code == 404
    assert bob.patch(url, content=VIDEO, headers={'Upload-Offset': '0'}).status_code == 404
    assert bob.post(url + '/complete').status_code == 404
    assert bob.delete(url).status_code == 404
    assert alice.patch(url, content=VIDEO).status_code == 400
    assert alice.patch(url, content=VIDEO + b'x' * 10, headers={'Upload-Offset': '0'}).status_code == 413
    assert alice.get(url).json()['offset'] == 0
    assert alice.patch(url, content=VIDEO[:16], headers={'Upload-Offset': '0'}).status_code == 200
    assert alice.delete(url).json()['status'] == 'CANCELLED'
    assert alice.patch(url, content=VIDEO[16:], headers={'Upload-Offset': '16'}).status_code == 409
    assert alice.post(url + '/complete').status_code == 409
    assert not (store.root / 'resumable' / upload['id'] / 'upload.part').exists()
    assert alice.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': 'huge.mp4', 'size_bytes': 65}).status_code == 413


def test_completion_recovers_a_crash_after_queue_commit_without_duplicate(accounts, monkeypatch):
    alice, _, _, store, _, _ = accounts
    session = alice.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': 'recover.mp4', 'size_bytes': len(VIDEO)}).json()
    url = '/api/uploads/' + session['id']
    alice.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO)
    original_create = store.create
    def commit_then_fail(*args, **kwargs):
        original_create(*args, **kwargs)
        raise OSError('Injected crash after durable job acceptance')
    monkeypatch.setattr(store, 'create', commit_then_fail)
    assert alice.post(url + '/complete').status_code == 507
    assert len(alice.get('/api/analyses').json()) == 1
    monkeypatch.setattr(store, 'create', original_create)
    response = alice.post(url + '/complete')
    assert response.status_code == 202
    restored = alice.get(url).json()
    assert restored['analysis_id'] == response.json()['id']
    assert restored['checksum'] == hashlib.sha256(VIDEO).hexdigest()
    assert len(alice.get('/api/analyses').json()) == 1


def test_competing_append_and_queue_claims_are_serialized(accounts):
    from concurrent.futures import ThreadPoolExecutor
    alice, _, _, store, _, _ = accounts
    session = alice.post('/api/uploads', headers={'Idempotency-Key': uuid4().hex}, json={'filename': 'parallel.mp4', 'size_bytes': len(VIDEO)}).json()
    url = '/api/uploads/' + session['id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: alice.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO[:16]), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert alice.get(url).json()['offset'] == 16
    alice.patch(url, headers={'Upload-Offset': '16'}, content=VIDEO[16:])
    alice.post(url + '/complete')
    alice.post('/api/analyses?filename=other.mp4', content=VIDEO)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: store.claim(), range(2)))
    assert len({claim['id'] for claim in claims}) == 2
    assert all(claim['status'] == 'PROCESSING' for claim in claims)


def test_coach_routes_require_authenticated_session_before_provider_work(accounts):
    _, _, guest, _, _, _ = accounts
    assert guest.get('/api/coach/status').status_code == 401
    assert guest.post('/api/coach', json={}).status_code == 401


def test_same_match_order_duplicate_check_and_one_file_failure(accounts):
    alice, _, _, _, _, _ = accounts
    files = [{'filename': 'a.mp4', 'size_bytes': len(VIDEO)}, {'filename': 'b.mp4', 'size_bytes': len(VIDEO)}]
    assert alice.post('/api/batches', headers={'Idempotency-Key': uuid4().hex}, json={'mode': 'same_match', 'files': files}).status_code == 422
    batch = alice.post('/api/batches', headers={'Idempotency-Key': uuid4().hex}, json={'mode': 'same_match', 'files': [
        {**files[0], 'clip_index': 0, 'start_offset_seconds': 0},
        {**files[1], 'clip_index': 1, 'start_offset_seconds': 15}]}).json()
    first, second = batch['uploads']
    for item in (first, second):
        url = '/api/uploads/' + item['id']
        assert alice.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO).status_code == 200
        assert alice.post(url + '/complete').status_code == 202
    summary = alice.get('/api/batches/' + batch['id']).json()
    assert summary['grouping_status'] == 'duplicates_detected' and summary['unified_report'] is None
    assert [item['clip_index'] for item in summary['uploads']] == [0, 1]
    assert len(alice.get('/api/analyses').json()) == 2
    independent = alice.post('/api/batches', headers={'Idempotency-Key': uuid4().hex}, json={'files': files}).json()
    good, bad = independent['uploads']
    for item, content in ((good, VIDEO), (bad, b'bad bytes' + b'x' * 23)):
        url = '/api/uploads/' + item['id']
        assert alice.patch(url, headers={'Upload-Offset': '0'}, content=content).status_code == 200
        response = alice.post(url + '/complete')
        assert response.status_code == (202 if item == good else 415)
    assert alice.get('/api/batches/' + independent['id']).json()['uploads'][0]['job']['status'] == 'UPLOADED'
    failed = alice.get('/api/uploads/' + bad['id']).json()
    assert failed['status'] == 'FAILED' and failed['error'] and failed['analysis_id'] is None
    assert alice.get('/api/batches/' + independent['id']).json()['uploads'][1]['status'] == 'FAILED'


def test_job_cancel_retry_preserves_source_and_old_evidence(accounts):
    alice, _, _, store, _, _ = accounts
    job = alice.post('/api/analyses?filename=retry.mp4', content=VIDEO).json()
    store.output_dir(job['id']).mkdir(parents=True)
    evidence = store.output_dir(job['id']) / 'processing.log'
    evidence.write_text('Original failure evidence')
    base = '/api/analyses/' + job['id']
    assert alice.post(base + '/retry', headers={'Idempotency-Key': 'ownership-retry-intent-0001'}).status_code == 409
    cancelled = alice.post(base + '/cancel').json()
    assert cancelled['status'] == 'FAILED' and cancelled['cancelled']
    queued = alice.post(base + '/retry', headers={'Idempotency-Key': 'ownership-retry-intent-0001'})
    assert queued.status_code == 202
    new = queued.json()
    assert new['id'] != job['id'] and new['retry_of'] == job['id']
    assert new['status'] == 'UPLOADED' and not new['cancel_requested']
    assert evidence.read_text() == 'Original failure evidence'
    assert (store.upload_dir(new['id']) / 'original.mp4').read_bytes() == VIDEO
    claimed = store.claim()
    assert claimed['id'] == new['id'] and claimed['attempt'] == 1
    alice.post('/api/analyses/' + new['id'] + '/cancel')
    with pytest.raises(JobCancelled):
        store.stage(new['id'], 'Safe boundary')
    store.finish(new['id'])
    assert store.get(new['id'])['status'] == 'FAILED'


def test_account_mode_curated_read_only_examples_and_revoked_manifest(accounts):
    alice, bob, guest, store, _, manifest = accounts
    _, _, entry = curated_job((alice, store, manifest))
    base = '/api/analyses/' + entry['analysisId']
    assert guest.get('/api/demos').status_code == 200
    for suffix in ('', '/result', '/media/annotated', '/download/original', '/download/report', '/download/readable-report', '/download/csv'):
        assert guest.get(base + suffix).status_code == 200
        assert bob.get(base + suffix).status_code == 200
    assert bob.post(base + '/cancel').status_code == 404
    assert bob.post(base + '/retry').status_code == 404
    manifest.write_text(json.dumps({'schemaVersion': 1, 'sources': []}))
    assert guest.get(base + '/result').status_code == 401
    assert bob.get(base + '/result').status_code == 404


def test_remote_requests_require_account_and_secure_cookie_mode(tmp_path):
    local = create_app(Settings(tmp_path / 'local', Path('unused')), start_worker=False)
    with TestClient(local, client=('203.0.113.10', 50000)) as client:
        assert client.get('/api/health').status_code == 403
    with TestClient(local, base_url='http://untrusted.example', client=('127.0.0.1', 50000)) as client:
        assert client.get('/api/analyses').status_code == 403
    insecure = create_app(Settings(tmp_path / 'insecure', Path('unused'), require_auth=True), start_worker=False)
    with TestClient(insecure, client=('203.0.113.10', 50000)) as client:
        assert client.get('/api/auth/session').status_code == 403
    secure = create_app(Settings(tmp_path / 'secure', Path('unused'), require_auth=True, secure_cookies=True), start_worker=False)
    with TestClient(secure, base_url='https://testserver', client=('203.0.113.10', 50000)) as client:
        response = client.post('/api/auth/register', json={'username': 'remote', 'password': PASSWORD})
        assert response.status_code == 201
        assert 'Secure' in response.headers['set-cookie'] and 'HttpOnly' in response.headers['set-cookie']
        assert client.get('/api/analyses').status_code == 200
