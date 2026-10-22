"""Real HTTP/SQLite owner admission; synthetic bytes and no model work."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import JobStore, Settings
from src.api import worker

VIDEO = b'\0\0\0\x18ftypisom' + b'x' * 20


def service(tmp_path, **limits):
    settings = Settings(tmp_path/'storage', tmp_path/'unused.yaml', max_upload_bytes=64, **limits)
    app = create_app(settings, start_worker=False, demos_manifest=tmp_path/'absent.json')
    return app, settings


def new_upload(client):
    return client.post('/api/uploads', json={'filename': 'match.mp4', 'size_bytes': len(VIDEO)})


@pytest.mark.parametrize('name', ['max_upload_bytes', 'max_duration_seconds',
                                  'max_owner_pending_jobs', 'max_owner_storage_bytes'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_settings_reject_nonpositive_or_noninteger_limits(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        Settings(tmp_path, tmp_path/'config', **{name: value})


def test_environment_limits_roundtrip(monkeypatch):
    monkeypatch.setenv('TENNIS_MAX_OWNER_PENDING_JOBS', '3')
    monkeypatch.setenv('TENNIS_MAX_OWNER_STORAGE_BYTES', '4096')
    settings = Settings.from_env()
    assert settings.max_owner_pending_jobs == 3 and settings.max_owner_storage_bytes == 4096


def test_batch_admission_is_all_or_none_and_cancellation_releases_pending_slot(tmp_path):
    app, _ = service(tmp_path, max_owner_pending_jobs=1)
    with TestClient(app) as client:
        response = client.post('/api/batches', json={'files': [
            {'filename': 'a.mp4', 'size_bytes': 32}, {'filename': 'b.mp4', 'size_bytes': 32}]})
        assert response.status_code == 429
        with app.state.store.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM upload_batches').fetchone()[0] == 0
            assert conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0] == 0
        accepted = new_upload(client)
        assert accepted.status_code == 201
        assert new_upload(client).status_code == 429
        assert client.post('/api/analyses?filename=another.mp4', content=VIDEO).status_code == 429
        assert client.delete('/api/uploads/'+accepted.json()['id']).status_code == 200
        assert new_upload(client).status_code == 201


def test_storage_reservation_persists_across_server_restart_and_is_owner_private(tmp_path):
    app, settings = service(tmp_path, max_owner_storage_bytes=48, require_auth=True)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        for client, name in [(alice, 'alice'), (bob, 'bob')]:
            assert client.post('/api/auth/register', json={
                'username': name, 'password': 'quota-test-password-only-123'}).status_code == 201
        assert new_upload(alice).status_code == 201
        assert new_upload(alice).status_code == 507
        assert new_upload(bob).status_code == 201
        usage = alice.get('/api/usage')
        assert usage.status_code == 200 and usage.json()['committed_bytes'] == 32
        assert usage.headers['cache-control'] == 'private, no-store'
        assert guest.get('/api/usage').status_code == 401
        assert set(usage.json()) == {'pending_jobs', 'pending_analyses', 'pending_corrections',
                                  'stored_bytes', 'reserved_input_bytes', 'committed_bytes', 'limits'}
        restarted = create_app(settings, start_worker=False, demos_manifest=tmp_path/'absent.json')
        with TestClient(restarted) as restored:
            restored.cookies.update(alice.cookies)
            assert restored.get('/api/usage').json() == usage.json()
            assert new_upload(restored).status_code == 507


def test_completion_deduplicates_pending_capacity_even_after_durable_commit_lost_ack(tmp_path, monkeypatch):
    app, _ = service(tmp_path, max_owner_pending_jobs=1, max_owner_storage_bytes=256)
    jobs = app.state.store
    with TestClient(app) as client:
        session = new_upload(client).json()
        url = '/api/uploads/'+session['id']
        assert client.patch(url, headers={'Upload-Offset': '0'}, content=VIDEO).status_code == 200
        original_create = jobs.create

        def commit_then_fail(*args, **kwargs):
            original_create(*args, **kwargs)
            raise OSError('test lost acknowledgement after durable commit')

        monkeypatch.setattr(jobs, 'create', commit_then_fail)
        assert client.post(url+'/complete').status_code == 507
        assert client.get('/api/usage').json()['pending_jobs'] == 1
        monkeypatch.setattr(jobs, 'create', original_create)
        first = client.post(url+'/complete')
        second = client.post(url+'/complete')
        assert first.status_code == second.status_code == 202
        assert first.json()['id'] == second.json()['id']
        assert len(jobs.list()) == 1 and new_upload(client).status_code == 429
        usage = client.get('/api/usage').json()
        assert usage['stored_bytes'] == 32 and usage['reserved_input_bytes'] == 0


def test_legacy_lost_queue_ack_retains_accepted_original_and_releases_reservation(tmp_path, monkeypatch):
    app, _ = service(tmp_path, max_owner_pending_jobs=1)
    jobs = app.state.store
    original_create = jobs.create

    def commit_then_fail(*args, **kwargs):
        original_create(*args, **kwargs)
        raise OSError('test lost acknowledgement')

    monkeypatch.setattr(jobs, 'create', commit_then_fail)
    with TestClient(app) as client:
        assert client.post('/api/analyses?filename=match.mp4', content=VIDEO).status_code == 507
        job, = jobs.list()
        assert (jobs.upload_dir(job['id'])/'original.mp4').read_bytes() == VIDEO
        assert client.get('/api/usage').json()['pending_jobs'] == 1
        with jobs.connect() as conn:
            assert conn.execute('SELECT active FROM resource_reservations').fetchone()[0] == 0


def test_unknown_length_reserves_maximum_and_stream_cannot_exceed_declared_bytes(tmp_path):
    app, _ = service(tmp_path, max_owner_storage_bytes=48)
    with TestClient(app) as client:
        unknown = client.post('/api/analyses?filename=match.mp4', content=iter([VIDEO]))
        assert unknown.status_code == 507
        bad = client.post('/api/analyses?filename=match.mp4', content=iter([VIDEO]),
                          headers={'Content-Length': '12'})
        assert bad.status_code == 413
        assert client.get('/api/usage').json()['committed_bytes'] == 0
        assert client.post('/api/analyses?filename=match.mp4', content=VIDEO).status_code == 202


def test_failure_frees_job_slot_but_retained_outputs_still_consume_storage(tmp_path):
    app, _ = service(tmp_path, max_owner_pending_jobs=1, max_owner_storage_bytes=96)
    jobs = app.state.store
    with TestClient(app) as client:
        accepted = client.post('/api/analyses?filename=match.mp4', content=VIDEO).json()
        jobs.finish(accepted['id'], 'synthetic failure')
        output = jobs.output_dir(accepted['id'])/'.result'/'old.files'
        output.mkdir(parents=True)
        (output/'source.mp4').write_bytes(b'x'*40)
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == 0 and usage['stored_bytes'] == 72
        assert client.post('/api/analyses/'+accepted['id']+'/retry').status_code == 507
        assert new_upload(client).status_code == 507
        assert len(jobs.list()) == 1
        assert list((jobs.root/'uploads').iterdir()) == [jobs.upload_dir(accepted['id'])]


def test_simultaneous_http_admissions_share_one_durable_slot(tmp_path):
    app, _ = service(tmp_path, max_owner_pending_jobs=1)
    barrier = Barrier(2)

    def submit():
        with TestClient(app) as client:
            barrier.wait(timeout=5)
            return new_upload(client).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(submit) for _ in range(2)]
        assert sorted(future.result(timeout=15) for future in futures) == [201, 429]
    with TestClient(app) as client:
        assert client.get('/api/usage').json()['pending_jobs'] == 1


def test_retry_admission_counts_logical_copy_before_reusing_source(tmp_path):
    app, _ = service(tmp_path, max_owner_pending_jobs=1, max_owner_storage_bytes=64)
    jobs = app.state.store
    with TestClient(app) as client:
        first = client.post('/api/analyses?filename=match.mp4', content=VIDEO).json()
        jobs.finish(first['id'], 'synthetic failure')
        retry = client.post('/api/analyses/'+first['id']+'/retry')
        assert retry.status_code == 202
        assert client.get('/api/usage').json()['committed_bytes'] == 64
        assert client.post('/api/analyses/'+first['id']+'/retry').status_code == 429
        assert (jobs.upload_dir(first['id'])/'original.mp4').read_bytes() == VIDEO


def test_worker_storage_refusal_precedes_model_asset_checks_and_preserves_input(tmp_path, monkeypatch):
    app, settings = service(tmp_path, max_owner_storage_bytes=16)
    jobs = app.state.store
    ident = jobs.new_id()
    jobs.create(ident, 'match.mp4', '.mp4', 32, settings.config)
    jobs.upload_dir(ident).mkdir(parents=True)
    (jobs.upload_dir(ident)/'original.mp4').write_bytes(VIDEO)
    claimed = jobs.claim()
    monkeypatch.setattr(worker, 'check_models', lambda *a: pytest.fail('over-quota job reached model checks'))
    worker.process_job(jobs, claimed, settings)
    job = jobs.get(ident)
    assert job['status'] == 'FAILED' and job['error'] == 'Your storage limit has been reached.'
    assert (jobs.upload_dir(ident)/'original.mp4').read_bytes() == VIDEO


def test_stage_log_checks_growth_at_pipeline_boundary():
    from io import StringIO
    from types import SimpleNamespace
    seen = []
    store = SimpleNamespace(stage=lambda *args: seen.append(args))
    log = worker.StageLog(StringIO(), store, 'fixture', storage_check=lambda: seen.append('quota'))
    log.write('[Step 3/11] Tracking\n')
    assert seen == [('fixture', 'Tracking', 3), 'quota']
