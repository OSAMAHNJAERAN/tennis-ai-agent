"""Owner recovery of abandoned admission capacity; synthetic bytes, no inference."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event

from fastapi.testclient import TestClient
import httpx
import pytest

from src.api.app import create_app
from src.api.jobs import JobStore, Settings


VIDEO = b'\0\0\0\x18ftypisom' + b'x' * 20
RETRY_KEY = 'retry-intent-test-000001'


def service(tmp_path, **limits):
    settings = Settings(tmp_path / 'storage', tmp_path / 'unused.yaml',
                        max_upload_bytes=64, max_owner_pending_jobs=1, **limits)
    return create_app(settings, start_worker=False, demos_manifest=tmp_path / 'absent.json'), settings


def reserve(app, ident, owner='local', *, managed=True, size=32):
    with app.state.store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        app.state.quotas.reserve(conn, ident, owner, size, recovery_version=int(managed))


def stored(app, relative, content):
    path = app.state.store.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def active(app, ident):
    with app.state.store.connect() as conn:
        return conn.execute('SELECT active FROM resource_reservations WHERE analysis_id=?',
                            (ident,)).fetchone()[0]


def test_restart_recovery_is_repeatable_and_keeps_all_owned_residue(tmp_path):
    app, settings = service(tmp_path)
    ident = 'match_abandoned'
    reserve(app, ident)
    partial = stored(app, f'uploads/{ident}/upload.part', b'partial')
    scratch = stored(app, f'analyses/{ident}/native_scratch/leftover.bin', b'extra')
    original_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (partial, scratch)]
    with TestClient(app) as client:
        before = client.get('/api/usage').json()
        assert before['pending_jobs'] == 1
        assert (before['stored_bytes'], before['reserved_input_bytes'], before['committed_bytes']) == (12, 25, 37)
    restarted = create_app(settings, start_worker=False, demos_manifest=tmp_path / 'absent.json')
    with TestClient(restarted) as client:
        first = client.post('/api/usage/reconcile')
        assert first.status_code == 200
        assert first.json()['reconciled_reservations'] == 1
        after = client.get('/api/usage').json()
        assert after['pending_jobs'] == after['reserved_input_bytes'] == 0
        assert after['stored_bytes'] == after['committed_bytes'] == 12
        again = client.post('/api/usage/reconcile')
        assert again.status_code == 200 and again.json()['reconciled_reservations'] == 0
        assert client.get('/api/usage').json() == after
    assert active(app, ident) == 0
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (partial, scratch)] == original_hashes
    with app.state.store.connect() as conn:
        row = conn.execute('SELECT owner_id,size_bytes FROM resource_reservations WHERE analysis_id=?',
                           (ident,)).fetchone()
        assert tuple(row) == ('local', 32)


def test_recovery_frees_only_unwritten_capacity_and_retained_bytes_still_block_storage(tmp_path):
    app, _ = service(tmp_path, max_owner_storage_bytes=35)
    reserve(app, 'match_full_orphan')
    original = stored(app, 'uploads/match_full_orphan/original.mp4', VIDEO)
    extra = stored(app, 'analyses/match_full_orphan/log.bin', b'log')
    with TestClient(app) as client:
        assert client.post('/api/usage/reconcile').status_code == 200
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == 0
        assert usage['stored_bytes'] == usage['committed_bytes'] == 35
        rejected = client.post('/api/uploads', headers={'Idempotency-Key': 'recovered-capacity-next-0001'},
                               json={'filename': 'next.mp4', 'size_bytes': 32})
        assert rejected.status_code == 507
    assert original.read_bytes() == VIDEO and extra.read_bytes() == b'log'


@pytest.mark.parametrize('status', ['UPLOADED', 'PROCESSING', 'COMPLETED', 'FAILED'])
def test_recovery_does_not_cancel_or_release_accepted_work(tmp_path, status):
    app, settings = service(tmp_path)
    ident = 'match_accepted'
    reserve(app, ident)
    app.state.store.create(ident, 'accepted.mp4', '.mp4', len(VIDEO), settings.config)
    original = stored(app, f'uploads/{ident}/original.mp4', VIDEO)
    with app.state.store.connect() as conn:
        conn.execute('UPDATE analyses SET status=? WHERE id=?', (status, ident))
    before = app.state.store.get(ident)
    with TestClient(app) as client:
        assert client.post('/api/usage/reconcile').status_code == 200
        assert client.post('/api/usage/reconcile').status_code == 200
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == int(status in {'UPLOADED', 'PROCESSING'})
        assert usage['stored_bytes'] == usage['committed_bytes'] == len(VIDEO)
    assert app.state.store.get(ident) == before
    assert original.read_bytes() == VIDEO


def test_unmanaged_historical_reservation_stays_conservative(tmp_path):
    app, _ = service(tmp_path)
    reserve(app, 'match_old_unknown_request', managed=False)
    with TestClient(app) as client:
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 200
        assert response.json()['unverified_reservations'] == 1
        assert response.json()['reconciled_reservations'] == 0
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == 1 and usage['committed_bytes'] == 32
    assert active(app, 'match_old_unknown_request') == 1


def test_recovery_authentication_and_owner_scope_do_not_accept_owner_override(tmp_path):
    app, _ = service(tmp_path, require_auth=True)
    with TestClient(app) as alice, TestClient(app) as bob, TestClient(app) as guest:
        owners = []
        for client, username in ((alice, 'alice'), (bob, 'bob')):
            response = client.post('/api/auth/register', json={
                'username': username, 'password': 'recovery-test-password-only-123'})
            assert response.status_code == 201
            owners.append(response.json()['user']['id'])
        reserve(app, 'match_alice_orphan', owners[0])
        reserve(app, 'match_bob_orphan', owners[1])
        assert guest.post('/api/usage/reconcile').status_code == 401
        response = alice.post('/api/usage/reconcile', params={'owner_id': owners[1]})
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'private, no-store'
        assert response.json()['usage']['pending_jobs'] == 0
        assert active(app, 'match_alice_orphan') == 0
        assert active(app, 'match_bob_orphan') == 1
        assert bob.get('/api/usage').json()['pending_jobs'] == 1
        assert owners[1] not in response.text and 'match_bob_orphan' not in response.text
        assert bob.post('/api/usage/reconcile').status_code == 200
        assert active(app, 'match_bob_orphan') == 0


@pytest.mark.parametrize('kind', ['legacy', 'retry'])
def test_recovery_skips_an_actual_live_request_before_queue_commit(tmp_path, monkeypatch, kind):
    app, _ = service(tmp_path)
    jobs = app.state.store
    with TestClient(app) as client:
        source = None
        if kind == 'retry':
            first = client.post('/api/analyses?filename=source.mp4', content=VIDEO)
            assert first.status_code == 202
            source = first.json()['id']
            jobs.finish(source, 'synthetic failure')
        create = jobs.create
        reached, proceed = Event(), Event()

        def paused_create(*args, **kwargs):
            reached.set()
            assert proceed.wait(timeout=15), 'test did not release the paused real request'
            return create(*args, **kwargs)

        monkeypatch.setattr(jobs, 'create', paused_create)

        def submit():
            with TestClient(app) as uploading:
                if kind == 'legacy':
                    return uploading.post('/api/analyses?filename=active.mp4', content=VIDEO)
                return uploading.post('/api/analyses/' + source + '/retry',
                                      headers={'Idempotency-Key': RETRY_KEY})

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(submit)
            try:
                assert reached.wait(timeout=15)
                recovered = client.post('/api/usage/reconcile')
                assert recovered.status_code == 200
                assert recovered.json()['busy_reservations'] == 1
                assert recovered.json()['reconciled_reservations'] == 0
                assert client.get('/api/usage').json()['pending_jobs'] == 1
                with jobs.connect() as conn:
                    selected = conn.execute('SELECT analysis_id FROM resource_reservations WHERE active=1').fetchone()[0]
                assert active(app, selected) == 1
                other = client.post('/api/uploads', headers={'Idempotency-Key': 'while-live-request-other-0001'},
                                    json={'filename': 'other.mp4', 'size_bytes': 32})
                assert other.status_code == 429
            finally:
                proceed.set()
            accepted = future.result(timeout=15)
        assert accepted.status_code == 202
        assert accepted.json()['id'] == selected
        assert client.get('/api/usage').json()['pending_jobs'] == 1
        assert client.post('/api/usage/reconcile').status_code == 200
    assert (jobs.upload_dir(selected) / 'original.mp4').read_bytes() == VIDEO


def test_recovery_preserves_upload_creation_acceptance_and_rejection_tombstones(tmp_path):
    app, _ = service(tmp_path)
    app.state.quotas.settings = replace(app.state.quotas.settings, max_owner_pending_jobs=8)
    old_key, accepted_key = 'rejected-creation-intent-0001', 'accepted-creation-intent-0001'
    with TestClient(app) as client:
        bad = client.post('/api/uploads', headers={'Idempotency-Key': old_key},
                          json={'filename': 'bad.txt', 'size_bytes': 32})
        assert bad.status_code == 415 and bad.headers['Upload-Creation-Rejected'] == old_key
        saved = client.post('/api/uploads', headers={'Idempotency-Key': accepted_key},
                            json={'filename': 'retained.mp4', 'size_bytes': 32})
        assert saved.status_code == 201
        reserve(app, 'match_other_abandoned')
        with app.state.store.connect() as conn:
            before = {table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table)]
                      for table in ('upload_sessions', 'upload_creation_requests', 'upload_creation_rejections')}
        assert client.post('/api/usage/reconcile').status_code == 200
        with app.state.store.connect() as conn:
            after = {table: [tuple(row) for row in conn.execute('SELECT * FROM ' + table)] for table in before}
        assert after == before
        rejected_again = client.post('/api/uploads', headers={'Idempotency-Key': old_key},
                                     json={'filename': 'corrected.mp4', 'size_bytes': 32})
        assert rejected_again.status_code == 409
        replay = client.post('/api/uploads', headers={'Idempotency-Key': accepted_key},
                             json={'filename': 'retained.mp4', 'size_bytes': 32})
        assert replay.status_code == 201 and replay.json()['id'] == saved.json()['id']


@pytest.mark.parametrize('failure', ['before', 'after'])
def test_actual_process_exit_then_recovery_preserves_retry_identity_and_one_job(tmp_path, failure):
    app, settings = service(tmp_path, max_owner_storage_bytes=1024)
    with TestClient(app) as client:
        first = client.post('/api/analyses?filename=source.mp4', content=VIDEO)
        assert first.status_code == 202
        source = first.json()['id']
        app.state.store.finish(source, 'synthetic failure')
    record_path = tmp_path / ('recovery_child_' + failure + '.json')
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).parent / 'helpers/retry_intent_crash_child.py'),
                             str(settings.root), source, failure, str(record_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == (85 if failure == 'before' else 86), result.stderr
    record = json.loads(record_path.read_text(encoding='utf-8'))
    assert record['pid'] != os.getpid()
    assert record['forbidden_import_attempts'] == record['forbidden_loaded'] == []
    evidence = os.environ.get('TENNIS_RECOVERY_EVIDENCE_DIR')
    if evidence:
        directory = Path(evidence)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ('retry_' + failure + '.json')).open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)
    child = record['selected']['analysis_id']
    restarted = create_app(settings, start_worker=False)
    with TestClient(restarted) as client:
        before = client.get('/api/usage').json()
        assert before['pending_jobs'] == 1 and before['stored_bytes'] == 64
        recovery = client.post('/api/usage/reconcile')
        assert recovery.status_code == 200
        assert recovery.json()['unverified_reservations'] == 0
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == int(failure == 'after')
        assert usage['stored_bytes'] == usage['committed_bytes'] == 64
        with restarted.state.store.connect() as conn:
            selected = dict(conn.execute('SELECT * FROM analysis_retry_intents').fetchone())
        assert selected == record['selected']
        for _ in range(2):
            accepted = client.post('/api/analyses/' + source + '/retry',
                                   headers={'Idempotency-Key': RETRY_KEY})
            assert accepted.status_code == 202 and accepted.json()['id'] == child
        assert client.get('/api/usage').json()['pending_jobs'] == 1
    assert (app.state.store.upload_dir(child) / 'original.mp4').read_bytes() == VIDEO
    assert [job['id'] for job in app.state.store.list() if job['retry_of'] == source] == [child]


def test_unreadable_retained_storage_remains_attributed_and_admission_fails_closed(tmp_path, monkeypatch):
    app, _ = service(tmp_path)
    reserve(app, 'match_unreadable')
    path = stored(app, 'uploads/match_unreadable/upload.part', b'leftover')
    native_stat = app.state.quotas._stat

    def blocked(candidate):
        if candidate == path:
            raise PermissionError('private path must never reach the response')
        return native_stat(candidate)

    monkeypatch.setattr(app.state.quotas, '_stat', blocked)
    with TestClient(app) as client:
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 503
        assert str(path) not in response.text and 'private path' not in response.text
        # Reconciliation may safely commit individual orphan retirements before
        # returning its fresh snapshot. Unreadable owned storage still blocks
        # new admission; retaining the row must not make those bytes invisible.
        upload = client.post('/api/uploads', headers={'Idempotency-Key': 'unreadable-storage-next-0001'},
                             json={'filename': 'next.mp4', 'size_bytes': 32})
        assert upload.status_code == 503
    with app.state.store.connect() as conn:
        row = conn.execute('SELECT owner_id FROM resource_reservations WHERE analysis_id=?',
                           ('match_unreadable',)).fetchone()
        assert row['owner_id'] == 'local'
    assert path.read_bytes() == b'leftover'


def test_recovery_sql_failure_keeps_reservation_active_and_error_private(tmp_path):
    app, _ = service(tmp_path)
    reserve(app, 'match_sql_failure')
    with app.state.store.connect() as conn:
        conn.execute("CREATE TRIGGER reject_recovery BEFORE UPDATE OF active ON resource_reservations "
                     "WHEN OLD.analysis_id='match_sql_failure' AND NEW.active=0 "
                     "BEGIN SELECT RAISE(ABORT,'private fixture database failure'); END")
    with TestClient(app) as client:
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 503
        assert 'private fixture' not in response.text
    assert active(app, 'match_sql_failure') == 1


def test_old_schema_migration_preserves_unverifiable_legacy_rows(tmp_path):
    settings = Settings(tmp_path / 'old_storage', tmp_path / 'unused.yaml', max_upload_bytes=64)
    jobs = JobStore(settings.root)
    with jobs.connect() as conn:
        conn.execute('CREATE TABLE resource_reservations ('
                     'analysis_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,size_bytes INTEGER NOT NULL,'
                     'kind TEXT NOT NULL,active INTEGER NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL)')
        conn.execute("INSERT INTO resource_reservations VALUES "
                     "('match_historical','local',32,'legacy',1,'original-time','original-time')")
    for _ in range(2):
        app = create_app(settings, start_worker=False)
        with TestClient(app) as client:
            response = client.post('/api/usage/reconcile')
            assert response.status_code == 200
            assert response.json()['unverified_reservations'] == 1
            assert response.json()['usage']['pending_jobs'] == 1
            assert response.json()['usage']['committed_bytes'] == 32
        with jobs.connect() as conn:
            row = dict(conn.execute('SELECT * FROM resource_reservations').fetchone())
        assert row == {'analysis_id': 'match_historical', 'owner_id': 'local', 'size_bytes': 32,
                       'kind': 'legacy', 'active': 1, 'created_at': 'original-time',
                       'updated_at': 'original-time', 'recovery_version': 0}


@pytest.mark.parametrize('redirect', ['directory', 'leaf'])
def test_redirected_recovery_lock_path_is_rejected_without_following_or_releasing(tmp_path, redirect):
    app, _ = service(tmp_path)
    ident = 'match_unsafe_lock'
    reserve(app, ident)
    target = tmp_path / 'other_owner'
    target.mkdir()
    sentinel = target / 'private.lock'
    sentinel.write_bytes(b'untouched-owner-data')
    lock_directory = app.state.store.root / 'reservation-locks'
    if redirect == 'directory':
        lock_directory.symlink_to(target, target_is_directory=True)
    else:
        lock_directory.mkdir()
        (lock_directory / (ident + '.lock')).symlink_to(sentinel)
    with TestClient(app) as client:
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 503
        assert str(target) not in response.text and 'private.lock' not in response.text
    assert active(app, ident) == 1
    assert sentinel.read_bytes() == b'untouched-owner-data'
    assert list(target.iterdir()) == [sentinel]


def test_recovery_never_retires_a_reservation_bound_to_a_resumable_session(tmp_path):
    app, _ = service(tmp_path)
    with TestClient(app) as client:
        session = client.post('/api/uploads', headers={'Idempotency-Key': 'resumable-owner-session-0001'},
                              json={'filename': 'incomplete.mp4', 'size_bytes': 32})
        assert session.status_code == 201
        with app.state.store.connect() as conn:
            before = dict(conn.execute('SELECT * FROM upload_sessions').fetchone())
        reserve(app, before['analysis_id'])
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 200
        assert response.json()['reconciled_reservations'] == 0
        assert response.json()['unverified_reservations'] == 1
        assert response.json()['usage']['pending_jobs'] == 1
        assert active(app, before['analysis_id']) == 1
        with app.state.store.connect() as conn:
            after = dict(conn.execute('SELECT * FROM upload_sessions').fetchone())
        assert after == before


def test_recovery_cannot_release_a_legacy_reservation_during_actual_body_streaming(tmp_path):
    app, _ = service(tmp_path)

    async def exercise():
        reached, proceed = asyncio.Event(), asyncio.Event()

        async def chunks():
            yield VIDEO[:16]
            reached.set()
            await proceed.wait()
            yield VIDEO[16:]

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://127.0.0.1') as client:
            uploading = asyncio.create_task(client.post('/api/analyses?filename=streaming.mp4',
                                                         content=chunks(), headers={'Content-Length': '32'}))
            try:
                await asyncio.wait_for(reached.wait(), timeout=15)
                # These are separate ASGI receive messages, unlike a generator
                # handed to synchronous TestClient, which coalesces its body.
                response = await client.post('/api/usage/reconcile')
                assert response.status_code == 200
                assert response.json()['busy_reservations'] == 1
                usage = response.json()['usage']
                assert usage['pending_jobs'] == 1
                # The writer may buffer a small chunk until close. Whichever
                # bytes reached disk, the complete input remains charged.
                assert usage['stored_bytes'] in {0, 16}
                assert usage['reserved_input_bytes'] == 32 - usage['stored_bytes']
                assert usage['committed_bytes'] == 32
                with app.state.store.connect() as conn:
                    selected = conn.execute('SELECT analysis_id FROM resource_reservations WHERE active=1').fetchone()[0]
                assert (app.state.store.upload_dir(selected) / 'upload.part').read_bytes() in {b'', VIDEO[:16]}
                assert active(app, selected) == 1
            finally:
                proceed.set()
            accepted = await asyncio.wait_for(uploading, timeout=15)
            assert accepted.status_code == 202 and accepted.json()['id'] == selected
            assert (app.state.store.upload_dir(selected) / 'original.mp4').read_bytes() == VIDEO
            assert (await client.get('/api/usage')).json()['pending_jobs'] == 1

    asyncio.run(exercise())


@pytest.mark.parametrize('stage', ['reserve', 'create'])
@pytest.mark.parametrize('cancel_scope', ['request', 'all_private'])
def test_native_task_cancellation_keeps_lease_until_committing_thread_has_settled(tmp_path, monkeypatch, stage, cancel_scope):
    app, _ = service(tmp_path)
    reached, proceed, exited = Event(), Event(), Event()
    jobs, quotas = app.state.store, app.state.quotas
    if stage == 'create':
        native_create = jobs.create

        def paused_create(*args, **kwargs):
            reached.set()
            try:
                assert proceed.wait(timeout=15)
                return native_create(*args, **kwargs)
            finally:
                exited.set()

        monkeypatch.setattr(jobs, 'create', paused_create)
    else:
        native_reserve = quotas.reserve

        def committed_reserve(conn, *args, **kwargs):
            result = native_reserve(conn, *args, **kwargs)
            # Reproduce the durable-commit/before-thread-return boundary while
            # keeping the reservation helper's real writer connection alive.
            conn.commit()
            reached.set()
            try:
                assert proceed.wait(timeout=15)
                return result
            finally:
                exited.set()

        monkeypatch.setattr(quotas, 'reserve', committed_reserve)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://127.0.0.1') as client:
            uploading = asyncio.create_task(client.post('/api/analyses?filename=cancelled.mp4', content=VIDEO))
            try:
                assert await asyncio.to_thread(reached.wait, 15)
                def cancel_requests():
                    if cancel_scope == 'request':
                        uploading.cancel()
                    else:
                        # Simulate shutdown cancelling request/operation Tasks,
                        # while this test's coordinator remains able to check
                        # the live lease and then release the executor thread.
                        for task in asyncio.all_tasks():
                            if task is not asyncio.current_task():
                                task.cancel()

                cancel_requests()
                await asyncio.sleep(0)
                # Cancellation must wait for the thread rather than releasing
                # the lease while that thread can still commit accepted work.
                recovered = await client.post('/api/usage/reconcile')
                assert recovered.status_code == 200
                assert recovered.json()['busy_reservations'] == 1
                assert recovered.json()['reconciled_reservations'] == 0
                assert recovered.json()['usage']['pending_jobs'] == 1
                cancel_requests()
                await asyncio.sleep(0)
                repeated = await client.post('/api/usage/reconcile')
                assert repeated.status_code == 200
                assert repeated.json()['busy_reservations'] == 1
                assert repeated.json()['reconciled_reservations'] == 0
                assert repeated.json()['usage']['pending_jobs'] == 1
                other = await client.post('/api/uploads', headers={'Idempotency-Key': 'native-cancel-other-intent-0001'},
                                          json={'filename': 'other.mp4', 'size_bytes': 32})
                assert other.status_code == 429
            finally:
                proceed.set()
                assert await asyncio.to_thread(exited.wait, 15)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(uploading, timeout=15)
            usage = (await client.get('/api/usage')).json()
            if stage == 'create':
                job, = jobs.list()
                assert job['status'] == 'UPLOADED'
                assert usage['pending_jobs'] == 1
                assert (jobs.upload_dir(job['id']) / 'original.mp4').read_bytes() == VIDEO
            else:
                assert jobs.list() == []
                assert usage['pending_jobs'] == usage['committed_bytes'] == 0
            with jobs.connect() as conn:
                assert conn.execute('SELECT COUNT(*) FROM resource_reservations WHERE active=1').fetchone()[0] == 0

    asyncio.run(exercise())


@pytest.mark.parametrize('corruption', ['owner', 'size', 'source_owner', 'key'])
def test_unverifiable_retry_mapping_stays_charged_and_is_not_rewritten(tmp_path, corruption):
    from src.api.retry_intents import RetryIntents
    app, settings = service(tmp_path)
    jobs = app.state.store
    with TestClient(app) as client:
        source_response = client.post('/api/analyses?filename=source.mp4', content=VIDEO)
        assert source_response.status_code == 202
        source = source_response.json()['id']
        jobs.finish(source, 'synthetic failure')
        intents = RetryIntents(jobs, app.state.quotas)
        with intents.lock('local', RETRY_KEY):
            child = intents.admit('local', RETRY_KEY, source, 32)
        if corruption == 'source_owner':
            jobs.create('match_foreign_source', 'foreign.mp4', '.mp4', 32, settings.config, owner_id='foreign-owner')
            jobs.finish('match_foreign_source', 'synthetic foreign failure')
        with jobs.connect() as conn:
            assignments = {'owner': ('owner_id', 'foreign-owner'), 'size': ('size_bytes', 33),
                           'source_owner': ('source_analysis_id', 'match_foreign_source'),
                           'key': ('intent_key', '../not-a-valid-key')}
            column, value = assignments[corruption]
            conn.execute('UPDATE analysis_retry_intents SET ' + column + '=? WHERE analysis_id=?', (value, child))
            before = dict(conn.execute('SELECT * FROM analysis_retry_intents').fetchone())
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 200
        assert response.json()['unverified_reservations'] == 1
        assert response.json()['reconciled_reservations'] == 0
        assert response.json()['usage']['pending_jobs'] == 1
        assert response.json()['usage']['committed_bytes'] == 64
        assert active(app, child) == 1
        with jobs.connect() as conn:
            after = dict(conn.execute('SELECT * FROM analysis_retry_intents').fetchone())
        assert after == before
        assert jobs.get(child) is None
        assert (jobs.upload_dir(source) / 'original.mp4').read_bytes() == VIDEO


def test_simultaneous_recovery_retires_one_orphan_once_and_preserves_owned_bytes(tmp_path):
    app, _ = service(tmp_path)
    ident = 'match_concurrent_recovery'
    reserve(app, ident)
    partial = stored(app, f'uploads/{ident}/upload.part', b'partial!')
    scratch = stored(app, f'analyses/{ident}/leftover.bin', b'retained')
    barrier = Barrier(2)

    def recover():
        with TestClient(app) as client:
            barrier.wait(timeout=15)
            response = client.post('/api/usage/reconcile')
            assert response.status_code == 200
            return response.json()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recover) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]
    assert sum(item['reconciled_reservations'] for item in results) == 1
    assert sum(item['unverified_reservations'] for item in results) == 0
    assert active(app, ident) == 0
    with TestClient(app) as client:
        usage = client.get('/api/usage').json()
        assert usage['pending_jobs'] == usage['reserved_input_bytes'] == 0
        assert usage['stored_bytes'] == usage['committed_bytes'] == 16
    assert partial.read_bytes() == b'partial!' and scratch.read_bytes() == b'retained'
