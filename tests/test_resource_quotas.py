"""Admission/software accounting only; no models, inference, or host probes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
from threading import Barrier
from types import SimpleNamespace

from fastapi import HTTPException
import pytest

from src.api.resource_quotas import ResourceQuotas


class Store:
    """The quota interface needs only the existing queue connection and root."""
    def __init__(self, root, *, optional_tables=True):
        self.root = root
        root.mkdir()
        self.db = root / 'queue.sqlite3'
        with self.connect() as conn:
            conn.execute('CREATE TABLE analyses '
                         '(id TEXT PRIMARY KEY,owner_id TEXT,status TEXT,size_bytes INTEGER)')
            if optional_tables:
                conn.execute('CREATE TABLE upload_sessions '
                             '(id TEXT PRIMARY KEY,owner_id TEXT,analysis_id TEXT,status TEXT,size_bytes INTEGER)')
                conn.execute('CREATE TABLE analysis_corrections '
                             '(id TEXT PRIMARY KEY,owner_id TEXT,status TEXT)')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


@pytest.fixture
def quota(tmp_path):
    store = Store(tmp_path / 'storage')
    settings = SimpleNamespace(max_owner_pending_jobs=8, max_owner_storage_bytes=1000, max_upload_bytes=100)
    return store, ResourceQuotas(store, settings)


def analysis(conn, ident, status='UPLOADED', owner='alice', size=32):
    conn.execute('INSERT INTO analyses VALUES (?,?,?,?)', (ident, owner, status, size))


def session(conn, ident, analysis_id, status='UPLOADING', owner='alice', size=32):
    conn.execute('INSERT INTO upload_sessions VALUES (?,?,?,?,?)', (ident, owner, analysis_id, status, size))


def write(store, relative, size):
    target = store.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'x' * size)
    return target


def test_snapshot_sees_callers_uncommitted_rows_without_opening_connections(quota, monkeypatch):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        analysis(conn, 'match_one')
        monkeypatch.setattr(store, 'connect', lambda: pytest.fail('snapshot opened another connection'))
        usage = quotas.snapshot(conn, 'alice')
    assert usage == {'pending_jobs': 1, 'pending_analyses': 1, 'pending_corrections': 0,
                     'stored_bytes': 0, 'reserved_input_bytes': 0, 'committed_bytes': 0,
                     'limits': {'pending_jobs': 8, 'storage_bytes': 1000}}


def test_pending_union_deduplicates_conversion_but_not_correction_ids(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_one', 'alice', 32)
        session(conn, 'upload_one', 'match_one')
        analysis(conn, 'match_one', 'PROCESSING')
        session(conn, 'upload_two', 'match_two')
        analysis(conn, 'match_done', 'COMPLETED')
        analysis(conn, 'match_failed', 'FAILED')
        conn.execute("INSERT INTO analysis_corrections VALUES ('match_one','alice','PROCESSING')")
        conn.execute("INSERT INTO analysis_corrections VALUES ('correction_done','alice','COMPLETED')")
        usage = quotas.snapshot(conn, 'alice')
    assert (usage['pending_jobs'], usage['pending_analyses'], usage['pending_corrections']) == (3, 2, 1)
    assert usage['reserved_input_bytes'] == 32  # Only the unqueued second session.


def test_partial_fulfils_reservation_but_chunk_temporary_remains_charged(quota):
    store, quotas = quota
    with store.connect() as conn:
        session(conn, 'upload_one', 'match_one')
        write(store, 'resumable/upload_one/upload.part', 10)
        write(store, 'resumable/upload_one/chunk_random.part', 5)
        usage = quotas.snapshot(conn, 'alice')
    assert (usage['stored_bytes'], usage['reserved_input_bytes'], usage['committed_bytes']) == (15, 22, 37)


def test_same_identity_reservations_use_maximum_declared_size_once(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_one', 'alice', 32)
        session(conn, 'upload_one', 'match_one', size=64)
        write(store, 'resumable/upload_one/upload.part', 16)
        usage = quotas.snapshot(conn, 'alice')
    assert usage['pending_jobs'] == 1
    assert (usage['stored_bytes'], usage['reserved_input_bytes'], usage['committed_bytes']) == (16, 48, 64)


def test_rename_before_queue_and_lost_ack_after_queue_are_not_double_charged(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_one', 'alice', 32)
        session(conn, 'upload_one', 'match_one')
        write(store, 'uploads/match_one/original.mp4', 32)
        assert quotas.snapshot(conn, 'alice')['committed_bytes'] == 32
        analysis(conn, 'match_one')
        write(store, 'analyses/match_one/.result/generations/old/result.json', 10)
        usage = quotas.snapshot(conn, 'alice')
        quotas.release(conn, 'match_one')
        quotas.release(conn, 'match_one')
        assert quotas.snapshot(conn, 'alice') == usage
    assert usage['pending_jobs'] == 1
    assert usage['stored_bytes'] == usage['committed_bytes'] == 42
    assert usage['reserved_input_bytes'] == 0


def test_cancel_and_failure_keep_all_retained_paths_attributed(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_orphan', 'alice', 32, 'retry')
        quotas.release(conn, 'match_orphan')
        analysis(conn, 'match_failed', 'FAILED')
        session(conn, 'upload_cancelled', 'match_cancelled', 'CANCELLED')
        conn.execute("INSERT INTO analysis_corrections VALUES ('correction_one','alice','CANCELLED')")
        write(store, 'uploads/match_orphan/original.mp4', 32)
        write(store, 'analyses/match_failed/.result/generations/old/report.html', 9)
        write(store, 'analyses/match_failed/native_scratch/leftover.bin', 4)
        write(store, 'analyses/match_failed.annotated_web_credit.json', 7)
        write(store, 'corrections/correction_one/attempt-old/processing.log', 6)
        write(store, 'resumable/upload_cancelled/chunk_leftover.part', 8)
        usage = quotas.snapshot(conn, 'alice')
    assert usage['pending_jobs'] == usage['reserved_input_bytes'] == 0
    assert usage['stored_bytes'] == usage['committed_bytes'] == 66


def test_hard_link_retry_is_charged_per_path(quota):
    store, quotas = quota
    with store.connect() as conn:
        analysis(conn, 'match_old', 'FAILED')
        analysis(conn, 'match_retry')
        original = write(store, 'uploads/match_old/original.mp4', 32)
        retried = store.root / 'uploads/match_retry/original.mp4'
        retried.parent.mkdir(parents=True)
        os.link(original, retried)
        assert quotas.snapshot(conn, 'alice')['stored_bytes'] == 64


def test_other_owners_files_reservations_and_work_are_excluded(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_alice', 'alice', 16)
        quotas.reserve(conn, 'match_bob', 'bob', 64)
        session(conn, 'upload_bob', 'match_bob', owner='bob', size=64)
        conn.execute("INSERT INTO analysis_corrections VALUES ('correction_bob','bob','QUEUED')")
        write(store, 'uploads/match_alice/original.mp4', 4)
        write(store, 'uploads/match_bob/original.mp4', 64)
        write(store, 'corrections/correction_bob/attempt-old/report.html', 12)
        usage = quotas.snapshot(conn, 'alice')
    assert usage['pending_jobs'] == 1 and usage['stored_bytes'] == 4
    assert usage['reserved_input_bytes'] == 12 and usage['committed_bytes'] == 16


@pytest.mark.parametrize(('pending_limit', 'storage_limit', 'status'), [(0, 1000, 429), (8, 31, 507)])
def test_rejected_reservation_leaves_no_row_even_if_caller_catches(quota, pending_limit, storage_limit, status):
    store, quotas = quota
    quotas.settings.max_owner_pending_jobs = pending_limit
    quotas.settings.max_owner_storage_bytes = storage_limit
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(HTTPException) as rejected:
            quotas.reserve(conn, 'match_one', 'alice', 32)
        assert rejected.value.status_code == status
        assert conn.execute('SELECT COUNT(*) FROM resource_reservations').fetchone()[0] == 0
        assert conn.in_transaction


def test_identical_reservation_is_idempotent_after_limits_are_lowered(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        first = quotas.reserve(conn, 'match_one', 'alice', 32)
        quotas.settings.max_owner_pending_jobs = 0
        again = quotas.reserve(conn, 'match_one', 'alice', 32)
        assert again['committed_bytes'] == first['committed_bytes'] == 32
        assert conn.execute('SELECT COUNT(*) FROM resource_reservations').fetchone()[0] == 1


@pytest.mark.parametrize(('owner', 'size', 'kind'), [('bob', 32, 'legacy'), ('alice', 33, 'legacy'),
                                                  ('alice', 32, 'retry')])
def test_reservation_identity_cannot_change_owner_size_or_kind(quota, owner, size, kind):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_one', 'alice', 32)
        with pytest.raises(HTTPException) as rejected:
            quotas.reserve(conn, 'match_one', owner, size, kind)
        assert rejected.value.status_code == 409
        quotas.release(conn, 'match_one')
        with pytest.raises(HTTPException) as released:
            quotas.reserve(conn, 'match_one', 'alice', 32)
        assert released.value.status_code == 409
        row = conn.execute('SELECT owner_id,active FROM resource_reservations').fetchone()
        assert tuple(row) == ('alice', 0)


@pytest.mark.parametrize('method', ['check', 'reserve', 'release'])
def test_admission_mutations_require_callers_transaction(quota, method):
    store, quotas = quota
    with store.connect() as conn:
        arguments = {'check': (conn, 'alice'), 'reserve': (conn, 'match_one', 'alice', 32),
                     'release': (conn, 'match_one')}
        with pytest.raises(RuntimeError, match='existing write transaction'):
            getattr(quotas, method)(*arguments[method])


def test_existing_transaction_gets_write_lock_before_any_snapshot_reads(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN')
        statements = []
        conn.set_trace_callback(statements.append)
        quotas.check(conn, 'alice', additional_jobs=1, additional_bytes=32)
    assert statements[0] == 'UPDATE resource_reservations SET active=active WHERE 0'


def test_batch_capacity_check_admits_whole_request_or_rejects_without_rows(quota):
    store, quotas = quota
    quotas.settings.max_owner_pending_jobs = 2
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.check(conn, 'alice', additional_jobs=2, additional_bytes=64)
        with pytest.raises(HTTPException) as rejected:
            quotas.check(conn, 'alice', additional_jobs=3, additional_bytes=96)
        assert rejected.value.status_code == 429
        assert conn.execute('SELECT COUNT(*) FROM resource_reservations').fetchone()[0] == 0


def test_unknown_length_reserves_maximum_while_resumed_partial_is_not_recharged(quota):
    store, quotas = quota
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        quotas.reserve(conn, 'match_unknown', 'alice', quotas.settings.max_upload_bytes)
        write(store, 'uploads/match_unknown/upload.part', 8)
        usage = quotas.snapshot(conn, 'alice')
        assert (usage['stored_bytes'], usage['reserved_input_bytes'], usage['committed_bytes']) == (8, 92, 100)
        assert quotas.reserve(conn, 'match_unknown', 'alice', 100)['committed_bytes'] == 100


def test_preexisting_crash_files_are_charged_during_new_reservation(quota):
    store, quotas = quota
    write(store, 'uploads/match_orphan/upload.part', 10)
    write(store, 'analyses/match_orphan/native_scratch/leftover.bin', 30)
    with store.connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        usage = quotas.reserve(conn, 'match_orphan', 'alice', 20)
    assert usage['stored_bytes'] == 40 and usage['reserved_input_bytes'] == 10
    assert usage['committed_bytes'] == 50


def test_worker_storage_guard_ignores_lowered_pending_limit_and_detects_growth(quota):
    store, quotas = quota
    quotas.settings.max_owner_pending_jobs = 0
    quotas.settings.max_owner_storage_bytes = 50
    with store.connect() as conn:
        analysis(conn, 'match_one', 'PROCESSING')
        write(store, 'uploads/match_one/original.mp4', 32)
        assert quotas.check_storage(conn, 'alice')['pending_jobs'] == 1
        write(store, 'analyses/match_one/native_scratch/growing.bin', 19)
        with pytest.raises(HTTPException) as rejected:
            quotas.check_storage(conn, 'alice')
    assert rejected.value.status_code == 507


@pytest.mark.parametrize('redirect', ['symlink', 'junction'])
def test_storage_redirects_fail_closed_without_following_them(quota, monkeypatch, redirect):
    store, quotas = quota
    target = write(store, 'uploads/match_one/original.mp4', 32).parent
    original_stat = Path.lstat

    def redirected(path):
        if path == target:
            return SimpleNamespace(st_mode=(stat.S_IFLNK if redirect == 'symlink' else stat.S_IFDIR),
                                   st_file_attributes=(0x400 if redirect == 'junction' else 0))
        return original_stat(path)

    monkeypatch.setattr(Path, 'lstat', redirected)
    with store.connect() as conn:
        analysis(conn, 'match_one')
        with pytest.raises(HTTPException) as rejected:
            quotas.snapshot(conn, 'alice')
    assert rejected.value.status_code == 503
    assert str(store.root) not in rejected.value.detail


@pytest.mark.parametrize('ident', ['../bob', '..', 'C:escape', 'match/bob', 'match\\bob', 'CON'])
def test_invalid_persisted_path_components_fail_closed(quota, ident):
    store, quotas = quota
    with store.connect() as conn:
        analysis(conn, ident, 'FAILED')
        with pytest.raises(HTTPException) as rejected:
            quotas.snapshot(conn, 'alice')
    assert rejected.value.status_code == 503
    assert ident not in rejected.value.detail


def test_unreadable_storage_fails_closed_with_sanitized_message(quota, monkeypatch):
    store, quotas = quota
    target = write(store, 'uploads/match_one/original.mp4', 32)
    original_stat = Path.lstat

    def unreadable(path):
        if path == target:
            raise PermissionError('private storage path and operating system details')
        return original_stat(path)

    monkeypatch.setattr(Path, 'lstat', unreadable)
    with store.connect() as conn:
        analysis(conn, 'match_one')
        with pytest.raises(HTTPException) as rejected:
            quotas.snapshot(conn, 'alice')
    assert rejected.value.status_code == 503
    assert 'private' not in rejected.value.detail and str(target) not in rejected.value.detail


def test_snapshot_supports_store_before_optional_router_tables_exist(tmp_path):
    store = Store(tmp_path / 'old_storage', optional_tables=False)
    settings = SimpleNamespace(max_owner_pending_jobs=32, max_owner_storage_bytes=20 * 1024 ** 3)
    quotas = ResourceQuotas(store, settings)
    with store.connect() as conn:
        analysis(conn, 'match_one', 'COMPLETED')
        write(store, 'uploads/match_one/original.mp4', 32)
        usage = quotas.snapshot(conn, 'alice')
    assert usage['pending_jobs'] == 0 and usage['stored_bytes'] == 32


def test_concurrent_admissions_serialize_against_same_owner_limit(quota):
    store, quotas = quota
    quotas.settings.max_owner_pending_jobs = 1
    together = Barrier(2)

    def admit(ident):
        with store.connect() as conn:
            together.wait(timeout=5)
            conn.execute('BEGIN IMMEDIATE')
            try:
                quotas.reserve(conn, ident, 'alice', 32)
            except HTTPException as error:
                return error.status_code
            return 202

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(admit, ['match_one', 'match_two']))
    assert sorted(outcomes) == [202, 429]
    with store.connect() as conn:
        assert quotas.snapshot(conn, 'alice')['pending_jobs'] == 1
        assert conn.execute('SELECT COUNT(*) FROM resource_reservations').fetchone()[0] == 1
