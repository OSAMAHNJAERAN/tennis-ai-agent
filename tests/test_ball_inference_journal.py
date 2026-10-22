"""Retained raw-frame storage and real process loss; no pipeline/model resume."""
from contextlib import closing
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.bbox_utils import BBox
from src.utils import ball_inference_journal as storage
from src.utils.ball_inference_journal import BallInferenceJournal


IDENTITY = {'analysis_id': 'synthetic-journal-only', 'source': 'a' * 64,
            'config': {'qualified': False, 'scale': 1.0}, 'devices': ['cuda:0', 'cpu']}


def create(directory, identity=None, total_frames=3):
    return BallInferenceJournal.create(directory, IDENTITY if identity is None else identity, total_frames)


def reopen(directory, identity=None, total_frames=3):
    return BallInferenceJournal.open(directory, IDENTITY if identity is None else identity, total_frames)


def candidate(index=0):
    return BallObservation(index + .125, index + .375, .9,
                           BBox(index, index + .25, index + 2., index + 3., .9, 0, None))


def edit(path, sql, values=()):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(sql, values)
        connection.commit()


def test_exact_raw_and_empty_frames_survive_reopen_without_write_through(tmp_path):
    smallest = float.fromhex('0x0.0000000000001p-1022')
    precise = float.fromhex('0x1.0000000000001p+0')
    original = [BallObservation(-0., precise, smallest,
                    BBox(-0., smallest, precise, float('inf'), float('nan'), 32, None)),
                BallObservation(float('nan'), float('-inf'), None)]
    with create(tmp_path) as journal:
        identity = journal.identity
        identity['config']['qualified'] = True
        assert journal.identity == IDENTITY and journal.total_frames == 3
        journal.append(0, original)
        journal.append(np.int64(1), [])
        assert journal.next_frame == len(journal) == 2
        snapshot = journal[0]
        snapshot[0].x_px = 999
        assert math.copysign(1., journal[0][0].x_px) == -1
        path = journal.path
        assert journal[1] == []
        with pytest.raises(TypeError):
            journal[0] = []
    assert path == tmp_path / storage.JOURNAL_NAME and path.is_file()
    with reopen(tmp_path) as journal:
        values = journal[0]
        assert journal.next_frame == len(journal) == 2
        assert math.copysign(1., values[0].x_px) == -1
        assert values[0].y_px.hex() == precise.hex()
        assert values[0].confidence.hex() == smallest.hex()
        assert values[0].bbox.y1.hex() == smallest.hex()
        assert values[0].bbox.x2.hex() == precise.hex()
        assert math.isinf(values[0].bbox.y2) and math.isnan(values[0].bbox.confidence)
        assert values[0].bbox.class_id == 32 and values[0].bbox.track_id is None
        assert math.isnan(values[1].x_px) and values[1].y_px == float('-inf')
        assert values[1].confidence is None and values[1].bbox is None
        assert journal[1:] == [[]]
        journal.append(2, [candidate(2)])
        assert list(journal)[2] == [candidate(2)]
        with pytest.raises(ValueError, match='exact next'):
            journal.append(3, [])
    with pytest.raises(RuntimeError, match='not open'):
        len(journal)


def test_full_sync_cache_configuration_and_close_retain_only_known_owned_state(tmp_path):
    unrelated = tmp_path / 'unrelated.sqlite3'
    unrelated.write_bytes(b'keep')
    with create(tmp_path) as journal:
        assert journal._execute('PRAGMA synchronous', fetch=True) == (2,)
        assert journal._execute('PRAGMA cache_size', fetch=True) == (-2048,)
        assert journal._execute('PRAGMA temp_store', fetch=True) == (1,)
        assert journal._execute('PRAGMA mmap_size', fetch=True) == (0,)
        assert journal._execute('PRAGMA journal_mode', fetch=True) == ('delete',)
        journal.append(0, [])
        path = journal.path
    assert path.is_file() and unrelated.read_bytes() == b'keep'
    with pytest.raises(RuntimeError, match='cannot be reused'):
        journal.__enter__()


@pytest.mark.parametrize('index', [True, False, np.bool_(True), 0., None, '0', -1, 1 << 63])
def test_invalid_global_index_cannot_start_a_transaction(tmp_path, index):
    with create(tmp_path) as journal:
        with pytest.raises(ValueError):
            journal.append(index, [])
        assert journal.next_frame == 0 and not journal._connection.in_transaction
        with pytest.raises(ValueError):
            journal[index]
        assert journal.next_frame == 0


def test_gaps_repeats_and_reads_beyond_the_frontier_are_rejected(tmp_path):
    with create(tmp_path) as journal:
        with pytest.raises(ValueError, match='exact next'):
            journal.append(1, [])
        journal.append(0, [candidate()])
        with pytest.raises(ValueError, match='exact next'):
            journal.append(0, [])
        with pytest.raises(IndexError):
            journal[1]
        assert journal[0] == [candidate()]


@pytest.mark.parametrize('index', [slice(True, None), slice(None, np.bool_(True)), slice(None, None, False)])
def test_slice_boolean_aliases_are_rejected(tmp_path, index):
    with create(tmp_path) as journal:
        journal.append(0, [])
        with pytest.raises(ValueError, match='Boolean'):
            journal[index]


@pytest.mark.parametrize('count', [True, False, np.bool_(True), 3., None, '3', 0, -1, 1 << 63])
def test_invalid_declared_count_is_rejected_before_creating_files(tmp_path, count):
    with pytest.raises(ValueError):
        create(tmp_path, total_frames=count)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('identity', [{}, [], {'x': float('nan')}, {'x': float('inf')},
                                    {1: 'x'}, {'x': (1, 2)}, {'x': Path('source.mp4')}])
def test_nonplain_identity_is_rejected_before_creating_files(tmp_path, identity):
    with pytest.raises(ValueError):
        create(tmp_path, identity=identity)
    assert list(tmp_path.iterdir()) == []


def test_circular_or_oversized_identity_is_rejected_without_a_journal(tmp_path):
    circular = {}
    circular['self'] = circular
    for identity in (circular, {'x': 'x' * storage.MAX_IDENTITY_BYTES}):
        with pytest.raises(ValueError):
            create(tmp_path, identity=identity)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('change', ['owner', 'boolean', 'float', 'list_order', 'count'])
def test_identity_and_declared_frame_count_must_match_exactly(tmp_path, change):
    with create(tmp_path) as journal:
        journal.append(0, [])
    identity = deepcopy(IDENTITY)
    count = 3
    if change == 'owner':
        identity['analysis_id'] = 'another-analysis'
    elif change == 'boolean':
        identity['config']['qualified'] = 0
    elif change == 'float':
        identity['config']['scale'] = 1
    elif change == 'list_order':
        identity['devices'] = ['cpu', 'cuda:0']
    else:
        count = 4
    with pytest.raises(ValueError, match='identity|frame count'):
        with reopen(tmp_path, identity=identity, total_frames=count):
            pytest.fail('Changed identity was admitted')
    with reopen(tmp_path) as journal:
        assert journal.next_frame == 1


def test_create_is_exclusive_and_open_never_creates_a_missing_journal(tmp_path):
    with pytest.raises(FileNotFoundError):
        with reopen(tmp_path):
            pass
    assert list(tmp_path.iterdir()) == []
    with create(tmp_path) as journal:
        journal.append(0, [])
        path = journal.path
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        with create(tmp_path):
            pass
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['version', 'application', 'extra_table', 'altered_table',
                                    'gap', 'frontier', 'metadata_identity_bound'])
def test_reopen_rejects_malformed_schema_identity_or_frontier(tmp_path, mutation):
    with create(tmp_path) as journal:
        for index in range(3):
            journal.append(index, [])
        path = journal.path
    if mutation == 'version':
        edit(path, 'PRAGMA user_version=2')
    elif mutation == 'application':
        edit(path, 'PRAGMA application_id=0')
    elif mutation == 'extra_table':
        edit(path, 'CREATE TABLE injected(value TEXT)')
    elif mutation == 'altered_table':
        edit(path, 'DROP TABLE raw_frames')
        edit(path, 'CREATE TABLE raw_frames(frame_index INTEGER PRIMARY KEY,payload TEXT,payload_sha256 TEXT)')
    elif mutation == 'gap':
        edit(path, 'DELETE FROM raw_frames WHERE frame_index=1')
    elif mutation == 'frontier':
        edit(path, 'UPDATE journal_metadata SET next_frame=2')
    else:
        edit(path, 'UPDATE journal_metadata SET identity=?', ('x' * (storage.MAX_IDENTITY_BYTES + 1),))
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        with reopen(tmp_path):
            pytest.fail('Malformed journal was admitted')
    assert path.is_file()


@pytest.mark.parametrize('statement', [
    'CREATE VIEW sqliteX_view AS SELECT frame_index FROM raw_frames',
    'CREATE INDEX sqliteX_index ON raw_frames(payload_sha256)',
    "CREATE TRIGGER sqliteX_trigger AFTER INSERT ON raw_frames BEGIN UPDATE journal_metadata SET identity='{}'; END",
])
def test_sqlite_lookalike_objects_are_not_hidden_by_a_schema_wildcard(tmp_path, statement):
    with create(tmp_path) as journal:
        journal.append(0, [])
        path = journal.path
    edit(path, statement)
    with pytest.raises(ValueError, match='schema'):
        with reopen(tmp_path):
            pytest.fail('A non-reserved sqliteX schema object survived validation')
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute('SELECT next_frame FROM journal_metadata').fetchone() == (1,)
        assert connection.execute('SELECT identity FROM journal_metadata').fetchone() == (storage._identity(IDENTITY),)


def test_trigger_added_after_admission_cannot_execute_on_a_new_raw_frame(tmp_path):
    with create(tmp_path) as journal:
        path = journal.path
        with closing(sqlite3.connect(path)) as other:
            other.execute('CREATE TABLE trigger_marker(value INTEGER)')
            other.execute("CREATE TRIGGER late_trigger AFTER INSERT ON raw_frames BEGIN "
                          "INSERT INTO trigger_marker VALUES(1); "
                          "UPDATE journal_metadata SET identity='{}'; END")
            other.commit()
        with pytest.raises(ValueError, match='schema changed'):
            journal.append(0, [candidate()])
        with pytest.raises(RuntimeError, match='unusable'):
            journal.next_frame
    with closing(sqlite3.connect(path)) as inspection:
        assert inspection.execute('SELECT count(*) FROM trigger_marker').fetchone() == (0,)
        assert inspection.execute('SELECT count(*) FROM raw_frames').fetchone() == (0,)
        assert inspection.execute('SELECT next_frame,identity FROM journal_metadata').fetchone() == (
            0, storage._identity(IDENTITY))
    with pytest.raises(ValueError, match='schema'):
        with reopen(tmp_path):
            pass


@pytest.mark.parametrize('mutation', ['payload', 'hash', 'noncanonical', 'oversized', 'blob'])
def test_raw_row_hash_codec_and_size_are_checked_before_decode(tmp_path, monkeypatch, mutation):
    with create(tmp_path) as journal:
        journal.append(0, [candidate()])
        path = journal.path
    if mutation == 'payload':
        edit(path, 'UPDATE raw_frames SET payload=? WHERE frame_index=0', ('[]',))
    elif mutation == 'hash':
        edit(path, 'UPDATE raw_frames SET payload_sha256=? WHERE frame_index=0', ('G' * 64,))
    elif mutation == 'noncanonical':
        payload = '[ ]'
        edit(path, 'UPDATE raw_frames SET payload=?,payload_sha256=? WHERE frame_index=0',
             (payload, storage._raw_hash(0, payload)))
    elif mutation == 'oversized':
        monkeypatch.setattr(storage, 'MAX_FRAME_BYTES', 2)
    else:
        edit(path, 'UPDATE raw_frames SET payload=? WHERE frame_index=0', (sqlite3.Binary(b'[]'),))
    if mutation in ('payload', 'hash', 'oversized', 'blob'):
        monkeypatch.setattr(storage, '_decode_frame', lambda payload: pytest.fail('Invalid row reached the codec'))
    with reopen(tmp_path) as journal:
        with pytest.raises(ValueError, match='corrupt|canonical'):
            journal[0]
        with pytest.raises(RuntimeError, match='unusable'):
            journal.append(1, [])


def test_failed_or_oversized_frame_producer_does_not_accept_a_partial_frame(tmp_path, monkeypatch):
    failure = KeyboardInterrupt('producer interrupted')
    closed = []

    def broken():
        try:
            yield candidate()
            raise failure
        finally:
            closed.append(True)

    with create(tmp_path) as journal:
        with pytest.raises(KeyboardInterrupt) as caught:
            journal.append(0, broken())
        assert caught.value is failure and closed == [True]
        assert journal.next_frame == 0 and not journal._connection.in_transaction
        monkeypatch.setattr(storage, 'MAX_CANDIDATES_PER_FRAME', 1)
        with pytest.raises(ValueError, match='candidate bound'):
            journal.append(0, [candidate(), candidate(1)])
        monkeypatch.setattr(storage, 'MAX_FRAME_BYTES', 2)
        with pytest.raises(ValueError, match='payload bound'):
            journal.append(0, [candidate()])
        journal.append(0, [])
        assert journal.next_frame == 1


@pytest.mark.parametrize('failure', [RuntimeError('transaction interrupted'), KeyboardInterrupt('interrupted')])
def test_interrupt_between_row_and_frontier_rolls_back_and_requires_reopen(tmp_path, monkeypatch, failure):
    with create(tmp_path) as journal:
        execute = journal._execute

        def interrupted(sql, *args, **kwargs):
            if sql.startswith('UPDATE journal_metadata'):
                raise failure
            return execute(sql, *args, **kwargs)

        monkeypatch.setattr(journal, '_execute', interrupted)
        with pytest.raises(type(failure)) as caught:
            journal.append(0, [candidate()])
        assert caught.value is failure and journal._connection is None
        with pytest.raises(RuntimeError, match='unusable'):
            journal.next_frame
    with reopen(tmp_path) as journal:
        assert journal.next_frame == len(journal) == 0
        journal.append(0, [])


def test_commit_acknowledgement_loss_requires_reopen_and_does_not_duplicate_frame(tmp_path, monkeypatch):
    failure = KeyboardInterrupt('commit acknowledgement lost')
    with create(tmp_path) as journal:
        execute = journal._execute

        def interrupted(sql, *args, **kwargs):
            result = execute(sql, *args, **kwargs)
            if sql == 'COMMIT':
                raise failure
            return result

        monkeypatch.setattr(journal, '_execute', interrupted)
        with pytest.raises(KeyboardInterrupt) as caught:
            journal.append(0, [candidate()])
        assert caught.value is failure and journal._connection is None
        with pytest.raises(RuntimeError, match='unusable'):
            len(journal)
    with reopen(tmp_path) as journal:
        assert journal.next_frame == 1 and journal[0] == [candidate()]
        with pytest.raises(ValueError, match='exact next'):
            journal.append(0, [])
        journal.append(1, [])
        assert journal.next_frame == 2


def test_another_owner_advancing_the_frontier_requires_reopen(tmp_path):
    with create(tmp_path) as first:
        with reopen(tmp_path) as second:
            second.append(0, [])
        with pytest.raises(ValueError, match='frontier changed'):
            first.append(0, [candidate()])
        with pytest.raises(RuntimeError, match='unusable'):
            first.next_frame
    with reopen(tmp_path) as journal:
        assert journal.next_frame == 1 and journal[0] == []


def test_active_owner_rechecks_its_bound_identity_before_a_new_commit(tmp_path):
    with create(tmp_path) as journal:
        edit(journal.path, 'UPDATE journal_metadata SET identity=?', ('{}',))
        with pytest.raises(ValueError, match='identity or frontier changed'):
            journal.append(0, [])
    with closing(sqlite3.connect(journal.path)) as connection:
        assert connection.execute('SELECT count(*) FROM raw_frames').fetchone() == (0,)


def test_cleanup_failures_preserve_the_original_transaction_failure(tmp_path, monkeypatch):
    failure = KeyboardInterrupt('keep original transaction failure')
    with create(tmp_path) as journal:
        connection = journal._connection

        class BrokenCleanup:
            @property
            def in_transaction(self):
                return connection.in_transaction

            def execute(self, *args, **kwargs):
                return connection.execute(*args, **kwargs)

            def rollback(self):
                connection.rollback()
                raise ValueError('injected rollback acknowledgement')

            def close(self):
                connection.close()
                raise ValueError('injected close acknowledgement')

        journal._connection = BrokenCleanup()
        execute = journal._execute

        def interrupted(sql, *args, **kwargs):
            if sql.startswith('UPDATE journal_metadata'):
                raise failure
            return execute(sql, *args, **kwargs)

        monkeypatch.setattr(journal, '_execute', interrupted)
        with pytest.raises(KeyboardInterrupt) as caught:
            journal.append(0, [])
        assert caught.value is failure
        assert caught.value.__notes__ == [
            'Journal rollback also failed: injected rollback acknowledgement',
            'Journal close also failed: injected close acknowledgement']
    with reopen(tmp_path) as journal:
        assert journal.next_frame == 0


def test_creation_cleanup_failure_keeps_the_original_fsync_interruption(tmp_path, monkeypatch):
    failure = KeyboardInterrupt('creation fsync interrupted')
    descriptor = None
    native_open, native_close = storage.os.open, storage.os.close

    def remember(*args, **kwargs):
        nonlocal descriptor
        descriptor = native_open(*args, **kwargs)
        return descriptor

    def interrupted(value):
        assert value == descriptor
        raise failure

    def bad_close(value):
        native_close(value)
        if value == descriptor:
            raise ValueError('injected descriptor close acknowledgement')

    monkeypatch.setattr(storage.os, 'open', remember)
    monkeypatch.setattr(storage.os, 'fsync', interrupted)
    monkeypatch.setattr(storage.os, 'close', bad_close)
    with pytest.raises(KeyboardInterrupt) as caught:
        with create(tmp_path):
            pass
    assert caught.value is failure
    assert caught.value.__notes__ == [
        'Journal creation descriptor cleanup also failed: injected descriptor close acknowledgement']
    assert (tmp_path / storage.JOURNAL_NAME).is_file()


@pytest.mark.parametrize('body_failure', [False, True])
def test_context_exit_never_commits_unaccepted_pending_work(tmp_path, body_failure):
    journal = create(tmp_path)
    failure = RuntimeError('body failure')

    def run():
        with journal:
            journal.append(0, [])
            journal._execute('BEGIN IMMEDIATE')
            journal._execute('INSERT INTO raw_frames VALUES(1,?,?)', ('[]', storage._raw_hash(1, '[]')))
            journal._execute('UPDATE journal_metadata SET next_frame=2')
            if body_failure:
                raise failure

    if body_failure:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is failure
    else:
        run()
    with reopen(tmp_path) as journal:
        assert journal.next_frame == 1


def test_parent_traversal_missing_directory_and_existing_sidecar_are_rejected(tmp_path):
    for path in (tmp_path / '..' / tmp_path.name, tmp_path / 'missing'):
        with pytest.raises(ValueError):
            create(path)
    sidecar = tmp_path / (storage.JOURNAL_NAME + '-journal')
    sidecar.write_bytes(b'unrelated journal')
    with pytest.raises(FileExistsError):
        with create(tmp_path):
            pass
    assert sidecar.read_bytes() == b'unrelated journal'
    assert not (tmp_path / storage.JOURNAL_NAME).exists()


@pytest.mark.parametrize('where', ['directory', 'database', 'sidecar'])
@pytest.mark.parametrize('kind', ['symlink', 'junction'])
def test_reported_symlink_or_junction_is_rejected_before_sqlite_open(tmp_path, monkeypatch, where, kind):
    with create(tmp_path) as journal:
        path = journal.path
    selected = tmp_path if where == 'directory' else path if where == 'database' else Path(str(path) + '-journal')
    # Explicit reported-link fault injection is portable to Windows accounts
    # without symlink privileges; no actual filesystem redirect is installed.
    method = 'is_symlink' if kind == 'symlink' else 'is_junction'
    original = getattr(Path, method, lambda self: False)
    monkeypatch.setattr(Path, method, lambda self: self == selected or original(self), raising=False)
    monkeypatch.setattr(storage.sqlite3, 'connect', lambda *args, **kwargs: pytest.fail('Redirect reached SQLite'))
    with pytest.raises(ValueError, match='links|junctions'):
        with reopen(tmp_path):
            pass


def test_hardlinked_database_is_rejected_when_filesystem_supports_it(tmp_path):
    with create(tmp_path) as journal:
        path = journal.path
    alias = tmp_path / 'another-name.sqlite3'
    os.link(path, alias)
    with pytest.raises(ValueError, match='unaliased'):
        with reopen(tmp_path):
            pass
    assert path.is_file() and alias.is_file()


CHILD = r'''
import builtins, importlib.abc, json, os, pathlib, sys
root, directory, mode, identity = sys.argv[1:]
sys.path.insert(0, root)
native_import = builtins.__import__
attempts = []
guard_path = pathlib.Path(directory) / ('child_imports_' + mode + '.json')
def save_guard():
    with guard_path.open('w') as output:
        json.dump({'attempts': attempts,
            'forbidden_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules))}, output)
        output.flush()
        os.fsync(output.fileno())
def blocked(name, mechanism):
    attempts.append({'name': name, 'mechanism': mechanism})
    save_guard()
    raise AssertionError('Journal child attempted a vision runtime import: ' + name)
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        blocked(name, 'builtins.__import__')
    return native_import(name, *args, **kwargs)
class ImportTripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'ultralytics'}:
            blocked(fullname, 'meta_path')
        return None
builtins.__import__ = guarded
sys.meta_path.insert(0, ImportTripwire())
save_guard()
from src.utils.ball_inference_journal import BallInferenceJournal
from src.tracking.temporal_ball_tracker import BallObservation
with BallInferenceJournal.open(directory, json.loads(identity), 3) as journal:
    execute = journal._execute
    def interrupted(sql, *args, **kwargs):
        if sql != 'COMMIT':
            return execute(sql, *args, **kwargs)
        if mode == 'after':
            execute(sql, *args, **kwargs)
        observation = {'scope': 'RAW_JOURNAL_PROCESS_LOSS_ONLY_NO_MODEL_OR_PIPELINE_RESUME',
            'pid': os.getpid(), 'mode': mode, 'in_transaction': journal._connection.in_transaction,
            'import_attempts': attempts,
            'forbidden_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules))}
        save_guard()
        with (pathlib.Path(directory) / ('child_' + mode + '.json')).open('x') as output:
            json.dump(observation, output)
            output.flush()
            os.fsync(output.fileno())
        os._exit(71 if mode == 'before' else 72)
    journal._execute = interrupted
    journal.append(1, [BallObservation(1.125, 1.375, .9)])
raise AssertionError('Child did not hit the commit frontier')
'''


@pytest.mark.parametrize('mode,expected_frontier,exit_code', [('before', 1, 71), ('after', 2, 72)])
def test_actual_child_process_loss_resolves_atomic_committed_frontier(tmp_path, mode, expected_frontier, exit_code):
    with create(tmp_path) as journal:
        journal.append(0, [])
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, '-B', '-c', CHILD, str(root), str(tmp_path), mode,
                            json.dumps(IDENTITY)], cwd=root,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'),
                           capture_output=True, text=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert child.returncode == exit_code, (child.stdout, child.stderr)
    observation = json.loads((tmp_path / ('child_' + mode + '.json')).read_text())
    guard = json.loads((tmp_path / ('child_imports_' + mode + '.json')).read_text())
    assert observation['pid'] != os.getpid() and observation['mode'] == mode
    assert observation['in_transaction'] is (mode == 'before')
    assert observation['import_attempts'] == observation['forbidden_loaded'] == []
    assert guard['attempts'] == guard['forbidden_loaded'] == []
    with reopen(tmp_path) as journal:
        assert journal.next_frame == len(journal) == expected_frontier
        assert journal[0] == []
        if mode == 'after':
            assert journal[1] == [BallObservation(1.125, 1.375, .9)]
        journal.append(expected_frontier, [])
        assert journal.next_frame == expected_frontier + 1
