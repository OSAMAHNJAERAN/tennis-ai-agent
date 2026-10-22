"""Finite owned activity snapshots and lifetime/failure contracts; no inference."""
from collections.abc import Sequence
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from src.events.activity_state import VisualActivityState as State
from src.utils import activity_state_sequence as module
from src.utils.activity_state_sequence import ActivityStateSequence


class ConnectionProxy:
    def __init__(self, actual): self.actual = actual
    def __getattr__(self, name): return getattr(self.actual, name)


def test_states_frontier_and_detached_private_evidence(tmp_path):
    raw = [State.UNKNOWN, State.ACTIVE_PLAY, State.UNKNOWN, State.DEAD_BALL_VISUAL, State.UNKNOWN]
    evidence = {'max_detection_confidence': .08, 'is_active': False, 'integer': 2**80,
                'nested': [{'tuple': (-0., None, True)}],
                'array': np.array([[1., -0.]], dtype='>f8'), 'scalar': np.float32(.25)}
    with ActivityStateSequence(tmp_path) as owner:
        assert owner.require_empty() is None and len(owner) == 0
        for state in raw: owner.append(state, evidence)
        with pytest.raises(ValueError, match='fresh'): owner.require_empty()
        with pytest.raises(RuntimeError, match='Seal'): len(owner.states)
        evidence['nested'][0]['tuple'] = (99,)
        evidence['array'][0, 0] = 99
        statements = []
        owner._connection.set_trace_callback(statements.append)
        assert owner.seal(preplay_before=3) is owner
        assert not any(statement.lstrip().upper().startswith('UPDATE') for statement in statements)
        assert list(owner.states) == [State.PRE_PLAY, State.ACTIVE_PLAY, State.PRE_PLAY,
                                      State.DEAD_BALL_VISUAL, State.UNKNOWN]
        first = owner.evidence[0]
        assert first['nested'] == [{'tuple': (-0., None, True)}]
        assert struct.pack('!d', first['nested'][0]['tuple'][0]) == struct.pack('!d', -0.)
        assert first['array'].dtype == np.dtype('>f8') and first['array'][0, 0] == 1.
        assert type(first['scalar']) is np.float32 and first['scalar'] == .25
        first['nested'].append('changed')
        first['array'][0, 0] = 123
        assert owner.evidence[0]['array'][0, 0] == 1.
        assert len(owner.evidence[0]['nested']) == 1
        with pytest.raises(ValueError, match='only once'): owner.seal()
        with pytest.raises(ValueError, match='Sealed'): owner.append(State.UNKNOWN, {})
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))


def test_sequence_slices_are_lazy_and_lifetime_bound(tmp_path, monkeypatch):
    owner = ActivityStateSequence(tmp_path)
    for operation in (lambda: len(owner), owner.require_empty, lambda: len(owner.states)):
        with pytest.raises(RuntimeError, match='not open'): operation()
    with owner:
        for i in range(7): owner.append(State.UNKNOWN, {'frame': i})
        for view in (owner.states, owner.evidence):
            for operation in (lambda: len(view), lambda: view[:], lambda: next(iter(view))):
                with pytest.raises(RuntimeError, match='Seal'): operation()
        owner.seal(preplay_before=2)
        decoded = []
        actual = module._evidence
        def observe(payload):
            decoded.append(True)
            return actual(payload)
        monkeypatch.setattr(module, '_evidence', observe)
        subset = owner.evidence[::-1][1::2]
        assert isinstance(subset, Sequence) and not isinstance(subset, list)
        assert len(subset) == 3 and decoded == []
        assert [row['frame'] for row in subset] == [5, 3, 1]
        assert len(decoded) == 3
        assert subset[-1] == {'frame': 1}
        assert list(owner.states[::2]) == [State.PRE_PLAY, State.UNKNOWN, State.UNKNOWN, State.UNKNOWN]
        for index in (7, -8):
            with pytest.raises(IndexError): owner.states[index]
        with pytest.raises(TypeError): owner.evidence['0']
        suspended = iter(owner.evidence)
        assert next(suspended) == {'frame': 0}
        empty = owner.states[0:0]
    for operation in (lambda: len(subset), lambda: subset[:], lambda: list(empty), lambda: next(suspended)):
        with pytest.raises(RuntimeError, match='not open'): operation()
    with pytest.raises(RuntimeError, match='reused'): owner.__enter__()


@pytest.mark.parametrize('size', [0, -1, 257, True, np.bool_(False), 1.5, '2'])
def test_transaction_size_is_bounded_and_typed(tmp_path, size):
    with pytest.raises(ValueError): ActivityStateSequence(tmp_path, transaction_size=size)


@pytest.mark.parametrize('frontier', [-1, 2, True, np.bool_(True), .5, '0'])
def test_invalid_frontier_does_not_seal_or_mutate(tmp_path, frontier):
    with ActivityStateSequence(tmp_path) as owner:
        owner.append(State.UNKNOWN, {})
        with pytest.raises(ValueError): owner.seal(preplay_before=frontier)
        owner.seal(preplay_before=1)
        assert list(owner.states) == [State.PRE_PLAY]


@pytest.mark.parametrize('state,evidence', [
    ('UNKNOWN', {}), (None, {}), (State.UNKNOWN, []),
    (State.UNKNOWN, {'bad': float('nan')}), (State.UNKNOWN, {'bad': math.inf}),
    (State.UNKNOWN, {'bad': [np.float32(-math.inf)]}),
    (State.UNKNOWN, {'bad': np.array([1., math.inf])}),
    (State.UNKNOWN, {'bad': np.array([object()], dtype=object)}),
    (State.UNKNOWN, {'bad': {1, 2}}),
])
def test_invalid_input_precedes_mutation_and_does_not_poison_owner(tmp_path, state, evidence):
    with ActivityStateSequence(tmp_path) as owner:
        with pytest.raises((TypeError, ValueError)): owner.append(state, evidence)
        assert len(owner) == 0 and owner.require_empty() is None
        owner.append(State.ACTIVE_PLAY, {'valid': True})
        owner.seal()
        assert list(owner.states) == [State.ACTIVE_PLAY]


def test_empty_timeline_can_be_sealed_but_never_reused(tmp_path):
    with ActivityStateSequence(tmp_path) as owner:
        owner.require_empty()
        owner.seal()
        assert list(owner.states) == list(owner.evidence) == []
        with pytest.raises(ValueError, match='fresh'): owner.require_empty()
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))


@pytest.mark.parametrize('size', [1, 17, 256])
def test_sqlite_memory_pragmas_and_commit_batches(tmp_path, size):
    commits = []
    with ActivityStateSequence(tmp_path, transaction_size=size) as owner:
        assert owner._execute('PRAGMA cache_size', one=True) == (-2048,)
        assert owner._execute('PRAGMA mmap_size', one=True) == (0,)
        assert owner._execute('PRAGMA temp_store', one=True) == (1,)
        class Observe(ConnectionProxy):
            def commit(self):
                commits.append(owner._pending)
                return self.actual.commit()
        owner._connection = Observe(owner._connection)
        for index in range(515): owner.append(State.UNKNOWN, {'index': index})
        owner.seal()
        assert max(commits) <= size <= 256 and sum(commits) == 515
        assert len(owner.states) == 515 and owner.evidence[-1] == {'index': 514}


@pytest.mark.parametrize('changed', [False, True])
def test_sql_write_failure_poisoning_includes_uncertain_partial_insert(tmp_path, changed):
    class Failing(ConnectionProxy):
        def execute(self, sql, args=()):
            if sql.startswith('INSERT'):
                if changed:
                    cursor = self.actual.execute(sql, args)
                    cursor.close()
                raise OSError('Supplied SQL insertion failure')
            return self.actual.execute(sql, args)
    with ActivityStateSequence(tmp_path) as owner:
        owner._connection = Failing(owner._connection)
        with pytest.raises(OSError, match='SQL insertion'): owner.append(State.UNKNOWN, {})
        for operation in (lambda: len(owner), owner.require_empty, owner.seal,
                          lambda: owner.append(State.UNKNOWN, {}), lambda: list(owner.states)):
            with pytest.raises(RuntimeError, match='storage failure'): operation()
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))


@pytest.mark.parametrize('committed', [False, True])
def test_uncertain_commit_poisoning_blocks_all_views_and_reuse(tmp_path, committed):
    class Failing(ConnectionProxy):
        def commit(self):
            if committed: self.actual.commit()
            raise OSError('Supplied uncertain commit')
    with ActivityStateSequence(tmp_path) as owner:
        owner.append(State.UNKNOWN, {'valid': True})
        owner._connection = Failing(owner._connection)
        with pytest.raises(OSError, match='uncertain commit'): owner.seal()
        for operation in (lambda: len(owner.states), lambda: owner.evidence[:], owner.require_empty,
                          lambda: len(owner), lambda: owner.append(State.UNKNOWN, {}), owner.seal):
            with pytest.raises(RuntimeError, match='storage failure'): operation()
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))


@pytest.mark.parametrize('damage', ['missing_row', 'invalid_state', 'invalid_evidence', 'nonfinite_evidence'])
def test_corrupt_read_poisoning(tmp_path, damage):
    with ActivityStateSequence(tmp_path) as owner:
        owner.append(State.UNKNOWN, {'valid': True}); owner.seal()
        if damage == 'missing_row': owner._execute('DELETE FROM activity')
        elif damage == 'invalid_state': owner._execute("UPDATE activity SET state='ARBITRARY'")
        elif damage == 'invalid_evidence': owner._execute("UPDATE activity SET evidence='{}'")
        else:
            payload = json.dumps({'schema_version': 1, 'evidence': module._encode({'bad': math.nan})})
            owner._execute('UPDATE activity SET evidence=?', (payload,))
        with pytest.raises((RuntimeError, ValueError)): owner.evidence[0]
        with pytest.raises(RuntimeError, match='storage failure'): owner.states[0]


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_context_failure_cleans_exact_paths_and_preserves_primary(tmp_path, failure):
    unrelated = tmp_path/'other.sqlite3'
    unrelated.write_bytes(b'not-owned')
    supplied = failure('Supplied classifier or cancellation failure')
    with pytest.raises(failure) as caught:
        with ActivityStateSequence(tmp_path) as owner:
            owner.append(State.UNKNOWN, {}); owner.seal()
            for suffix in ('-journal', '-wal', '-shm'):
                Path(str(owner.path)+suffix).write_bytes(b'supplied-owned-sidecar')
            raise supplied
    assert caught.value is supplied and unrelated.read_bytes() == b'not-owned'
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))
    with pytest.raises(RuntimeError, match='storage failure'): len(owner)


def test_cleanup_error_is_not_allowed_to_replace_primary_failure(tmp_path, monkeypatch):
    primary = KeyboardInterrupt('Primary supplied cancellation')
    original_unlink = Path.unlink
    owner = ActivityStateSequence(tmp_path)
    def fail_one(path, *args, **kwargs):
        if path == owner.path: raise OSError('Supplied cleanup failure')
        return original_unlink(path, *args, **kwargs)
    with pytest.raises(KeyboardInterrupt) as caught:
        with owner:
            owner.append(State.UNKNOWN, {})
            monkeypatch.setattr(Path, 'unlink', fail_one)
            raise primary
    assert caught.value is primary
    assert any('Supplied cleanup failure' in note for note in primary.__notes__)
    monkeypatch.setattr(Path, 'unlink', original_unlink)
    assert owner.path.parent == tmp_path.resolve()
    owner.path.unlink()


def test_initialization_failure_cleans_owned_file_and_cannot_reenter(tmp_path, monkeypatch):
    execute = ActivityStateSequence._execute
    def fail_schema(self, sql, *args, **kwargs):
        if sql.startswith('CREATE TABLE'): raise OSError('Supplied schema failure')
        return execute(self, sql, *args, **kwargs)
    monkeypatch.setattr(ActivityStateSequence, '_execute', fail_schema)
    owner = ActivityStateSequence(tmp_path)
    with pytest.raises(OSError, match='schema'): owner.__enter__()
    with pytest.raises(RuntimeError, match='reused'): owner.__enter__()
    assert not list(tmp_path.glob('activity_states_*.sqlite3*'))
