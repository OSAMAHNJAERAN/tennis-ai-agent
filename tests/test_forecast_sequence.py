"""Owned causal forecast storage; synthetic software checks, no model runtime."""
from collections.abc import MutableMapping, Sequence
from enum import IntEnum
import builtins
import gc
import json
from pathlib import Path
import sqlite3
import time
import tracemalloc

import pytest

from src.utils.forecast_sequence import ForecastSequence, MAX_HISTORY_BYTES


CAPACITY_OBSERVATIONS = []


@pytest.fixture(autouse=True)
def no_numerical_or_model_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in ('torch', 'ultralytics', 'numpy', 'pandas', 'cv2'):
            raise AssertionError('Forecast storage must not import heavy runtimes: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def test_actual_sqlite_configuration_indexed_history_and_detached_point_tuples(tmp_path):
    with ForecastSequence(tmp_path, transaction_size=3) as owner:
        assert owner._open().execute('PRAGMA cache_size').fetchone()[0] == -2048
        assert owner._open().execute('PRAGMA mmap_size').fetchone()[0] == 0
        assert owner._open().execute('PRAGMA temp_store').fetchone()[0] == 1
        assert owner._open().execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        assert isinstance(owner.history, Sequence) and isinstance(owner.point_index, MutableMapping)
        assert not owner.history and not owner.point_index
        value = {'status': 'uncalibrated', 'probabilities': {'match': {'player_1': .6, 'player_2': .4}}}
        owner.history.append(value)
        value['probabilities']['match']['player_1'] = .9
        assert owner.history[-1]['probabilities']['match']['player_1'] == .6
        detached = owner.history[0]
        detached['status'] = 'CHANGED_CALLER_COPY'
        assert owner.history[0]['status'] == 'uncalibrated'
        owner.point_index['point-1'] = (1, 2, 1.125)
        assert owner.point_index['point-1'] == (1, 2, 1.125)
        assert type(owner.point_index['point-1']) is tuple
        assert 'point-1' in owner.point_index and owner.point_index.get('absent') is None
        assert owner.pending == 2
        owner.point_index['point-1'] = (1, 2, 1.125)
        assert owner.pending == 2 and len(owner.point_index) == 1
        owner.history.append({'status': 'unavailable'})
        assert owner.pending == 0  # Three SQL writes across both collections.
        assert [row['status'] for row in owner.history] == ['uncalibrated', 'unavailable']
        assert list(owner.point_index) == ['point-1']
        assert owner.history[-1] == {'status': 'unavailable'}
        assert owner.history[::-1][0] == {'status': 'unavailable'}
        assert list(owner.history[::2]) == [owner.history[0]]
        with pytest.raises(TypeError): owner.history[::2].append({})
        with pytest.raises(TypeError): owner.history[True]
        with pytest.raises(IndexError): owner.history[2]
        with pytest.raises(IndexError): owner.history[-3]
        assert owner.seal() is owner and owner.sealed
        with pytest.raises(ValueError, match='sealed'): owner.history.append({})
        with pytest.raises(ValueError, match='sealed'): owner.point_index['point-2'] = (1, 1, 2.)
        with pytest.raises(ValueError, match='sealed'): owner.point_index['point-1'] = (1, 2, 1.125)
        assert owner.history[-1]['status'] == 'unavailable'
        path, history, index = owner.path, owner.history[:], owner.point_index
    assert not path.exists() and not list(tmp_path.iterdir())
    for action in (lambda: len(history), lambda: history[0], lambda: history[-1],
                   lambda: len(index), lambda: index.get('point-1'), lambda: 'point-1' in index,
                   lambda: next(iter(index)), lambda: owner._open()):
        with pytest.raises(RuntimeError, match='not open'):
            action()
    with pytest.raises(ValueError, match='reused'): owner.__enter__()


@pytest.mark.parametrize('key', ['', ' ', '\t\r\n', 1, 1.0, True, None, 'x' * 201])
def test_point_id_primitives_are_strict_on_insert_lookup_and_membership(tmp_path, key):
    with ForecastSequence(tmp_path) as owner:
        for action in (lambda: owner.point_index.__setitem__(key, (1, 2, 1.)),
                       lambda: owner.point_index.__getitem__(key), lambda: key in owner.point_index):
            with pytest.raises(ValueError, match='point ID'):
                action()
        assert len(owner.point_index) == 0


@pytest.mark.parametrize('value', [None, [1, 2, 1.], (1, 2), (1, 2, 1., 0),
    (True, 2, 1.), (1., 2, 1.), (1, False, 1.), (1, 2., 1.), (3, 1, 1.),
    (1, 3, 1.), (1, 2, True), (1, 2, -1), (1, 2, float('nan')),
    (1, 2, float('inf')), (1, 2, float('-inf')), (1, 2, '1')])
def test_point_value_container_participants_and_clock_are_strict(tmp_path, value):
    with ForecastSequence(tmp_path) as owner:
        with pytest.raises(ValueError): owner.point_index['p'] = value
        assert len(owner.point_index) == 0


def test_point_ids_cannot_be_rewritten_deleted_or_popped(tmp_path):
    with ForecastSequence(tmp_path) as owner:
        owner.point_index['p'] = (1, 2, 4)
        owner.point_index['p'] = (1, 2, 4.)  # Same timestamp, retain original representation.
        assert type(owner.point_index['p'][2]) is int
        for value in ((2, 2, 4), (1, 1, 4), (1, 2, 5)):
            with pytest.raises(ValueError, match='different outcome'): owner.point_index['p'] = value
        for action in (lambda: owner.point_index.__delitem__('p'), lambda: owner.point_index.clear(),
                       lambda: owner.point_index.pop('p'), lambda: owner.point_index.pop('absent', None),
                       lambda: owner.point_index.popitem()):
            with pytest.raises(TypeError, match='append-only'): action()
        assert owner.point_index['p'] == (1, 2, 4) and len(owner.point_index) == 1
        with pytest.raises(KeyError): owner.point_index['absent']


@pytest.mark.parametrize('player_ids', [(1, 1), (True, 2), (1., 2), (0, 2), (-1, 2),
    (1,), (1, 2, 3), '12', None])
def test_invalid_player_configuration_is_rejected_before_storage(tmp_path, player_ids):
    with pytest.raises(ValueError): ForecastSequence(tmp_path, player_ids=player_ids)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('size', [True, 0, 257, 1.0, '1', None])
def test_invalid_transaction_configuration_is_rejected(tmp_path, size):
    with pytest.raises(ValueError): ForecastSequence(tmp_path, transaction_size=size)


def test_nondefault_participants_are_bound_to_the_index(tmp_path):
    with ForecastSequence(tmp_path, player_ids=[17, 43], transaction_size=1) as owner:
        owner.point_index['p'] = (43, 17, .25)
        assert owner.point_index['p'] == (43, 17, .25) and owner.pending == 0
        with pytest.raises(ValueError): owner.point_index['q'] = (1, 2, .5)


class NumericEnum(IntEnum):
    ONE = 1


@pytest.mark.parametrize('value', [[], None, {'number': float('nan')}, {'number': float('inf')},
    {1: 'integer-key'}, {'tuple': (1, 2)}, {'set': {1}}, {'bytes': b'x'},
    {'enum': NumericEnum.ONE}, {'model': object()}])
def test_history_accepts_only_plain_finite_json_objects(tmp_path, value):
    with ForecastSequence(tmp_path) as owner:
        with pytest.raises(ValueError): owner.history.append(value)
        assert len(owner.history) == 0


def test_history_byte_limit_includes_utf8_encoding_and_preserves_exact_boundary(tmp_path):
    with ForecastSequence(tmp_path) as owner:
        # Compact encoding is {"x":"..."}: eight bytes besides its string.
        owner.history.append({'x': 'a' * (MAX_HISTORY_BYTES - 8)})
        assert len(json.dumps(owner.history[0], separators=(',', ':')).encode()) == MAX_HISTORY_BYTES
        for value in ({'x': 'a' * (MAX_HISTORY_BYTES - 7)}, {'x': 'é' * (MAX_HISTORY_BYTES // 2)},
                      {'x': 'a' * (MAX_HISTORY_BYTES * 2)}):
            with pytest.raises(ValueError, match='64 KiB'): owner.history.append(value)
        assert len(owner.history) == 1


def test_circular_and_invalid_unicode_history_is_rejected(tmp_path):
    circular = {}
    circular['again'] = circular
    with ForecastSequence(tmp_path) as owner:
        with pytest.raises(ValueError, match='circular'): owner.history.append(circular)
        with pytest.raises(ValueError, match='encodable'): owner.history.append({'invalid': '\ud800'})
        assert not owner.history


@pytest.mark.parametrize('error', [RuntimeError('EXPLICIT_BODY_FAILURE'), KeyboardInterrupt('EXPLICIT_CANCEL')])
def test_failure_and_cancellation_close_suspended_readers_and_remove_exact_owned_files(tmp_path, error):
    unrelated = tmp_path / 'keep.txt'
    unrelated.write_text('keep', encoding='utf-8')
    with pytest.raises(type(error), match=str(error)):
        with ForecastSequence(tmp_path) as owner:
            owner.history.append({'sequence': 1})
            owner.history.append({'sequence': 2})
            owner.point_index['p'] = (1, 2, 1.)
            rows, points = iter(owner.history), iter(owner.point_index)
            assert next(rows) == {'sequence': 1} and next(points) == 'p'
            assert len(owner._readers) == 2
            path = owner.path
            raise error
    assert not path.exists() and list(tmp_path.iterdir()) == [unrelated]
    assert owner.connection is None and not owner._readers
    with pytest.raises(RuntimeError): next(rows)
    with pytest.raises(RuntimeError): next(points)


def test_cleanup_failure_preserves_the_primary_failure_and_explains_cleanup(tmp_path, monkeypatch):
    original = Path.unlink
    with monkeypatch.context() as patch:
        def fail_owned(path, *args, **kwargs):
            if path.name.startswith('forecast_sequence_') and path.suffix == '.sqlite3':
                raise PermissionError('EXPLICIT_DELETE_FAILURE')
            return original(path, *args, **kwargs)
        patch.setattr(Path, 'unlink', fail_owned)
        with pytest.raises(RuntimeError, match='EXPLICIT_PRIMARY_FAILURE') as captured:
            with ForecastSequence(tmp_path) as owner:
                owner.history.append({})
                path = owner.path
                raise RuntimeError('EXPLICIT_PRIMARY_FAILURE')
        assert any('EXPLICIT_DELETE_FAILURE' in note for note in captured.value.__notes__)
        assert owner.connection is None
    path.unlink()
    assert not list(tmp_path.iterdir())


def test_a_failed_database_commit_invalidates_views_until_cleanup(tmp_path):
    class FailingConnection:
        def __init__(self, connection): self.connection = connection
        def commit(self): raise sqlite3.OperationalError('EXPLICIT_COMMIT_FAILURE')
        def rollback(self): return self.connection.rollback()
        def close(self): return self.connection.close()
    with ForecastSequence(tmp_path) as owner:
        owner.history.append({'before': True})
        owner.connection = FailingConnection(owner.connection)
        with pytest.raises(sqlite3.OperationalError, match='EXPLICIT_COMMIT_FAILURE'): owner.flush()
        for action in (lambda: len(owner.history), lambda: owner.history.append({}),
                       lambda: len(owner.point_index), lambda: owner.seal()):
            with pytest.raises(RuntimeError, match='transaction failure'): action()
    assert not list(tmp_path.iterdir())


def test_directory_admission_fails_without_creating_or_removing_other_files(tmp_path):
    missing = tmp_path / 'missing'
    for directory in (missing, tmp_path / '..' / tmp_path.name):
        with pytest.raises(ValueError):
            with ForecastSequence(directory): pytest.fail('Unsafe directory yielded')
    assert not list(tmp_path.iterdir())


def test_growing_history_and_point_cardinality_keeps_owned_python_memory_bounded(tmp_path):
    measurements = []
    for count in (257, 8193):
        directory = tmp_path / str(count)
        directory.mkdir()
        gc.collect()
        started = time.perf_counter()
        tracemalloc.start()
        with ForecastSequence(directory) as owner:
            for index in range(count):
                point_id = 'point-' + str(index)
                owner.point_index[point_id] = (1 if index % 2 else 2, 1, index / 30.)
                owner.history.append({'point_id': point_id, 'sequence': index,
                    'probabilities': {'match': {'player_1': .6, 'player_2': .4}},
                    'status': 'uncalibrated', 'calibration_status': 'not_calibrated'})
                assert owner.pending < owner.transaction_size
            owner.seal()
            assert len(owner.history) == len(owner.point_index) == count
            assert owner.history[-1]['sequence'] == count - 1
            assert owner.point_index['point-' + str(count - 1)][2] == (count - 1) / 30.
            assert sum(1 for _ in owner.history) == count
            assert sum(1 for _ in owner.point_index) == count
            gc.collect()
            retained, peak = tracemalloc.get_traced_memory()
            measurements.append({'history_rows': count, 'point_ids': count,
                'retained_bytes': retained, 'peak_bytes': peak,
                'database_bytes': owner.path.stat().st_size,
                'elapsed_seconds': time.perf_counter() - started,
                'scope': 'WHOLE_STORAGE_CONTEXT_WITH_GROWING_HISTORY_AND_POINT_IDS; SQLITE_NATIVE_CACHE_EXCLUDED'})
        tracemalloc.stop()
        assert not list(directory.iterdir())
    CAPACITY_OBSERVATIONS.extend(measurements)
    assert measurements[1]['retained_bytes'] - measurements[0]['retained_bytes'] < 128 * 1024
    assert measurements[1]['peak_bytes'] - measurements[0]['peak_bytes'] < 512 * 1024
    assert measurements[1]['database_bytes'] > measurements[0]['database_bytes'] * 10
