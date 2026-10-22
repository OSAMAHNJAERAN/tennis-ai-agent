"""Software/storage contracts; these checks do not qualify tracking accuracy."""
from contextlib import closing
from dataclasses import replace
import json
import math
import sqlite3
import struct
import tracemalloc

import numpy as np
import pytest

from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils import temporal_ball_point_sequence as module
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence


def point(index):
    return TemporalBallPoint(index, index / 25, index + .125, index + .25,
                             confidence=.75, state=BallState.DETECTED,
                             source='high_conf_detector', velocity_px_per_sec=(1.25, -2.5))


def stored_payloads(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute('SELECT payload FROM temporal_points ORDER BY position').fetchall()


def test_single_pass_large_producer_bounded_commits_and_lazy_slice_do_not_retain_points(tmp_path, monkeypatch):
    with TemporalBallPointSequence(tmp_path) as points:
        finalized = []
        def producer():
            try:
                for index in range(12000):
                    assert len(points) == index  # Eager producer materialization fails here.
                    yield point(index)
            finally:
                finalized.append(True)
        tracemalloc.start()
        try:
            points.extend(producer())
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2 * 1024 ** 2
        assert finalized == [True]
        assert len(points) == 12000
        assert len(stored_payloads(points.path)) == 12000 // 256 * 256
        assert points._pending_mutations == 12000 % 256
        for pragma, expected in [('cache_size', -2048), ('mmap_size', 0), ('temp_store', 1)]:
            assert points._connection.execute(f'PRAGMA {pragma}').fetchone()[0] == expected
        decoded = []
        real_decode = module._decode_point
        monkeypatch.setattr(module, '_decode_point', lambda payload: decoded.append(True) or real_decode(payload))
        whole_match_view = points[:]
        reverse_view = whole_match_view[::-1]
        assert len(reverse_view) == 12000
        assert decoded == []
        assert reverse_view[0] == point(11999)
        assert decoded == [True]
        assert points[0] == point(0)
        owned = points.path
    assert not owned.exists()


def test_ieee_bits_numpy_scalars_nulls_source_and_all_states_survive_typed_storage(tmp_path):
    nan_bits = bytes.fromhex('7ff8000000000123')
    payload_nan = struct.unpack('!d', nan_bits)[0]
    precise = float.fromhex('0x1.0000000000001p+0')
    subnormal = float.fromhex('0x0.0000000000001p-1022')
    numpy_nan = np.array([0x7FC00123], dtype=np.uint32).view(np.float32)[0]
    original = TemporalBallPoint(2 ** 70, -0., precise, payload_nan, court_x_m=np.float64(subnormal),
                                 court_y_m=numpy_nan, confidence=None, state=BallState.OCCLUDED,
                                 source='source with unicode 🎾\n\u0000',
                                 velocity_px_per_sec=(float('inf'), float('-inf')), speed_kmh=np.float32(-0.))
    with TemporalBallPointSequence(tmp_path) as points:
        points.append(original)
        restored = points[0]
        assert isinstance(restored, TemporalBallPoint)
        assert restored.frame_index == 2 ** 70
        assert restored.source == original.source
        assert restored.state is BallState.OCCLUDED
        assert struct.pack('!d', restored.timestamp_seconds) == struct.pack('!d', -0.)
        assert struct.pack('!d', restored.y_px) == nan_bits
        assert restored.x_px.hex() == precise.hex()
        assert type(restored.court_x_m) is np.float64
        assert restored.court_x_m.tobytes() == original.court_x_m.tobytes()
        assert type(restored.court_y_m) is np.float32
        assert restored.court_y_m.tobytes() == numpy_nan.tobytes()
        assert math.copysign(1., restored.speed_kmh) == -1.
        assert restored.velocity_px_per_sec == (float('inf'), float('-inf'))
        assert restored.confidence is None
        for state in (*BallState, None):
            points.append(TemporalBallPoint(0, 0., None, None, state=state, source='none'))
            assert points[-1].state is state
            assert points[-1].velocity_px_per_sec is None
        points.flush()
        # Scratch is finite tagged JSON even when the original IEEE value is not.
        for (payload,) in stored_payloads(points.path):
            json.loads(payload, parse_constant=lambda value: pytest.fail(f'Nonfinite JSON token {value}'))


def test_indexed_snapshots_negative_indices_views_and_explicit_writeback(tmp_path):
    with TemporalBallPointSequence(tmp_path, transaction_size=4) as points:
        original = point(0)
        points.append(original)
        original.x_px = 999.
        assert points[0] == point(0)
        points.extend(point(index) for index in range(1, 9))
        assert points[-1] == point(8)
        assert points[-9] == point(0)
        assert list(points[2:7:2]) == [point(index) for index in (2, 4, 6)]
        assert list(points[::-1][1::3]) == [point(index) for index in (7, 4, 1)]
        assert len(points[100:200]) == 0
        snapshot = points[3]
        snapshot.x_px = 888.
        snapshot.source = 'explicit correction'
        assert points[3] == point(3)
        points[-6] = snapshot
        assert points[3] == snapshot
        view = points[8:0:-2]
        replacement = replace(view[1], state=BallState.PREDICTED, source='kinematic_kalman')
        view[1] = replacement
        assert points[6] == replacement
        for index, value in enumerate(points):
            value.court_x_m = float(index)
            points[index] = value
        assert [value.court_x_m for value in points] == list(range(9))
        with pytest.raises(TypeError, match='one ball point'):
            points[:] = point(0)
        with pytest.raises(TypeError, match='one ball point'):
            view[:] = point(0)
        for invalid in (-10, 9):
            with pytest.raises(IndexError):
                _ = points[invalid]
            with pytest.raises(IndexError):
                points[invalid] = point(0)


def test_updates_share_the_bounded_commit_batch_and_explicit_flush_persists(tmp_path):
    with TemporalBallPointSequence(tmp_path, transaction_size=2) as points:
        points.extend([point(0), point(1)])
        snapshot = replace(points[0], source='changed')
        points[0] = snapshot
        assert module._decode_point(stored_payloads(points.path)[0][0]).source == 'high_conf_detector'
        points[1] = replace(points[1], state=BallState.MISSING)
        assert module._decode_point(stored_payloads(points.path)[0][0]).source == 'changed'
        points.append(point(2))
        assert len(stored_payloads(points.path)) == 2
        points.flush()
        assert len(stored_payloads(points.path)) == 3


def test_failure_retains_accepted_closed_scratch_and_closes_producer(tmp_path):
    sequence = TemporalBallPointSequence(tmp_path)
    finalized = []
    def producer():
        try:
            yield point(17)
            raise LookupError('source timing producer failure')
        finally:
            finalized.append(True)
    with pytest.raises(LookupError, match='source timing producer failure'):
        with sequence:
            sequence.extend(producer())
    assert finalized == [True]
    assert sequence._connection is None
    assert sequence.path.exists()
    assert module._decode_point(stored_payloads(sequence.path)[0][0]) == point(17)
    with pytest.raises(RuntimeError, match='not open'):
        sequence.append(point(0))
    with pytest.raises(RuntimeError, match='cannot be reused'):
        with sequence:
            pass


def test_success_removes_only_owned_exact_scratch_names(tmp_path):
    unrelated = tmp_path / 'temporal_ball_points_unrelated.sqlite3'
    unrelated.write_bytes(b'other output')
    with TemporalBallPointSequence(tmp_path) as points:
        points.append(point(1))
        owned = points.path
        for suffix in ('-wal', '-shm'):
            owned.with_name(owned.name + suffix).write_bytes(b'owned sidecar')
    assert unrelated.read_bytes() == b'other output'
    assert not owned.exists()
    assert not owned.with_name(owned.name + '-wal').exists()
    assert not owned.with_name(owned.name + '-shm').exists()


def test_setup_failure_retains_exclusive_file_without_entering_producer(tmp_path, monkeypatch):
    def fail(path):
        raise sqlite3.OperationalError('storage unavailable')
    monkeypatch.setattr(module.sqlite3, 'connect', fail)
    sequence = TemporalBallPointSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match='storage unavailable'):
        with sequence:
            pytest.fail('Setup failure entered producer')
    assert sequence._connection is None
    assert sequence.path.parent == tmp_path.resolve()
    assert sequence.path.exists()


def test_cleanup_commit_failure_preserves_original_error_and_closes_connection(tmp_path, monkeypatch):
    real_connect = module.sqlite3.connect
    connections = []
    class Connection:
        fail = False
        closed = False
        def __init__(self, path):
            self.connection = real_connect(path)
        def execute(self, *args):
            return self.connection.execute(*args)
        def commit(self):
            if self.fail:
                raise sqlite3.OperationalError('commit unavailable')
            return self.connection.commit()
        def close(self):
            self.closed = True
            self.connection.close()
    def connect(path):
        connection = Connection(path)
        connections.append(connection)
        return connection
    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    sequence = TemporalBallPointSequence(tmp_path)
    with pytest.raises(LookupError, match='original failure') as caught:
        with sequence:
            sequence.append(point(0))
            connections[0].fail = True
            raise LookupError('original failure')
    assert connections[0].closed
    assert sequence._connection is None
    assert sequence.path.exists()
    assert any('commit unavailable' in note for note in caught.value.__notes__)


def test_successful_body_commit_failure_remains_failure_and_retains_scratch(tmp_path, monkeypatch):
    real_connect = module.sqlite3.connect
    class Connection:
        def __init__(self, path):
            self.connection = real_connect(path)
            self.commits = 0
            self.closed = False
        def execute(self, *args):
            return self.connection.execute(*args)
        def commit(self):
            self.commits += 1
            if self.commits > 1:
                raise sqlite3.OperationalError('final commit unavailable')
            return self.connection.commit()
        def close(self):
            self.closed = True
            self.connection.close()
    connections = []
    def connect(path):
        value = Connection(path)
        connections.append(value)
        return value
    monkeypatch.setattr(module.sqlite3, 'connect', connect)
    sequence = TemporalBallPointSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match='final commit unavailable'):
        with sequence:
            sequence.append(point(0))
    assert connections[0].closed
    assert sequence.path.exists()


@pytest.mark.parametrize('value', [0, -1, 257, True, np.bool_(True), 1.5, '256'])
def test_transaction_limit_cannot_create_unbounded_batches(tmp_path, value):
    with pytest.raises(ValueError, match='1 through 256'):
        TemporalBallPointSequence(tmp_path, transaction_size=value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('changes', [{'frame_index': True}, {'state': 'DETECTED'}, {'source': None},
                                   {'timestamp_seconds': None}, {'x_px': [1.]},
                                   {'velocity_px_per_sec': (1.,)}, {'velocity_px_per_sec': (1., None)}])
def test_invalid_record_is_rejected_before_replacing_an_accepted_point(tmp_path, changes):
    with TemporalBallPointSequence(tmp_path) as points:
        points.append(point(0))
        with pytest.raises(TypeError):
            points[0] = replace(point(1), **changes)
        assert points[0] == point(0)
        assert len(points) == 1


def test_missing_directory_is_not_created(tmp_path):
    missing = tmp_path / 'missing'
    with pytest.raises(FileNotFoundError):
        with TemporalBallPointSequence(missing):
            pass
    assert not missing.exists()


def test_missing_row_and_malformed_codec_fail_instead_of_manufacturing_a_point(tmp_path):
    with TemporalBallPointSequence(tmp_path) as points:
        points.append(point(0))
        points._connection.execute('DELETE FROM temporal_points WHERE position=0').close()
        with pytest.raises(RuntimeError, match='missing'):
            _ = points[0]
    document = json.loads(module._encode_point(point(0)))
    document['point']['unexpected'] = 'attribute'
    with pytest.raises(ValueError, match='Malformed'):
        module._decode_point(json.dumps(document))
    with pytest.raises(ValueError, match='numeric field'):
        module._decode_number(['numpy', 'O', '00'])
