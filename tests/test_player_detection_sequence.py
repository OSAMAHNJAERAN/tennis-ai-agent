"""Typed anatomical observations, bounded disk state and owned failure cleanup."""
from contextlib import closing
import json
import math
import sqlite3
import struct
import tracemalloc
import weakref

import numpy as np
import pytest

from src.detection.yolo26_adapters import PoseBBox
from src.utils.bbox_utils import BBox
from src.utils import player_detection_sequence as module
from src.utils.player_detection_sequence import PlayerDetectionSequence, _encode_frame


def person(index=0):
    return BBox(np.float32(index + .125), np.float64(-0.), index + 20., index + 80.,
                confidence=np.float32(.9), class_id=np.int16(0), track_id=index)


def rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute('SELECT COUNT(*) FROM player_frames').fetchone()[0]


def test_bbox_and_pose_annotations_preserve_numeric_bits_and_types(tmp_path):
    precise = float.fromhex('0x1.0000000000001p+0')
    subnormal = float.fromhex('0x0.0000000000001p-1022')
    nan = struct.unpack('!d', bytes.fromhex('fff8000000000042'))[0]
    base = BBox(-0., subnormal, precise, float('inf'), confidence=nan, class_id=32, track_id=None)
    base.ground_point_px = [np.float32(3.125), np.float64(4.25)]
    base.ground_point_source = 'MANUAL_FOOT_POINT'
    base.ground_point_quality = 'GROUND_CONTACT_UNVERIFIED'
    pose = PoseBBox(np.float32(.125), np.float64(2.5), 20., 90., confidence=np.float32(.8),
                    class_id=np.int16(0), track_id=2 ** 80,
                    keypoints_px=tuple((index + .125, index + .375, np.float32(.7)) for index in range(17)),
                    ground_point_px=(precise, -0.), ground_point_source='POSE_ANKLE_MIDPOINT_ESTIMATE',
                    ground_point_quality='ANKLES_VISIBLE_GROUND_CONTACT_UNVERIFIED', ground_point_confidence=np.float32(.7))
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.append([base, pose, person()])
        restored = sequence[0]
        assert _encode_frame(restored) == _encode_frame([base, pose, person()])
        assert type(restored[0]) is BBox and type(restored[1]) is PoseBBox
        assert restored[0].ground_point_px is not base.ground_point_px
        assert isinstance(restored[0].ground_point_px, list)
        assert math.copysign(1., restored[0].x1) == -1
        assert restored[0].y1.hex() == subnormal.hex()
        assert struct.pack('!d', restored[0].confidence).hex() == 'fff8000000000042'
        assert restored[1].track_id == 2 ** 80 and type(restored[1].x1) is np.float32
        assert type(restored[1].class_id) is np.int16 and type(restored[1].keypoints_px[0][2]) is np.float32


@pytest.mark.parametrize('keypoints', [tuple(), [], np.empty((0,), dtype='>f8'),
                                      np.arange(51, dtype='>f4').reshape(17, 3),
                                      [[index, index + 1, .5] for index in range(17)]])
def test_pose_point_container_dtype_endianness_and_snapshot_semantics(tmp_path, keypoints):
    pose = PoseBBox(1., 2., 10., 30., keypoints_px=keypoints)
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.append([pose])
        first, second = sequence[0][0], sequence[0][0]
        assert _encode_frame([first]) == _encode_frame([pose])
        assert first is not second and first is not pose
        if isinstance(keypoints, np.ndarray):
            assert first.keypoints_px.dtype.str == keypoints.dtype.str
            assert not np.shares_memory(first.keypoints_px, second.keypoints_px)
            if first.keypoints_px.size:
                first.keypoints_px.flat[0] = 999
                assert sequence[0][0].keypoints_px.flat[0] != 999


def test_order_empty_frames_random_indices_slices_and_explicit_replacement(tmp_path):
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.extend([[person(index)] if index % 3 else [] for index in range(12)])
        assert len(sequence) == 12 and sequence[0] == []
        assert [sequence[index][0].track_id for index in (8, 2, 4, 11, 1)] == [8, 2, 4, 11, 1]
        assert sequence[-1][0].track_id == 11
        assert [_encode_frame(frame) for frame in sequence[::-1]] == [_encode_frame(sequence[index]) for index in range(11, -1, -1)]
        snapshot = sequence[4]
        snapshot[0].x2 = 999
        snapshot.append(person(88))
        assert sequence[4][0].x2 != 999 and len(sequence[4]) == 1
        sequence[4] = snapshot
        assert _encode_frame(sequence[4]) == _encode_frame(snapshot)
        for index in (-13, 12):
            with pytest.raises(IndexError):
                sequence[index]
            with pytest.raises(IndexError):
                sequence[index] = []
        with pytest.raises(TypeError):
            sequence[1.5]
        with pytest.raises(TypeError):
            sequence[1:3] = []


def test_transactions_page_cache_and_owned_success_cleanup(tmp_path):
    unrelated = tmp_path / 'player_detections_unrelated.sqlite3'
    unrelated.write_bytes(b'preserve unrelated evidence')
    with PlayerDetectionSequence(tmp_path, transaction_size=2) as sequence:
        connection = sequence._connection
        assert connection.execute('PRAGMA cache_size').fetchone()[0] == -2048
        assert connection.execute('PRAGMA temp_store').fetchone()[0] == 1
        assert connection.execute('PRAGMA mmap_size').fetchone()[0] == 0
        assert connection.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        sequence.append([])
        assert rows(sequence.path) == 0
        sequence.append([person()])
        assert rows(sequence.path) == 2
        owned = sequence.path
        for suffix in ('-wal', '-shm'):
            type(owned)(str(owned) + suffix).write_bytes(b'owned sidecar')
    assert not owned.exists() and not type(owned)(str(owned) + '-wal').exists()
    assert unrelated.read_bytes() == b'preserve unrelated evidence'
    with pytest.raises(RuntimeError, match='cannot be reused'):
        sequence.__enter__()


def test_failed_lazy_producer_is_closed_and_accepted_records_retained(tmp_path):
    closed = []
    primary = RuntimeError('original producer failure')
    sequence = PlayerDetectionSequence(tmp_path)
    def producer():
        try:
            yield [person()]
            yield []
            raise primary
        finally:
            closed.append(True)
    with pytest.raises(RuntimeError) as raised:
        with sequence:
            sequence.extend(producer())
    assert raised.value is primary and closed == [True]
    assert sequence._connection is None and rows(sequence.path) == 2
    renamed = sequence.path.with_suffix('.diagnostic')
    sequence.path.rename(renamed)
    assert renamed.exists()


def test_producer_cleanup_failure_does_not_replace_original_storage_error(tmp_path):
    primary = TypeError('Player storage accepts only BBox and PoseBBox observations.')
    def producer():
        try:
            yield [person()]
            yield [object()]
        finally:
            raise RuntimeError('producer close failure')
    sequence = PlayerDetectionSequence(tmp_path)
    with pytest.raises(TypeError) as raised:
        with sequence:
            sequence.extend(producer())
    assert str(raised.value) == str(primary)
    assert any('producer close failure' in note for note in raised.value.__notes__)
    assert rows(sequence.path) == 1 and sequence._connection is None


@pytest.mark.parametrize('change,error', [
    ('unknown_attribute', TypeError), ('bad_keypoint_count', ValueError), ('object_points', ValueError),
    ('fractional_track', TypeError), ('bool_track', TypeError), ('noninteger_class', TypeError),
    ('complex_coordinate', TypeError), ('string_ground', TypeError), ('bad_ground_count', ValueError),
])
def test_invalid_schema_never_changes_a_previously_stored_record(tmp_path, change, error):
    value = PoseBBox(1., 2., 10., 20., track_id=1)
    if change == 'unknown_attribute':
        value.future_annotation = 'do not silently drop'
    elif change == 'bad_keypoint_count':
        value.keypoints_px = ((1., 2., .5),) * 16
    elif change == 'object_points':
        value.keypoints_px = np.full((17, 3), object(), dtype=object)
    elif change == 'fractional_track':
        value.track_id = 1.5
    elif change == 'bool_track':
        value.track_id = True
    elif change == 'noninteger_class':
        value.class_id = .5
    elif change == 'complex_coordinate':
        value.x1 = 1 + 2j
    elif change == 'string_ground':
        value.ground_point_px = 'foot'
    else:
        value.ground_point_px = (1., 2., 3.)
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.append([person()])
        before = _encode_frame(sequence[0])
        with pytest.raises(error):
            sequence[0] = [value]
        assert _encode_frame(sequence[0]) == before and len(sequence) == 1


def test_frame_bound_stops_an_unbounded_observation_generator(tmp_path):
    consumed, closed = [], []
    def unbounded():
        try:
            index = 0
            while True:
                consumed.append(index)
                yield person(index)
                index += 1
        finally:
            closed.append(True)
    with PlayerDetectionSequence(tmp_path, max_detections_per_frame=3) as sequence:
        observations = unbounded()
        with pytest.raises(ValueError, match='observation bound'):
            sequence.append(observations)
        assert len(consumed) == 4 and len(sequence) == 0 and closed == [True]


@pytest.mark.parametrize('payload', [
    '{"schemaVersion":true,"detections":[]}',
    '{"schemaVersion":1,"detections":[{"kind":"ExecuteArbitraryClass","box":{},"annotations":{}}]}',
    '{"schemaVersion":1,"detections":[],"unexpected":true}',
])
def test_malformed_scratch_schema_cannot_construct_arbitrary_objects(tmp_path, payload):
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.append([])
        sequence._execute('UPDATE player_frames SET payload=? WHERE frame_index=0', (payload,))
        with pytest.raises(ValueError):
            sequence[0]


def test_missing_pose_annotations_or_duplicate_fields_are_corruption(tmp_path):
    with PlayerDetectionSequence(tmp_path) as sequence:
        sequence.append([PoseBBox(1., 2., 10., 20.)])
        payload = json.loads(sequence._execute('SELECT payload FROM player_frames WHERE frame_index=0')[0])
        del payload['detections'][0]['annotations']['ground_point_quality']
        sequence._execute('UPDATE player_frames SET payload=? WHERE frame_index=0', (json.dumps(payload),))
        with pytest.raises(ValueError, match='Malformed typed'):
            sequence[0]
        sequence._execute('UPDATE player_frames SET payload=? WHERE frame_index=0',
                          ('{"schemaVersion":1,"schemaVersion":1,"detections":[]}',))
        with pytest.raises(ValueError, match='Duplicate typed'):
            sequence[0]


def test_sqlite_failure_preserves_primary_error_and_closed_diagnostic(tmp_path, monkeypatch):
    original = module.sqlite3.connect
    class Connection:
        def __init__(self, actual):
            self.actual = actual
        def execute(self, sql, values=()):
            if sql.startswith('INSERT'):
                raise sqlite3.OperationalError('injected disk full')
            return self.actual.execute(sql, values)
        def commit(self):
            self.actual.commit()
        def close(self):
            self.actual.close()
    monkeypatch.setattr(module.sqlite3, 'connect', lambda *args, **kwargs: Connection(original(*args, **kwargs)))
    sequence = PlayerDetectionSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match='injected disk full'):
        with sequence:
            sequence.append([person()])
    assert sequence._connection is None and sequence.path.exists()
    with closing(original(sequence.path)) as connection:
        assert connection.execute('SELECT COUNT(*) FROM player_frames').fetchone()[0] == 0


@pytest.mark.parametrize('parameter', ['transaction_size', 'max_detections_per_frame'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_resource_bounds_fail_before_creating_files(tmp_path, parameter, value):
    with pytest.raises(ValueError):
        PlayerDetectionSequence(tmp_path, **{parameter: value})
    assert list(tmp_path.iterdir()) == []


def test_twenty_five_thousand_frames_retain_only_current_boxes(tmp_path):
    counters = {'live': 0, 'peak': 0, 'closed': False}
    def released():
        counters['live'] -= 1
    with PlayerDetectionSequence(tmp_path) as sequence:
        def producer():
            try:
                for index in range(25000):
                    assert len(sequence) == index
                    value = person(index)
                    counters['live'] += 1
                    counters['peak'] = max(counters['peak'], counters['live'])
                    weakref.finalize(value, released)
                    yield [value]
            finally:
                counters['closed'] = True
        tracemalloc.start()
        sequence.extend(producer())
        first_current, first_peak = tracemalloc.get_traced_memory()
        assert counters['closed'] and counters['live'] == 0 and counters['peak'] < 3
        assert len(sequence) == 25000 and rows(sequence.path) == 25000 // 256 * 256
        assert sequence[24999][0].track_id == 24999
        assert sum(len(frame) for frame in sequence) == 25000
        final_current, final_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Includes Python codec/cursor allocation. SQLite's 2 MiB page cache is
        # independently asserted above; this is not process/GPU RSS evidence.
        assert max(first_peak, final_peak) < 2_000_000
        (tmp_path / 'player_storage_memory_fixture.json').write_text(json.dumps({
            'frames': 25000, 'peak_live_source_boxes': counters['peak'],
            'python_current_after_write_bytes': first_current, 'python_peak_after_write_bytes': first_peak,
            'python_current_after_read_bytes': final_current, 'python_peak_after_read_bytes': final_peak,
            'sqlite_cache_kib': 2048, 'scope': 'SYNTHETIC_STORAGE_FIXTURE_NOT_PIPELINE_RSS_OR_MATCH_CAPACITY'}, indent=2))
