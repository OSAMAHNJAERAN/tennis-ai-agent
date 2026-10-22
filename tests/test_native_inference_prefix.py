"""Typed completed-player prefix storage; no models or recovery claims."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import stat
import struct
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from src.court.calibration import CourtCalibration
from src.detection.yolo26_adapters import PoseBBox, _native_crop_options
from src.utils.frame_registration_sequence import FrameRegistration
from src.utils.json_array_artifact import JsonArrayArtifact
from src.utils import native_inference_prefix as module
from src.utils.native_inference_prefix import (NativePrefixFrame, DATABASE_NAME, SEAL_NAME,
    AUDIT_NAMES, write_native_inference_prefix, open_native_inference_prefix)


IDENTITY = {'analysis_id': 'synthetic-prefix', 'source': 'a' * 64, 'runtime': {'revision': 1}}


def video_metadata(count=3):
    return {'frame_count': count, 'fps': 60000 / 1001, 'width': 1920, 'height': 1080}


def crop_record():
    return {'schema_version': 1, 'role': 'players', 'selected_model_path': 'selected-player.pt',
            'tracking_policy': 'DISABLED_BOUND_SOURCE_CROPS', 'configured_device': 'auto',
            'resolved_device': 'cuda:2', 'input_spec': _native_crop_options({'imgsz': 640}, 'cuda:2'),
            'prediction_calls': 0, 'completed_prediction_calls': 0, 'failed': False}


def calibration(index=0):
    matrix = np.arange(9, dtype='>f8').reshape(3, 3) + index
    matrix[0, 0] = -0.
    bits = np.array([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    matrix[1, 0] = bits
    return CourtCalibration(matrix, (True, np.bool_(False), True),
                            struct.unpack('!d', bytes.fromhex('fff8000000000123'))[0],
                            np.float32(-0.), float('inf'), 'REJECTED_RAW_MATRIX_RETAINED')


def court_points(index=0):
    points = np.arange(28, dtype='>f4').reshape(14, 2) + index
    points[0, 0] = -0.
    points[1, 0] = np.array([0xFFC12345], dtype=np.uint32).view(np.float32)[0]
    return points


def prefix_frame(index):
    point = PoseBBox(np.float32(10.25), 20., 40., 90., confidence=.8123456789012345,
                    track_id=np.int64(7), keypoints_px=np.arange(51, dtype='>f4').reshape(17, 3),
                    ground_point_px=(np.float32(-0.), .125), ground_point_confidence=np.float32(.75))
    # Equal snapshots and identical local IDs are not exact proposal identities.
    p1, p2 = (None, point) if index % 3 == 1 else (point, point)
    if index % 3 == 2:
        p1 = p2 = None
    return NativePrefixFrame(p1, p2, {1: None if p1 is None else 3, 2: None if p2 is None else 7},
        FrameRegistration(calibration(index), np.asfortranarray(np.eye(3, dtype='>f4'))),
        {0: {'frame_index': index, 'source_pts': str(index * 1001), 'source_time_base': '1/60000',
             'source_timestamp_seconds': index * 1001 / 60000, 'measurement_transition_allowed': False},
         1: {'frame_index': index, 'observations': [], 'checkpoint': {'epoch': index, 'tuple': (-0., None)},
             'nested_integer_keys': {1: np.float32(.125)}},
         2: court_points(index)})


def audits(directory, count=3):
    for name in AUDIT_NAMES:
        with JsonArrayArtifact(directory / name, {'scope': 'SYNTHETIC_ONLY'}, allow_nan=False) as writer:
            for index in range(count):
                writer.append({'frame_index': index, 'synthetic': True})


def write(directory, *, count=3, records=None, identity=IDENTITY, metadata=None, crop=None):
    return write_native_inference_prefix(directory, identity,
        metadata=metadata if metadata is not None else video_metadata(count),
        initial_calibration=calibration(), initial_court_keypoints=court_points(),
        crop_execution=crop if crop is not None else crop_record(),
        frames=records if records is not None else (prefix_frame(index) for index in range(count)))


def assert_exact(left, right):
    assert type(left) is type(right)
    if isinstance(left, (np.ndarray, np.generic)):
        assert left.dtype.str == right.dtype.str and left.shape == right.shape
        assert left.tobytes(order='C') == right.tobytes(order='C')
    elif type(left) is float:
        assert struct.pack('!d', left) == struct.pack('!d', right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    elif hasattr(left, '__dataclass_fields__'):
        for name in left.__dataclass_fields__:
            assert_exact(getattr(left, name), getattr(right, name))
    else:
        assert left == right


def reseal_database(directory):
    # Deliberately forge the outer integrity binding in a synthetic fixture to
    # exercise inner schema/bound checks rather than only the file-hash guard.
    seal = json.loads((directory / SEAL_NAME).read_text())
    raw = (directory / DATABASE_NAME).read_bytes()
    seal['database'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (directory / SEAL_NAME).write_text(json.dumps(seal))


def test_exact_completed_prefix_round_trip_all_views_and_source_proposal_indices(tmp_path):
    audits(tmp_path)
    saved = write(tmp_path)
    assert saved['stage'] == 'NATIVE_PLAYER_PREFIX_COMPLETED'
    assert set(saved['audits']) == set(AUDIT_NAMES)
    with open_native_inference_prefix(tmp_path, IDENTITY, expected_metadata=video_metadata()) as prefix:
        assert len(prefix) == len(prefix.players[1]) == len(prefix.registrations.calibrations) == 3
        for index in (2, 0, 1):
            expected = prefix_frame(index)
            assert_exact(prefix[index], expected)
            assert_exact(prefix.players[1][index], expected.player1)
            assert_exact(prefix.players[2][index], expected.player2)
            assert_exact(prefix.players.selected_detection_indices[index], expected.source_indices)
            assert_exact(prefix.registrations[index], expected.registration)
            assert_exact(prefix.registrations.calibrations[index], expected.registration.calibration)
            assert_exact(prefix.registrations.transforms[index], expected.registration.anchor_to_frame_px)
            assert_exact(prefix.candidate_frames[index], expected.candidate_frame)
        assert prefix.players.index(1, 0) == 3 and prefix.players.index(2, 0) == 7
        assert prefix.players[1][0].track_id == prefix.players[2][0].track_id == 7
        assert prefix.players[1][1] is None and prefix.players[2][1] is not None
        assert prefix.players[1][2] is prefix.players[2][2] is None
        assert_exact(prefix.initial_calibration, calibration())
        assert_exact(prefix.initial_court_keypoints, court_points())
        assert prefix.crop_execution == crop_record() and prefix.metadata == video_metadata()
        assert prefix.seal == saved
        assert prefix[-1].source_indices == {1: None, 2: None}
        assert len(prefix.registrations[::-1]) == 3
    assert (tmp_path / DATABASE_NAME).exists() and (tmp_path / SEAL_NAME).exists()
    with pytest.raises(RuntimeError, match='not open'):
        prefix.players[1][0]
    with pytest.raises(RuntimeError, match='cannot be reused'):
        prefix.__enter__()


def test_detached_rows_arrays_headers_and_crop_policy_cannot_mutate_retained_state(tmp_path):
    audits(tmp_path)
    write(tmp_path)
    with open_native_inference_prefix(tmp_path, IDENTITY) as prefix:
        row = prefix[0]
        row.player1.keypoints_px[:] = 99
        row.registration.calibration.image_to_court[:] = 88
        row.candidate_frame[2][:] = 77
        row.candidate_frame[1]['checkpoint']['epoch'] = 999
        prefix.initial_court_keypoints[:] = 66
        prefix.initial_calibration.image_to_court[:] = 55
        prefix.crop_execution['input_spec']['device'] = 'cpu'
        prefix.metadata['fps'] = 1
        prefix.seal['database']['sha256'] = '0' * 64
        assert_exact(prefix[0], prefix_frame(0))
        assert_exact(prefix.initial_calibration, calibration())
        assert_exact(prefix.initial_court_keypoints, court_points())
        assert prefix.crop_execution['resolved_device'] == 'cuda:2'
        assert prefix.metadata['fps'] == video_metadata()['fps']
    with open_native_inference_prefix(tmp_path, IDENTITY) as reopened:
        assert_exact(reopened[0], prefix_frame(0))


@pytest.mark.parametrize('identity', [{'analysis_id': 'different'}, {**IDENTITY, 'runtime': {'revision': 1.}},
                                    {**IDENTITY, 'runtime': {'revision': True}}])
def test_changed_identity_is_rejected_with_strict_number_types(tmp_path, identity):
    audits(tmp_path)
    write(tmp_path)
    with pytest.raises(ValueError, match='identity/schema'):
        with open_native_inference_prefix(tmp_path, identity):
            pytest.fail('Changed identity admitted')


def test_changed_video_metadata_and_duplicate_creation_are_rejected(tmp_path):
    audits(tmp_path)
    write(tmp_path)
    database = (tmp_path / DATABASE_NAME).read_bytes()
    marker = (tmp_path / SEAL_NAME).read_bytes()
    with pytest.raises(ValueError, match='metadata differs'):
        with open_native_inference_prefix(tmp_path, IDENTITY, expected_metadata={**video_metadata(), 'fps': 30.}):
            pass
    with pytest.raises(FileExistsError, match='retained state'):
        write(tmp_path)
    assert (tmp_path / DATABASE_NAME).read_bytes() == database
    assert (tmp_path / SEAL_NAME).read_bytes() == marker


@pytest.mark.parametrize('name', AUDIT_NAMES)
def test_changed_audit_cannot_be_reopened(tmp_path, name):
    audits(tmp_path)
    write(tmp_path)
    with (tmp_path / name).open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='audit bindings differ'):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass


@pytest.mark.parametrize('fault', ['count', 'index', 'missing'])
def test_incomplete_or_unaligned_audits_never_publish_a_prefix(tmp_path, fault):
    audits(tmp_path)
    path = tmp_path / 'match_identity.json'
    if fault == 'missing':
        path.rename(tmp_path / 'retained_wrong_name.json')
    else:
        doc = json.loads(path.read_text())
        if fault == 'count':
            doc['frames'].pop()
        else:
            doc['frames'][1]['frame_index'] = 99
        path.write_text(json.dumps(doc))
    with pytest.raises((ValueError, FileNotFoundError)):
        write(tmp_path)
    assert not (tmp_path / SEAL_NAME).exists()
    assert not (tmp_path / DATABASE_NAME).exists()
    assert list(tmp_path.glob('.native_prefix.*.sqlite3'))  # Retained diagnostics.


@pytest.mark.parametrize('fault', ['fewer', 'more', 'indices', 'box', 'candidate_frame', 'boolean_frame'])
def test_bad_frame_alignment_does_not_publish_a_seal(tmp_path, fault):
    audits(tmp_path)
    rows = [prefix_frame(index) for index in range(3)]
    if fault == 'fewer':
        rows.pop()
    elif fault == 'more':
        rows.append(prefix_frame(3))
    elif fault == 'indices':
        rows[0] = replace(rows[0], source_indices={1: 3, 2: 3})
    elif fault == 'box':
        rows[0] = replace(rows[0], player1=None)
    else:
        rows[0].candidate_frame[1]['frame_index'] = True if fault == 'boolean_frame' else 55
    with pytest.raises(ValueError):
        write(tmp_path, records=iter(rows))
    assert not (tmp_path / SEAL_NAME).exists()


@pytest.mark.parametrize('change', [
    lambda value: value.update(frame_count=True), lambda value: value.update(frame_count=np.int64(3)),
    lambda value: value.update(frame_count=0), lambda value: value.update(fps=True),
    lambda value: value.update(fps=float('nan')), lambda value: value.update(width=0),
    lambda value: value.update(height=1.5),
])
def test_invalid_video_metadata_creates_no_prefix_files(tmp_path, change):
    metadata = video_metadata()
    change(metadata)
    with pytest.raises(ValueError):
        write(tmp_path, metadata=metadata)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('change', [
    lambda value: value.update(prediction_calls=1), lambda value: value.update(prediction_calls=False),
    lambda value: value.update(failed=True), lambda value: value.update(resolved_device='auto'),
    lambda value: value['input_spec'].update(device='cpu'), lambda value: value['input_spec'].update(nms=True),
])
def test_prefix_requires_clean_unused_explicit_crop_policy(tmp_path, change):
    value = crop_record()
    change(value)
    with pytest.raises(ValueError):
        write(tmp_path, crop=value)
    assert list(tmp_path.iterdir()) == []


def test_missing_marker_or_changed_database_cannot_be_read(tmp_path):
    audits(tmp_path)
    write(tmp_path)
    marker = tmp_path / SEAL_NAME
    backup = tmp_path / 'retained_seal.json'
    marker.rename(backup)
    with pytest.raises(FileNotFoundError):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass
    backup.rename(marker)
    with (tmp_path / DATABASE_NAME).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='database hash differs'):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass


@pytest.mark.parametrize('sql', ['DELETE FROM frames WHERE frame_index=1',
    'ALTER TABLE frames ADD COLUMN unexpected TEXT', 'DELETE FROM initial',
    "UPDATE header SET payload='{}'", "UPDATE frames SET candidate='corrupt' WHERE frame_index=0",
    'CREATE VIEW extra_view AS SELECT frame_index FROM frames',
    'CREATE INDEX extra_index ON frames(role_1)',
    'CREATE TRIGGER extra_trigger AFTER INSERT ON frames BEGIN SELECT 1; END'])
def test_schema_gaps_and_row_hash_corruption_are_rejected_beyond_outer_hash(tmp_path, sql):
    audits(tmp_path)
    write(tmp_path)
    with sqlite3.connect(tmp_path / DATABASE_NAME) as connection:
        connection.execute(sql)
    reseal_database(tmp_path)
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        with open_native_inference_prefix(tmp_path, IDENTITY) as prefix:
            prefix[0]


def test_oversized_row_is_rejected_before_full_row_fetch(tmp_path, monkeypatch):
    audits(tmp_path)
    write(tmp_path)
    with sqlite3.connect(tmp_path / DATABASE_NAME) as connection:
        connection.execute('UPDATE frames SET candidate=? WHERE frame_index=0', ('x' * (module.MAX_CANDIDATE_BYTES + 1),))
    reseal_database(tmp_path)
    original = module._connect
    statements = []

    def tracked(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(module, '_connect', tracked)
    with pytest.raises(ValueError, match='stored row exceeds'):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass
    assert not any(statement.startswith('SELECT frame_index,role_1,') for statement in statements)


def test_unsafe_candidate_array_shape_is_rejected_before_buffer_decode(tmp_path):
    audits(tmp_path)
    write(tmp_path)
    with sqlite3.connect(tmp_path / DATABASE_NAME) as connection:
        row = list(connection.execute('SELECT ' + ','.join(module._FRAME_COLUMNS) + ' FROM frames WHERE frame_index=0').fetchone())
        node = json.loads(row[7])
        # Replace the encoded integer-key 2 court observation with an object
        # array tag, then forge its row digest to reach the schema guard.
        assert node[0] == 'dict'
        target = next(pair for pair in node[1] if pair[0] == ['int', 2])
        target[1] = ['array', '|O', [14, 2], '00' * (28 * np.dtype('O').itemsize)]
        row[7] = json.dumps(node)
        row[8] = module._digest(row[:8])
        connection.execute('UPDATE frames SET candidate=?,row_sha256=? WHERE frame_index=0', (row[7], row[8]))
    reseal_database(tmp_path)
    with open_native_inference_prefix(tmp_path, IDENTITY) as prefix:
        with pytest.raises(ValueError, match='Unsafe native candidate'):
            prefix[0]


def test_marker_audit_count_boolean_cannot_alias_an_integer(tmp_path):
    audits(tmp_path, 1)
    write(tmp_path, count=1)
    seal = json.loads((tmp_path / SEAL_NAME).read_text())
    seal['audits'][AUDIT_NAMES[0]]['frames'] = True
    (tmp_path / SEAL_NAME).write_text(json.dumps(seal))
    with pytest.raises(ValueError, match='audit bindings differ'):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass


def test_interruption_before_marker_retains_unsealed_database_and_never_overwrites(tmp_path, monkeypatch):
    audits(tmp_path)
    unrelated = tmp_path / 'unrelated.evidence'
    unrelated.write_bytes(b'keep')
    native_link = module.os.link
    failure = KeyboardInterrupt('injected pre-seal process interruption')

    def interrupt(source, destination):
        if Path(destination).name == SEAL_NAME:
            raise failure
        return native_link(source, destination)

    monkeypatch.setattr(module.os, 'link', interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        write(tmp_path)
    assert caught.value is failure
    assert (tmp_path / DATABASE_NAME).is_file() and not (tmp_path / SEAL_NAME).exists()
    assert unrelated.read_bytes() == b'keep'
    with pytest.raises(FileNotFoundError):
        with open_native_inference_prefix(tmp_path, IDENTITY):
            pass
    with pytest.raises(FileExistsError):
        write(tmp_path)


def test_producer_failure_and_cleanup_failure_preserve_primary_exception(tmp_path):
    audits(tmp_path)
    primary = RuntimeError('injected player-prefix producer failure')

    class Producer:
        calls = closes = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.calls += 1
            if self.calls == 1:
                return prefix_frame(0)
            raise primary

        def close(self):
            self.closes += 1
            raise ValueError('injected producer close failure')

    source = Producer()
    with pytest.raises(RuntimeError) as caught:
        write(tmp_path, records=source)
    assert caught.value is primary and source.closes == 1
    assert primary.__notes__ == ['Native prefix producer cleanup also failed: injected producer close failure']
    assert not (tmp_path / SEAL_NAME).exists()
    for path in tmp_path.glob('.native_prefix.*.sqlite3'):
        with sqlite3.connect(path) as connection:
            connection.execute('PRAGMA quick_check').fetchone()


def test_typed_rows_are_consumed_lazily_and_reads_keep_no_snapshot_cache(tmp_path):
    count = 65
    audits(tmp_path, count)
    live, peak = 0, 0

    def released():
        nonlocal live
        live -= 1

    def producer():
        nonlocal live, peak
        for index in range(count):
            assert live < 3
            value = prefix_frame(index)
            live += 1
            peak = max(peak, live)
            weakref.finalize(value, released)
            yield value

    write(tmp_path, count=count, records=producer())
    assert live == 0 and peak <= 2
    with open_native_inference_prefix(tmp_path, IDENTITY) as prefix:
        assert len(prefix) == count
        for index in (64, 0, 32, 1):
            assert prefix[index].candidate_frame[0]['frame_index'] == index
        assert not hasattr(prefix, '_cache')
        assert prefix._one('PRAGMA cache_size') == (-2048,)
        assert prefix._one('PRAGMA mmap_size') == (0,)
        assert prefix._one('PRAGMA query_only') == (1,)


@pytest.mark.parametrize('kind', ['symlink', 'reparse'])
def test_directory_link_or_reparse_ancestor_is_rejected_before_writes(tmp_path, monkeypatch, kind):
    native_lstat = Path.lstat

    def linked(path):
        details = native_lstat(path)
        if path == tmp_path:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == 'symlink' else details.st_mode,
                                   st_file_attributes=0x400 if kind == 'reparse' else 0)
        return details

    monkeypatch.setattr(Path, 'lstat', linked)
    with pytest.raises(ValueError, match='links, junctions or reparse'):
        write(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_parent_traversal_and_oversized_identity_are_rejected_before_writes(tmp_path):
    with pytest.raises(ValueError, match='parent traversal'):
        write(tmp_path / 'nested' / '..')
    with pytest.raises(ValueError, match='64 KiB'):
        write(tmp_path, identity={'too_large': 'x' * module.MAX_IDENTITY_BYTES})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('where', ['one', 'schema_objects', 'table_columns'])
def test_cursor_read_and_close_failures_preserve_original_keyboard_interrupt(where):
    primary = KeyboardInterrupt('injected prefix cursor interruption')
    close_failure = ValueError('injected prefix cursor close failure')
    failed_cursors = []

    class Cursor:
        def __init__(self, *, one=None, many=None, fail=False):
            self.one, self.many, self.fail, self.closes = one, many, fail, 0
            if fail:
                failed_cursors.append(self)

        def fetchone(self):
            if self.fail:
                raise primary
            return self.one

        def fetchmany(self, maximum):
            assert maximum <= 10
            if self.fail:
                raise primary
            return self.many[:maximum]

        def fetchall(self):
            pytest.fail('Prefix schema read attempted unbounded fetchall')

        def close(self):
            self.closes += 1
            if self.fail:
                raise close_failure

    class Connection:
        def execute(self, statement, parameters=()):
            if where == 'one':
                return Cursor(fail=True)
            if statement.startswith('SELECT COUNT(*) FROM sqlite_master'):
                return Cursor(one=(3,))
            if statement.startswith('SELECT type,name FROM sqlite_master'):
                return Cursor(many=[('table', 'frames'), ('table', 'header'), ('table', 'initial')],
                              fail=where == 'schema_objects')
            if statement == 'PRAGMA table_info(frames)':
                return Cursor(fail=True)
            pytest.fail('Unexpected prefix schema probe: ' + statement)

    owner = object.__new__(module.NativeInferencePrefix)
    owner._open = lambda: Connection()
    with pytest.raises(KeyboardInterrupt) as caught:
        owner._one('SELECT injected') if where == 'one' else owner._validate_database()
    assert caught.value is primary and len(failed_cursors) == 1
    assert failed_cursors[0].closes == 1
    assert primary.__notes__ == ['Native prefix cursor cleanup also failed: injected prefix cursor close failure']


def test_cursor_close_failure_after_successful_read_still_surfaces():
    failure = ValueError('injected successful-read close failure')
    closes = []

    class Cursor:
        def fetchone(self):
            return (1,)

        def close(self):
            closes.append(True)
            raise failure

    owner = object.__new__(module.NativeInferencePrefix)
    owner._open = lambda: SimpleNamespace(execute=lambda statement, parameters=(): Cursor())
    with pytest.raises(ValueError) as caught:
        owner._one('SELECT 1')
    assert caught.value is failure and closes == [True]


def test_directory_sync_and_close_failures_preserve_original_interrupt(tmp_path, monkeypatch):
    primary = KeyboardInterrupt('injected directory fsync interruption')
    descriptor = 424242
    events = []

    def opened(path, flags):
        assert Path(path) == tmp_path
        events.append('open')
        return descriptor

    def synced(value):
        assert value == descriptor
        events.append('fsync')
        raise primary

    def closed(value):
        assert value == descriptor
        events.append('close')
        raise ValueError('injected directory descriptor close failure')

    # Exercise the POSIX cleanup branch without changing global os.name, which
    # would change pathlib's platform dispatch on the Windows test host.
    monkeypatch.setattr(module, '_DIRECTORY_SYNC_SUPPORTED', True)
    monkeypatch.setattr(module.os, 'open', opened)
    monkeypatch.setattr(module.os, 'fsync', synced)
    monkeypatch.setattr(module.os, 'close', closed)
    with pytest.raises(KeyboardInterrupt) as caught:
        module._sync_directory(tmp_path)
    assert caught.value is primary and events == ['open', 'fsync', 'close']
    assert primary.__notes__ == [
        'Native prefix directory descriptor cleanup also failed: injected directory descriptor close failure']


def test_directory_close_failure_after_successful_sync_surfaces(tmp_path, monkeypatch):
    failure = ValueError('injected directory descriptor close failure')
    descriptor = 424243
    events = []
    monkeypatch.setattr(module, '_DIRECTORY_SYNC_SUPPORTED', True)
    monkeypatch.setattr(module.os, 'open', lambda path, flags: descriptor)
    monkeypatch.setattr(module.os, 'fsync', lambda value: events.append(('fsync', value)))

    def closed(value):
        events.append(('close', value))
        raise failure

    monkeypatch.setattr(module.os, 'close', closed)
    with pytest.raises(ValueError) as caught:
        module._sync_directory(tmp_path)
    assert caught.value is failure
    assert events == [('fsync', descriptor), ('close', descriptor)]
