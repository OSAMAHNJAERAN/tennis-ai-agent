"""Deterministic equivalence and bounded storage; no player accuracy claim."""

from collections.abc import Sequence
from contextlib import closing
import gc
import json
import math
import sqlite3
import tracemalloc
import weakref

import numpy as np
import pytest

from src.court.calibration import CourtCalibration
from src.detection.yolo26_adapters import PoseBBox
from src.tracking.court_player_tracker import CourtPlayerTracker
from src.tracking.player_identity import select_court_players_disk
from src.utils.bbox_utils import BBox
from src.utils.player_detection_sequence import PlayerDetectionSequence, _encode_frame


def calibration(dx=0.):
    return CourtCalibration(np.array([[1 / 30, 0., -(100. + dx) / 30],
                                     [0., 1 / 30, -100. / 30], [0., 0., 1.]]))


def box(x, y, track_id, *, confidence=.9, dx=0.):
    center, foot = x * 30 + 100 + dx, y * 30 + 100
    return BBox(center - 12, foot - 60, center + 12, foot, confidence=confidence, track_id=track_id)


def assert_exact_selection(tracker, frames, tmp_path, *, fps=30., calibrations=None):
    expected = tracker.select(frames, fps, calibrations)
    with select_court_players_disk(tracker, frames, fps, tmp_path, calibrations) as selected:
        assert tuple(selected) == (1, 2)
        assert len(selected) == 2
        for role in (1, 2):
            assert isinstance(selected[role], Sequence)
            assert len(selected[role]) == len(frames)
            for index, original in enumerate(expected[role]):
                actual = selected[role][index]
                expected_index = (None if original is None else
                                  next(i for i, candidate in enumerate(frames[index]) if candidate is original))
                assert selected.index(role, index) == expected_index
                assert selected.selected_detection_indices[index][role] == expected_index
                if original is None:
                    assert actual is None
                else:
                    assert actual is not original
                    assert _encode_frame([actual]) == _encode_frame([original])
            assert len(list(selected[role])) == len(frames)
        paths = selected.storage_paths
        assert all(path.is_file() for path in paths)
    assert all(not path.exists() for path in paths)
    return expected


def test_dynamic_geometry_gaps_memory_loss_and_source_indices_match_legacy(tmp_path):
    frames = [
        [box(-5, -6, 99), box(5, 25, 1), box(5, -2, 2)],
        [box(5.1, 25, 1, dx=220), box(5.1, -2, 2, dx=220)],
        [box(0, 25, 77, dx=220), box(5.1, -2, 2, dx=220)],
        [],
        [box(5, 25, 3), box(5, -2, 4)],
        [box(0, 25, 77), box(5, -2, 4)],
    ]
    geometry = [calibration(), calibration(220), calibration(220),
                CourtCalibration(None, rejection_reason='CAMERA_LOST'), calibration(), calibration()]
    expected = assert_exact_selection(CourtPlayerTracker(calibration()), frames, tmp_path, calibrations=geometry)
    assert expected[1][2] is None
    assert expected[1][3] is None
    assert expected[1][4].track_id == 3
    assert expected[1][5] is None


def test_duplicate_counts_equal_proposals_none_ids_ties_and_mutual_exclusion(tmp_path):
    # Presence counts include repeated proposals, and ties retain source order.
    tie_one, tie_two = box(5, 25, None), box(5, 25, None)
    equal_one, equal_two = box(5, 25, 8), box(5, 25, 8)
    frames = [[tie_one, tie_two, box(5, -2, None)], [],
              [equal_one, equal_two, box(5, -2, 8), box(5, -2, 9)],
              [box(5, 25, 8), box(5, 25, 8), box(5, -2, 9)]]
    expected = assert_exact_selection(CourtPlayerTracker(calibration(), memory_seconds=0), frames, tmp_path)
    assert expected[1][0] is tie_one
    assert expected[1][2] is equal_one
    assert expected[2][2].track_id == 9


def test_whole_segment_presence_prior_is_preserved_including_duplicates(tmp_path):
    # A repeated farther ID wins the initial prior over the nearer one-off ID.
    frames = [[box(5.5, 25, 1), box(6, 25, 2), box(6, 25, 2)]]
    frames += [[box(6, 25, 2), box(6, 25, 2)] for _ in range(10)]
    expected = assert_exact_selection(CourtPlayerTracker(calibration()), frames, tmp_path)
    assert expected[1][0].track_id == 2


@pytest.mark.parametrize('track_id', [np.int64(17), np.uint64(2**64 - 1), 2**180, -(2**90)])
def test_numpy_and_wide_integral_id_keys_preserve_exact_scalar_types(tmp_path, track_id):
    first = box(5, 25, track_id)
    frames = [[first, box(5, -2, None)], [box(5.1, 25, track_id)]]
    assert_exact_selection(CourtPlayerTracker(calibration()), frames, tmp_path)


@pytest.mark.parametrize('annotation_form', ['tuple', 'list', 'numpy'])
def test_pose_and_attached_ground_point_fields_survive_selection(tmp_path, annotation_form):
    first = box(0, 20, 5)
    # The observed anatomical ground estimate, rather than bbox bottom, drives
    # the exact same candidate point in both implementations.
    first.ground_point_px = (250., 850.)
    first.ground_point_source = 'POSE_ANKLE_MIDPOINT_ESTIMATE'
    first.ground_point_quality = 'ANKLES_VISIBLE_GROUND_CONTACT_UNVERIFIED'
    first.ground_point_confidence = np.float32(.5625)
    pose = PoseBBox(np.float32(238), np.float32(790), np.float32(262), np.float32(850),
                    confidence=np.float32(.875), track_id=np.int32(6),
                    keypoints_px=tuple((250., 800., .5) for _ in range(17)),
                    ground_point_px=(250., 850.), ground_point_confidence=.5)
    if annotation_form == 'list':
        pose.keypoints_px = [list(point) for point in pose.keypoints_px]
        pose.ground_point_px = list(pose.ground_point_px)
    elif annotation_form == 'numpy':
        pose.keypoints_px = np.asarray(pose.keypoints_px, dtype='>f4')
        pose.ground_point_px = np.asarray(pose.ground_point_px, dtype=np.float32)
    frames = [[first, box(5, -2, 2)], [pose, box(5, -2, 2)]]
    assert_exact_selection(CourtPlayerTracker(calibration()), frames, tmp_path)


def test_invalid_candidates_and_all_invalid_geometry_remain_missing(tmp_path):
    frames = [[box(5, 25, 1), BBox(float('nan'), 0., 1., 2.),
               box(100, 100, 9), BBox(1., 1., 1., 2.), box(5, -2, None)], []]
    assert_exact_selection(CourtPlayerTracker(calibration()), frames, tmp_path)
    invalid = CourtCalibration(None, rejection_reason='UNKNOWN_VIEW')
    expected = assert_exact_selection(CourtPlayerTracker(invalid), frames, tmp_path)
    assert expected == {1: [None, None], 2: [None, None]}


def test_randomized_admitted_observations_match_all_legacy_choices(tmp_path):
    rng = np.random.default_rng(493)
    frames, geometry = [], []
    for index in range(160):
        dx = (index % 7) * 13.125
        frames.append([box(float(rng.uniform(-5, 15)), float(rng.uniform(-8, 32)),
                           None if proposal % 5 == 0 else int(rng.integers(0, 12)),
                           confidence=float(rng.uniform(.1, 1.)), dx=dx)
                       for proposal in range(int(rng.integers(0, 12)))])
        geometry.append(calibration(dx) if index % 17 != 0 else CourtCalibration(None))
    assert_exact_selection(CourtPlayerTracker(calibration(), memory_seconds=.125), frames, tmp_path,
                           fps=59.94, calibrations=geometry)


def test_read_only_sequences_independent_snapshots_and_index_validation(tmp_path):
    with select_court_players_disk(CourtPlayerTracker(calibration()),
                                   [[box(5, 25, 1)], [], [box(5, -2, 2)]], 30, tmp_path) as selected:
        assert selected[1][-1] is None
        assert selected[2][-1].track_id == 2
        assert selected.index(2, -1) == 0
        assert selected[1][::-1][0] is None
        snapshot = selected[1][0]
        snapshot.x1 = 999.
        assert selected[1][0].x1 != 999.
        indices = selected.selected_detection_indices[0]
        indices[1] = 55
        assert selected.index(1, 0) == 0
        with pytest.raises(TypeError):
            selected[1][0] = snapshot
        with pytest.raises(TypeError):
            selected.selected_detection_indices[0] = indices
        with pytest.raises(KeyError):
            selected.index(3, 0)
        with pytest.raises(IndexError):
            selected[1][3]
        with pytest.raises(TypeError):
            selected[1][.5]
    with pytest.raises(RuntimeError, match='not open'):
        selected[2][2]


@pytest.mark.parametrize('fps', [0, -1, math.nan, math.inf])
def test_invalid_fps_rejects_before_allocating_storage(tmp_path, fps):
    with pytest.raises(ValueError, match='FPS'):
        with select_court_players_disk(CourtPlayerTracker(calibration()), [], fps, tmp_path):
            pass
    assert not list(tmp_path.iterdir())


def test_calibration_alignment_failure_closes_and_retains_accepted_input(tmp_path):
    closed = []

    def frames():
        try:
            yield [box(5, 25, 1)]
            yield []
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match='align'):
        with select_court_players_disk(CourtPlayerTracker(calibration()), frames(), 30, tmp_path, []):
            pass
    assert closed == [True]
    paths = list(tmp_path.glob('*.sqlite3'))
    assert len(paths) == 3
    for path in paths:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute('BEGIN EXCLUSIVE')
            if path.name.startswith('player_detections_'):
                count = connection.execute('SELECT COUNT(*) FROM player_frames').fetchone()[0]
                assert count in (0, 2)


def test_producer_failure_closes_generator_and_retains_partial_rows(tmp_path):
    closed = []

    def frames():
        try:
            yield [box(5, 25, 1)]
            yield []
            raise RuntimeError('source failed')
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match='source failed'):
        with select_court_players_disk(CourtPlayerTracker(calibration()), frames(), 30, tmp_path):
            pass
    assert closed == [True]
    paths = list(tmp_path.glob('*.sqlite3'))
    assert len(paths) == 1
    with closing(sqlite3.connect(paths[0])) as connection:
        connection.execute('BEGIN EXCLUSIVE')
        assert connection.execute('SELECT COUNT(*) FROM player_frames').fetchone()[0] == 2


def test_selection_and_consumer_failure_retain_diagnostics_and_unrelated_files(tmp_path, monkeypatch):
    unrelated = tmp_path / 'keep.sqlite3'
    unrelated.write_bytes(b'unrelated')
    tracker = CourtPlayerTracker(calibration())
    candidate = tracker._candidate
    calls = 0

    def fail_second_pass(*args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('candidate failed')
        return candidate(*args)

    monkeypatch.setattr(tracker, '_candidate', fail_second_pass)
    with pytest.raises(RuntimeError, match='candidate failed'):
        with select_court_players_disk(tracker, [[box(5, 25, 1)] for _ in range(3)], 30, tmp_path):
            pass
    assert unrelated.read_bytes() == b'unrelated'
    selection_db = next(tmp_path.glob('court_player_selection_*.sqlite3'))
    with closing(sqlite3.connect(selection_db)) as connection:
        connection.execute('BEGIN EXCLUSIVE')
        assert connection.execute('SELECT SUM(count) FROM presence').fetchone()[0] == 3
    monkeypatch.setattr(tracker, '_candidate', candidate)
    with pytest.raises(RuntimeError, match='consumer failed'):
        with select_court_players_disk(tracker, [[box(5, 25, 1)]], 30, tmp_path) as selected:
            paths = selected.storage_paths
            raise RuntimeError('consumer failed')
    assert all(path.exists() for path in paths)
    for path in paths:
        with closing(sqlite3.connect(path)) as connection:
            connection.execute('BEGIN EXCLUSIVE')


def test_external_input_store_remains_open_and_is_not_owned_by_selector(tmp_path):
    with PlayerDetectionSequence(tmp_path) as source:
        source.extend([[box(5, 25, 1)], [box(5, -2, 2)]])
        with select_court_players_disk(CourtPlayerTracker(calibration()), source, 30, tmp_path) as selected:
            assert len(selected.storage_paths) == 2
            assert source.path not in selected.storage_paths
            paths = selected.storage_paths
            assert selected.index(1, 0) == 0
        assert all(not path.exists() for path in paths)
        assert source[1][0].track_id == 2
        assert source.path.exists()
    assert not source.path.exists()


def test_empty_input_has_aligned_empty_role_and_index_sequences(tmp_path):
    with select_court_players_disk(CourtPlayerTracker(calibration()), iter(()), 30, tmp_path) as selected:
        assert len(selected[1]) == len(selected[2]) == len(selected.selected_detection_indices) == 0
        assert list(selected[1]) == []
    assert not list(tmp_path.iterdir())


def test_25000_frame_new_ids_keep_python_observations_and_counts_bounded(tmp_path):
    live = {'count': 0, 'peak': 0, 'closed': False}

    def released():
        live['count'] -= 1

    def frames():
        try:
            for index in range(25_000):
                observation = box(5, 25, index)
                live['count'] += 1
                live['peak'] = max(live['peak'], live['count'])
                weakref.finalize(observation, released)
                yield [observation]
        finally:
            live['closed'] = True

    gc.collect()
    tracemalloc.start()
    try:
        with select_court_players_disk(CourtPlayerTracker(calibration()), frames(), 30, tmp_path) as selected:
            current, peak = tracemalloc.get_traced_memory()
            assert len(selected[1]) == len(selected.selected_detection_indices) == 25_000
            assert selected[1][24_999].track_id == 24_999
            assert selected.index(1, 24_999) == 0
            assert selected.index(2, 24_999) is None
            assert live['closed'] and live['count'] == 0 and live['peak'] <= 2
            # Track cardinality grows in the bounded-cache disk table, not a
            # retained Python Counter or match-long selected/candidate list.
            ledger = selected._ledger
            assert ledger._connection.execute('SELECT COUNT(*) FROM presence').fetchone()[0] == 25_000
            assert ledger._connection.execute('PRAGMA cache_size').fetchone()[0] == -2048
            assert ledger._connection.execute('PRAGMA temp_store').fetchone()[0] == 1
            assert ledger._connection.execute('PRAGMA mmap_size').fetchone()[0] == 0
            assert not any(isinstance(value, (list, dict)) and len(value) > 2
                           for value in vars(ledger).values())
            assert current < 1_000_000
            assert peak < 2_000_000
            fixture = {
                'schemaVersion': 1,
                'fixture': '25000_synthetic_frames_one_unique_track_id_per_frame',
                'frame_count': len(selected[1]),
                'presence_rows': ledger._connection.execute('SELECT COUNT(*) FROM presence').fetchone()[0],
                'live_source_boxes_after_spooling': live['count'],
                'peak_live_source_boxes': live['peak'],
                'source_producer_closed': live['closed'],
                'tracemalloc_current_bytes': current,
                'tracemalloc_peak_bytes': peak,
                'measurement_scope': 'Python allocations traced during source spooling and both candidate passes',
                'unmeasured_scope': 'Native SQLite allocations, process RSS, neural inference and full pipeline capacity',
                'sqlite_cache_kib_per_database': 2048,
                'sqlite_mmap_bytes': 0,
                'sqlite_temp_store': 'FILE',
                'storage_snapshot_scope': 'Before normal context close; bounded final transactions may remain in journals',
                'databases': {
                    label: {
                        'filename': path.name,
                        'database_bytes': path.stat().st_size,
                        'sidecar_bytes': {
                            suffix: sidecar.stat().st_size if sidecar.exists() else 0
                            for suffix in ('-journal', '-wal', '-shm')
                            for sidecar in (type(path)(str(path) + suffix),)
                        },
                    }
                    for label, path in zip(('source', 'selected', 'ledger'), selected.storage_paths)
                },
            }
            paths = selected.storage_paths
    finally:
        tracemalloc.stop()
    assert all(not path.exists() for path in paths)
    assert not list(tmp_path.iterdir())
    fixture['owned_databases_removed_on_success'] = True
    fixture_path = tmp_path / 'disk_selection_memory_fixture.json'
    fixture_path.write_text(json.dumps(fixture, indent=2) + '\n', encoding='utf-8')
