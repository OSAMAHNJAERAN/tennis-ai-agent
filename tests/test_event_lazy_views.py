"""Disk timelines avoid duration-sized projections while preserving event math."""
from collections.abc import Sequence
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from src.events.event_detector import EventDetectorSettings, TennisEventDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox
from src.utils.mapped_sequence_view import MappedSequenceView, attribute_sequence
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence


class VirtualPoints(Sequence):
    def __init__(self, count):
        self.count, self.reads = count, 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError('A lazy projection must not slice its source')
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        return SimpleNamespace(timestamp_seconds=index / 60., x_px=index, y_px=index + 1)


def test_million_frame_view_and_nested_slices_read_only_requested_items():
    source = VirtualPoints(1_000_000)
    view = attribute_sequence(source, 'timestamp_seconds')
    sliced = view[20:-10:3][::-2]
    assert source.reads == 0
    assert isinstance(view, MappedSequenceView)
    assert len(sliced) == len(range(1_000_000)[20:-10:3][::-2])
    assert sliced[0] == range(1_000_000)[20:-10:3][::-2][0] / 60.
    assert sliced[-1] == range(1_000_000)[20:-10:3][::-2][-1] / 60.
    assert source.reads == 2
    with pytest.raises(IndexError):
        _ = sliced[len(sliced)]
    assert source.reads == 2


def test_legacy_list_attribute_projection_remains_a_snapshot():
    point = SimpleNamespace(timestamp_seconds=2.)
    values = attribute_sequence([point], 'timestamp_seconds', convert=float)
    point.timestamp_seconds = 3.
    assert type(values) is list
    assert values == [2.]


@pytest.mark.parametrize('enabled,offsets', [(False, None), (False, [(2., 3.)] * 7),
                                          (True, None), (True, [(2., 3.), None] * 3 + [(2., 3.)])])
def test_disk_position_projection_matches_legacy_missing_and_compensated_coordinates(tmp_path, enabled, offsets):
    points = [TemporalBallPoint(index, index / 25., float(index), float(index + 10),
                               state=BallState.DETECTED, confidence=.8) for index in range(7)]
    points[2].x_px = None
    points[5].y_px = None
    detector = TennisEventDetector(enable_camera_motion_compensation=enabled)
    expected, expected_state = detector._positions(points, offsets)
    with TemporalBallPointSequence(tmp_path) as stored:
        for point in points:
            stored.append(point)
        actual, actual_state = detector._positions(stored, offsets)
        assert isinstance(actual, MappedSequenceView)
        assert list(actual) == expected
        assert actual_state == expected_state
        assert list(actual[::-2]) == expected[::-2]


def test_disk_position_offset_misalignment_fails_before_reading_source():
    points = VirtualPoints(1_000_000)
    detector = TennisEventDetector()
    with pytest.raises(ValueError, match='align'):
        detector._positions(points, [(0., 0.)])
    assert points.reads == 0


def _bytes(value):
    return json.dumps(value, sort_keys=True, default=lambda item: item.value, allow_nan=True).encode()


@pytest.mark.parametrize('compensate', [False, True])
def test_complete_disk_event_analysis_and_features_match_legacy_irregular_timeline(tmp_path, compensate):
    points = []
    for index in range(50):
        time = index / 25. + (index % 3) * .0001
        y = 180. + 8. * abs(index - 24)
        points.append(TemporalBallPoint(index, time, 300. + 3. * index, y,
                                       state=BallState.DETECTED, confidence=.85))
    points[9].x_px = points[9].y_px = None
    points[9].state = BallState.MISSING
    boxes1 = [BBox(350., 130., 420., 320., .9, 1)] * len(points)
    boxes2 = [None] * len(points)
    offsets = [(index * .1, index * .05) if index != 17 else None for index in range(len(points))]
    settings = EventDetectorSettings(enable_camera_motion_compensation=compensate)
    legacy = TennisEventDetector(config=asdict(settings))
    expected_positions, _ = legacy._positions(points, offsets)
    expected_features = legacy._features(points, (1280, 720), expected_positions)
    expected_analysis = legacy.analyze(points, boxes1, boxes2, fps=25., frame_size=(1280, 720), camera_offsets_px=offsets)
    with TemporalBallPointSequence(tmp_path) as stored:
        for point in points:
            stored.append(point)
        detector = TennisEventDetector(config=asdict(settings))
        positions, _ = detector._positions(stored, offsets)
        features = detector._features(stored, (1280, 720), positions)
        analysis = detector.analyze(stored, boxes1, boxes2, fps=25., frame_size=(1280, 720), camera_offsets_px=offsets)
        assert _bytes([asdict(item) for item in features]) == _bytes([asdict(item) for item in expected_features])
        assert _bytes(asdict(analysis)) == _bytes(asdict(expected_analysis))
        assert detector._player_features(24, stored, boxes1, boxes2, .14) == legacy._player_features(24, points, boxes1, boxes2, .14)
        assert detector._frame_size(stored, None) == legacy._frame_size(points, None)
