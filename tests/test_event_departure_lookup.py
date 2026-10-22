"""Departure lookup preserves legacy values without scanning every frame per contact."""
from collections.abc import Sequence
import math

import numpy as np
import pytest

from src.events.event_detector import EventCandidate, TennisEventDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox
from src.utils.mapped_sequence_view import attribute_sequence


def points_at(timestamps):
    return [TemporalBallPoint(index, timestamp, 5. + 5. * index, 0.,
                              state=BallState.DETECTED, confidence=.9)
            for index, timestamp in enumerate(timestamps)]


def player_boxes(count):
    return [BBox(-10., -50., 0., 50., .9, 1)] * count


@pytest.mark.parametrize('timestamps,frame,window', [
    ([0., .037, .081, .132, .193], 1, .094),
    ([0., 2., 4.], 0, 1.),  # midpoint chooses the first equal distance
    ([0., 1., 2.], 0, float(2 ** 55)),  # rounded-distance plateau chooses frame 0
    ([0., 1., 2.], 0, 1e308),
    ([-1e308, 0., 1e308], 2, 1e308),  # target overflow retains the scan fallback
    ([np.float32(0.), np.float32(.035), np.float32(.083), np.float32(.141)], 1, .057),
    ([np.float64(0.), np.float64(.04), np.float64(.089), np.float64(.139)], 1, .062),
    ([0., .04, .08], 0, math.inf),
    ([0., .04, .08], 0, math.nan),
])
def test_ordered_departure_matches_original_first_minimum(timestamps, frame, window):
    points = points_at(timestamps)
    boxes = player_boxes(len(points))
    detector = TennisEventDetector(departure_window_seconds=window)
    expected = detector._departure(frame, 1, points, boxes)
    view = attribute_sequence(tuple(points), 'timestamp_seconds')
    assert detector._departure(frame, 1, tuple(points), boxes, ordered_timestamps=view) == expected


class CountedPoints(Sequence):
    def __init__(self, count):
        self.count, self.reads = count, 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError('Departure lookup must not slice its source')
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        return TemporalBallPoint(index, index / 60., 10., 0., state=BallState.DETECTED)


def test_ordered_departure_reads_only_logarithmic_number_of_points():
    points = CountedPoints(1_000_000)
    timestamps = attribute_sequence(points, 'timestamp_seconds')
    boxes = player_boxes(1)

    class Boxes(Sequence):
        def __len__(self):
            return len(points)

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            return boxes[0]

    detector = TennisEventDetector()
    result = detector._departure(500_000, 1, points, Boxes(), ordered_timestamps=timestamps)
    assert result == detector._clip(.1 / detector.settings.player_reach_normalized)
    assert points.reads <= 2 * math.ceil(math.log2(len(points))) + 8


def test_analysis_certification_stays_local_and_leaves_legacy_and_nonfinite_hooks_unchanged(monkeypatch):
    detector = TennisEventDetector(enable_event_time_refinement=False)
    candidate = EventCandidate(1, .04, .8, (10., 0.), BallState.DETECTED.value)
    monkeypatch.setattr(detector, 'detect_candidates', lambda *args, **kwargs: [candidate])
    passed = []

    def verify(*args, **kwargs):
        passed.append(kwargs)
        return None, {}

    monkeypatch.setattr(detector, '_verify', verify)
    timelines = [
        points_at([0., .04, .08, .12]),
        tuple(points_at([0., .04, .083, .123])),
        tuple(points_at([0., .04, math.nan, .12])),
        points_at([0., .04, .08, .12]),
    ]
    for points in timelines:
        detector.analyze(points, player_boxes(len(points)), [None] * len(points), frame_size=(1280, 720))
    assert passed[0] == passed[2] == passed[3] == {}
    assert list(passed[1]['ordered_timestamps']) == [point.timestamp_seconds for point in timelines[1]]


def test_verify_forwards_ordered_departure_only_when_supplied(monkeypatch):
    points = points_at([0., .04, .08, .12])
    detector = TennisEventDetector()
    positions, _ = detector._positions(points, None)
    feature = detector._features(points, (1280, 720), positions)[1]
    candidate = EventCandidate(1, .04, .8, (10., 0.), BallState.DETECTED.value)
    received = []

    def departure(*args, **kwargs):
        received.append(kwargs)
        return 0.

    monkeypatch.setattr(detector, '_departure', departure)
    arguments = (1, candidate, feature, points, player_boxes(len(points)), [None] * len(points),
                 None, 'DISABLED', 720, math.hypot(1280, 720))
    detector._verify(*arguments)
    timestamps = attribute_sequence(tuple(points), 'timestamp_seconds')
    detector._verify(*arguments, ordered_timestamps=timestamps)
    assert received[0] == {}
    assert received[1] == {'ordered_timestamps': timestamps}
