"""The real event feature path must avoid per-query full-timeline scans.

A delegating counting Sequence changes no values or nearest-index algorithm.
The exact preserved production module pins every feature field and general
nearest-index fallback behavior. These synthetic numeric tests qualify no
tennis inference, accuracy or full-video capacity.
"""
from collections.abc import Sequence
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

from src.events.event_detector import TennisEventDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/validation/product_execution/event_timestamp_lookup_2026-10-06/src/events/event_detector.py'
BASELINE_SHA256 = '078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
FRAME_SIZE = (1920, 1080)


class CountedTimestamps(Sequence):
    """Read-only value-preserving view; counts only successful scalar reads."""

    def __init__(self, timestamps, counts):
        self.timestamps = timestamps
        self.counts = counts

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        value = self.timestamps[index]
        self.counts['timestampReads'] += 1
        return value


@pytest.fixture(scope='module')
def frozen():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
    name = 'frozen_event_timestamp_lookup_20261006'
    spec = importlib.util.spec_from_file_location(name, BASELINE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trajectory(timestamps, *, with_gaps=False):
    result = []
    states = (BallState.DETECTED, BallState.TRACKED, BallState.PREDICTED,
              BallState.INTERPOLATED, BallState.OCCLUDED)
    for index, timestamp in enumerate(timestamps):
        # Changes in direction, rebound signatures and off-grid decimal
        # positions exercise all populated derivative/strength fields.
        phase = index % 31
        x = 200.125 + (phase if phase <= 15 else 30 - phase) * 4.25
        y = 310.75 + abs((index % 23) - 11) * 5.125
        missing = with_gaps and index % 11 == 7
        partial = with_gaps and index % 17 == 9
        result.append(TemporalBallPoint(
            frame_index=index, timestamp_seconds=timestamp,
            x_px=None if missing else x, y_px=None if missing or partial else y,
            court_x_m=None, court_y_m=None,
            confidence=None if missing or index % 7 == 4 else .875,
            state=BallState.MISSING if missing else states[index % len(states)],
            source='numeric-regression'))
    return result


def feature_bytes(features):
    # allow_nan pins existing diagnostic output semantics for nonfinite
    # fallbacks; this is not an assertion that they are strict API JSON.
    return json.dumps([asdict(feature) for feature in features],
                      sort_keys=True, allow_nan=True).encode('utf-8')


def preserve_diagnostic(name, data):
    directory = os.environ.get('EVENT_TIMESTAMP_LOOKUP_EVIDENCE')
    if directory:
        path = Path(directory) / name
        with path.open('xb') as stream:
            stream.write(data)


def test_actual_features_nearest_timestamp_reads_are_subquadratic(monkeypatch, frozen, record_property):
    measurements = []
    for count in (128, 256, 512):
        timestamps = [index / 60.0 for index in range(count)]
        points = trajectory(timestamps)
        current = TennisEventDetector()
        reference = frozen.TennisEventDetector()
        positions, _ = current._positions(points, None)
        expected = feature_bytes(reference._features(points, FRAME_SIZE, positions))
        counts = {'timestampReads': 0, 'nearestQueries': 0}
        # Root may add a validated sorted helper, preserving the general helper
        # for fallbacks. Observe both real implementations when present.
        for name in ('_nearest_index', '_nearest_ordered_index'):
            if not hasattr(current, name):
                continue
            implementation = getattr(current, name)

            def observed(values, target, *args, _implementation=implementation, **kwargs):
                counts['nearestQueries'] += 1
                return _implementation(CountedTimestamps(values, counts), target, *args, **kwargs)

            monkeypatch.setattr(current, name, observed)
        actual = feature_bytes(current._features(points, FRAME_SIZE, positions))
        assert actual == expected, 'Every serialized _Feature field must match the frozen detector'
        preserve_diagnostic(f'features_{count}_current.json', actual)
        preserve_diagnostic(f'features_{count}_frozen.json', expected)
        # Generous to binary search and earliest rounded-distance plateau
        # searches; still sharply catches a full N scan for each query.
        read_budget = 4 * count * (4 * math.ceil(math.log2(count + 1)) + 8)
        measurements.append({'frames': count, **counts, 'timestampReadBudget': read_budget,
                             'exactFrozenFeatureBytes': True, 'featureBytes': len(actual),
                             'featureSha256': hashlib.sha256(actual).hexdigest()})
    message = {'measurements': measurements,
               'doublingReadRatios': [measurements[i + 1]['timestampReads'] /
                                      measurements[i]['timestampReads']
                                      for i in range(len(measurements) - 1)]}
    print(json.dumps(message, allow_nan=False), flush=True)
    record_property('nearestTimestampMeasurements', json.dumps(message, allow_nan=False))
    preserve_diagnostic('nearest_timestamp_measurements.json',
                        json.dumps(message, indent=2, allow_nan=False).encode('utf-8'))
    # Check after collecting all sizes, so the genuine RED receipt retains the
    # growth evidence rather than only the first failing scalar.
    assert all(row['timestampReads'] <= row['timestampReadBudget'] for row in measurements), message


def nearest_outcome(implementation, timestamps, target):
    try:
        return ('index', implementation(timestamps, target))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


@pytest.mark.parametrize('timestamps,target', [
    ([], 0.0),
    ([1.0], -100.0),
    ([0.0, 2.0], 1.0),                         # exact midpoint: earliest wins
    ([0.0, 1.0, 1.0, 2.0], 1.0),              # duplicate timestamp: first duplicate
    ([-0.0, 0.0, 2.0], 0.0),
    ([0.0, 1.0, 2.0], -100.0),
    ([0.0, 1.0, 2.0], 100.0),
    ([0.0, 1.0, 2.0], float(2 ** 55)),          # rounded distance plateau: index 0
    ([0.0, 1.0, 2.0], 1e308),                 # huge finite plateau beyond range
    ([-1e308, 0.0, 1.0], 1e308),              # subtraction overflow
    ([3.0, -1.0, 2.0, 0.0], .5),              # unsorted legacy fallback
    ([math.nan, 1.0, 2.0], 1.9),
    ([0.0, math.nan, 2.0], 1.9),
    ([0.0, 1.0, math.nan], 1.9),
    ([-math.inf, 0.0, 1.0, math.inf], .5),
    ([0.0, 1.0, 2.0], math.nan),
    ([0.0, 1.0, 2.0], math.inf),
    ([0.0, 1.0, 2.0], -math.inf),
])
def test_general_nearest_index_preserves_legacy_fallbacks(frozen, timestamps, target):
    expected = nearest_outcome(frozen.TennisEventDetector._nearest_index, timestamps, target)
    actual = nearest_outcome(TennisEventDetector._nearest_index, timestamps, target)
    assert actual == expected
    if timestamps == [0.0, 1.0, 2.0] and target == float(2 ** 55):
        assert actual == ('index', 0)


FEATURE_CASES = [
    ('native60_with_gaps', [index / 60.0 for index in range(67)], .067, True),
    ('irregular_finite', [index / 29.97 + (index % 4) * .0000125 for index in range(71)], .113, True),
    ('all_finite_tight', [index * 1e-7 for index in range(19)], .067, False),
    ('huge_finite_rounded', [0.0, 1.0, 2.0, float(2 ** 55)], float(2 ** 55), False),
    ('finite_overflow_target', [-1e308, 0.0, 1e308], 1e308, False),
    ('duplicate_rejected', [0.0, .1, .1, .3], .067, False),
    ('unsorted_rejected', [0.0, .2, .1, .3], .067, False),
    ('nan_first_fallback', [math.nan, .1, .2, .3], .067, False),
    ('nan_interior_fallback', [0.0, .1, math.nan, .3], .067, False),
    ('nan_last_fallback', [0.0, .1, .2, math.nan], .067, False),
    ('negative_infinity_fallback', [-math.inf, 0.0, .1, .2], .067, False),
    ('positive_infinity_fallback', [0.0, .1, .2, math.inf], .067, False),
    ('infinite_window_fallback', [0.0, .1, .2, .3], math.inf, False),
    ('nan_window_fallback', [0.0, .1, .2, .3], math.nan, False),
]


def features_outcome(detector, points):
    positions, _ = detector._positions(points, None)
    try:
        return ('features', feature_bytes(detector._features(points, FRAME_SIZE, positions)))
    except Exception as error:
        return ('error', type(error).__name__, str(error))


@pytest.mark.parametrize('name,timestamps,half_window,with_gaps', FEATURE_CASES,
                         ids=[case[0] for case in FEATURE_CASES])
def test_actual_features_preserve_numeric_and_rejection_semantics(
        frozen, name, timestamps, half_window, with_gaps):
    points = trajectory(timestamps, with_gaps=with_gaps)
    config = {'feature_half_window_seconds': half_window}
    expected = features_outcome(frozen.TennisEventDetector(config=config), points)
    actual = features_outcome(TennisEventDetector(config=config), points)
    assert actual == expected
    if name.endswith('_rejected'):
        assert actual == ('error', 'ValueError', 'trajectory timestamps must be strictly increasing')

