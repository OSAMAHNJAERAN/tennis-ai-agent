"""Disk ranking preserves Python ordering/suppression without full-feature lists."""
from dataclasses import asdict, replace
import json
import math
import weakref

import pytest

from src.events import event_detector as module


def feature(frame, timestamp, **changes):
    values = dict(frame=frame, timestamp=timestamp, raw_position=(50., 100.),
                  feature_position=(50., 100.), state='DETECTED', confidence=.9,
                  score=.8, normalized_speed=.1, strengths={'kinematic': .8})
    values.update(changes)
    return module._Feature(**values)


def candidate_bytes(candidates):
    return json.dumps([asdict(item) for item in candidates], sort_keys=True, allow_nan=True).encode()


def compare(tmp_path, features, interval):
    detector = module.TennisEventDetector(candidate_min_interval_seconds=interval)
    expected = detector._candidates_from_features(features, 25., 1280, 720, 'DISABLED')
    with module._FeatureSequence(tmp_path) as stored:
        for item in features:
            stored.append(item)
        actual = detector._candidates_from_features(stored, 25., 1280, 720, 'DISABLED')
        assert candidate_bytes(actual) == candidate_bytes(expected)
    assert not list(tmp_path.glob('event_features_*'))
    return actual


@pytest.mark.parametrize('interval', [-.01, 0., .1, .2])
def test_mixed_ranking_keys_missing_states_and_stable_ties_match_python(tmp_path, interval):
    features = [feature(index, (index % 6) * .1, score=.8 + (index % 3) * .03,
                        vertical_inversion=index % 2 == 0,
                        direction_change=None if index % 4 == 0 else float(index % 4),
                        normalized_velocity_change=float(index % 5),
                        normalized_acceleration=float(index % 7)) for index in range(24)]
    # The source ordinal, rather than the feature's frame value, resolves a
    # completely tied Python key. A negative interval accepts equal timestamps.
    features.extend([feature(900, 3., score=.95), feature(3, 3., score=.95),
                     feature(25, 4., state='MISSING', score=1.),
                     feature(26, 4.1, raw_position=None, score=1.),
                     feature(27, 4.2, normalized_speed=0.), feature(28, 4.3, score=.01)])
    candidates = compare(tmp_path, features, interval)
    tied = [item.frame_index for item in candidates if item.timestamp_s == 3.]
    assert tied == ([900, 3] if interval < 0 else [900])
    assert not {25, 26, 27, 28}.intersection(item.frame_index for item in candidates)


@pytest.mark.parametrize('timestamps,interval,expected', [
    ([.3, .4, 1.], .1, [0, 1, 2]),  # subtraction exceeds .1 although .3+.1 == .4
    ([.5, .625, 1.], .125, [0, 2]),  # the inclusive suppression endpoint rejects
    ([0., math.nextafter(.1, 0.), .2], .1, [0, 2]),
    ([0., math.nextafter(.1, math.inf), .3], .1, [0, 1, 2]),
])
def test_suppression_uses_exact_absolute_subtraction_not_rounded_range_bounds(tmp_path, timestamps, interval, expected):
    features = [feature(index, timestamp, score=.9 - index * .1)
                for index, timestamp in enumerate(timestamps)]
    assert [item.frame_index for item in compare(tmp_path, features, interval)] == expected


@pytest.mark.parametrize('origin,interval', [(0., 0.), (2.**53, 1.), (1.e308, 1.e292),
                                          (-1.e308, 1.e292), (1.e308, 1.e308)])
def test_indexed_suppression_outward_bounds_match_extreme_finite_python_rules(tmp_path, origin, interval):
    timestamps = [origin, math.nextafter(origin, math.inf), math.nextafter(origin, -math.inf),
                  origin / 2., -origin]
    compare(tmp_path, [feature(index, timestamp, score=.95 - index * .01)
                       for index, timestamp in enumerate(timestamps)], interval)


def test_accepted_time_range_uses_disk_index_for_suppression(tmp_path):
    with module._FeatureSequence(tmp_path) as stored:
        plan = stored._open().execute(
            'EXPLAIN QUERY PLAN SELECT 1 FROM accepted WHERE timestamp>=? AND timestamp<=? '
            'AND abs(timestamp-?)<=? LIMIT 1', (0., .2, .1, .1)).fetchall()
        assert any('SEARCH accepted USING COVERING INDEX accepted_time' in row[-1] for row in plan)


def test_selection_crosses_commit_boundary_with_every_frame_accepted(tmp_path):
    features = [feature(index, index / 25.) for index in range(257)]
    assert len(compare(tmp_path, features, -.01)) == 257


@pytest.mark.parametrize('name', ['score', 'timestamp', 'direction_change',
                                  'normalized_velocity_change', 'normalized_acceleration', 'normalized_speed'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_active_keys_fail_clearly_and_remove_only_owned_scratch(tmp_path, name, value):
    unrelated = tmp_path / 'keep.sqlite3'
    unrelated.write_bytes(b'keep')
    stored = module._FeatureSequence(tmp_path)
    with pytest.raises(ValueError, match='finite binary64'):
        with stored:
            for index in range(3):
                stored.append(feature(index, index / 25., **{name: value}) if name != 'timestamp'
                              else feature(index, value))
            module.TennisEventDetector()._candidates_from_features(stored, 25., 1280, 720, 'DISABLED')
    assert not stored.path.exists()
    assert unrelated.read_bytes() == b'keep'
    with pytest.raises(RuntimeError, match='not open'):
        stored[0]


def test_missing_features_preserve_nonfinite_payloads_without_becoming_ranked_candidates(tmp_path):
    features = [feature(0, math.nan, state='MISSING', score=math.nan, direction_change=math.inf),
                feature(1, math.inf, raw_position=None, score=math.inf),
                feature(2, .08)]
    assert [item.frame_index for item in compare(tmp_path, features, .1)] == [2]


def test_legacy_nonfinite_selector_keeps_original_permissive_behavior():
    features = [feature(0, 0., score=math.inf), feature(1, .2), feature(2, .4)]
    candidates = module.TennisEventDetector()._candidates_from_features(features, 25., 1280, 720, 'DISABLED')
    assert [item.frame_index for item in candidates] == [0, 1, 2]
    assert math.isinf(candidates[0].score)


def test_rank_and_selection_keep_constant_live_features_across_commit_boundary(tmp_path, monkeypatch):
    original = module._Feature
    live = weakref.WeakSet()
    counts = {'peak': 0}

    class ObservedFeature(original):
        __hash__ = object.__hash__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live.add(self)
            counts['peak'] = max(counts['peak'], len(live))

    monkeypatch.setattr(module, '_Feature', ObservedFeature)
    with module._FeatureSequence(tmp_path) as stored:
        for index in range(257):
            stored.append(feature(index, index / 25.))
        assert stored._execute('PRAGMA cache_size', fetch=True) == (-2048,)
        assert stored._execute('PRAGMA mmap_size', fetch=True) == (0,)
        assert stored._execute('PRAGMA temp_store', fetch=True) == (1,)
        candidates = module.TennisEventDetector()._candidates_from_features(stored, 25., 1280, 720, 'DISABLED')
        assert len(candidates) == 86
        assert counts['peak'] < 8
        assert len(live) == 0
        # Repeated selection rebuilds disk state instead of retaining previous
        # selected snapshots or allowing accepted timestamps to bleed through.
        repeat = module.TennisEventDetector()._candidates_from_features(stored, 25., 1280, 720, 'DISABLED')
        assert candidate_bytes(repeat) == candidate_bytes(candidates)
        assert stored._pending == 0
    assert not list(tmp_path.glob('event_features_*'))


@pytest.mark.parametrize('transaction_size', [0, -1, 257, True, 1.5])
def test_feature_transactions_cannot_exceed_bounded_batch_size(tmp_path, transaction_size):
    with pytest.raises(ValueError, match='1 through 256'):
        module._FeatureSequence(tmp_path, transaction_size=transaction_size)


@pytest.mark.parametrize('changes', [{'frame': True}, {'timestamp': None}, {'state': 'UNKNOWN'},
                                    {'raw_position': [1., 2.]}, {'edge_continuity': None},
                                    {'strengths': {'invalid': None}}])
def test_feature_schema_rejects_untyped_fields_before_accepting_a_write(tmp_path, changes):
    item = replace(feature(0, 0.), **changes)
    with module._FeatureSequence(tmp_path) as stored:
        with pytest.raises(TypeError):
            stored.append(item)
        assert len(stored) == 0
