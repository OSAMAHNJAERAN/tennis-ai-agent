"""The candidate event path spools exact features once, without changing gates."""
from dataclasses import asdict
import json
import struct
import weakref

import pytest

from src.events import event_detector as module
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence


def serialized(value):
    return json.dumps(value, sort_keys=True, default=lambda item: item.value, allow_nan=True).encode()


def points():
    return [TemporalBallPoint(index, index / 25. + (index % 3) * .0001,
                             300. + 3. * index if index != 9 else None,
                             180. + 8. * abs(index - 24) if index != 9 else None,
                             state=BallState.DETECTED if index != 9 else BallState.MISSING,
                             confidence=.85 if index != 9 else None) for index in range(50)]


def test_typed_feature_snapshots_preserve_ieee_tuples_and_nested_strengths(tmp_path):
    nan = struct.unpack('!d', bytes.fromhex('7ff8000000000041'))[0]
    feature = module._Feature(3, -0., (nan, 2.), None, 'MISSING', None, strengths={'signed': -0.})
    with module._FeatureSequence(tmp_path) as stored:
        stored.append(feature)
        restored = stored[0]
        assert struct.pack('!d', restored.timestamp) == struct.pack('!d', feature.timestamp)
        assert type(restored.raw_position) is tuple
        assert struct.pack('!d', restored.raw_position[0]) == struct.pack('!d', nan)
        assert struct.pack('!d', restored.strengths['signed']) == struct.pack('!d', -0.)
        restored.strengths['signed'] = 4.
        assert stored[0].strengths['signed'] == -0.
        assert list(stored[:0]) == []
        assert stored[::-1][0].frame == 3
    assert not list(tmp_path.glob('event_features_*'))


@pytest.mark.parametrize('compensate', [False, True])
def test_spooled_candidate_analysis_equals_legacy_and_computes_features_once(tmp_path, monkeypatch, compensate):
    trajectory = points()
    boxes = [None] * len(trajectory)
    offsets = [(index * .1, index * .05) if index != 17 else None for index in range(len(trajectory))]
    legacy = module.TennisEventDetector(enable_camera_motion_compensation=compensate)
    expected = legacy.analyze(trajectory, boxes, boxes, fps=25., frame_size=(1280, 720), camera_offsets_px=offsets)
    with TemporalBallPointSequence(tmp_path) as stored:
        for point in trajectory:
            stored.append(point)
        detector = module.TennisEventDetector(enable_camera_motion_compensation=compensate)
        implementation = detector._features
        feature_passes = []

        def observed(*args, **kwargs):
            result = implementation(*args, **kwargs)
            feature_passes.append(type(result))
            return result

        monkeypatch.setattr(detector, '_features', observed)
        actual = detector.analyze(stored, boxes, boxes, fps=25., frame_size=(1280, 720),
                                  camera_offsets_px=offsets, scratch_directory=tmp_path)
        assert serialized(asdict(actual)) == serialized(asdict(expected))
        assert feature_passes == [module._FeatureSequence]
        assert detector.last_analysis is actual
        assert not list(tmp_path.glob('event_features_*'))


def test_feature_output_retains_constant_live_objects_for_thousand_missing_frames(tmp_path, monkeypatch):
    trajectory = [TemporalBallPoint(index, index / 60., None, None) for index in range(1000)]
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
        detector = module.TennisEventDetector()
        result = detector._features(trajectory, (1280, 720), [None] * len(trajectory), output=stored)
        assert result is stored
        assert len(stored) == 1000
        assert stored[-1].frame == 999
        assert counts['peak'] < 8
    assert not list(tmp_path.glob('event_features_*'))


def test_failure_during_verification_closes_and_removes_only_owned_feature_scratch(tmp_path, monkeypatch):
    unrelated = tmp_path / 'unrelated.sqlite3'
    unrelated.write_bytes(b'preserve')
    trajectory = points()
    detector = module.TennisEventDetector()
    assert detector.detect_candidates(trajectory, fps=25., frame_size=(1280, 720))

    def fail(*args, **kwargs):
        raise RuntimeError('verification failed')

    monkeypatch.setattr(detector, '_verify', fail)
    with pytest.raises(RuntimeError, match='verification failed'):
        detector.analyze(trajectory, [None] * len(trajectory), [None] * len(trajectory),
                         fps=25., frame_size=(1280, 720), scratch_directory=tmp_path)
    assert unrelated.read_bytes() == b'preserve'
    assert not list(tmp_path.glob('event_features_*'))


def test_default_legacy_feature_and_candidate_overrides_keep_original_call_contract(monkeypatch):
    trajectory = points()
    detector = module.TennisEventDetector()
    original = detector._features
    calls = []

    def old_feature_override(trajectory, frame_size, positions):
        calls.append('features')
        return original(trajectory, frame_size, positions)

    def old_candidate_override(trajectory, fps, frame_size=None, camera_offsets_px=None):
        calls.append('candidates')
        return []

    monkeypatch.setattr(detector, '_features', old_feature_override)
    monkeypatch.setattr(detector, 'detect_candidates', old_candidate_override)
    result = detector.analyze(trajectory, [None] * len(trajectory), [None] * len(trajectory),
                             fps=25., frame_size=(1280, 720))
    assert calls == ['features', 'candidates']
    assert result.candidates == []
