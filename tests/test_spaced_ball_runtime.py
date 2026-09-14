import copy

import numpy as np
import pytest

from scripts.evaluate.temporal_spacing_stream import predict_spaced_tiled_stream as frozen_stream
from scripts.evaluate.replay_ball_temporal_detours import reject_detours as frozen_detours
from src.detection.spaced_ball_stream import predict_spaced_tiled_stream, temporal_stride, validate_spaced_ball_config
from src.tracking.selected_temporal_detours import reject_detours
from src.tracking.temporal_ball_tracker import BallObservation


@pytest.mark.parametrize('length', [0, 1, 2, 3, 5, 6, 7, 11])
@pytest.mark.parametrize('stride', [1, 2, 3])
def test_runtime_stream_matches_frozen_research_alignment(length, stride):
    class Detector:
        def predict_stream(self, frames):
            for frame in frames:
                value = float(frame[0, 0, 0])
                yield [] if value % 3 == 1 else [BallObservation(1, 2, value)]
    frames = [np.full((10, 10, 3), i, np.uint8) for i in range(length)]
    def coordinates(stream):
        return [[(p.x_px, p.y_px, p.confidence) for p in row] for row in stream]
    actual = coordinates(predict_spaced_tiled_stream(Detector(), iter(frames), (10, 10), stride))
    expected = coordinates(frozen_stream(Detector(), iter(frames), (10, 10), stride))
    assert actual == expected
    assert len(actual) == length


@pytest.mark.parametrize('fps,stride', [(24,1), (25,1), (29.97,1), (30,1), (59.94,2), (60,2), (120,4)])
def test_stride_uses_native_fps(fps, stride):
    assert temporal_stride(fps) == stride


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), True])
def test_invalid_native_fps_is_rejected(fps):
    with pytest.raises(ValueError): temporal_stride(fps)


def test_detour_rejection_matches_frozen_research_rule():
    points = [[10 + i * 4, 100] for i in range(30)]
    points[7] = [400, 20]
    points[12] = None
    actual = reject_detours(points, 60, [512, 288])
    assert actual == frozen_detours(points, 60, [512, 288])
    assert actual[0] == {7}


def test_unvalidated_authority_and_tracking_combinations_fail_before_models_load():
    config = {'ball_detection': {'backend': 'wasb', 'temporal_step': 1,
              'spatial_crops': {'enabled': True}, 'patch_persistence': {'enabled': True},
              'temporal_spacing': {'enabled': True}, 'temporal_detours': {'enabled': True}},
              'temporal_tracking': {'strategy': 'model_top1'},
              'event_detection': {'authoritative_enabled': False}}
    validate_spaced_ball_config(config)
    for section, field, value in [('ball_detection', 'backend', 'yolo11'),
                                  ('ball_detection', 'temporal_step', 3),
                                  ('ball_detection', 'spatial_crops', {'enabled': False}),
                                  ('ball_detection', 'patch_persistence', {'enabled': False}),
                                  ('ball_detection', 'temporal_spacing', {'enabled': 'false'}),
                                  ('temporal_tracking', 'strategy', 'kalman'),
                                  ('event_detection', 'authoritative_enabled', True)]:
        changed = copy.deepcopy(config)
        changed[section][field] = value
        with pytest.raises(ValueError): validate_spaced_ball_config(changed)
    validate_spaced_ball_config({})
