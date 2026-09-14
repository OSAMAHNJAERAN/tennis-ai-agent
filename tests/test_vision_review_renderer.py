import numpy as np
import pytest

from src.visualization.vision_review_renderer import VisionReviewRenderer
from scripts.run.render_vision_review import aligned_motion_samples


@pytest.mark.parametrize('summary', [False, True])
def test_review_accepts_legacy_arrays_and_summary_samples(summary):
    samples = [{'timestamp_s': 0., 'ground_position_m': None}]
    value = {'samples': samples} if summary else samples
    motion = {'player_1': value, 'player_2': value}
    assert aligned_motion_samples(motion, 1) == {'1': samples, '2': samples}
    with pytest.raises(ValueError, match='length differs'):
        aligned_motion_samples(motion, 2)


def test_missing_observations_break_history_and_do_not_add_occupancy():
    renderer = VisionReviewRenderer(30, True)
    frame = np.zeros((360, 640, 3), np.uint8)
    motion = {'1': {'ground_position_m': [5., 26.]}, '2': {'ground_position_m': None}}
    first = renderer.render(frame, 0, {'ball': {'position_px': [120, 150]}}, {}, {}, motion)
    assert first.shape == frame.shape
    assert len(renderer.ball_trail) == 1
    assert len(renderer.player_trails['1']) == 1
    assert renderer.occupancy.sum() == pytest.approx(1 / 30)
    motion['1']['ground_position_m'] = None
    renderer.render(frame, 1, {'ball': {'position_px': None}}, {}, {}, motion)
    assert not renderer.ball_trail
    assert not renderer.player_trails['1']
    assert renderer.occupancy.sum() == pytest.approx(1 / 30)
    renderer.render(frame, 2, {'ball': {'position_px': [240, 260]}}, {}, {}, motion)
    assert len(renderer.ball_trail) == 1


def test_invalid_calibration_abstains_and_nonsequential_render_is_rejected():
    renderer = VisionReviewRenderer(25, False)
    frame = np.zeros((240, 320, 3), np.uint8)
    motion = {key: {'ground_position_m': [5., 10.], 'speed_kmh': 5.} for key in ('1', '2')}
    renderer.render(frame, 0, {}, {}, {}, motion)
    assert renderer.occupancy.sum() == 0
    assert not renderer.player_trails['1']
    with pytest.raises(ValueError, match='sequential'):
        renderer.render(frame, 2, {}, {}, {}, motion)


def test_registration_loss_hides_ground_positions_even_if_stale_motion_is_supplied():
    renderer = VisionReviewRenderer(30, True)
    frame = np.zeros((360, 640, 3), np.uint8)
    motion = {key: {'ground_position_m': [5., 10.], 'speed_kmh': 5.} for key in ('1', '2')}
    renderer.render(frame, 0, {'ball': {'position_px': [20, 30]}}, {}, {}, motion)
    occupancy = renderer.occupancy.sum()
    renderer.render(frame, 1, {'court_registration_valid': False, 'ball': {'position_px': [30, 40]}}, {}, {}, motion)
    assert not renderer.current_calibration_valid
    assert not renderer.player_trails['1']
    assert renderer.occupancy.sum() == occupancy
    assert len(renderer.ball_trail) == 1
