import numpy as np
import pytest

from src.tracking.player_motion_tracking import decode_pose, motion_samples, summarize_motion


def test_pose_offsets_and_low_confidence_joints_remain_unknown():
    data = np.tile([10., 20., .9], (17, 1))
    data[9, 2] = .1
    result = decode_pose(data, (100, 200))
    assert len(result) == 17
    assert result["left_shoulder"]["position_px"] == [110., 220.]
    assert result["left_wrist"]["position_px"] is None


def test_known_motion_and_stationary_direction():
    result = motion_samples([(0, 0), (.2, 0), (.4, 0)], [0., .1, .2])
    assert result[1]["speed_kmh"] == pytest.approx(7.2)
    assert result[2]["acceleration_mps2"] == pytest.approx([0, 0])
    assert result[1]["direction_deg"] == 0
    assert motion_samples([(1, 1), (1, 1)], [0., .1])[1]["direction_deg"] is None


def test_acceleration_uses_effective_window_times_during_startup():
    times = np.arange(8, dtype=float) * .05
    positions = np.column_stack((times ** 2, np.zeros_like(times)))
    samples = motion_samples(positions.tolist(), times.tolist())
    for sample in samples[2:]:
        assert sample['velocity_mps'][0] == pytest.approx(2 * sample['velocity_effective_timestamp_s'])
        assert sample['acceleration_mps2'] == pytest.approx([2., 0.])


def test_motion_does_not_bridge_missing_positions_or_long_gaps():
    result = motion_samples([(0, 0), None, (1, 0), (2, 0)], [0, .1, .2, 1.])
    assert all(item["speed_kmh"] is None for item in result)


def test_impossible_jump_and_bad_timestamps_rejected():
    assert motion_samples([(0, 0), (20, 0)], [0, .033])[1]["speed_kmh"] is None
    with pytest.raises(ValueError):
        motion_samples([(0, 0), (1, 1)], [0, 0])


def test_distance_summary_excludes_gaps_and_preserves_unknown():
    summary = summarize_motion([(0, 0), (.2, 0), None, (10, 0)], [0, .1, .2, .3])
    assert summary["observed_distance_m"] == pytest.approx(.2)
    assert summary["mean_speed_kmh"] == pytest.approx(7.2)
    assert summary["accepted_intervals"] == 1
    assert summarize_motion([(0, 0), None, (2, 0)], [0, .1, .2])["observed_distance_m"] is None


def test_stationary_localization_jitter_is_not_accumulated_as_running_distance():
    times = np.arange(301) / 60
    positions = np.random.default_rng(7).normal(0, .03, (301, 2))
    result = summarize_motion(positions.tolist(), times.tolist())
    assert result['raw_observed_polyline_distance_m'] > 10
    assert result['observed_distance_m'] < 2
    assert result['mean_speed_kmh'] < 1.5


@pytest.mark.parametrize('fps', [25, 30, 60])
def test_noisy_fifteen_metre_run_remains_close_to_known_distance_across_frame_rates(fps):
    times = np.arange(fps * 5 + 1) / fps
    positions = np.column_stack((times * 3, times * 0))
    positions += np.random.default_rng(7).normal(0, .03, positions.shape)
    result = summarize_motion(positions.tolist(), times.tolist())
    assert result['observed_distance_m'] == pytest.approx(15., abs=.5)
    assert result['measured_duration_s'] == pytest.approx(5.)
