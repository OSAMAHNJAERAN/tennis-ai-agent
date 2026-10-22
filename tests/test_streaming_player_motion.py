"""Analytic motion, source gaps and recovery without whole-match frame lists."""
import json

import numpy as np
import pytest

from src.tracking.player_motion_tracking import motion_samples, summarize_motion
from src.tracking.streaming_player_motion import GroundMotionAccumulator


def test_irregular_source_times_measure_motion_and_gap_barriers():
    motion = GroundMotionAccumulator()
    times = [1791500000.0 + value for value in (0, .04, .09, .14, .28)]
    samples = [motion.observe(time, (3 * (time - times[0]), 0)) for time in times]
    assert all(sample['speed_kmh'] == pytest.approx(10.8) for sample in samples[1:])
    assert motion.summary()['observed_distance_m'] == pytest.approx(3 * (times[-1] - times[0]))
    assert motion.observe(times[-1] + .04, (30, 0), transition_allowed=False)['speed_kmh'] is None
    assert motion.observe(times[-1] + .08, None)['ground_position_m'] is None
    assert motion.observe(times[-1] + .12, (30, 0))['speed_kmh'] is None
    assert motion.observe(times[-1] + 1, (31, 0))['speed_kmh'] is None


def test_quadratic_path_acceleration_uses_window_effective_time():
    motion = GroundMotionAccumulator()
    for index in range(15):
        time = index * .05
        sample = motion.observe(time, (time ** 2, 0))
        if index >= 2:
            assert sample['velocity_mps'][0] == pytest.approx(2 * sample['velocity_effective_timestamp_s'])
            assert sample['acceleration_mps2'] == pytest.approx([2., 0.])


@pytest.mark.parametrize('fps', [25, 30, 60])
def test_proxy_estimator_agrees_with_retained_batch_contract(fps):
    times = (np.arange(201) / fps).tolist()
    points = np.column_stack((np.array(times) * 3, np.zeros(201)))
    points += np.random.default_rng(31).normal(0, .02, points.shape)
    points = points.tolist()
    points[45] = None
    points[110] = None
    expected = motion_samples(points, times)
    motion = GroundMotionAccumulator()
    for point, time, reference in zip(points, times, expected):
        sample = motion.observe(time, point)
        assert set(sample) == set(reference)
        for key in sample:
            assert sample[key] == pytest.approx(reference[key]) if reference[key] is not None else sample[key] is None
    summary = summarize_motion(points, times)
    for key, value in motion.summary().items():
        assert value == pytest.approx(summary[key]) if isinstance(value, (float, int)) else value == summary[key]


def test_checkpoint_resumes_identical_totals_and_rejects_duplicate_pts():
    first = GroundMotionAccumulator()
    for index in range(12):
        first.observe(index * .14, (index * .14, 0))
    resumed = GroundMotionAccumulator.restore(json.loads(json.dumps(first.checkpoint(), allow_nan=False)))
    for index in range(12, 20):
        timestamp = index * .14
        point = None if index == 16 else (timestamp, 0)
        assert resumed.observe(timestamp, point) == first.observe(timestamp, point)
    assert resumed.summary() == first.summary()
    with pytest.raises(ValueError, match='strictly increasing'):
        resumed.observe(19 * .14, (3, 0))


def test_long_input_retains_only_window_and_finite_checkpoint():
    motion = GroundMotionAccumulator()
    for index in range(20000):
        time = index / 60
        motion.observe(time, (time, 0))
        assert len(motion.window) <= 11
    assert motion.summary()['observed_distance_m'] == pytest.approx(19999 / 60)
    checkpoint = motion.checkpoint()
    assert len(json.dumps(checkpoint, allow_nan=False)) < 2500
    assert len(GroundMotionAccumulator.restore(checkpoint).window) <= 11


def test_unknown_nonfinite_and_impossible_ground_motion_stay_unmeasured():
    motion = GroundMotionAccumulator()
    motion.observe(0, (0, 0))
    assert motion.observe(.04, (30, 0))['speed_kmh'] is None
    assert motion.observe(.08, (np.nan, 0))['ground_position_m'] is None
    assert motion.summary()['observed_distance_m'] is None
    with pytest.raises(ValueError, match='two coordinates'):
        motion.observe(.12, (1, 2, 3))
    with pytest.raises(TypeError):
        motion.observe(.12, (0, 0), transition_allowed=1)


def test_unexpected_source_density_fails_instead_of_dropping_samples():
    motion = GroundMotionAccumulator(max_window_samples=4)
    for index in range(4):
        motion.observe(index / 100, (index / 100, 0))
    with pytest.raises(ValueError, match='no silent dropping'):
        motion.observe(.04, (.04, 0))
    corrupted = motion.checkpoint()
    corrupted['window'] *= 3
    with pytest.raises(ValueError, match='bounded window'):
        GroundMotionAccumulator.restore(corrupted)


def test_checkpoint_cannot_introduce_a_future_fit_time_or_unearned_duration():
    motion = GroundMotionAccumulator()
    for time in (0, .0625, .125):
        motion.observe(time, (time, 0))
    checkpoint = json.loads(json.dumps(motion.checkpoint()))
    checkpoint['previous_velocity'][0] = checkpoint['last_timestamp']
    with pytest.raises(ValueError, match='prior fit differs'):
        GroundMotionAccumulator.restore(checkpoint)
    empty = GroundMotionAccumulator().checkpoint()
    empty['duration'] = 10
    with pytest.raises(ValueError, match='aggregates disagree'):
        GroundMotionAccumulator.restore(empty)


def test_acceleration_is_unknown_when_wall_clock_resolution_collapses_fit_times():
    motion = GroundMotionAccumulator()
    origin = 1e9
    tick = np.spacing(origin)
    for index in range(8):
        sample = motion.observe(origin + index * tick, (index * tick, 0))
        json.dumps(sample, allow_nan=False)
    GroundMotionAccumulator.restore(json.loads(json.dumps(motion.checkpoint(), allow_nan=False)))
