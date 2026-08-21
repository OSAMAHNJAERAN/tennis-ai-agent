import pytest
import numpy as np
from src.analytics.player_analytics import (
    calculate_distance,
    calculate_speed,
    calculate_cumulative_distance
)

def test_zero_movement_zero_distance():
    positions = [(5.0, 5.0) for _ in range(10)]
    dist = calculate_distance(positions)
    assert dist == 0.0

def test_known_displacement():
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    dist = calculate_distance(positions)
    assert np.isclose(dist, 2.0)

def test_speed_from_timestamps():
    # 2 meters in 0.1 seconds -> 20 m/s
    positions = [(0.0, 0.0), (2.0, 0.0)]
    timestamps = [0.0, 0.1]
    speeds = calculate_speed(positions, timestamps, window=2)
    valid_speeds = [s for s in speeds if s is not None]
    assert len(valid_speeds) > 0
    assert np.isclose(valid_speeds[0], 20.0, atol=1.0)

def test_outlier_rejection():
    # jump 10 meters in 1 frame (impossible)
    positions = [(0.0, 0.0), (10.0, 0.0), (0.1, 0.0)]
    dist = calculate_distance(positions)
    # The 10m jump should be rejected (d <= 5.0 threshold)
    assert dist < 10.0

def test_missing_positions_handled():
    positions = [(0.0, 0.0), None, (1.0, 0.0)]
    dist = calculate_distance(positions)
    assert np.isclose(dist, 1.0)

def test_cumulative_distance():
    positions = [(0.0, 0.0), (1.0, 0.0), (1.0, 2.0)]
    cum_dist = calculate_cumulative_distance(positions)
    assert len(cum_dist) == 3
    assert np.isclose(cum_dist[0], 0.0)
    assert np.isclose(cum_dist[1], 1.0)
    assert np.isclose(cum_dist[2], 3.0)
