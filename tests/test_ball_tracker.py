import pytest
import numpy as np
from src.tracking.ball_tracker import BallTracker, BallPointState
from src.utils.bbox_utils import BBox

def test_interpolation_gap_limit():
    tracker = BallTracker(max_interpolation_gap=5)
    # 1 detected, 6 missing frames, 1 detected
    detections = [BBox(100, 100, 110, 110)] + [None]*6 + [BBox(200, 200, 210, 210)]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert traj_interp[0].state == BallPointState.DETECTED
    # Gaps > 5 should remain MISSING
    for i in range(1, 7):
        assert traj_interp[i].state == BallPointState.MISSING
    assert traj_interp[7].state == BallPointState.DETECTED

def test_short_gap_interpolated():
    tracker = BallTracker(max_interpolation_gap=5)
    # 1 detected, 3 missing frames (<= 5), 1 detected
    detections = [BBox(100, 100, 110, 110)] + [None]*3 + [BBox(200, 200, 210, 210)]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert traj_interp[0].state == BallPointState.DETECTED
    for i in range(1, 4):
        assert traj_interp[i].state == BallPointState.INTERPOLATED
        assert traj_interp[i].x_px is not None
        assert traj_interp[i].y_px is not None
    assert traj_interp[4].state == BallPointState.DETECTED

def test_no_backfill():
    tracker = BallTracker(max_interpolation_gap=5)
    detections = [None, None, BBox(100, 100, 110, 110)]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert traj_interp[0].state == BallPointState.MISSING
    assert traj_interp[1].state == BallPointState.MISSING
    assert traj_interp[2].state == BallPointState.DETECTED

def test_all_detected():
    tracker = BallTracker(max_interpolation_gap=5)
    detections = [BBox(10, 10, 20, 20), BBox(20, 20, 30, 30)]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert all(p.state == BallPointState.DETECTED for p in traj_interp)

def test_all_missing():
    tracker = BallTracker(max_interpolation_gap=5)
    detections = [None, None, None]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert all(p.state == BallPointState.MISSING for p in traj_interp)

def test_state_tracking():
    tracker = BallTracker(max_interpolation_gap=5)
    detections = [BBox(10, 10, 20, 20), None, BBox(20, 20, 30, 30)]
    traj = tracker.create_trajectory(detections, fps=30.0)
    traj_interp = tracker.interpolate_trajectory(traj)
    
    assert traj_interp[0].state == BallPointState.DETECTED
    assert traj_interp[1].state == BallPointState.INTERPOLATED
    assert traj_interp[2].state == BallPointState.DETECTED
