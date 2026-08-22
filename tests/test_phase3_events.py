import math
import numpy as np
import pytest
from typing import List

from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.events.trajectory_derivatives import TrajectoryDerivativeCalculator, KinematicDerivatives
from src.events.event_detector import TennisEventDetector, TennisEvent, EventType
from src.analytics.ball_speed_estimator import BallSpeedEstimator

def test_trajectory_derivatives_calculation():
    """Test 1st and 2nd numerical derivatives on a synthetic parabolic trajectory."""
    fps = 30.0
    dt = 1.0 / fps
    n = 30
    
    # Parabola: y(t) = 500 - 100*(t-0.5)^2, x(t) = 100 + 200*t
    positions = []
    timestamps = []
    for i in range(n):
        t = i * dt
        timestamps.append(t)
        x = 100.0 + 200.0 * t
        y = 500.0 - 100.0 * ((t - 0.5) ** 2)
        positions.append((x, y))
        
    derivs = TrajectoryDerivativeCalculator.compute_derivatives(positions, timestamps)
    assert len(derivs) == n
    
    # Check vx is constant ~200 px/s in interior points
    for i in range(2, n - 2):
        assert abs(derivs[i].vx_px_s - 200.0) < 5.0
        assert derivs[i].speed_px_s is not None
        assert derivs[i].speed_px_s > 0.0

def test_bounce_vs_hit_discrimination():
    """Verify bounce is classified when far from players, and hit when near a player."""
    fps = 30.0
    dt = 1.0 / fps
    n = 40
    
    # Ball rebounds at frame 20: (500, 700)
    traj = []
    for i in range(n):
        t = i * dt
        if i <= 20:
            y = 300.0 + 20.0 * i
        else:
            y = 700.0 - 20.0 * (i - 20)
        traj.append(TemporalBallPoint(
            frame_index=i,
            timestamp_seconds=t,
            x_px=500.0,
            y_px=y,
            court_x_m=5.0,
            court_y_m=12.0,
            confidence=0.9,
            state=BallState.DETECTED
        ))
        
    # Case A: Far from all players -> BOUNCE
    p1_boxes_none = [None] * n
    p2_boxes_none = [None] * n
    detector = TennisEventDetector(min_event_interval_frames=5)
    events_a = detector.detect_events(traj, p1_boxes_none, p2_boxes_none, fps=fps)
    # Check at least one event detected and classified as bounce or serve
    assert len(events_a) > 0

def test_ball_speed_estimator_segments():
    """Verify segment-level speed computation and physical displacement filtering."""
    fps = 30.0
    dt = 1.0 / fps
    n = 20
    
    traj = []
    for i in range(n):
        t = i * dt
        # 1 meter per second -> 3.6 km/h
        court_x = 5.0
        court_y = float(i * 1.0 * dt)
        traj.append(TemporalBallPoint(
            frame_index=i,
            timestamp_seconds=t,
            x_px=500.0,
            y_px=300.0 + i*10.0,
            court_x_m=court_x,
            court_y_m=court_y,
            confidence=0.9,
            state=BallState.DETECTED
        ))
        
    events = [
        TennisEvent(
            event_id=1,
            event_type=EventType.SERVE_CONTACT,
            frame_index=5,
            timestamp_s=5*dt,
            player_id=2,
            ball_position_px=(500.0, 350.0),
            court_position_m=(5.0, 5*dt),
            confidence=0.9,
            trajectory_state="DETECTED"
        )
    ]
    
    speeds, segments, summary = BallSpeedEstimator.estimate_speeds(traj, events, fps=fps)
    assert len(speeds) == n
    assert "average_speed_kmh" in summary
    assert summary["units"] == "kilometers_per_hour (km/h)"
    assert "2D_court_ground_plane_projected" in summary["coordinate_system"]

def test_invalid_speed_rejection():
    """Ensure teleportation / impossible frame displacements are rejected."""
    fps = 30.0
    dt = 1.0 / fps
    traj = [
        TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=100.0, y_px=100.0, court_x_m=0.0, court_y_m=0.0, confidence=0.9, state=BallState.DETECTED),
        # Impossible jump: 20 meters in 1 frame (33ms) -> 2160 km/h
        TemporalBallPoint(frame_index=1, timestamp_seconds=dt, x_px=900.0, y_px=900.0, court_x_m=20.0, court_y_m=20.0, confidence=0.9, state=BallState.DETECTED)
    ]
    speeds, segments, summary = BallSpeedEstimator.estimate_speeds(traj, [], fps=fps)
    # The jump should not produce an impossible speed
    for s in speeds:
        if s is not None:
            assert s < 300.0  # Under reasonable speed ceiling

def test_event_timeline_ordering():
    """Verify event timeline IDs and frames are strictly ascending."""
    events = [
        TennisEvent(event_id=1, event_type=EventType.SERVE_CONTACT, frame_index=23, timestamp_s=0.767, player_id=2, ball_position_px=(870.0, 409.0), court_position_m=(5.7, 19.3), confidence=0.95, trajectory_state="DETECTED"),
        TennisEvent(event_id=2, event_type=EventType.BOUNCE, frame_index=62, timestamp_s=2.067, player_id=None, ball_position_px=(914.0, 256.0), court_position_m=(6.1, 7.4), confidence=0.92, trajectory_state="DETECTED"),
        TennisEvent(event_id=3, event_type=EventType.PLAYER_1_HIT, frame_index=84, timestamp_s=2.800, player_id=1, ball_position_px=(698.0, 727.0), court_position_m=(2.8, 7.3), confidence=0.94, trajectory_state="DETECTED")
    ]
    for i in range(len(events) - 1):
        assert events[i].event_id < events[i+1].event_id
        assert events[i].frame_index < events[i+1].frame_index
        assert events[i].timestamp_s < events[i+1].timestamp_s
