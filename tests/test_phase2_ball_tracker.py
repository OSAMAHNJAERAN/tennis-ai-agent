import pytest
import numpy as np
from src.tracking.temporal_ball_tracker import (
    TemporalBallTracker,
    KinematicFilter,
    BallState,
    BallObservation,
    TemporalBallPoint
)
from src.utils.bbox_utils import BBox

def test_kinematic_filter_linear_motion():
    kf = KinematicFilter(dt=1.0/30.0)
    kf.initialize(x=100.0, y=100.0, vx=300.0, vy=0.0)
    
    # Predict 3 frames ahead
    for _ in range(3):
        pred_x, pred_y = kf.predict()
        
    # At 300 px/s with dt=1/30s, each frame moves 10 px
    assert np.isclose(pred_x, 130.0, atol=2.0)
    assert np.isclose(pred_y, 100.0, atol=2.0)

def test_temporal_tracker_state_transitions():
    tracker = TemporalBallTracker(high_conf_thresh=0.20, low_conf_thresh=0.03, max_prediction_gap=3)
    
    # Frame 0: High conf detection -> DETECTED
    # Frame 1: Low conf candidate near prediction -> TRACKED
    # Frame 2: No candidate -> PREDICTED
    # Frame 3: High conf detection -> DETECTED
    candidates = [
        [BallObservation(100.0, 100.0, 0.85)],
        [BallObservation(110.0, 100.0, 0.08)],  # matches prediction ~110
        [],                                      # no candidate -> predict ~120
        [BallObservation(130.0, 100.0, 0.90)]
    ]
    
    traj = tracker.track_video_candidates(candidates, fps=30.0)
    
    assert traj[0].state == BallState.DETECTED
    assert traj[0].confidence == 0.85
    
    assert traj[1].state == BallState.TRACKED
    assert traj[1].confidence is not None
    
    assert traj[2].state == BallState.PREDICTED
    assert traj[2].confidence is not None
    
    assert traj[3].state == BallState.DETECTED
    assert traj[3].confidence == 0.90

def test_impossible_trajectory_rejection():
    tracker = TemporalBallTracker(max_valid_speed_px_per_frame=50.0)
    
    # Impossible jump of 500 px in single frame
    candidates = [
        [BallObservation(100.0, 100.0, 0.85)],
        [BallObservation(600.0, 600.0, 0.05)]  # Outlier low-conf noise
    ]
    
    traj = tracker.track_video_candidates(candidates, fps=30.0)
    
    assert traj[0].state == BallState.DETECTED
    # Outlier should not be tracked as valid candidate
    assert traj[1].state in (BallState.PREDICTED, BallState.MISSING)

def test_confidence_modeling():
    tracker = TemporalBallTracker(high_conf_thresh=0.20, low_conf_thresh=0.03)
    
    candidates = [
        [BallObservation(100.0, 100.0, 0.80)],
        [],
        [],
        [BallObservation(130.0, 100.0, 0.80)]
    ]
    traj = tracker.track_video_candidates(candidates, fps=30.0)
    
    assert traj[0].confidence == 0.80
    assert traj[3].confidence == 0.80

def test_all_missing_handled_safely():
    tracker = TemporalBallTracker()
    candidates = [[] for _ in range(10)]
    traj = tracker.track_video_candidates(candidates, fps=30.0)
    
    assert len(traj) == 10
    assert all(p.state == BallState.MISSING for p in traj)
    assert all(p.x_px is None for p in traj)
