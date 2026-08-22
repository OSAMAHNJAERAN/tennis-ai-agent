"""
Unit and regression tests for Phase 6.4: Ball-Tracking & Event-Window Observability Recovery.
Validates multi-candidate association, static distractor filtering, adaptive uncertainty gating,
scale normalization (720p vs 1080p), short-gap ballistic reacquisition, and provenance preservation.
"""

import math
import pytest
import numpy as np

from src.tracking.temporal_ball_tracker import (
    BallObservation,
    BallState,
    TemporalBallPoint,
    TemporalBallTracker,
    KinematicFilter,
)
from src.utils.bbox_utils import BBox


def test_static_distractor_suppression():
    """Verify that static false-positive objects (zero velocity across multiple frames) are suppressed."""
    tracker = TemporalBallTracker(
        high_conf_thresh=0.08,
        low_conf_thresh=0.01,
        enable_multi_candidate_association=True,
    )
    
    # 6 frames where static distractor sits at (411.0, 270.0) with conf 0.50,
    # but a real moving ball moves from (500.0, 500.0) to (580.0, 580.0) with conf 0.15.
    detections = []
    for f in range(6):
        c_static = BallObservation(411.0, 270.0, 0.50)
        c_moving = BallObservation(500.0 + f * 16.0, 500.0 + f * 16.0, 0.15)
        detections.append([c_static, c_moving])
        
    trajectory = tracker.track_video_candidates(detections, fps=30.0, frame_size=(1280, 720))
    
    # The tracker should NOT track the static distractor at (411.0, 270.0)
    for p in trajectory:
        if p.x_px is not None:
            assert abs(p.x_px - 411.0) > 20.0, f"Static distractor tracked at frame {p.frame_index}!"


def test_adaptive_scale_normalization_1080p():
    """Verify that Kalman gating and speed limits scale appropriately for 1080p vs 720p."""
    tracker_1080 = TemporalBallTracker(
        base_gating_radius_px=45.0,
        max_valid_speed_px_per_frame=80.0,
        enable_scale_normalization=True,
    )
    
    # In 1080p, a high-speed ball moving 100 px/frame should NOT be rejected as an outlier
    detections_1080 = [
        [BallObservation(500.0, 500.0, 0.50)],
        [BallObservation(570.0, 570.0, 0.40)],  # step ~99 px
        [BallObservation(640.0, 640.0, 0.40)],  # step ~99 px
    ]
    
    traj_1080 = tracker_1080.track_video_candidates(
        detections_1080, fps=30.0, frame_size=(1920, 1080)
    )
    assert traj_1080[1].state in (BallState.DETECTED, BallState.TRACKED)
    assert traj_1080[2].state in (BallState.DETECTED, BallState.TRACKED)


def test_short_gap_ballistic_reacquisition():
    """Verify that after a 2-frame occlusion/gap, the track is cleanly reacquired."""
    tracker = TemporalBallTracker(
        high_conf_thresh=0.08,
        low_conf_thresh=0.01,
        max_prediction_gap=4,
        max_interpolation_gap=3,
        enable_short_gap_reacquisition=True,
    )
    
    # 3 initial frames moving at +20 px/frame, then 2 missing frames, then 3 reappearing frames
    detections = [
        [BallObservation(100.0, 200.0, 0.60)],
        [BallObservation(120.0, 220.0, 0.50)],
        [BallObservation(140.0, 240.0, 0.50)],
        [],  # frame 3: missing
        [],  # frame 4: missing
        [BallObservation(200.0, 300.0, 0.06)],  # frame 5: reappears with low conf
        [BallObservation(220.0, 320.0, 0.08)],  # frame 6: continuation
        [BallObservation(240.0, 340.0, 0.12)],  # frame 7: continuation
    ]
    
    traj = tracker.track_video_candidates(detections, fps=30.0, frame_size=(1280, 720))
    
    # Verify initial frames are DETECTED
    assert traj[0].state == BallState.DETECTED
    assert traj[1].state == BallState.DETECTED
    assert traj[2].state == BallState.DETECTED
    
    # Verify gap frames 3 and 4 are PREDICTED
    assert traj[3].state == BallState.PREDICTED
    assert traj[4].state == BallState.PREDICTED
    
    # Verify frame 5, 6, 7 are successfully TRACKED / DETECTED (reacquired)
    assert traj[5].state in (BallState.TRACKED, BallState.DETECTED)
    assert traj[6].state in (BallState.TRACKED, BallState.DETECTED)
    assert traj[7].state in (BallState.TRACKED, BallState.DETECTED)


def test_gap_limited_interpolation_preserves_provenance():
    """Verify that gaps <= 3 frames that were unpredicted become INTERPOLATED with confidence=None."""
    tracker = TemporalBallTracker(
        max_prediction_gap=1,
        max_interpolation_gap=3,
    )
    
    # 2 initial frames, 2 missing frames (frame 2 is predicted, frame 3 exceeds max_prediction_gap -> MISSING -> interpolated), 2 final frames
    detections_short = [
        [BallObservation(100.0, 100.0, 0.80)],
        [BallObservation(120.0, 120.0, 0.80)],
        [],
        [],
        [BallObservation(180.0, 180.0, 0.80)],
        [BallObservation(200.0, 200.0, 0.80)],
    ]
    traj_short = tracker.track_video_candidates(detections_short, fps=30.0)
    assert traj_short[2].state in (BallState.PREDICTED, BallState.INTERPOLATED)
    assert traj_short[3].state == BallState.INTERPOLATED
    assert traj_short[3].confidence is None


def test_all_ball_states_in_contract():
    """Verify all BallState enum values adhere to the system contract."""
    expected_states = {"DETECTED", "TRACKED", "PREDICTED", "INTERPOLATED", "OCCLUDED", "MISSING"}
    actual_states = {s.value for s in BallState}
    assert actual_states == expected_states
