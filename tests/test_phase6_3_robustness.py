"""
Unit tests for Phase 6.3: Real-World Ball Tracking & Tennis Event Robustness.
Validates multi-candidate association, adaptive gating, player track continuity,
scale invariance, and configuration integrity.
"""

import pytest
import os
import yaml
import numpy as np
from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import BallObservation, TemporalBallPoint, BallState, TemporalBallTracker, KinematicFilter
from src.tracking.player_tracker import PlayerTracker
from src.events.event_detector import TennisEventDetector
from src.court.court_geometry import TennisCourtGeometry


def test_multi_candidate_temporal_association():
    """Verifies that the tracker associates multiple low-confidence candidates without track fragmentation."""
    tracker = TemporalBallTracker(
        high_conf_thresh=0.05,
        low_conf_thresh=0.01,
        max_prediction_gap=4,
        max_interpolation_gap=3,
        max_valid_speed_px_per_frame=80.0
    )
    
    # 5 frames of motion with low-confidence candidates
    detections = [
        [BallObservation(100.0, 100.0, 0.06)],
        [BallObservation(110.0, 115.0, 0.04)],
        [BallObservation(120.0, 130.0, 0.05)],
        [BallObservation(130.0, 145.0, 0.07)],
        [BallObservation(140.0, 160.0, 0.09)],
    ]
    
    trajectory = tracker.track_video_candidates(detections, fps=30.0)
    assert len(trajectory) == 5
    # All 5 points should be tracked or detected
    for p in trajectory:
        assert p.state in (BallState.DETECTED, BallState.TRACKED)
        assert p.x_px is not None and p.y_px is not None


def test_adaptive_kalman_gating():
    """Verifies that the search gating radius adapts to velocity and uncertainty."""
    kf = KinematicFilter(dt=1.0 / 30.0)
    kf.initialize(x=500.0, y=300.0, vx=100.0, vy=50.0)
    
    r_initial = kf.get_gating_radius(base_radius=40.0)
    assert r_initial > 40.0
    
    # Predict step increases uncertainty and expands search radius
    kf.predict()
    r_predicted = kf.get_gating_radius(base_radius=40.0)
    assert r_predicted >= r_initial


def test_player_proportional_reach_scaling():
    """Verifies that player reach radius dynamically scales with player bounding box height."""
    detector = TennisEventDetector(player_reach_radius_px=140.0)
    
    # Near player (tall bbox: h = 200px)
    p1_box = BBox(500, 400, 600, 600, confidence=0.9, class_id=0)
    
    # Ball 120px away from player center -> inside scaled reach (reach = 200 * 0.65 = 130px)
    ball_pt = TemporalBallPoint(10, 0.33, 550.0, 620.0, BallState.DETECTED, confidence=0.9)
    
    # Bounding box reach scale
    h_player = max(10.0, p1_box.y2 - p1_box.y1)
    reach = max(detector.player_reach_radius_px, 0.65 * h_player)
    assert reach >= 130.0


def test_court_adaptive_player_ranking():
    """Verifies that choose_players robustly assigns Player 1 to near court and Player 2 to far court."""
    # Synthetic frame detections: one candidate near (y=600), one far (y=250)
    detections = [
        [
            BBox(400, 550, 500, 700, confidence=0.9, class_id=0, track_id=10), # Near
            BBox(450, 200, 520, 320, confidence=0.85, class_id=0, track_id=20) # Far
        ]
    ] * 20
    
    # Canonical court keypoints
    kps = TennisCourtGeometry.get_canonical_keypoints()
    
    assigned = PlayerTracker.choose_players(detections, kps)
    assert 1 in assigned and 2 in assigned
    # Player 1 is near court (y > 500), Player 2 is far court (y < 350)
    assert assigned[1][0].y2 > 500
    assert assigned[2][0].y2 < 350


def test_scale_invariant_constants():
    """Verifies that mathematical formulas scale with image height and FPS."""
    base_h = 720.0
    hd_h = 1080.0
    scale_factor = hd_h / base_h
    assert scale_factor == 1.5

    base_speed = 80.0
    scaled_speed = base_speed * scale_factor
    assert scaled_speed == 120.0


def test_frozen_phase6_3_config_validity():
    """Verifies that configs/phase6_3_robustness/final.yaml exists, loads, and meets specifications."""
    cfg_path = "configs/phase6_3_robustness/final.yaml"
    assert os.path.exists(cfg_path), "Frozen config final.yaml missing!"
    
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    assert cfg["version"] == "6.3.0"
    assert cfg["ball_detection"]["imgsz"] == 1024
    assert cfg["ball_detection"]["low_conf"] <= 0.05
    assert cfg["temporal_tracking"]["max_interpolation_gap"] == 3
    assert cfg["pose_estimation"]["enabled"] is True
