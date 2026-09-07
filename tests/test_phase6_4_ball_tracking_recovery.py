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
from src.tracking.ball_proposal_filter import (
    BallProposalFilter,
    ProposalFilterSettings,
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


def test_rapid_ball_movement():
    """Verify high-velocity tennis ball (fast smash/serve) within speed threshold is reliably tracked."""
    tracker = TemporalBallTracker(
        max_valid_speed_px_per_frame=80.0,
        base_gating_radius_px=50.0,
    )
    # Ball moving at 55 px/frame downwards
    detections = [
        [BallObservation(640.0, 100.0 + f * 55.0, 0.70)]
        for f in range(8)
    ]
    trajectory = tracker.track_video_candidates(detections, fps=30.0, frame_size=(1280, 720))
    for f in range(8):
        assert trajectory[f].state in (BallState.DETECTED, BallState.TRACKED)
        assert trajectory[f].y_px is not None
        assert abs(trajectory[f].y_px - (100.0 + f * 55.0)) < 1.0


def test_small_ball_detection():
    """Verify small ball proposals within physical bounding limits are accepted, while subpixel noise is rejected."""
    filter_mod = BallProposalFilter(ProposalFilterSettings(min_bbox_dim_px=3.0, max_bbox_dim_px=30.0))

    obs_valid_small = BallObservation(500.0, 300.0, 0.5, bbox=BBox(498.0, 298.0, 502.0, 302.0))  # 4x4
    obs_subpixel_noise = BallObservation(500.0, 300.0, 0.5, bbox=BBox(499.5, 299.5, 500.5, 300.5))  # 1x1

    filtered = filter_mod.filter_video_candidates([[obs_valid_small, obs_subpixel_noise]])
    assert len(filtered[0]) == 1
    assert filtered[0][0] == obs_valid_small


def test_prediction_timeout():
    """Verify tracker stops predicting after max_prediction_gap and transitions cleanly to MISSING."""
    tracker = TemporalBallTracker(max_prediction_gap=3, max_interpolation_gap=2)
    detections = [
        [BallObservation(100.0, 100.0, 0.8)],
        [BallObservation(120.0, 120.0, 0.8)],
        [],  # frame 2: predicted (gap 1)
        [],  # frame 3: predicted (gap 2)
        [],  # frame 4: predicted (gap 3)
        [],  # frame 5: timeout -> MISSING
        [],  # frame 6: MISSING
    ]
    traj = tracker.track_video_candidates(detections, fps=30.0)
    assert traj[0].state == BallState.DETECTED
    assert traj[1].state == BallState.DETECTED
    assert traj[2].state == BallState.PREDICTED
    assert traj[3].state == BallState.PREDICTED
    assert traj[4].state == BallState.PREDICTED
    assert traj[5].state == BallState.MISSING
    assert traj[6].state == BallState.MISSING


def test_synthetic_gap_speed_rejection():
    """Verify Pass 5 interpolation rejects disparate distant points to prevent synthetic trajectories."""
    tracker = TemporalBallTracker(
        max_prediction_gap=1,
        max_interpolation_gap=4,
        max_valid_speed_px_per_frame=40.0,
    )
    # Dying track at (200, 200), then 3 empty frames, then a new object at (900, 600)
    # Gap = 4 frames, Euclidean dist = sqrt(700^2 + 400^2) = 806 px -> ~201 px/frame >> 40 px/frame
    detections = [
        [BallObservation(190.0, 190.0, 0.7)],
        [BallObservation(200.0, 200.0, 0.7)],
        [],
        [],
        [],
        [BallObservation(900.0, 600.0, 0.7)],
        [BallObservation(910.0, 610.0, 0.7)],
    ]
    traj = tracker.track_video_candidates(detections, fps=30.0)
    # Frames 2, 3, 4 should NOT be interpolated across this 800px teleportation
    for f in (2, 3, 4):
        assert traj[f].state != BallState.INTERPOLATED, f"Frame {f} was illegally interpolated across disparate objects!"


def test_low_confidence_preservation():
    """Verify two-stage association recovers kinematically continuous low-confidence ball observations."""
    tracker = TemporalBallTracker(
        high_conf_thresh=0.40,
        low_conf_thresh=0.03,
        enable_multi_candidate_association=True,
    )
    detections = [
        [BallObservation(100.0, 200.0, 0.85)],
        [BallObservation(120.0, 210.0, 0.80)],
        [BallObservation(140.0, 220.0, 0.08)],  # low confidence motion blur
        [BallObservation(160.0, 230.0, 0.06)],  # low confidence motion blur
        [BallObservation(180.0, 240.0, 0.75)],  # recovered high confidence
    ]
    traj = tracker.track_video_candidates(detections, fps=30.0)
    for f in range(5):
        assert traj[f].state in (BallState.DETECTED, BallState.TRACKED), f"Frame {f} failed association!"


def test_multi_framerate_adaptation():
    """Verify velocity and prediction kinematics adjust consistently across 24, 30, and 60 FPS."""
    for fps in [24.0, 30.0, 60.0]:
        tracker = TemporalBallTracker()
        dt = 1.0 / fps
        # Ball travelling at 600 px/sec -> dx per frame = 600 * dt
        dx = 600.0 * dt
        detections = [
            [BallObservation(100.0 + f * dx, 200.0, 0.75)]
            for f in range(6)
        ]
        traj = tracker.track_video_candidates(detections, fps=fps)
        assert traj[-1].state in (BallState.DETECTED, BallState.TRACKED)
        assert traj[-1].x_px is not None
        assert abs(traj[-1].x_px - (100.0 + 5 * dx)) < 1.5


def test_resolution_scaling_720p_1080p():
    """Verify proposal filter and tracker scale gates proportionally from 720p to 1080p."""
    filt_720 = BallProposalFilter(ProposalFilterSettings(max_bbox_dim_px=30.0))
    filt_1080 = BallProposalFilter(ProposalFilterSettings(max_bbox_dim_px=30.0))

    # In 1080p (scale=1.5), a 40px bounding box is equivalent to 26.6px in 720p and should be valid
    obs_40px = BallObservation(960.0, 540.0, 0.6, bbox=BBox(940.0, 520.0, 980.0, 560.0))  # 40x40

    f720 = filt_720.filter_video_candidates([[obs_40px]], frame_size=(1280, 720), scale=1.0)
    f1080 = filt_1080.filter_video_candidates([[obs_40px]], frame_size=(1920, 1080), scale=1.5)

    assert len(f720[0]) == 0, "40px box should be rejected at 720p (limit 30px)"
    assert len(f1080[0]) == 1, "40px box should be accepted at 1080p (limit 45px)"


def test_oversized_and_margin_rejection():
    """Verify BallProposalFilter rejects oversized distractors (players/racquets) and margin artifacts."""
    filt = BallProposalFilter(ProposalFilterSettings(
        max_bbox_dim_px=35.0,
        top_margin_fraction=0.05,
        bottom_margin_fraction=0.05,
        side_margin_fraction=0.05,
    ))

    # 1. Valid ball in court
    valid_ball = BallObservation(600.0, 400.0, 0.8, bbox=BBox(595.0, 395.0, 605.0, 405.0))
    # 2. Player head (oversized: 60x60)
    player_head = BallObservation(620.0, 350.0, 0.9, bbox=BBox(590.0, 320.0, 650.0, 380.0))
    # 3. Racquet (elongated: 80x20, aspect ratio 4.0)
    racquet = BallObservation(630.0, 370.0, 0.85, bbox=BBox(590.0, 360.0, 670.0, 380.0))
    # 4. Top scoreboard margin artifact (y=10 on 720p frame)
    scoreboard = BallObservation(100.0, 10.0, 0.8, bbox=BBox(95.0, 5.0, 105.0, 15.0))

    filtered = filt.filter_video_candidates(
        [[valid_ball, player_head, racquet, scoreboard]],
        frame_size=(1280, 720),
        scale=1.0,
    )
    assert len(filtered[0]) == 1
    assert filtered[0][0] == valid_ball
