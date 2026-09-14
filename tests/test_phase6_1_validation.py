"""
Unit and Regression Tests for Phase 6.1: Real-World Shot Classification Validation
Tests benchmark splits, handedness inversion, near/far court normalization, pose fallback, and rally analytics.
"""

import json
import pytest
from src.shot_analysis.shot_types import (
    ShotType, ShotDirection, PlayerHandedness, CourtDepthZone, CourtLateralZone, CourtZone3x3, ShotClassificationSource, ShotEventEvidence
)
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.court_zones import CourtZoneEngine
from src.shot_analysis.shot_linker import TennisShotLinker
from src.analytics.rally_analyzer import RallyAnalyzer
from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.events.event_detector import TennisEvent, EventType


def test_benchmark_split_independence():
    """Verify that dev, val, and held-out test splits have 0 overlapping rally IDs."""
    with open("data/benchmarks/shot_classification_real/splits.json", "r") as f:
        splits = json.load(f)["splits"]

    dev_rallies = set(splits["development"]["rally_ids"])
    val_rallies = set(splits["validation"]["rally_ids"])
    test_rallies = set(splits["held_out_test"]["rally_ids"])

    # Strict pairwise disjointness
    assert len(dev_rallies.intersection(val_rallies)) == 0, "Leakage: Dev and Val overlap!"
    assert len(dev_rallies.intersection(test_rallies)) == 0, "Leakage: Dev and Test overlap!"
    assert len(val_rallies.intersection(test_rallies)) == 0, "Leakage: Val and Test overlap!"


def test_real_label_provenance_and_support():
    """Verify benchmark has balanced real live Forehand, Backhand, and Serve strokes."""
    with open("data/benchmarks/shot_classification_real/real_rallies_ground_truth.json", "r") as f:
        bench = json.load(f)

    strokes = bench["strokes"]
    forehands = [s for s in strokes if s["shot_type"] == "FOREHAND"]
    backhands = [s for s in strokes if s["shot_type"] == "BACKHAND"]
    serves = [s for s in strokes if s["shot_type"] == "SERVE"]

    assert len(forehands) >= 10, f"Insufficient Forehand samples: {len(forehands)}"
    assert len(backhands) >= 10, f"Insufficient Backhand samples: {len(backhands)}"
    assert len(serves) >= 5, f"Insufficient Serve samples: {len(serves)}"


def test_handedness_metadata_inversion():
    """Verify Left-Handed vs Right-Handed stroke logic is cleanly inverted."""
    clf_right = TennisShotClassifier(
        handedness_map={1: PlayerHandedness.RIGHT_HANDED}, court_orientation_map={1: 1.0}
    )
    clf_left = TennisShotClassifier(
        handedness_map={1: PlayerHandedness.LEFT_HANDED}, court_orientation_map={1: 1.0}
    )

    p_box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=1)
    # Ball to the right of player (+35 px)
    b_pt = TemporalBallPoint(50, 1.5, 185.0, 200.0, BallState.DETECTED)

    st_r, _, src_r, _, _ = clf_right.classify_shot("PLAYER_1_HIT", 50, 1, p_box, b_pt)
    assert st_r == ShotType.FOREHAND

    st_l, _, src_l, _, _ = clf_left.classify_shot("PLAYER_1_HIT", 50, 1, p_box, b_pt)
    assert st_l == ShotType.BACKHAND


def test_near_far_court_normalization():
    """Verify Near (Player 1) and Far (Player 2) court sides are properly normalized."""
    clf = TennisShotClassifier(
        handedness_map={1: PlayerHandedness.RIGHT_HANDED, 2: PlayerHandedness.RIGHT_HANDED},
        court_orientation_map={1: 1.0, 2: -1.0},
    )

    # Near Player 1: cx=150. Ball on right side (x=185) -> FOREHAND
    p1_box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=1)
    b1_pt = TemporalBallPoint(50, 1.5, 185.0, 200.0, BallState.DETECTED)
    st1, _, _, _, _ = clf.classify_shot("PLAYER_1_HIT", 50, 1, p1_box, b1_pt)
    assert st1 == ShotType.FOREHAND

    # Far Player 2 (facing camera/downward): Ball on camera left (x=115) is player's right hand -> FOREHAND
    p2_box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=2)
    b2_pt = TemporalBallPoint(60, 1.8, 115.0, 200.0, BallState.DETECTED)
    st2, _, _, _, _ = clf.classify_shot("PLAYER_2_HIT", 60, 2, p2_box, b2_pt)
    assert st2 == ShotType.FOREHAND


def test_pose_unavailable_safe_fallback():
    """Verify classifier falls back safely when pose model is not available."""
    clf = TennisShotClassifier(
        pose_extractor=None,
        handedness_map={1: PlayerHandedness.RIGHT_HANDED},
        court_orientation_map={1: 1.0},
    )
    p_box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=1)
    b_pt = TemporalBallPoint(50, 1.5, 185.0, 200.0, BallState.DETECTED)

    st, conf, src, comps, reason = clf.classify_shot(
        event_type="PLAYER_1_HIT",
        hit_frame=50,
        player_id=1,
        player_box=p_box,
        ball_point=b_pt
    )
    assert st == ShotType.FOREHAND
    assert src == ShotClassificationSource.GEOMETRY_BASELINE


def test_hit_to_bounce_linking():
    """Verify shot linker correctly links sequential hits and bounces."""
    events = [
        TennisEvent(event_id=1, event_type=EventType.SERVE_CONTACT, frame_index=23, timestamp_s=0.77, player_id=2, ball_position_px=(100, 200), court_position_m=(4.28, 6.71), confidence=0.95, trajectory_state="DETECTED"),
        TennisEvent(event_id=2, event_type=EventType.BOUNCE, frame_index=81, timestamp_s=2.70, player_id=None, ball_position_px=(150, 400), court_position_m=(3.11, 20.56), confidence=0.92, trajectory_state="DETECTED")
    ]
    ball_traj = [
        TemporalBallPoint(frame_index=23, timestamp_seconds=0.77, x_px=100.0, y_px=200.0, court_x_m=4.28, court_y_m=6.71, state=BallState.DETECTED),
        TemporalBallPoint(frame_index=81, timestamp_seconds=2.70, x_px=150.0, y_px=400.0, court_x_m=3.11, court_y_m=20.56, state=BallState.DETECTED)
    ]
    p_boxes = {23: {2: BBox(80, 150, 120, 250, 0.9, 2)}}

    linker = TennisShotLinker(classifier=TennisShotClassifier())
    shots = linker.link_shots(
        events=events,
        ball_trajectory=ball_traj,
        player1_boxes=[None]*100,
        player2_boxes=[BBox(80, 150, 120, 250, 0.9, 2) if i == 23 else None for i in range(100)],
        raw_frames=None,
        dead_event_ids=set()
    )

    assert len(shots) == 1
    assert shots[0].shot_type == ShotType.SERVE
    assert shots[0].bounce_frame == 81
    assert shots[0].landing_zone.value == "MID_LEFT"


def test_dead_ball_exclusion_regression():
    """Verify dead-ball shots are suppressed and excluded from rally segmentation."""
    shots = [
        ShotEventEvidence(1, 1, 23, 0.77, 2, ShotType.SERVE, 0.95, ShotClassificationSource.EVENT_PASSTHROUGH, ShotDirection.DOWN_THE_LINE, 0.9, "GEOMETRY_DERIVED", is_dead_ball=False),
        ShotEventEvidence(2, 2, 84, 2.80, 1, ShotType.UNKNOWN, 0.0, ShotClassificationSource.ABSTENTION_UNKNOWN, ShotDirection.UNKNOWN, 0.0, "GEOMETRY_DERIVED", is_dead_ball=True)
    ]
    rallies = RallyAnalyzer.analyze_rallies(shots)

    assert len(rallies) == 1
    # Dual stroke counts
    assert rallies[0].total_strokes_including_serve == 1
    assert rallies[0].rally_hits_excluding_serve == 0


def test_direction_normalization():
    """Verify shot direction across both sides."""
    # Near player cross-court (from X=2.0m to X=8.5m)
    dir_cc, _, _ = ShotDirectionClassifier.classify_direction((2.0, 22.0), (8.5, 4.0))
    assert dir_cc == ShotDirection.CROSS_COURT

    # Far player down-the-line (from X=8.5m to X=8.6m)
    dir_dtl, _, _ = ShotDirectionClassifier.classify_direction((8.5, 3.0), (8.6, 21.0))
    assert dir_dtl == ShotDirection.DOWN_THE_LINE

    # Middle landing (X=5.5m)
    dir_mid, _, _ = ShotDirectionClassifier.classify_direction((2.0, 22.0), (5.5, 4.0))
    assert dir_mid == ShotDirection.MIDDLE


def test_court_zone_orientation():
    """Verify canonical 3x3 court zone mapping."""
    # Near baseline left
    z1 = CourtZoneEngine.get_3x3_zone(2.0, 22.0)
    assert z1 == CourtZone3x3.DEEP_LEFT

    # Near service box center
    z2 = CourtZoneEngine.get_3x3_zone(5.5, 14.5)
    assert z2 == CourtZone3x3.SHORT_CENTER

    # Far court right mid
    z3 = CourtZoneEngine.get_3x3_zone(8.5, 5.0)
    assert z3 == CourtZone3x3.MID_RIGHT


def test_serve_target_definitions():
    """Verify service placement categories (Wide, Body, T)."""
    # Deuce court T serve (near center service line X=5.485m, landing at X=5.10m -> dist = 0.385m <= 0.85m)
    target_t = CourtZoneEngine.classify_service_placement(5.10, 15.0)
    assert target_t == "T"

    # Wide serve (near sideline, landing at X=2.20m -> dist from center = 3.285m >= 2.50m)
    target_w = CourtZoneEngine.classify_service_placement(2.20, 15.0)
    assert target_w == "WIDE"

    # Body serve (intermediate landing at X=3.80m)
    target_b = CourtZoneEngine.classify_service_placement(3.80, 15.0)
    assert target_b == "BODY"
