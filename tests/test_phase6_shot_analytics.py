import pytest
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.events.event_detector import TennisEvent, EventType
from src.shot_analysis.shot_types import (
    ShotType,
    ShotDirection,
    CourtDepthZone,
    CourtLateralZone,
    CourtZone3x3,
    PlayerHandedness,
    ShotOutcome,
    ShotClassificationSource,
    ShotEventEvidence
)
from src.shot_analysis.court_zones import CourtZoneEngine
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_linker import TennisShotLinker
from src.analytics.rally_analyzer import RallyAnalyzer, RallySegment
from src.analytics.serve_analyzer import ServeAnalyzer
from src.analytics.shot_statistics import ShotStatisticsAnalyzer
from src.analytics.match_analytics import MatchAnalyticsAggregator
from src.scoring.match_state import MatchState

def test_court_zoning_depth_lateral_and_3x3():
    """Tests 3x3 court zoning across short/mid/deep and left/center/right corridors."""
    # Near Court: Y in [11.885, 23.77]
    # SHORT_LEFT: X=2.0m, Y=14.0m (dist from net = 2.115m < 4.5m)
    assert CourtZoneEngine.get_3x3_zone(2.0, 14.0) == CourtZone3x3.SHORT_LEFT
    
    # DEEP_CENTER: X=5.5m, Y=22.0m (dist from net = 10.115m > 9.0m)
    assert CourtZoneEngine.get_3x3_zone(5.5, 22.0) == CourtZone3x3.DEEP_CENTER

    # MID_RIGHT: X=8.5m, Y=18.0m (dist from net = 6.115m)
    assert CourtZoneEngine.get_3x3_zone(8.5, 18.0) == CourtZone3x3.MID_RIGHT

    # Far Court: Y in [0, 11.885]
    # DEEP_LEFT: X=2.0m, Y=1.5m (dist from net = 10.385m > 9.0m)
    assert CourtZoneEngine.get_3x3_zone(2.0, 1.5) == CourtZone3x3.DEEP_LEFT

    # Out of bounds
    assert CourtZoneEngine.get_3x3_zone(-2.0, 14.0) == CourtZone3x3.OUT_OF_BOUNDS

def test_service_placement_classification():
    """Tests classification of serve placement into Wide, Body, or T."""
    # Near Center line (X=5.485m) -> T
    assert CourtZoneEngine.classify_service_placement(5.30, 15.0) == "T"
    assert CourtZoneEngine.classify_service_placement(5.80, 8.0) == "T"

    # Near singles sideline (X=1.37m or X=9.60m) -> WIDE
    assert CourtZoneEngine.classify_service_placement(2.0, 15.0) == "WIDE"
    assert CourtZoneEngine.classify_service_placement(8.8, 8.0) == "WIDE"

    # Mid body area -> BODY
    assert CourtZoneEngine.classify_service_placement(3.8, 15.0) == "BODY"

def test_shot_direction_classification():
    """Tests geometry-derived Cross-court, Down-the-line, and Middle classification."""
    # Struck from left (X=2.0m) -> Lands on right (X=8.5m) -> CROSS_COURT
    dir_cc, conf_cc, _ = ShotDirectionClassifier.classify_direction((2.0, 22.0), (8.5, 4.0))
    assert dir_cc == ShotDirection.CROSS_COURT
    assert conf_cc >= 0.70

    # Struck from left (X=2.0m) -> Lands on left (X=2.5m) -> DOWN_THE_LINE
    dir_dtl, conf_dtl, _ = ShotDirectionClassifier.classify_direction((2.0, 22.0), (2.5, 4.0))
    assert dir_dtl == ShotDirection.DOWN_THE_LINE
    assert conf_dtl >= 0.70

    # Struck from right (X=8.5m) -> Lands in center (X=5.5m) -> MIDDLE
    dir_mid, conf_mid, _ = ShotDirectionClassifier.classify_direction((8.5, 22.0), (5.5, 4.0))
    assert dir_mid == ShotDirection.MIDDLE

    # Insufficient depth displacement
    dir_unk, _, _ = ShotDirectionClassifier.classify_direction((2.0, 22.0), (2.5, 21.0))
    assert dir_unk == ShotDirection.UNKNOWN

def test_serve_event_passthrough():
    """Tests SERVE_CONTACT event automatically classifies as SERVE."""
    clf = TennisShotClassifier()
    st, conf, src, comps, reason = clf.classify_shot(
        event_type="SERVE_CONTACT",
        hit_frame=23,
        player_id=2,
        player_box=None,
        ball_point=None
    )
    assert st == ShotType.SERVE
    assert conf >= 0.90
    assert src == ShotClassificationSource.EVENT_PASSTHROUGH

def test_dead_ball_stroke_suppression():
    """CRITICAL REGRESSION: Dead-ball hits (e.g. Frame 84) must be suppressed to UNKNOWN."""
    clf = TennisShotClassifier()
    st, conf, src, comps, reason = clf.classify_shot(
        event_type="PLAYER_1_HIT",
        hit_frame=84,
        player_id=1,
        player_box=BBox(100, 100, 200, 300, confidence=0.9, class_id=1),
        ball_point=TemporalBallPoint(84, 2.8, 180, 200, BallState.DETECTED),
        is_dead_ball=True
    )
    assert st == ShotType.UNKNOWN
    assert src == ShotClassificationSource.ABSTENTION_UNKNOWN

def test_geometry_baseline_handedness_inversion():
    """Tests handedness-aware geometry classification for Right vs Left-handed players."""
    # Player 1 (Near court, facing up, Right-handed): Ball on right side (x_ball > x_center) -> FOREHAND
    clf_right = TennisShotClassifier(handedness_map={1: PlayerHandedness.RIGHT_HANDED})
    p_box = BBox(100, 100, 200, 300, confidence=0.9, class_id=1) # cx = 150, bw = 100
    b_pt_right = TemporalBallPoint(50, 1.5, 185.0, 200.0, BallState.DETECTED) # dx = +35 px -> FOREHAND
    
    st_r, _, src_r, _, _ = clf_right.classify_shot("PLAYER_1_HIT", 50, 1, p_box, b_pt_right)
    assert st_r == ShotType.FOREHAND
    assert src_r == ShotClassificationSource.GEOMETRY_BASELINE

    # Player 1 (Near court, facing up, Left-handed): Same ball on right side -> BACKHAND
    clf_left = TennisShotClassifier(handedness_map={1: PlayerHandedness.LEFT_HANDED})
    st_l, _, src_l, _, _ = clf_left.classify_shot("PLAYER_1_HIT", 50, 1, p_box, b_pt_right)
    assert st_l == ShotType.BACKHAND
    assert src_l == ShotClassificationSource.GEOMETRY_BASELINE

def test_safe_abstention_on_ambiguity():
    """Tests classifier safely returns UNKNOWN when lateral offset is near center."""
    clf = TennisShotClassifier()
    p_box = BBox(100, 100, 200, 300, confidence=0.9, class_id=1) # cx = 150, bw = 100
    b_pt_center = TemporalBallPoint(50, 1.5, 152.0, 200.0, BallState.DETECTED) # dx = +2 px -> Ambiguous!
    
    st, conf, src, _, _ = clf.classify_shot("PLAYER_1_HIT", 50, 1, p_box, b_pt_center)
    assert st == ShotType.UNKNOWN
    assert src == ShotClassificationSource.ABSTENTION_UNKNOWN

def test_rally_analyzer_dual_stroke_metrics():
    """Tests rally segmentation computes dual stroke metrics and excludes dead-ball shots."""
    # Create simulated shots
    s1 = ShotEventEvidence(
        shot_id=1, match_event_id=1, frame_index=23, timestamp_s=0.77, player_id=2,
        shot_type=ShotType.SERVE, shot_confidence=0.95, classification_source=ShotClassificationSource.EVENT_PASSTHROUGH,
        direction=ShotDirection.CROSS_COURT, direction_confidence=0.9, direction_source="GEOMETRY_DERIVED",
        is_dead_ball=False
    )
    s2 = ShotEventEvidence(
        shot_id=2, match_event_id=2, frame_index=84, timestamp_s=2.80, player_id=1,
        shot_type=ShotType.UNKNOWN, shot_confidence=0.0, classification_source=ShotClassificationSource.ABSTENTION_UNKNOWN,
        direction=ShotDirection.UNKNOWN, direction_confidence=0.0, direction_source="GEOMETRY_DERIVED",
        is_dead_ball=True # DEAD BALL!
    )
    
    rallies = RallyAnalyzer.analyze_rallies([s1, s2], server_id=2, receiver_id=1)
    assert len(rallies) == 1
    # Only live serve counted in live rally strokes
    assert rallies[0].total_strokes_including_serve == 1
    assert rallies[0].rally_hits_excluding_serve == 0

def test_serve_analyzer_accuracy():
    """Tests serve statistics calculation."""
    s1 = ShotEventEvidence(
        shot_id=1, match_event_id=1, frame_index=23, timestamp_s=0.77, player_id=2,
        shot_type=ShotType.SERVE, shot_confidence=0.95, classification_source=ShotClassificationSource.EVENT_PASSTHROUGH,
        direction=ShotDirection.CROSS_COURT, direction_confidence=0.9, direction_source="GEOMETRY_DERIVED",
        landing_court_position_m=(2.5, 15.0), # WIDE
        bounce_frame=81,
        is_dead_ball=False
    )
    from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision
    from src.line_calling.line_geometry import CourtLineType
    lc = LineCallEvidence(
        event_id=2,
        bounce_frame=81,
        decision=LineCallDecision.SERVE_FAULT,
        decision_context="SERVE",
        nearest_line=CourtLineType.NEAR_SERVICE_LINE,
        center_signed_distance_cm=-50.0,
        ball_edge_margin_cm=-50.0,
        contact_patch_model="EMPIRICAL_PATCH",
        contact_patch_radius_cm=1.25,
        position_uncertainty_cm=1.0,
        spatial_tier="NEAR",
        refinement_method="PIECEWISE_IMPACT_INTERSECTION",
        bounce_position_px=(0, 0),
        bounce_position_m=(2.5, 15.0),
        tracker_state="DETECTED",
        confidence=0.95,
        reason=""
    )
    stats = ServeAnalyzer.compute_serve_statistics([s1], [lc], server_id=2)
    assert stats["first_serves_attempted"] == 1
    assert stats["first_serves_faulted"] == 1
    assert stats["first_serve_percentage"] == 0.0
    assert stats["placement_distribution"]["WIDE"] == 1

def test_shot_statistics_and_heatmap_bins():
    """Tests player shot stats and 2D spatial heatmap binning."""
    s1 = ShotEventEvidence(
        shot_id=1, match_event_id=1, frame_index=50, timestamp_s=1.6, player_id=1,
        shot_type=ShotType.FOREHAND, shot_confidence=0.88, classification_source=ShotClassificationSource.YOLO11_POSE_TEMPORAL,
        direction=ShotDirection.CROSS_COURT, direction_confidence=0.92, direction_source="GEOMETRY_DERIVED",
        landing_court_position_m=(3.0, 5.0), landing_zone=CourtZone3x3.DEEP_LEFT
    )
    s2 = ShotEventEvidence(
        shot_id=2, match_event_id=2, frame_index=90, timestamp_s=3.0, player_id=1,
        shot_type=ShotType.BACKHAND, shot_confidence=0.85, classification_source=ShotClassificationSource.YOLO11_POSE_TEMPORAL,
        direction=ShotDirection.DOWN_THE_LINE, direction_confidence=0.90, direction_source="GEOMETRY_DERIVED",
        landing_court_position_m=(8.5, 4.0), landing_zone=CourtZone3x3.DEEP_RIGHT
    )

    stats = ShotStatisticsAnalyzer.compute_player_shot_stats([s1, s2], player_id=1)
    assert stats["shot_type_counts"]["total"] == 2
    assert stats["shot_type_counts"]["forehand"] == 1
    assert stats["shot_type_counts"]["backhand"] == 1
    assert stats["direction_counts"]["cross_court"] == 1
    assert stats["direction_counts"]["down_the_line"] == 1

    # Occupancy heatmap
    positions = [(5.0, 20.0), (5.2, 20.1), (5.5, 20.2)]
    heatmap = ShotStatisticsAnalyzer.compute_court_occupancy_heatmap(positions, grid_x_bins=5, grid_y_bins=5)
    assert len(heatmap) > 0
    assert sum(bin_["count"] for bin_ in heatmap) == 3

def test_match_analytics_aggregator_schema():
    """Tests master match analytics JSON serialization schema."""
    st = MatchState(match_id="match_test", server_id=2, receiver_id=1)
    s1 = ShotEventEvidence(
        shot_id=1, match_event_id=1, frame_index=23, timestamp_s=0.77, player_id=2,
        shot_type=ShotType.SERVE, shot_confidence=0.95, classification_source=ShotClassificationSource.EVENT_PASSTHROUGH,
        direction=ShotDirection.CROSS_COURT, direction_confidence=0.9, direction_source="GEOMETRY_DERIVED"
    )
    r1 = RallySegment(
        rally_id=1, point_id=1, start_frame=23, end_frame=81, start_time_s=0.77, end_time_s=2.70, duration_s=1.93,
        server_id=2, receiver_id=1, serve_attempt=1, total_strokes_including_serve=1, rally_hits_excluding_serve=0,
        shots_p1=0, shots_p2=1
    )
    
    match_json = MatchAnalyticsAggregator.build_match_analytics(
        match_state=st,
        shots=[s1],
        rallies=[r1],
        line_calls=[],
        p1_positions=[(5.0, 20.0)],
        p2_positions=[(5.0, 4.0)],
        p1_distance_m=10.5,
        p2_distance_m=12.3,
        p1_speed_kmh=8.4,
        p2_speed_kmh=9.1
    )
    assert match_json["match_id"] == "match_test"
    assert "players" in match_json
    assert "player_1" in match_json["players"]
    assert "player_2" in match_json["players"]
    assert "serve_analytics" in match_json
    assert match_json["overview"]["total_shots_recorded"] == 1
