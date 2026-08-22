import math
import numpy as np
import pytest

from src.line_calling.line_geometry import CourtLineGeometry, CourtLineType, ServiceBoxType
from src.line_calling.line_call_engine import TennisLineCallEngine, LineCallDecision, LineCallContext
from src.line_calling.contact_refinement import BounceContactRefiner
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

def test_court_line_geometry_dimensions():
    """Verify canonical singles and service court geometry dimensions against ITF standards."""
    assert CourtLineGeometry.SINGLES_LEFT_X == 1.37
    assert CourtLineGeometry.SINGLES_RIGHT_X == 9.60
    assert CourtLineGeometry.FAR_BASELINE_Y == 0.00
    assert CourtLineGeometry.NEAR_BASELINE_Y == 23.77
    assert CourtLineGeometry.NET_Y == 11.885
    assert CourtLineGeometry.BALL_RADIUS_CM == 3.35

def test_signed_distance_inside_court():
    """Points inside court produce positive signed distances."""
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=5.485, court_y_m=11.885)
    assert res.is_center_inside is True
    assert res.is_footprint_inside_or_touching is True
    assert res.signed_center_distance_cm > 100.0  # Center is >1m from sidelines

def test_signed_distance_outside_court():
    """Points outside singles court produce negative signed distances."""
    # Point in right doubles alley (X=10.0m)
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=10.00, court_y_m=15.00)
    assert res.is_center_inside is False
    assert res.nearest_line == CourtLineType.RIGHT_SIDELINE
    assert res.signed_center_distance_cm == pytest.approx(-40.0, abs=0.1)

def test_ball_footprint_line_touching_rule():
    """ITF Rule 12: A ball whose footprint touches the line is legally IN."""
    # Ball center 2.0 cm outside right singles sideline (X=9.62m)
    # Ball radius = 3.35 cm -> Edge margin = -2.0 + 3.35 = +1.35 cm
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=9.62, court_y_m=15.00)
    assert res.is_center_inside is False
    assert res.is_footprint_inside_or_touching is True
    assert res.ball_edge_margin_cm == pytest.approx(1.35, abs=0.1)

def test_service_box_near_deuce_evaluation():
    """Test service box boundaries for Near Deuce serve."""
    # Inside Near Deuce box
    res_in = CourtLineGeometry.evaluate_service_box_boundary(
        court_x_m=3.00, court_y_m=15.00, target_box=ServiceBoxType.NEAR_DEUCE
    )
    assert res_in.is_center_inside is True

    # Fault: Long past service line (Y=19.0m, service line is Y=18.285m)
    res_long = CourtLineGeometry.evaluate_service_box_boundary(
        court_x_m=3.00, court_y_m=19.00, target_box=ServiceBoxType.NEAR_DEUCE
    )
    assert res_long.is_center_inside is False
    assert res_long.nearest_line == CourtLineType.NEAR_SERVICE_LINE

def test_spatial_uncertainty_tiers():
    """Verify calibrated near, mid, and far court spatial uncertainty values."""
    tier_near, unc_near = CourtLineGeometry.get_spatial_uncertainty_tier(15.0)
    assert tier_near == "NEAR"
    assert unc_near == 0.008  # 0.8 cm

    tier_mid, unc_mid = CourtLineGeometry.get_spatial_uncertainty_tier(8.0)
    assert tier_mid == "MID"
    assert unc_mid == 0.025   # 2.5 cm

    tier_far, unc_far = CourtLineGeometry.get_spatial_uncertainty_tier(2.0)
    assert tier_far == "FAR"
    assert unc_far == 0.350   # 35.0 cm

def test_predicted_state_mandatory_abstention():
    """Any bounce contact with PREDICTED tracker state triggers mandatory REVIEW_REQUIRED."""
    engine = TennisLineCallEngine()
    
    # Mock trajectory where contact frame is PREDICTED
    pts = [
        TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=1000.0, y_px=700.0, state=BallState.DETECTED, confidence=0.8),
        TemporalBallPoint(frame_index=1, timestamp_seconds=0.033, x_px=1010.0, y_px=710.0, state=BallState.PREDICTED, confidence=0.3),
        TemporalBallPoint(frame_index=2, timestamp_seconds=0.067, x_px=1020.0, y_px=720.0, state=BallState.PREDICTED, confidence=0.2)
    ]
    H = np.eye(3, dtype=np.float32)

    call = engine.evaluate_bounce(
        event_id=1,
        bounce_frame=1,
        ball_trajectory=pts,
        homography_matrix=H,
        context=LineCallContext.RALLY
    )

    assert call.decision == LineCallDecision.REVIEW_REQUIRED
    assert "PREDICTED" in call.reason
    assert call.tracker_state == "PREDICTED"

def test_far_court_ambiguity_abstention():
    """A marginal bounce in far court with high perspective uncertainty must abstain safely."""
    # Near top baseline: X=2.00, Y=0.20 (Center is 20cm inside top baseline)
    # But far-court uncertainty is ±35cm -> Safety margin is 1.5 * 35 = 52.5cm
    # Since margin (23.35cm) < 52.5cm, line intersects uncertainty envelope -> REVIEW_REQUIRED
    engine = TennisLineCallEngine()
    pts = [
        TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8),
        TemporalBallPoint(frame_index=1, timestamp_seconds=0.033, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8),
        TemporalBallPoint(frame_index=2, timestamp_seconds=0.067, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8)
    ]
    
    # Simple homography mapping (500, 305) to (2.0, 0.20)
    H = np.array([
        [0.004, 0.0, 0.0],
        [0.0, 0.0006557, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    call = engine.evaluate_bounce(
        event_id=1,
        bounce_frame=1,
        ball_trajectory=pts,
        homography_matrix=H,
        context=LineCallContext.RALLY
    )

    assert call.decision == LineCallDecision.REVIEW_REQUIRED
    assert call.spatial_tier == "FAR"

def test_line_call_evidence_serialization():
    """Verify evidence schema serialization to dict."""
    engine = TennisLineCallEngine()
    pts = [
        TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=960.0, y_px=668.0, state=BallState.DETECTED, confidence=0.9),
        TemporalBallPoint(frame_index=1, timestamp_seconds=0.033, x_px=960.0, y_px=668.0, state=BallState.DETECTED, confidence=0.9),
        TemporalBallPoint(frame_index=2, timestamp_seconds=0.067, x_px=960.0, y_px=668.0, state=BallState.DETECTED, confidence=0.9)
    ]
    H = np.eye(3, dtype=np.float32)
    call = engine.evaluate_bounce(event_id=1, bounce_frame=1, ball_trajectory=pts, homography_matrix=H)
    d = call.to_dict()
    assert "decision" in d
    assert "nearest_line" in d
    assert "position_uncertainty_cm" in d
    assert "reason" in d
