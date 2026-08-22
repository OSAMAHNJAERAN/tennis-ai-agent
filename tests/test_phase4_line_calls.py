import math
import numpy as np
import pytest

from src.line_calling.line_geometry import (
    CourtLineGeometry,
    CourtLineType,
    ServiceBoxType,
    ContactPatchModelType,
    CourtLineStrip
)
from src.line_calling.line_call_engine import TennisLineCallEngine, LineCallDecision, LineCallContext
from src.line_calling.contact_refinement import BounceContactRefiner
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

def test_court_line_geometry_dimensions():
    """Verify canonical singles and service court geometry dimensions against ITF 2026 standards."""
    assert CourtLineGeometry.SINGLES_LEFT_X == 1.37
    assert CourtLineGeometry.SINGLES_RIGHT_X == 9.60
    assert CourtLineGeometry.FAR_BASELINE_Y == 0.00
    assert CourtLineGeometry.NEAR_BASELINE_Y == 23.77
    assert CourtLineGeometry.NET_Y == 11.885
    assert CourtLineGeometry.BALL_RADIUS_CM == 3.35
    assert CourtLineGeometry.DEFAULT_CONTACT_PATCH_RADIUS_CM == 1.25

def test_court_line_strips():
    """Verify explicit 2D line strips and outside-of-line measurement convention."""
    # Left sideline: outer=1.37, width=0.05 -> inner=1.42
    strip_left = CourtLineGeometry.get_line_strip(CourtLineType.LEFT_SIDELINE)
    assert strip_left.outer_edge_m == 1.37
    assert strip_left.inner_edge_m == 1.42
    assert strip_left.width_m == 0.05
    assert strip_left.polygon_bounds_m == (1.37, 1.42, 0.00, 23.77)

    # Right sideline: outer=9.60, width=0.05 -> inner=9.55
    strip_right = CourtLineGeometry.get_line_strip(CourtLineType.RIGHT_SIDELINE)
    assert strip_right.outer_edge_m == 9.60
    assert strip_right.inner_edge_m == 9.55
    assert strip_right.polygon_bounds_m == (9.55, 9.60, 0.00, 23.77)

    # Near baseline: outer=23.77, width=0.10 -> inner=23.67
    strip_near_base = CourtLineGeometry.get_line_strip(CourtLineType.NEAR_BASELINE)
    assert strip_near_base.outer_edge_m == 23.77
    assert strip_near_base.inner_edge_m == 23.67
    assert strip_near_base.width_m == 0.10

    # Center service line: centered at 5.485, width=0.05 -> [5.460, 5.510]
    strip_center = CourtLineGeometry.get_line_strip(CourtLineType.CENTER_SERVICE_LINE)
    assert strip_center.centerline_m == 5.485
    assert strip_center.width_m == 0.05
    assert strip_center.polygon_bounds_m[0] == pytest.approx(5.460, abs=1e-4)
    assert strip_center.polygon_bounds_m[1] == pytest.approx(5.510, abs=1e-4)

def test_signed_distance_inside_court():
    """Points inside court produce positive signed distances."""
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=5.485, court_y_m=11.885)
    assert res.is_center_inside is True
    assert res.is_contact_inside_or_touching is True
    assert res.signed_center_distance_cm > 100.0

def test_signed_distance_outside_court():
    """Points outside singles court produce negative signed distances."""
    # Point in right doubles alley (X=10.0m)
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=10.00, court_y_m=15.00)
    assert res.is_center_inside is False
    assert res.nearest_line == CourtLineType.RIGHT_SIDELINE
    assert res.signed_center_distance_cm == pytest.approx(-40.0, abs=0.1)

def test_painted_line_touching_rule():
    """ITF Rule 12: A ball landing on the painted line strip is legally IN."""
    # Point on right sideline strip (X=9.58m, strip is [9.55, 9.60])
    res = CourtLineGeometry.evaluate_singles_rally_boundary(court_x_m=9.58, court_y_m=15.00)
    assert res.is_center_inside is True
    assert res.is_center_on_painted_line is True
    assert res.is_contact_inside_or_touching is True
    assert res.signed_center_distance_cm == pytest.approx(2.0, abs=0.1)

def test_contact_patch_models():
    """Verify different contact patch models on a ball center 2.0 cm outside line (X=9.62m)."""
    # 1. Point contact (r_c = 0.0) -> edge margin = -2.0 cm
    res_pt = CourtLineGeometry.evaluate_singles_rally_boundary(
        court_x_m=9.62, court_y_m=15.00, contact_model=ContactPatchModelType.POINT_CONTACT
    )
    assert res_pt.ball_edge_margin_cm == pytest.approx(-2.0, abs=0.01)

    # 2. Empirical patch (r_c = 1.25 cm) -> edge margin = -2.0 + 1.25 = -0.75 cm
    res_emp = CourtLineGeometry.evaluate_singles_rally_boundary(
        court_x_m=9.62, court_y_m=15.00, contact_model=ContactPatchModelType.EMPIRICAL_PATCH
    )
    assert res_emp.ball_edge_margin_cm == pytest.approx(-0.75, abs=0.01)

def test_piecewise_impact_refinement():
    """Verify Piecewise Trajectory Intersection computes contact change-point."""
    refiner = BounceContactRefiner(window_radius=3, fps=30.0)
    
    # Simulate V-shape trajectory with bounce at frame 10
    pts = []
    for f in range(20):
        # Incoming: y drops from 700 to 750; Outgoing: y rises from 750 to 700
        y = 700.0 + (f * 5.0) if f <= 10 else 750.0 - ((f - 10) * 5.0)
        x = 500.0 + (f * 2.0)
        pts.append(TemporalBallPoint(
            frame_index=f, timestamp_seconds=f/30.0, x_px=x, y_px=y,
            state=BallState.DETECTED, confidence=0.9
        ))
        
    refined = refiner.refine_bounce_contact(10, pts)
    assert refined.refinement_method == "PIECEWISE_IMPACT_INTERSECTION"
    assert refined.sub_frame_time == pytest.approx(10.0, abs=0.5)

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
    engine = TennisLineCallEngine()
    pts = [
        TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8),
        TemporalBallPoint(frame_index=1, timestamp_seconds=0.033, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8),
        TemporalBallPoint(frame_index=2, timestamp_seconds=0.067, x_px=500.0, y_px=305.0, state=BallState.DETECTED, confidence=0.8)
    ]
    
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
    assert "line_strip_info" in d
