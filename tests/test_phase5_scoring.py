import pytest
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from src.scoring.match_state import (
    MatchState,
    MatchFormat,
    SetFormat,
    ServiceSide,
    PointState,
    BallPlayState,
    ServerCourtEnd
)
from src.scoring.scoring_rules import TennisScoringRules
from src.scoring.point_outcome_resolver import PointOutcomeResolver, PointOutcomeType
from src.scoring.scoring_engine import TennisScoringEngine
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision
from src.line_calling.line_geometry import CourtLineType, ServiceBoxType

def test_standard_game_scoring_love_to_game():
    """Tests clean 4-0 game progression: Love -> 15 -> 30 -> 40 -> Game."""
    st = MatchState(server_id=1, receiver_id=2)
    assert st.get_points_display_p1() == "0"
    assert st.get_points_display_p2() == "0"
    assert st.service_side == ServiceSide.DEUCE

    # Point 1 (P1 wins)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.points_p1 == 1
    assert st.get_points_display_p1() == "15"
    assert st.service_side == ServiceSide.AD

    # Point 2 (P1 wins)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.points_p1 == 2
    assert st.get_points_display_p1() == "30"
    assert st.service_side == ServiceSide.DEUCE

    # Point 3 (P1 wins)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.points_p1 == 3
    assert st.get_points_display_p1() == "40"
    assert st.service_side == ServiceSide.AD

    # Point 4 (P1 wins -> Game won)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.games_p1 == 1
    assert st.points_p1 == 0
    assert st.points_p2 == 0
    # Server rotates to Player 2
    assert st.server_id == 2
    assert st.receiver_id == 1
    assert st.service_side == ServiceSide.DEUCE

def test_deuce_and_advantage_cycles():
    """Tests 40-40 Deuce, Advantage P1, back to Deuce, Advantage P2, Game P2."""
    st = MatchState(points_p1=3, points_p2=3, server_id=1, receiver_id=2) # 40-40 Deuce
    assert st.get_points_display_p1() == "40"
    assert st.get_points_display_p2() == "40"

    # P1 wins point -> Advantage P1
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.points_p1 == 4 and st.points_p2 == 3
    assert st.get_points_display_p1() == "AD"
    assert st.get_points_display_p2() == "40"
    assert st.games_p1 == 0

    # P2 wins point -> Back to Deuce
    st = TennisScoringRules.award_point(st, winner_id=2)
    assert st.points_p1 == 4 and st.points_p2 == 4
    assert st.get_points_display_p1() == "40"
    assert st.get_points_display_p2() == "40"

    # P2 wins point -> Advantage P2
    st = TennisScoringRules.award_point(st, winner_id=2)
    assert st.points_p1 == 4 and st.points_p2 == 5
    assert st.get_points_display_p1() == "40"
    assert st.get_points_display_p2() == "AD"

    # P2 wins point -> Game P2 (Break of serve!)
    st = TennisScoringRules.award_point(st, winner_id=2)
    assert st.games_p2 == 1
    assert st.points_p1 == 0 and st.points_p2 == 0
    assert st.server_id == 2  # Server rotates

def test_first_fault_vs_double_fault_mechanics():
    """Tests first fault does not award point, and second consecutive fault awards point to receiver."""
    st = MatchState(server_id=1, receiver_id=2)
    assert st.serve_attempt == 1

    # First fault
    st, is_df = TennisScoringRules.record_fault(st, server_id=1, fault_reason="Long past service line")
    assert not is_df
    assert st.serve_attempt == 2
    assert st.ball_state == BallPlayState.DEAD
    assert st.points_p1 == 0 and st.points_p2 == 0 # No point awarded

    # Second fault (Double Fault)
    st, is_df = TennisScoringRules.record_fault(st, server_id=1, fault_reason="Net fault")
    assert is_df
    assert st.serve_attempt == 1 # Resets after point completion
    assert st.points_p2 == 1     # Receiver Player 2 wins point
    assert st.points_p1 == 0
    assert st.get_points_display_p2() == "15"

def test_service_let_replay():
    """Tests service let allows replay with no score alteration."""
    st = MatchState(server_id=1, receiver_id=2, serve_attempt=1)
    st = TennisScoringRules.record_service_let(st)
    assert st.serve_attempt == 1
    assert st.ball_state == BallPlayState.LET_REPLAY
    assert st.points_p1 == 0 and st.points_p2 == 0

def make_fake_line_call(
    event_id: int,
    bounce_frame: int,
    decision: LineCallDecision,
    decision_context: str = "SERVE",
    nearest_line: CourtLineType = CourtLineType.NEAR_SERVICE_LINE,
    bounce_position_px: Tuple[float, float] = (500.0, 500.0),
    bounce_position_m: Tuple[float, float] = (3.0, 15.0),
    tracker_state: str = "DETECTED",
    center_signed_distance_cm: float = 0.0,
    contact_patch_model: str = "EMPIRICAL_PATCH",
    contact_patch_radius_cm: float = 1.25,
    ball_edge_margin_cm: float = 0.0,
    position_uncertainty_cm: float = 1.48,
    spatial_tier: str = "NEAR",
    confidence: float = 0.98,
    reason: str = "Test line call",
    refinement_method: str = "PIECEWISE_IMPACT_INTERSECTION"
) -> LineCallEvidence:
    return LineCallEvidence(
        event_id=event_id,
        bounce_frame=bounce_frame,
        decision=decision,
        decision_context=decision_context,
        nearest_line=nearest_line,
        bounce_position_px=bounce_position_px,
        bounce_position_m=bounce_position_m,
        tracker_state=tracker_state,
        center_signed_distance_cm=center_signed_distance_cm,
        contact_patch_model=contact_patch_model,
        contact_patch_radius_cm=contact_patch_radius_cm,
        ball_edge_margin_cm=ball_edge_margin_cm,
        position_uncertainty_cm=position_uncertainty_cm,
        spatial_tier=spatial_tier,
        confidence=confidence,
        reason=reason,
        refinement_method=refinement_method
    )

def test_frame_81_84_dead_ball_regression():
    """
    CRITICAL REGRESSION TEST:
    Frame 81 Serve Fault -> Ball DEAD -> Frame 84 dead-ball return shot ignored -> NO POINT AWARDED.
    """
    engine = TennisScoringEngine(player_1_id=1, player_2_id=2, initial_server_id=2)

    # Frame 23: Player 2 Serve Contact
    st1, out1 = engine.process_match_event(
        event_id=1,
        event_type="SERVE_CONTACT",
        timestamp_seconds=0.77,
        player_id=2
    )
    assert st1.ball_state == BallPlayState.LIVE or st1.point_state == PointState.RALLY_LIVE

    # Frame 81: Ball Bounces Out (SERVE_FAULT)
    fake_line_call = make_fake_line_call(
        event_id=2,
        bounce_frame=81,
        decision=LineCallDecision.SERVE_FAULT,
        decision_context="SERVE",
        nearest_line=CourtLineType.NEAR_SERVICE_LINE,
        bounce_position_px=(718.16, 730.58),
        bounce_position_m=(3.12, 20.43),
        ball_edge_margin_cm=-212.71,
        position_uncertainty_cm=1.48,
        confidence=0.98,
        reason="Ball landed 212.7 cm past near service line."
    )
    st2, out2 = engine.process_match_event(
        event_id=2,
        event_type="BOUNCE",
        timestamp_seconds=2.70,
        line_call=fake_line_call
    )
    assert out2.outcome_type == PointOutcomeType.FIRST_SERVE_FAULT
    assert st2.serve_attempt == 2
    assert st2.ball_state == BallPlayState.DEAD
    assert st2.points_p1 == 0 and st2.points_p2 == 0 # NO POINT AWARDED!

    # Frame 84: Player 1 Hits Dead Ball
    st3, out3 = engine.process_match_event(
        event_id=3,
        event_type="PLAYER_1_HIT",
        timestamp_seconds=2.80,
        player_id=1
    )
    assert out3.outcome_type == PointOutcomeType.DEAD_BALL_IGNORED
    assert st3.points_p1 == 0 and st3.points_p2 == 0 # Score unchanged!
    assert st3.serve_attempt == 2

def test_rally_out_point_resolution():
    """Tests live rally shot landing OUT awards point to non-hitter."""
    engine = TennisScoringEngine(player_1_id=1, player_2_id=2, initial_server_id=1)

    # 1. Serve Contact
    engine.process_match_event(1, "SERVE_CONTACT", 0.0, player_id=1)
    
    # 2. Serve IN
    serve_in_call = make_fake_line_call(
        event_id=2, bounce_frame=20, decision=LineCallDecision.SERVE_IN, decision_context="SERVE",
        nearest_line=CourtLineType.NEAR_SERVICE_LINE, bounce_position_px=(500, 500), bounce_position_m=(3.0, 15.0),
        tracker_state="DETECTED", ball_edge_margin_cm=100.0, position_uncertainty_cm=1.0, confidence=0.99
    )
    engine.process_match_event(2, "BOUNCE", 0.67, line_call=serve_in_call)

    # 3. Player 2 Returns Hit
    engine.process_match_event(3, "PLAYER_2_HIT", 1.0, player_id=2)

    # 4. Player 2's shot lands OUT
    out_call = make_fake_line_call(
        event_id=4, bounce_frame=50, decision=LineCallDecision.OUT, decision_context="RALLY",
        nearest_line=CourtLineType.FAR_BASELINE, bounce_position_px=(787, 249), bounce_position_m=(5.0, -2.0),
        tracker_state="DETECTED", ball_edge_margin_cm=-200.0, position_uncertainty_cm=2.0, confidence=0.95,
        reason="Landed long past baseline."
    )
    st, out = engine.process_match_event(4, "BOUNCE", 1.67, line_call=out_call)
    
    assert out.outcome_type == PointOutcomeType.POINT_WON_PLAYER_1
    assert out.winner_id == 1
    assert st.points_p1 == 1
    assert st.get_points_display_p1() == "15"

def test_review_required_mandatory_pause():
    """Tests line call with REVIEW_REQUIRED halts scoring without awarding points."""
    engine = TennisScoringEngine(player_1_id=1, player_2_id=2, initial_server_id=1)
    engine.process_match_event(1, "SERVE_CONTACT", 0.0, player_id=1)

    review_call = make_fake_line_call(
        event_id=2, bounce_frame=25, decision=LineCallDecision.REVIEW_REQUIRED, decision_context="SERVE",
        nearest_line=CourtLineType.NEAR_SERVICE_LINE, bounce_position_px=(500, 500), bounce_position_m=(3.0, 18.28),
        tracker_state="INTERPOLATED", ball_edge_margin_cm=0.2, position_uncertainty_cm=4.5, confidence=0.45,
        reason="Close call within uncertainty envelope."
    )
    st, out = engine.process_match_event(2, "BOUNCE", 0.83, line_call=review_call)

    assert out.outcome_type == PointOutcomeType.POINT_REVIEW_REQUIRED
    assert st.point_state == PointState.POINT_REVIEW_PENDING
    assert st.points_p1 == 0 and st.points_p2 == 0 # Score paused, no points awarded

def test_set_scoring_7_5_win():
    """Tests 5-5 -> 6-5 -> 7-5 set win without entering tiebreak."""
    st = MatchState(games_p1=5, games_p2=5, server_id=1, receiver_id=2)

    # P1 wins game 11 -> 6-5 (set continues!)
    for _ in range(4):
        st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.games_p1 == 6 and st.games_p2 == 5
    assert len(st.completed_sets) == 0 # Set not won yet!

    # P1 wins game 12 -> 7-5 (P1 wins set!)
    for _ in range(4):
        st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.sets_won_p1 == 1
    assert len(st.completed_sets) == 1
    assert st.completed_sets[0].games_p1 == 7
    assert st.completed_sets[0].games_p2 == 5

def test_tie_break_6_6_activation_and_extended_scoring():
    """Tests 6-6 tie-break activation, service rotation, and 12-10 extended resolution."""
    st = MatchState(games_p1=6, games_p2=5, server_id=2, receiver_id=1)
    
    # P2 wins game to make it 6-6
    for _ in range(4):
        st = TennisScoringRules.award_point(st, winner_id=2)
    
    assert st.games_p1 == 6 and st.games_p2 == 6
    assert st.tie_break_active
    assert st.tie_break_points_p1 == 0 and st.tie_break_points_p2 == 0

    # Test Tie-break point 1 (Server P1 serves 1 point from Deuce)
    initial_server = st.server_id
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.tie_break_points_p1 == 1
    # Point 2 should be served by opponent from Ad side
    assert st.server_id != initial_server
    assert st.service_side == ServiceSide.AD

    # Fast forward to 10-10
    st.tie_break_points_p1 = 10
    st.tie_break_points_p2 = 10

    # P1 wins to make it 11-10 (Tie-break continues!)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.tie_break_points_p1 == 11 and st.tie_break_points_p2 == 10
    assert len(st.completed_sets) == 0

    # P1 wins to make it 12-10 (P1 wins tie-break and set 7-6!)
    st = TennisScoringRules.award_point(st, winner_id=1)
    assert st.sets_won_p1 == 1
    assert len(st.completed_sets) == 1
    assert st.completed_sets[0].tie_break_p1 == 12
    assert st.completed_sets[0].tie_break_p2 == 10

def test_best_of_three_match_win():
    """Tests winning 2 sets concludes Best-of-3 match."""
    st = MatchState(format=MatchFormat.BEST_OF_3, sets_won_p1=1, games_p1=5, games_p2=4, server_id=1, receiver_id=2)
    
    # P1 wins game -> wins 2nd set (6-4) -> Wins Match!
    for _ in range(4):
        st = TennisScoringRules.award_point(st, winner_id=1)

    assert st.sets_won_p1 == 2
    assert st.match_complete
    assert st.winner_id == 1

def test_event_idempotency():
    """Tests duplicate event IDs are rejected and do not double-count."""
    engine = TennisScoringEngine(player_1_id=1, player_2_id=2, initial_server_id=1)

    # 1. Serve contact
    engine.process_match_event(101, "SERVE_CONTACT", 0.0, player_id=1)
    
    # 2. Serve IN
    serve_in_call = make_fake_line_call(
        event_id=102, bounce_frame=20, decision=LineCallDecision.SERVE_IN, decision_context="SERVE",
        nearest_line=CourtLineType.FAR_SERVICE_LINE, bounce_position_px=(500, 500), bounce_position_m=(3.0, 15.0),
        ball_edge_margin_cm=50.0, position_uncertainty_cm=1.0, confidence=0.99
    )
    engine.process_match_event(102, "BOUNCE", 0.67, line_call=serve_in_call)

    # 3. Player 2 Hit
    engine.process_match_event(103, "PLAYER_2_HIT", 1.0, player_id=2)

    # 4. Out Shot (Player 1 wins point)
    out_call = make_fake_line_call(
        event_id=104, bounce_frame=40, decision=LineCallDecision.OUT, decision_context="RALLY",
        nearest_line=CourtLineType.FAR_BASELINE, bounce_position_px=(0, 0), bounce_position_m=(0, 0),
        tracker_state="DETECTED", ball_edge_margin_cm=-10.0, position_uncertainty_cm=1.0, confidence=0.9
    )
    st1, _ = engine.process_match_event(104, "BOUNCE", 1.5, line_call=out_call)
    assert st1.points_p1 == 1 and st1.points_p2 == 0 # P1 awarded point

    # Send SAME event_id 104 again (Duplicate!)
    st2, out2 = engine.process_match_event(104, "BOUNCE", 1.5, line_call=out_call)
    assert st2.points_p1 == 1 # Must remain 1, NOT increment to 2!
    assert out2.outcome_type == PointOutcomeType.UNKNOWN_EVENT



def test_state_invariants():
    """Tests invalid match state configurations are flagged."""
    st_valid = MatchState()
    assert len(TennisScoringRules.validate_invariants(st_valid)) == 0

    st_invalid = MatchState(games_p1=-1, server_id=1, receiver_id=1, serve_attempt=3)
    violations = TennisScoringRules.validate_invariants(st_invalid)
    assert len(violations) >= 3
