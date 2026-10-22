"""Rules/math correctness checks; synthetic inputs do not qualify vision or calibration."""
import copy
import math
from collections import defaultdict

import pytest

from src.prediction.tennis import (ServePointPrior, TennisWinProbability, _RuleProbabilities,
                                   game_probability, replay_predictions, tiebreak_probability)
from src.scoring.match_state import (BallPlayState, MatchFormat, MatchState, PointState,
                                     ServerCourtEnd, ServiceSide, SetFormat)
from src.scoring.point_ledger import PointLedger, PointRecord
from src.scoring.point_outcome_resolver import PointOutcomeType
from src.scoring.scoring_engine import TennisScoringEngine
from src.scoring.scoring_rules import TennisScoringRules
from src.line_calling.line_call_engine import LineCallDecision
from tests.test_phase5_scoring import make_fake_line_call


def call(event, decision):
    return make_fake_line_call(event, event, decision)


def test_second_serve_starts_after_fault_and_dead_ball_hit_is_ignored():
    engine = TennisScoringEngine()
    engine.process_match_event(1, "SERVE_CONTACT", 0, player_id=1)
    engine.process_match_event(2, "BOUNCE", 1, line_call=call(2, LineCallDecision.SERVE_FAULT))
    state, outcome = engine.process_match_event(3, "PLAYER_2_HIT", 2, player_id=2)
    assert outcome.outcome_type == PointOutcomeType.DEAD_BALL_IGNORED
    assert state.serve_attempt == 2 and state.points_p2 == 0
    state, outcome = engine.process_match_event(4, "SERVE_CONTACT", 3, player_id=1)
    assert outcome.outcome_type == PointOutcomeType.SERVE_IN_PLAY
    state, outcome = engine.process_match_event(5, "BOUNCE", 4,
        line_call=call(5, LineCallDecision.SERVE_FAULT), point_id="rally-one")
    assert outcome.outcome_type == PointOutcomeType.DOUBLE_FAULT
    assert state.points_p2 == 1 and state.serve_attempt == 1
    state, outcome = engine.process_match_event(6, "SERVE_CONTACT", 5, player_id=1)
    assert outcome.outcome_type == PointOutcomeType.SERVE_IN_PLAY


def test_second_serve_let_replays_second_attempt():
    engine = TennisScoringEngine()
    engine.process_match_event(1, "SERVE_CONTACT", 0)
    engine.process_match_event(2, "BOUNCE", 1, line_call=call(2, LineCallDecision.SERVE_FAULT))
    engine.process_match_event(3, "SERVE_CONTACT", 2)
    state, _ = engine.process_match_event(4, "SERVICE_LET", 3)
    assert state.serve_attempt == 2 and state.ball_state == BallPlayState.LET_REPLAY
    engine.process_match_event(5, "SERVE_CONTACT", 4)
    state, outcome = engine.process_match_event(6, "BOUNCE", 5, line_call=call(6, LineCallDecision.SERVE_FAULT))
    assert outcome.outcome_type == PointOutcomeType.DOUBLE_FAULT and state.points_p2 == 1


def test_review_freeze_persists_and_correction_replays_dependent_events():
    engine = TennisScoringEngine()
    engine.process_match_event(1, "SERVE_CONTACT", 0)
    engine.process_match_event(2, "BOUNCE", 1, line_call=call(2, LineCallDecision.REVIEW_REQUIRED))
    engine.process_match_event(3, "PLAYER_2_HIT", 2, player_id=2)
    frozen, _ = engine.process_match_event(4, "BOUNCE", 3, line_call=call(4, LineCallDecision.OUT))
    assert frozen.points_p1 == frozen.points_p2 == 0 and not frozen.score_confirmed
    state = engine.correct_event(2, reason="Independent serve review", expected_revision=0,
                                line_call=call(2, LineCallDecision.SERVE_IN))
    assert state.points_p1 == 1 and state.score_confirmed
    assert engine.revision == 1 and engine.corrections[0]["source"] == "human_review"
    assert engine.corrections[0]["previous_history"][3]["new_state"]["score_confirmed"] is False
    with pytest.raises(ValueError, match="revision"):
        engine.correct_event(2, reason="stale update", expected_revision=0)


def test_consequential_gap_freezes_later_match_evidence():
    engine = TennisScoringEngine()
    engine.process_match_event(1, "CONSEQUENTIAL_GAP", 0)
    state, outcome = engine.process_match_event(2, "SERVE_CONTACT", 1)
    assert outcome.outcome_type == PointOutcomeType.POINT_REVIEW_REQUIRED
    assert state.point_state == PointState.POINT_REVIEW_PENDING and not state.score_confirmed


def test_stable_point_id_prevents_different_events_awarding_twice():
    engine = TennisScoringEngine()
    for event, time in ((1, 0), (3, 2)):
        engine.process_match_event(event, "SERVE_CONTACT", time)
        engine.process_match_event(event + 1, "BOUNCE", time + 1,
                                   line_call=call(event + 1, LineCallDecision.SERVE_FAULT), point_id="same-point")
    before = engine.get_current_state().to_dict()
    state, outcome = engine.process_match_event(5, "BOUNCE", 4,
        line_call=call(5, LineCallDecision.SERVE_FAULT), point_id="same-point")
    assert state.to_dict() == before
    assert outcome.outcome_type == PointOutcomeType.DEAD_BALL_IGNORED


@pytest.mark.parametrize("timestamp", [-1, math.nan, math.inf])
def test_scoring_rejects_invalid_timing(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        TennisScoringEngine().process_match_event(1, "SERVE_CONTACT", timestamp)


def test_scoring_rejects_out_of_order_events_and_wrong_serve():
    engine = TennisScoringEngine()
    engine.process_match_event(1, "SERVE_CONTACT", 1)
    with pytest.raises(ValueError, match="source-time"):
        engine.process_match_event(2, "BOUNCE", 0)
    wrong = TennisScoringEngine()
    state, outcome = wrong.process_match_event(1, "SERVE_CONTACT", 0, player_id=2)
    assert outcome.outcome_type == PointOutcomeType.POINT_REVIEW_REQUIRED and not state.score_confirmed
    with pytest.raises(ValueError, match="current server"):
        TennisScoringRules.record_fault(MatchState(), 2)
    with pytest.raises(ValueError, match="participant"):
        TennisScoringRules.award_point(MatchState(), 9)


@pytest.mark.parametrize("first", [1, 2])
@pytest.mark.parametrize("loser_points", range(6))
def test_next_set_server_is_opponent_of_first_tiebreak_server(first, loser_points):
    state = MatchState(games_p1=6, games_p2=6, tie_break_active=True,
                       initial_server_id=first, server_id=first, receiver_id=3-first)
    # Different lengths exercise different servers on the final point.
    winners = [1, 2] * loser_points + [1] * (7-loser_points)
    for winner in winners:
        state = TennisScoringRules.award_point(state, winner)
    assert state.current_set == 2 and state.server_id == 3-first
    assert state.service_side == ServiceSide.DEUCE and not state.tie_break_active


def test_tiebreak_server_side_and_end_order():
    state = MatchState(games_p1=6, games_p2=6, tie_break_active=True)
    expected_servers = [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1]
    for index, server in enumerate(expected_servers):
        assert state.server_id == server
        assert state.service_side == (ServiceSide.DEUCE if index % 2 == 0 else ServiceSide.AD)
        assert state.player_1_court_end == (ServerCourtEnd.NEAR_COURT if (index // 6) % 2 == 0 else ServerCourtEnd.FAR_COURT)
        state = TennisScoringRules.award_point(state, 1 if index % 2 == 0 else 2)


@pytest.mark.parametrize("match_format,final_set,sets", [(MatchFormat.BEST_OF_3, 3, 1), (MatchFormat.BEST_OF_5, 5, 2)])
def test_deciding_set_ten_point_tiebreak(match_format, final_set, sets):
    state = MatchState(format=match_format, current_set=final_set, sets_won_p1=sets, sets_won_p2=sets,
                       games_p1=6, games_p2=6, tie_break_active=True, deciding_set_tie_break_points_to_win=10)
    for _ in range(7):
        state = TennisScoringRules.award_point(state, 1)
    assert not state.match_complete
    for _ in range(3):
        state = TennisScoringRules.award_point(state, 1)
    assert state.match_complete and state.winner_id == 1
    assert state.completed_sets[-1].tie_break_p1 == 10


def test_advantage_set_extended_score_and_end_changes():
    state = MatchState(set_format=SetFormat.ADVANTAGE_SET, games_p1=6, games_p2=6)
    for winner in [1, 2, 1, 1]:
        for _ in range(4):
            state = TennisScoringRules.award_point(state, winner)
    assert state.completed_sets[-1].games_p1 == 9 and state.completed_sets[-1].games_p2 == 7
    assert state.player_1_court_end == ServerCourtEnd.NEAR_COURT


def test_midmatch_initial_state_is_preserved_and_unknown_initial_state_has_no_match_winner():
    state = MatchState(current_set=2, sets_won_p1=1, games_p1=5, games_p2=4,
                       points_p1=3, server_id=2, receiver_id=1, initial_server_id=2,
                       initial_state_confirmed=False, score_confirmed=False)
    original = copy.deepcopy(state)
    state = TennisScoringRules.award_point(state, 1)
    assert state.match_complete and state.winner_id is None and state.match_status == "PARTIAL"
    assert original.points_p1 == 3 and not original.match_complete


@pytest.mark.parametrize("p", [0, .1, .3, .5, .67, .9, 1])
def test_game_probability_matches_independent_closed_expression(p):
    q = 1-p
    expected = p**4 * (1 + 4*q + 10*q*q) + 20*p**3*q**3 * p*p / (p*p + q*q)
    assert game_probability(p) == pytest.approx(expected, abs=1e-14)
    assert game_probability(p, 50, 50) == pytest.approx(p*p/(p*p+q*q))
    assert game_probability(p, 50, 49) == pytest.approx(p+q*p*p/(p*p+q*q))


def enumerate_tiebreak(p1, p2, first, target, a=0, b=0, steps=100):
    """Independent forward probability mass propagation; residual gives bounds."""
    states, won = {(a, b): 1.0}, 0.0
    for _ in range(steps):
        next_states = defaultdict(float)
        for (x, y), mass in states.items():
            total = x+y
            schedule = [first] + [3-first, 3-first, first, first] * ((total+4)//4+1)
            p = p1 if schedule[total] == 1 else 1-p2
            for nx, ny, weight in ((x+1, y, p), (x, y+1, 1-p)):
                if max(nx, ny) >= target and abs(nx-ny) >= 2:
                    won += mass*weight if nx > ny else 0
                else:
                    next_states[nx, ny] += mass*weight
        states = next_states
    return won, won + sum(states.values())


@pytest.mark.parametrize("first,target,a,b", [(1,7,0,0), (2,7,6,6), (1,7,30,29), (2,10,0,0), (2,10,9,10)])
def test_tiebreak_math_inside_forward_enumeration_bounds(first, target, a, b):
    lo, hi = enumerate_tiebreak(.65, .55, first, target, a, b)
    actual = tiebreak_probability(.65, .55, first_server=first, target=target, points_p1=a, points_p2=b)
    assert lo - 1e-13 <= actual <= hi + 1e-13
    assert hi-lo < 1e-10


@pytest.mark.parametrize("kind", [SetFormat.TIE_BREAK_SET, SetFormat.ADVANTAGE_SET])
@pytest.mark.parametrize("format", [MatchFormat.BEST_OF_3, MatchFormat.BEST_OF_5])
@pytest.mark.parametrize("server", [1, 2])
def test_match_probability_obeys_point_transition_identity(kind, format, server):
    state = MatchState(format=format, set_format=kind, games_p1=5, games_p2=5,
                       points_p1=3, points_p2=2, server_id=server, receiver_id=3-server)
    p1, p2 = .62, .57
    actual = _RuleProbabilities(state, p1, p2).current()["match"]
    win_state = TennisScoringRules.award_point(state, 1)
    lose_state = TennisScoringRules.award_point(state, 2)
    q = p1 if server == 1 else 1-p2
    expected = q * _RuleProbabilities(win_state, p1, p2).current()["match"] + (1-q) * _RuleProbabilities(lose_state, p1, p2).current()["match"]
    assert actual == pytest.approx(expected, abs=1e-13)


def test_prior_shrinkage_idempotency_future_rejection_and_session_isolation():
    engine = TennisWinProbability()
    prior = engine.snapshot(MatchState())
    assert prior["status"] == "prior_only" and prior["probabilities"]["match"]["player_1"] == pytest.approx(.5)
    data = dict(point_id="p1", server_id=1, winner_id=1, timestamp_seconds=10)
    engine.observe_point(**data)
    engine.observe_point(**data)
    current = engine.snapshot(TennisScoringRules.award_point(MatchState(), 1), timestamp_seconds=10)
    assert current["sample_count"] == 1 and current["serve_estimates"][0]["probability"] == pytest.approx(3/5)
    assert current["status"] == "uncalibrated" and current["calibration_status"] == "not_calibrated"
    assert TennisWinProbability().snapshot(MatchState())["sample_count"] == 0
    with pytest.raises(ValueError, match="future"):
        engine.snapshot(MatchState(), timestamp_seconds=9)
    with pytest.raises(ValueError, match="different outcome"):
        engine.observe_point(**{**data, "winner_id": 2})
    with pytest.raises(ValueError, match="source-time"):
        engine.observe_point(**{**data, "point_id": "earlier", "timestamp_seconds": 1})
    with pytest.raises(ValueError):
        ServePointPrior(alpha=0)


def test_predictions_are_prefix_causal_and_recompute_after_correction():
    points = [{"point_id": f"point-{i}", "winner": 1 if i%2 else 2, "timestamp_seconds": i+1} for i in range(10)]
    full = replay_predictions(MatchState(), points, revision=1)
    prefix = replay_predictions(MatchState(), points[:4], revision=1)
    assert full["history"][:4] == prefix["history"]
    corrected = copy.deepcopy(points)
    corrected[4]["winner"] = 3-corrected[4]["winner"]
    revised = replay_predictions(MatchState(), corrected, revision=1)
    assert revised["history"][:4] == full["history"][:4]
    assert revised["history"][4] != full["history"][4]


def test_ledger_gap_freezes_and_human_correction_rebuilds_score_and_predictions():
    ledger = PointLedger(MatchState())
    ledger.append(PointRecord("one", 1, 1, "confirmed", evidence_ids=("bounce-one",)))
    ledger.append(PointRecord("missing", 2))
    ledger.append(PointRecord("three", 3, 2, "confirmed"))
    frozen = ledger.snapshot()
    assert frozen["state"]["player_1"]["points_raw"] == 1 and frozen["state"]["player_2"]["points_raw"] == 0
    assert frozen["current"]["status"] == "unavailable" and frozen["current"]["sample_count"] == 1
    revised = ledger.correct("missing", winner=2, reason="Reviewed original footage", expected_revision=3,
                             evidence_ids=("human-reviewed-frame",), loss_reason="double_fault")
    assert revised["state"]["player_1"]["points_raw"] == 1 and revised["state"]["player_2"]["points_raw"] == 2
    assert revised["current"]["sample_count"] == 3 and revised["current"]["revision"] == 4
    assert ledger.snapshot()["audit"][-1]["before"]["status"] == "unknown"
    assert revised["history"][1]["provenance"] == "human_review"
    assert not ledger.append(ledger.points[0])
    with pytest.raises(ValueError, match="conflicting"):
        ledger.append(PointRecord("one", 1, 2, "confirmed"))


def test_unknown_initial_score_and_match_completion_are_distinct_from_forecast():
    ledger = PointLedger()
    assert ledger.snapshot()["current"]["status"] == "unavailable"
    ledger.confirm_initial_state(MatchState(), source="user confirmed scoreboard", reason="clip start", expected_revision=0)
    assert ledger.snapshot()["current"]["status"] == "prior_only"
    for winner in [1]*48:
        ledger.append(PointRecord(f"p-{ledger.revision}", ledger.revision, winner, "confirmed", "manual"))
    final = ledger.snapshot()["current"]
    assert final["status"] == "confirmed_result" and final["winner_id"] == 1 and final["probabilities"] is None
    assert TennisWinProbability().snapshot(MatchState(match_status="RETIRED"))["status"] == "unavailable"
    assert TennisWinProbability().snapshot(MatchState(match_status="SUSPENDED"))["status"] == "unavailable"
