"""Actual scoring classes and Phase6 context block without model/image imports.

Rule inputs and line-call decisions are explicit synthetic fixtures, never
claims about event detection, service-box identity or geometric accuracy.
"""
from __future__ import annotations

import ast
import builtins
import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/validation/yolo26_execution_2026-10-16/service_phase_repair_v1/baseline_scoring_engine.py"
BASELINE_SHA = "46e3345062a7e756f440df0f76152dc93c9dca95fe8c07464493121316165d9d"
FORBIDDEN = {"torch", "ultralytics", "numpy", "cv2", "PIL"}


def actual_classes(*, baseline=False):
    """Execute unmodified real definitions, omitting transitive package imports."""
    name = "actual_scoring_service_phase_baseline" if baseline else "actual_scoring_service_phase_current"
    module = ModuleType(name)
    sys.modules[name] = module
    namespace = module.__dict__
    namespace.update({key: globals()[key] for key in ("Enum", "dataclass", "field", "asdict", "copy", "json", "math",
        "Any", "Dict", "List", "Optional", "Set", "Tuple")})
    sources = [
        (ROOT / "src/line_calling/line_geometry.py", {"CourtLineType", "ServiceBoxType"}),
        (ROOT / "src/line_calling/line_call_engine.py", {"LineCallDecision", "LineCallContext", "LineCallEvidence"}),
        (ROOT / "src/scoring/match_state.py", None),
        (ROOT / "src/scoring/scoring_rules.py", None),
        (ROOT / "src/scoring/point_outcome_resolver.py", None),
        (ROOT / "src/scoring/event_history.py", None),
        (BASELINE if baseline else ROOT / "src/scoring/scoring_engine.py", None)]
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in FORBIDDEN:
            raise AssertionError("No model/image runtime permitted in service-phase tests: " + name)
        return original_import(name, *args, **kwargs)
    try:
        builtins.__import__ = guarded
        for path, selected in sources:
            if path == BASELINE:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE_SHA
            nodes = [node for node in ast.parse(path.read_text(encoding="utf-8")).body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (selected is None or node.name in selected)]
            if selected is not None:
                assert {node.name for node in nodes} == selected
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    finally:
        builtins.__import__ = original_import
    return SimpleNamespace(**namespace)


def actual_phase6_context(api, engine):
    """Execute the exact caller's pre-bounce context/target selection nodes."""
    path = ROOT / "src/pipeline/phase6_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Phase6Pipeline")
    run = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == "_run_with_frame_exports")
    loop = next(n for n in run.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "ev")
    bounce = next(n for n in loop.body if isinstance(n, ast.If) and "EventType.BOUNCE" in ast.unparse(n.test))
    nodes = []
    for node in bounce.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "evidence" for t in node.targets):
            break
        nodes.append(node)
    assert len(nodes) == 4 and "curr_st.ball_state" in ast.unparse(nodes[-1].test)
    namespace = vars(api).copy()
    namespace["self"] = SimpleNamespace(scoring_engine=engine)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["call_ctx"], namespace["tgt_box"]


def call(api, decision, ident=1, context="SERVE"):
    return api.LineCallEvidence(ident, ident, api.LineCallDecision(decision), api.LineCallContext(context),
        api.CourtLineType.NEAR_SERVICE_LINE, (500., 400.), (3., 6.), "SYNTHETIC_ONLY", 20.,
        "SYNTHETIC_ONLY", 1., 19., 0., "SYNTHETIC_ONLY", .99,
        "EXPLICIT_SYNTHETIC_RULE_INPUT_NOT_GEOMETRY_GT", "SYNTHETIC_ONLY")


def process(engine, ident, kind, **kwargs):
    return engine.process_match_event(ident, kind, ident / 10., **kwargs)


def assert_phase(api, engine, ball, point):
    assert engine.state.ball_state == api.BallPlayState(ball)
    assert engine.state.point_state == api.PointState(point)
    assert engine.resolver.ball_play_state == api.BallPlayState(ball)


def test_actual_contact_keeps_serve_started_and_native_first_bounce_requests_serve():
    api = actual_classes()
    engine = api.TennisScoringEngine()
    _, outcome = process(engine, 1, "SERVE_CONTACT", player_id=1)
    assert outcome.outcome_type == api.PointOutcomeType.SERVE_IN_PLAY
    assert_phase(api, engine, "SERVE_STARTED", "FIRST_SERVE_IN_PROGRESS")
    context, target = actual_phase6_context(api, engine)
    assert context == api.LineCallContext.SERVE and target == api.ServiceBoxType.FAR_DEUCE
    assert engine.state.points_p1 == engine.state.points_p2 == 0


@pytest.mark.parametrize("legal_decision", ["SERVE_IN", "IN"])
def test_first_fault_second_contact_in_then_rally_uses_exact_native_context(legal_decision):
    api = actual_classes()
    engine = api.TennisScoringEngine()
    process(engine, 1, "SERVE_CONTACT", player_id=1)
    contexts = [actual_phase6_context(api, engine)[0].value]
    _, fault = process(engine, 2, "BOUNCE", line_call=call(api, "SERVE_FAULT", 2))
    assert fault.outcome_type == api.PointOutcomeType.FIRST_SERVE_FAULT
    assert_phase(api, engine, "DEAD", "SECOND_SERVE_WAITING")
    assert engine.state.serve_attempt == 2 and engine.state.points_p2 == 0
    _, dead = process(engine, 3, "PLAYER_2_HIT", player_id=2)
    assert dead.outcome_type == api.PointOutcomeType.DEAD_BALL_IGNORED
    process(engine, 4, "SERVE_CONTACT", player_id=1)
    assert_phase(api, engine, "SERVE_STARTED", "SECOND_SERVE_IN_PROGRESS")
    contexts.append(actual_phase6_context(api, engine)[0].value)
    _, landed = process(engine, 5, "BOUNCE", line_call=call(api, legal_decision, 5))
    assert landed.outcome_type == api.PointOutcomeType.SERVE_IN_PLAY
    assert_phase(api, engine, "LIVE", "RALLY_LIVE")
    assert engine.state.serve_attempt == 2
    process(engine, 6, "PLAYER_2_HIT", player_id=2)
    contexts.append(actual_phase6_context(api, engine)[0].value)
    _, winner = process(engine, 7, "BOUNCE", line_call=call(api, "OUT", 7, "RALLY"), point_id="synthetic-point")
    assert contexts == ["SERVE", "SERVE", "RALLY"]
    assert winner.winner_id == 1 and engine.state.points_p1 == 1 and engine.state.points_p2 == 0
    assert engine.state.serve_attempt == 1 and engine.state.ball_state == api.BallPlayState.POINT_COMPLETE
    assert engine.resolver.ball_play_state == api.BallPlayState.DEAD


@pytest.mark.parametrize("attempt", [1, 2])
def test_first_and_second_serve_lets_replay_same_attempt_without_live_phase(attempt):
    api = actual_classes()
    engine = api.TennisScoringEngine()
    ident = 1
    if attempt == 2:
        process(engine, ident, "SERVE_CONTACT"); ident += 1
        process(engine, ident, "BOUNCE", line_call=call(api, "SERVE_FAULT", ident)); ident += 1
    process(engine, ident, "SERVE_CONTACT"); ident += 1
    expected = "FIRST_SERVE_IN_PROGRESS" if attempt == 1 else "SECOND_SERVE_IN_PROGRESS"
    assert_phase(api, engine, "SERVE_STARTED", expected)
    _, outcome = process(engine, ident, "SERVICE_LET"); ident += 1
    assert outcome.outcome_type == api.PointOutcomeType.SERVICE_LET
    assert engine.state.serve_attempt == attempt and engine.state.ball_state == api.BallPlayState.LET_REPLAY
    _, unknown = process(engine, ident, "BOUNCE", line_call=call(api, "IN", ident, "RALLY")); ident += 1
    assert unknown.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT
    assert engine.state.ball_state == api.BallPlayState.LET_REPLAY
    process(engine, ident, "SERVE_CONTACT"); ident += 1
    assert_phase(api, engine, "SERVE_STARTED", expected)
    assert actual_phase6_context(api, engine)[0] == api.LineCallContext.SERVE
    process(engine, ident, "BOUNCE", line_call=call(api, "SERVE_IN", ident))
    assert_phase(api, engine, "LIVE", "RALLY_LIVE")
    assert engine.state.points_p1 == engine.state.points_p2 == 0


def test_second_fault_awards_receiver_once_dead_ball_and_next_attempt_reset_preserved():
    api = actual_classes()
    engine = api.TennisScoringEngine()
    for ident in (1, 3):
        process(engine, ident, "SERVE_CONTACT")
        assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
        _, outcome = process(engine, ident + 1, "BOUNCE", line_call=call(api, "SERVE_FAULT", ident + 1))
    assert outcome.outcome_type == api.PointOutcomeType.DOUBLE_FAULT and outcome.winner_id == 2
    assert engine.state.points_p2 == 1 and engine.state.serve_attempt == 1
    assert engine.state.ball_state == api.BallPlayState.POINT_COMPLETE
    snapshot = copy.deepcopy(engine.state.to_dict())
    _, dead = process(engine, 5, "BOUNCE", line_call=call(api, "OUT", 5, "RALLY"))
    assert dead.outcome_type == api.PointOutcomeType.DEAD_BALL_IGNORED and engine.state.to_dict() == snapshot
    process(engine, 6, "SERVE_CONTACT")
    assert_phase(api, engine, "SERVE_STARTED", "FIRST_SERVE_IN_PROGRESS")


def test_duplicate_contact_and_bounce_ids_cannot_advance_phase_or_history_twice():
    api = actual_classes()
    engine = api.TennisScoringEngine()
    process(engine, 1, "SERVE_CONTACT")
    snapshot = copy.deepcopy(engine.state.to_dict())
    _, duplicate = process(engine, 1, "BOUNCE", line_call=call(api, "SERVE_IN", 1))
    assert duplicate.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT and engine.state.to_dict() == snapshot
    assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED and len(engine.history.entries) == 1
    process(engine, 2, "BOUNCE", line_call=call(api, "SERVE_IN", 2))
    snapshot = copy.deepcopy(engine.state.to_dict())
    process(engine, 2, "SERVE_CONTACT")
    assert engine.state.to_dict() == snapshot and len(engine.history.entries) == 2


@pytest.mark.parametrize("unknown", ["AD_BREAK", "BOUNCE"])
def test_unknown_or_unverified_service_bounce_cannot_turn_contact_into_live_rally(unknown):
    api = actual_classes()
    engine = api.TennisScoringEngine()
    process(engine, 1, "SERVE_CONTACT")
    before = copy.deepcopy(engine.state.to_dict())
    _, outcome = process(engine, 2, unknown)
    assert outcome.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT
    assert engine.state.to_dict() == before
    assert_phase(api, engine, "SERVE_STARTED", "FIRST_SERVE_IN_PROGRESS")
    assert actual_phase6_context(api, engine)[0] == api.LineCallContext.SERVE


@pytest.mark.parametrize("gap", ["CONSEQUENTIAL_GAP", "POINT_OUTCOME_UNKNOWN"])
def test_review_gap_freezes_score_and_later_in_bounce_cannot_resume_authority(gap):
    api = actual_classes()
    engine = api.TennisScoringEngine()
    process(engine, 1, "SERVE_CONTACT")
    process(engine, 2, gap)
    assert engine.state.point_state == api.PointState.POINT_REVIEW_PENDING and engine.state.score_confirmed is False
    _, outcome = process(engine, 3, "BOUNCE", line_call=call(api, "SERVE_IN", 3))
    assert outcome.outcome_type == api.PointOutcomeType.POINT_REVIEW_REQUIRED
    assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
    assert engine.resolver.ball_play_state == api.BallPlayState.SERVE_STARTED
    assert engine.state.points_p1 == engine.state.points_p2 == 0


def test_unknown_line_decision_and_premature_return_withhold_live_phase():
    api = actual_classes()
    engine = api.TennisScoringEngine()
    process(engine, 1, "SERVE_CONTACT")
    _, unknown = process(engine, 2, "BOUNCE", line_call=call(api, "UNKNOWN", 2))
    assert unknown.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT
    assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
    _, review = process(engine, 3, "PLAYER_2_HIT", player_id=2)
    assert review.outcome_type == api.PointOutcomeType.POINT_REVIEW_REQUIRED
    assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
    assert engine.state.points_p1 == engine.state.points_p2 == 0


def test_no_serve_contact_and_review_required_line_evidence_cannot_create_live_play():
    api = actual_classes()
    engine = api.TennisScoringEngine()
    _, unknown = process(engine, 1, "BOUNCE", line_call=call(api, "IN", 1))
    assert unknown.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT
    assert engine.state.ball_state == api.BallPlayState.NOT_STARTED
    process(engine, 2, "SERVE_CONTACT")
    _, outcome = process(engine, 3, "BOUNCE", line_call=call(api, "REVIEW_REQUIRED", 3))
    assert outcome.outcome_type == api.PointOutcomeType.POINT_REVIEW_REQUIRED
    assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
    assert engine.state.point_state == api.PointState.POINT_REVIEW_PENDING
    assert engine.state.points_p1 == engine.state.points_p2 == 0


def test_narrow_fix_keeps_all_rule_outcomes_scores_attempts_winners_versions_and_history():
    pairs = []
    sequence = [("SERVE_CONTACT", None), ("BOUNCE", "SERVE_FAULT"), ("SERVE_CONTACT", None),
        ("SERVICE_LET", None), ("SERVE_CONTACT", None), ("BOUNCE", "SERVE_IN"),
        ("PLAYER_2_HIT", None), ("BOUNCE", "OUT"), ("SERVE_CONTACT", None),
        ("BOUNCE", "SERVE_FAULT"), ("SERVE_CONTACT", None), ("BOUNCE", "SERVE_FAULT")]
    for baseline in (True, False):
        api = actual_classes(baseline=baseline)
        engine = api.TennisScoringEngine()
        records = []
        for ident, (kind, decision) in enumerate(sequence, 1):
            _, outcome = process(engine, ident, kind, line_call=None if decision is None else call(api, decision, ident),
                                 player_id=2 if kind == "PLAYER_2_HIT" else None)
            state = engine.state.to_dict()
            state.pop("ball_state"); state.pop("point_state")
            history = engine.history.entries[-1].to_dict()
            for key in ("previous_state", "new_state"):
                history[key].pop("ball_state"); history[key].pop("point_state")
            records.append((outcome.to_dict(), state, history))
        pairs.append(records)
    assert pairs[0] == pairs[1]
