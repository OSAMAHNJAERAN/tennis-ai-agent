"""Exercise the actual line-call definitions without loading image/model runtimes.

Explicit synthetic metric points establish software routing, never line accuracy.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"torch", "ultralytics", "numpy", "cv2", "pandas", "PIL"}


def actual_api():
    module = ModuleType("actual_service_target_abstention")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update({key: globals()[key] for key in (
        "asdict", "dataclass", "Enum", "math", "Any", "Dict", "List", "Optional", "Tuple")})
    namespace.update(np=SimpleNamespace(ndarray=object),
                     BounceContactRefiner=object, TemporalBallPoint=object,
                     RefinedContactPoint=object)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in FORBIDDEN:
            raise AssertionError("Model/image runtime forbidden in pure line-call test: " + name)
        return original(name, *args, **kwargs)
    try:
        builtins.__import__ = guarded
        for path, selected in (
            ("src/tracking/temporal_ball_tracker.py", {"BallState"}),
            ("src/line_calling/line_geometry.py", None),
            ("src/line_calling/line_call_engine.py", None),
        ):
            source = ROOT / path
            nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                     if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                     and (selected is None or node.name in selected)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    finally:
        builtins.__import__ = original
    return SimpleNamespace(**namespace)


class ForbiddenAccess:
    def __getitem__(self, key):
        raise AssertionError("Unavailable service target must not read trajectory/frame data")


class ForbiddenRefiner:
    def refine_bounce_contact(self, *args):
        raise AssertionError("Unavailable service target must not refine a contact")


@pytest.mark.parametrize("target", [None, "", "UNKNOWN", "near_deuce", 0, True, [], {}])
def test_unavailable_target_abstains_before_any_geometry_or_observation_access(target):
    api = actual_api()
    api.transform_point = lambda *args: pytest.fail("Projection is forbidden without service context")
    engine = api.TennisLineCallEngine(refiner=ForbiddenRefiner())
    evidence = engine.evaluate_bounce(19, 987, ForbiddenAccess(), ForbiddenAccess(),
                                    api.LineCallContext.SERVE, target, ForbiddenAccess())
    assert evidence.decision == api.LineCallDecision.REVIEW_REQUIRED
    assert evidence.decision_context == api.LineCallContext.SERVE
    assert evidence.event_id == 19 and evidence.bounce_frame == 987
    assert evidence.reason_code == "SERVICE_TARGET_UNAVAILABLE"
    assert evidence.confidence == 0.0
    assert evidence.nearest_line == api.CourtLineType.NONE
    assert evidence.refinement_method == "NOT_PERFORMED"
    assert evidence.tracker_state == "UNKNOWN"
    for field in ("bounce_position_px", "bounce_position_m", "center_signed_distance_cm",
                  "ball_edge_margin_cm", "position_uncertainty_cm", "contact_patch_radius_cm"):
        assert getattr(evidence, field) is None
    payload = json.loads(json.dumps(evidence.to_dict(), allow_nan=False))
    assert payload["decision"] == "REVIEW_REQUIRED" and payload["bounce_position_m"] is None
    assert payload["line_strip_info"] is None


@pytest.mark.parametrize("box_name,point", [
    ("NEAR_DEUCE", (7., 15.)), ("NEAR_AD", (3., 15.)),
    ("FAR_DEUCE", (3., 8.)), ("FAR_AD", (7., 8.)),
])
@pytest.mark.parametrize("as_string", [False, True])
def test_explicit_correct_target_preserves_real_geometry(box_name, point, as_string):
    api = actual_api()
    box = api.ServiceBoxType(box_name)
    class Refiner:
        calls = 0
        def refine_bounce_contact(self, *args):
            self.calls += 1
            return SimpleNamespace(x_px=point[0], y_px=point[1], refinement_method="SYNTHETIC_POINT")
    refiner = Refiner()
    api.TennisLineCallEngine.evaluate_bounce.__globals__["transform_point"] = lambda xy, matrix: xy
    trajectory = [SimpleNamespace(state=api.BallState.DETECTED)]
    evidence = api.TennisLineCallEngine(refiner=refiner).evaluate_bounce(
        1, 0, trajectory, object(), api.LineCallContext.SERVE, box_name if as_string else box)
    assert refiner.calls == 1
    assert evidence.decision == api.LineCallDecision.SERVE_IN
    assert evidence.bounce_position_m == point and evidence.confidence > 0
    assert evidence.ball_edge_margin_cm > 0
    assert evidence.reason_code is None


def test_explicit_opposite_box_still_faults_and_rally_needs_no_service_target():
    api = actual_api()
    refiner = SimpleNamespace(refine_bounce_contact=lambda *args:
        SimpleNamespace(x_px=3., y_px=6., refinement_method="SYNTHETIC_POINT"))
    api.TennisLineCallEngine.evaluate_bounce.__globals__["transform_point"] = lambda xy, matrix: xy
    engine = api.TennisLineCallEngine(refiner=refiner)
    trajectory = [SimpleNamespace(state=api.BallState.DETECTED)]
    explicit = engine.evaluate_bounce(1, 0, trajectory, object(), api.LineCallContext.SERVE,
                                      api.ServiceBoxType.NEAR_DEUCE)
    rally = engine.evaluate_bounce(2, 0, trajectory, object(), api.LineCallContext.RALLY)
    assert explicit.decision == api.LineCallDecision.SERVE_FAULT
    assert rally.decision == api.LineCallDecision.IN


def test_caller_can_record_missing_current_end_evidence_without_calling_geometry():
    api = actual_api()
    evidence = api.LineCallEvidence.unavailable(
        event_id=23, bounce_frame=43, context=api.LineCallContext.SERVE,
        reason="Current identity-to-court-end binding is unavailable.",
        reason_code="SERVICE_END_BINDING_UNAVAILABLE")
    assert evidence.reason_code == "SERVICE_END_BINDING_UNAVAILABLE"
    assert evidence.decision == api.LineCallDecision.REVIEW_REQUIRED
    assert evidence.bounce_position_px is None and evidence.bounce_position_m is None
    text = evidence.overlay_text()
    assert "REVIEW_REQUIRED" in text and "unavailable" in text.lower()
    assert "0.0cm" not in text and "None" not in text


def test_measured_evidence_overlay_keeps_the_existing_numeric_text():
    api = actual_api()
    evidence = api.LineCallEvidence(
        1, 2, api.LineCallDecision.IN, api.LineCallContext.RALLY, api.CourtLineType.LEFT_SIDELINE,
        (5., 6.), (3., 6.), "DETECTED", 4., "EMPIRICAL_PATCH", 1.25, 5.25, 1.48,
        "MID", .8, "Synthetic only", "SYNTHETIC_ONLY")
    assert evidence.overlay_text() == "LINE CALL: [IN] (LEFT_SIDELINE +5.2cm, Unc: +- 1.5cm)"


@pytest.mark.parametrize("relative", ["src/pipeline/phase4_pipeline.py", "src/pipeline/phase5_pipeline.py"])
def test_existing_pipeline_overlay_executes_with_nullable_evidence(relative):
    api = actual_api()
    evidence = api.LineCallEvidence.unavailable(
        1, 2, api.LineCallContext.SERVE, "Service target unavailable.")
    path = ROOT / relative
    assignments = [node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "call_text" for target in node.targets)]
    assert len(assignments) == 1
    namespace = {"evidence": evidence}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), namespace)
    assert "REVIEW_REQUIRED" in namespace["call_text"]
    assert "None" not in namespace["call_text"]


@pytest.mark.parametrize("near,far", [("NEAR_DEUCE", "FAR_DEUCE"), ("NEAR_AD", "FAR_AD")])
def test_service_targets_rotate_with_the_servers_right_or_left_half(near, far):
    api = actual_api()
    geometry = api.CourtLineGeometry
    a = geometry.SERVICE_BOX_BOUNDS[api.ServiceBoxType(far)]
    b = geometry.SERVICE_BOX_BOUNDS[api.ServiceBoxType(near)]
    assert (10.97 - a[1], 10.97 - a[0], 23.77 - a[3], 23.77 - a[2]) == pytest.approx(b)
    center = (10.97 - (a[0]+a[1])/2, 23.77 - (a[2]+a[3])/2)
    result = geometry.evaluate_service_box_boundary(*center, api.ServiceBoxType(near))
    assert result.is_center_inside and result.is_contact_inside_or_touching


@pytest.mark.parametrize("box,x,y,line", [
    ("NEAR_DEUCE", 9.70, 15., "RIGHT_SIDELINE"),
    ("NEAR_DEUCE", 5.40, 15., "CENTER_SERVICE_LINE"),
    ("NEAR_AD", 1.27, 15., "LEFT_SIDELINE"),
    ("NEAR_AD", 5.57, 15., "CENTER_SERVICE_LINE"),
    ("FAR_DEUCE", 1.27, 8., "LEFT_SIDELINE"),
    ("FAR_AD", 9.70, 8., "RIGHT_SIDELINE"),
])
def test_service_boundary_names_follow_actual_polygon_side(box, x, y, line):
    api = actual_api()
    result = api.CourtLineGeometry.evaluate_service_box_boundary(x, y, api.ServiceBoxType(box))
    assert result.nearest_line.value == line
    assert result.signed_center_distance_cm < 0

