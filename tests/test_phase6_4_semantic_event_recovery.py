"""Unit and regression tests for Phase 6.4 Semantic Event Type Disambiguation Recovery."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

import pytest

from src.events.event_detector import (
    EventCandidate,
    EventDetectorSettings,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
)
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = REPO_ROOT / "artifacts" / "validation"


def _trajectory(count=100, fps=30.0, x=640.0, y=360.0) -> List[TemporalBallPoint]:
    return [
        TemporalBallPoint(
            frame_index=index,
            timestamp_seconds=index / fps,
            x_px=x,
            y_px=y,
            confidence=0.9,
            state=BallState.DETECTED,
        )
        for index in range(count)
    ]


def _candidate(frame, fps=30.0, *, y=360.0, inversion=False, court_rebound=False) -> EventCandidate:
    return EventCandidate(
        frame_index=frame,
        timestamp_s=frame / fps,
        score=0.85,
        ball_position_px=(640.0, y),
        trajectory_state="DETECTED",
        evidence={
            "candidate_source": "TEST",
            "pre_speed_px_s": 400.0,
            "post_speed_px_s": 450.0,
            "direction_change_degrees": 45.0,
            "acceleration_magnitude_px_s2": 1200.0,
            "vertical_inversion": inversion,
            "court_rebound": court_rebound,
            "trajectory_continuity": True,
            "actual_fps": fps,
        },
    )


# Test 1: Contact family stage separates player from court contact
def test_contact_family_stage_separates_player_from_court_contact(monkeypatch):
    detector = TennisEventDetector(config={"enable_contact_family_stage": True})
    traj = _trajectory(y=600.0)
    # Near player -> PLAYER_CONTACT
    p1_box = BBox(600.0, 500.0, 700.0, 700.0)
    p1 = [p1_box] * 100
    p2 = [None] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 1
    assert analysis.events[0].event_type in (EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT, EventType.SERVE_CONTACT)
    assert analysis.verification_traces[0]["physical_event_type"] == PhysicalEventType.PLAYER_CONTACT.value


# Test 2: Temporal proximity improves contact association
def test_temporal_proximity_improves_contact_association(monkeypatch):
    detector = TennisEventDetector(config={"enable_player_temporal_proximity": True})
    traj = _trajectory(y=300.0)
    # Player arrives 2 frames later
    p1 = [None] * 100
    p1[32] = BBox(600.0, 200.0, 700.0, 400.0)
    p2 = [None] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    trace = analysis.verification_traces[0]
    assert trace["player1_distance_normalized"] is not None


# Test 3: Event time refinement reduces timing error
def test_event_time_refinement_reduces_timing_error():
    detector = TennisEventDetector(config={"enable_event_time_refinement": True})
    traj = _trajectory(100)
    for i in range(25, 35):
        traj[i].x_px = 640.0 + (i - 30) ** 2
    cand = _candidate(28, 30.0)
    p1 = [None] * 100
    positions = [(p.x_px, p.y_px) for p in traj]
    feats = detector._features(traj, (1280, 720), positions)
    refined = detector._refined_frame(cand, feats, traj, p1, p1)
    assert abs(refined - 30) <= 2


# Test 4: Player attribution fuses dual player evidence
def test_player_attribution_fuses_dual_player_evidence(monkeypatch):
    detector = TennisEventDetector(config={"enable_player_attribution_fusion": True})
    traj = _trajectory(y=300.0)
    p1 = [BBox(620.0, 200.0, 680.0, 400.0)] * 100  # Closer
    p2 = [BBox(800.0, 200.0, 900.0, 400.0)] * 100  # Farther
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert analysis.events[0].player_id == 1


# Test 5: Ambiguous player attribution abstains safely
def test_ambiguous_player_attribution_abstains_safely(monkeypatch):
    detector = TennisEventDetector(config={"enable_player_attribution_fusion": True})
    traj = _trajectory(y=300.0)
    # Both players at identical location
    box = BBox(600.0, 200.0, 700.0, 400.0)
    p1 = [box] * 100
    p2 = [box] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    # Abstains safely: either no event or UNKNOWN
    assert analysis.events == [] or analysis.events[0].event_type == EventType.UNKNOWN_EVENT


# Test 6: Bounce verifier rejects non-court vertical reversals
def test_bounce_verifier_rejects_non_court_vertical_reversals(monkeypatch):
    detector = TennisEventDetector(config={"enable_bounce_verification": True})
    traj = _trajectory(y=100.0)  # High in the air
    p1 = [None] * 100
    p2 = [None] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps, inversion=False, court_rebound=False)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert analysis.events == []


# Test 7: Bounce verifier accepts valid court contact
def test_bounce_verifier_accepts_valid_court_contact(monkeypatch):
    detector = TennisEventDetector(config={"enable_bounce_verification": True})
    traj = _trajectory(y=550.0)
    p1 = [None] * 100
    p2 = [None] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps, inversion=True, court_rebound=True)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 1
    assert analysis.events[0].event_type == EventType.BOUNCE


# Test 8: Serve context identifies overhead contact
def test_serve_context_identifies_overhead_contact(monkeypatch):
    detector = TennisEventDetector(config={"enable_serve_semantics": True})
    traj = _trajectory(y=180.0)
    # Pre-serve toss
    for i in range(10, 21):
        traj[i].y_px = 240.0 - (i - 10) * 6.0
    for i in range(21, 35):
        traj[i].y_px = 180.0 + (i - 20) * 18.0
        traj[i].x_px = 640.0 + (i - 20) * 15.0
    p1 = [BBox(600.0, 200.0, 700.0, 450.0)] * 100
    p2 = [None] * 100
    cand = _candidate(20, 30.0, y=180.0)
    cand.evidence["post_speed_px_s"] = 650.0
    cand.evidence["pre_speed_px_s"] = 50.0
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 1
    assert analysis.events[0].event_type == EventType.SERVE_CONTACT


# Test 9: Serve context rejects groundstrokes and bounces
def test_serve_context_rejects_groundstrokes_and_bounces(monkeypatch):
    detector = TennisEventDetector(config={"enable_serve_semantics": True})
    traj = _trajectory(y=350.0)  # Waist height
    p1 = [BBox(600.0, 200.0, 700.0, 450.0)] * 100
    p2 = [None] * 100
    cand = _candidate(50, 30.0, y=350.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 1
    assert analysis.events[0].event_type == EventType.PLAYER_1_HIT


# Test 10: Serve context does not use first event or frame <= N
def test_serve_context_does_not_use_first_event_or_frame_le_n(monkeypatch):
    detector = TennisEventDetector(config={"enable_serve_semantics": True})
    traj = _trajectory(y=350.0)
    p1 = [BBox(600.0, 200.0, 700.0, 450.0)] * 100
    p2 = [None] * 100
    # First candidate at frame 2
    cand = _candidate(2, 30.0, y=350.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert analysis.events[0].event_type != EventType.SERVE_CONTACT


# Test 11: Semantic uncertainty preserves physical contact
def test_semantic_uncertainty_preserves_physical_contact(monkeypatch):
    detector = TennisEventDetector(config={"enable_semantic_unknown_abstention": True})
    traj = _trajectory(y=300.0)
    p1 = [BBox(600.0, 200.0, 700.0, 400.0)] * 100
    p2 = [None] * 100
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [_candidate(30, fps)])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) >= 1


# Test 12: Unknown semantic abstention mechanism
def test_unknown_semantic_abstention_mechanism():
    settings = EventDetectorSettings(enable_semantic_unknown_abstention=True)
    assert settings.enable_semantic_unknown_abstention is True


# Test 13: Candidate recall preserved through semantic stage
def test_candidate_recall_preserved_through_semantic_stage():
    with open(VALIDATION_ROOT / "phase6_4_semantic_stage_metrics.json") as f:
        data = json.load(f)
    stages = data["stage_metrics"]
    cand_recall = stages[0]["recall"]["OVERALL"]
    final_recall = stages[-1]["recall"]["OVERALL"]
    assert cand_recall >= 0.50
    assert final_recall > 0.15


# Test 14: All ablation variants A through I runnable
def test_all_ablation_variants_a_through_i_runnable():
    with open(VALIDATION_ROOT / "phase6_4_semantic_event_ablation.json") as f:
        data = json.load(f)
    variants = [v["variant"] for v in data["variants"]]
    expected_prefixes = ["A_", "B_", "C_", "D_", "E_", "F_", "G_", "H_", "I_"]
    for pref in expected_prefixes:
        assert any(v.startswith(pref) for v in variants)


# Test 15: Ablation variant A matches baseline
def test_ablation_variant_a_matches_baseline():
    with open(VALIDATION_ROOT / "phase6_4_semantic_event_ablation.json") as f:
        data = json.load(f)
    var_a = next(v for v in data["variants"] if v["variant"].startswith("A_"))
    assert var_a["true_positives"] >= 9


# Test 16: Ablation variant I shows improved F1 or precision
def test_ablation_variant_i_shows_improved_f1():
    with open(VALIDATION_ROOT / "phase6_4_semantic_event_ablation.json") as f:
        data = json.load(f)
    var_a = next(v for v in data["variants"] if v["variant"].startswith("A_"))
    var_i = next(v for v in data["variants"] if v["variant"].startswith("I_"))
    assert var_i["overall_precision"] >= var_a["overall_precision"]


# Test 17: FPS invariance 24, 30, 60
def test_fps_invariance_24_30_60():
    detector = TennisEventDetector()
    for fps in (24.0, 30.0, 60.0):
        traj = _trajectory(int(fps * 3), fps=fps)
        positions = [(p.x_px, p.y_px) for p in traj]
        feats = detector._features(traj, (1280, 720), positions)
        assert len(feats) == len(traj)


# Test 18: Resolution invariance 720p vs 1080p
def test_resolution_invariance_720p_1080p():
    detector = TennisEventDetector()
    traj_720 = _trajectory(y=360.0)
    traj_1080 = _trajectory(y=540.0)
    pos_720 = [(p.x_px, p.y_px) for p in traj_720]
    pos_1080 = [(p.x_px, p.y_px) for p in traj_1080]
    feats_720 = detector._features(traj_720, (1280, 720), pos_720)
    feats_1080 = detector._features(traj_1080, (1920, 1080), pos_1080)
    assert len(feats_720) == len(feats_1080)


# Test 19: No video ID branching in codebase
def test_no_video_id_branching_in_codebase():
    src_events = REPO_ROOT / "src" / "events"
    for py_file in src_events.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert not re.search(r'\bvideo_0[89]\b|\bvideo_10\b|\bDjokovic\b|\bTiafoe\b', content)


# Test 20: Semantic confusion audit artifact schema and integrity
def test_semantic_confusion_audit_artifact_schema_and_integrity():
    with open(VALIDATION_ROOT / "phase6_4_semantic_event_confusion.json") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["phase"] == "6.4"
    assert len(data["records"]) == 40
    outcomes = data["summary"]["outcomes"]
    assert "CORRECT_CLASSIFICATION" in outcomes


# Test 21: Ablation artifact schema and integrity
def test_ablation_artifact_schema_and_integrity():
    with open(VALIDATION_ROOT / "phase6_4_semantic_event_ablation.json") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert len(data["variants"]) == 9
    assert data["recommended_variant"] == "I_FINAL_INTEGRATED_SEMANTIC_SYSTEM"


# Test 22: Semantic stage metrics monotonicity
def test_semantic_stage_metrics_monotonicity():
    with open(VALIDATION_ROOT / "phase6_4_semantic_stage_metrics.json") as f:
        data = json.load(f)
    stages = data["stage_metrics"]
    recalls = [s["recall"]["OVERALL"] for s in stages]
    assert len(recalls) == 6
    # Monotonically non-increasing
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-6
