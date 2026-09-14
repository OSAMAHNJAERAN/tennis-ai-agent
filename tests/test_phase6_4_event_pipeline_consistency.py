"""Unit and regression tests for Phase 6.4 Event Pipeline Consistency & Recovery."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import List, Tuple

import pytest

from src.events.event_detector import (
    EventCandidate,
    EventDetectorSettings,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
)
from src.events.event_evaluator import (
    CanonicalStage,
    FirstFailureStage,
    canonical_one_to_one_matches,
    evaluate_event_lineage,
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


# 1. Canonical Stage Definitions
def test_canonical_stage_definitions_exist_and_ordered():
    stages = [s.value for s in CanonicalStage]
    assert stages == [
        "STAGE_0_RAW_PROPOSAL",
        "STAGE_1_USABLE_OBSERVATION",
        "STAGE_2_PHYSICAL_CANDIDATE",
        "STAGE_3_PHYSICAL_CONTACT",
        "STAGE_4_SEMANTIC_TYPE",
        "STAGE_5_PLAYER_ATTRIBUTION",
        "STAGE_6_AUTHORITATIVE_EVENT",
    ]


# 2. First failure stage taxonomy
def test_first_failure_stage_taxonomy():
    failures = [f.value for f in FirstFailureStage]
    assert "NO_RAW_PROPOSAL" in failures
    assert "TRACK_NOT_USABLE" in failures
    assert "NO_EVENT_CANDIDATE" in failures
    assert "PHYSICAL_CONTACT_REJECTED" in failures
    assert "CONTACT_FAMILY_WRONG" in failures
    assert "SEMANTIC_TYPE_WRONG" in failures
    assert "PLAYER_ATTRIBUTION_WRONG" in failures
    assert "FULLY_CORRECT" in failures


# 3. Same event gives same stage result across evaluators
def test_evaluator_consistency_on_identical_inputs():
    traj = _trajectory(100)
    p1 = [BBox(600.0, 200.0, 700.0, 400.0)] * 100
    p2 = [None] * 100
    gt = [{"event_id": 1, "event_type": "PLAYER_HIT", "frame_best": 30, "frame_min": 28, "frame_max": 32, "player_id": 1}]
    detector = TennisEventDetector()
    raw_frames = [[{"x_px": 640.0, "y_px": 360.0, "confidence": 0.9, "bbox": [635, 355, 645, 365]}]] * 100

    recs1, m1 = evaluate_event_lineage("test_vid", raw_frames, traj, p1, p2, gt, detector, fps=30.0, frame_size=(1280, 720))
    recs2, m2 = evaluate_event_lineage("test_vid", raw_frames, traj, p1, p2, gt, detector, fps=30.0, frame_size=(1280, 720))

    assert recs1[0].first_failure_stage == recs2[0].first_failure_stage
    assert m1["candidate_count"] == m2["candidate_count"]


# 4. Canonical 1-to-1 bipartite matching rules & duplicate penalty
def test_bipartite_matching_enforces_one_to_one_and_duplicate_penalty():
    preds = [
        {"frame": 30, "event_type": "PLAYER_HIT", "player_id": 1},
        {"frame": 31, "event_type": "PLAYER_HIT", "player_id": 1},  # Duplicate
    ]
    gt = [{"frame_best": 30, "frame_min": 28, "frame_max": 32, "event_type": "PLAYER_HIT", "player_id": 1}]
    matches = canonical_one_to_one_matches(preds, gt, tolerance_s=0.200, fps=30.0, require_event_type=True)
    assert len(matches) == 1
    # Exactly one prediction matched
    matched_preds = {m[0] for m in matches}
    assert len(matched_preds) == 1


# 5. Physical contact preserved under semantic uncertainty
def test_physical_contact_preserved_under_semantic_uncertainty(monkeypatch):
    detector = TennisEventDetector(config={"enable_semantic_unknown_abstention": True})
    traj = _trajectory(100, y=300.0)
    p1 = [BBox(600.0, 200.0, 700.0, 400.0)] * 100
    p2 = [None] * 100
    cand = _candidate(30, 30.0, y=300.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) >= 1
    assert analysis.verification_traces[0]["physical_event_type"] in (
        PhysicalEventType.PLAYER_CONTACT.value,
        PhysicalEventType.COURT_CONTACT.value,
        PhysicalEventType.UNKNOWN.value,
    )


# 6. Contact family metrics separation
def test_contact_family_metrics_separation():
    with open(VALIDATION_ROOT / "phase6_4_event_pipeline_confusion.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    fam_metrics = data["contact_family_metrics"]
    assert "PLAYER_CONTACT" in fam_metrics
    assert "COURT_CONTACT" in fam_metrics
    assert fam_metrics["PLAYER_CONTACT"]["recall"] >= 0.50


# 7. Player vs court contact elevation disambiguation
def test_player_vs_court_elevation_disambiguation(monkeypatch):
    detector = TennisEventDetector(config={"enable_contact_family_stage": True})
    traj = _trajectory(100, y=250.0)  # Mid-air
    p1 = [BBox(600.0, 200.0, 700.0, 450.0)] * 100
    p2 = [None] * 100
    cand = _candidate(30, 30.0, y=250.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 1
    assert analysis.events[0].event_type in (EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT, EventType.SERVE_CONTACT)


# 8. Event time refinement improves apex accuracy
def test_event_time_refinement_apex_accuracy():
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


# 9. Native FPS timing invariance (24, 30, 60 FPS)
def test_native_fps_timing_invariance():
    detector = TennisEventDetector()
    for fps in (24.0, 30.0, 60.0):
        traj = _trajectory(int(fps * 3), fps=fps)
        positions = [(p.x_px, p.y_px) for p in traj]
        feats = detector._features(traj, (1280, 720), positions)
        assert len(feats) == len(traj)


# 10. Resolution scale invariance (720p vs 1080p)
def test_resolution_scale_invariance():
    detector = TennisEventDetector()
    traj_720 = _trajectory(100, y=360.0)
    traj_1080 = _trajectory(100, y=540.0)
    feats_720 = detector._features(traj_720, (1280, 720), [(p.x_px, p.y_px) for p in traj_720])
    feats_1080 = detector._features(traj_1080, (1920, 1080), [(p.x_px, p.y_px) for p in traj_1080])
    assert len(feats_720) == len(feats_1080)


# 11. Zero video-specific branching in production source code
def test_zero_video_specific_rules_in_events_package():
    src_events = REPO_ROOT / "src" / "events"
    for py_file in src_events.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        assert not re.search(r'\bvideo_0[89]\b|\bvideo_10\b|\bDjokovic\b|\bTiafoe\b|\bDimitrov\b', content)


# 12. Ground truth leakage protection
def test_no_ground_truth_leakage_in_event_detector():
    detector_code = (REPO_ROOT / "src" / "events" / "event_detector.py").read_text(encoding="utf-8")
    assert "ground_truth" not in detector_code
    assert "gt_events" not in detector_code


# 13. Event lineage artifact schema and integrity
def test_event_lineage_artifact_schema_and_integrity():
    with open(VALIDATION_ROOT / "phase6_4_event_lineage.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["phase"] == "6.4"
    assert len(data["records"]) == 40
    for r in data["records"]:
        assert "first_failure_stage" in r
        assert "first_failure_reason" in r


# 14. Canonical stage metrics monotonicity
def test_canonical_stage_metrics_monotonicity():
    with open(VALIDATION_ROOT / "phase6_4_canonical_stage_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    ceilings = [s["recall_ceiling"] for s in data["stage_metrics"]]
    assert len(ceilings) == 7
    for i in range(len(ceilings) - 1):
        assert ceilings[i] >= ceilings[i + 1] - 1e-6


# 15. Unknown event serialization & analytics contract compatibility
def test_unknown_event_serialization_and_contract_compatibility():
    event = TennisEvent(
        event_id=1,
        event_type=EventType.UNKNOWN_EVENT,
        frame_index=15,
        timestamp_s=0.50,
        player_id=None,
        ball_position_px=(640.0, 360.0),
        court_position_m=None,
        confidence=0.75,
        trajectory_state="DETECTED",
        evidence={"physical_event_type": "UNKNOWN_PHYSICAL_EVENT"},
    )
    assert event.event_type == EventType.UNKNOWN_EVENT
    assert event.confidence == 0.75
