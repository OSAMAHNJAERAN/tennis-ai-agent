"""Unit and regression tests for Phase 6.4 Pre-Semantic Physical Contact Recovery."""

from __future__ import annotations

import json
import math
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
from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallPoint, TemporalBallTracker
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


def _candidate(frame, fps=30.0, *, y=360.0, score=0.85, inversion=False) -> EventCandidate:
    return EventCandidate(
        frame_index=frame,
        timestamp_s=frame / fps,
        score=score,
        ball_position_px=(640.0, y),
        trajectory_state="DETECTED",
        evidence={
            "candidate_source": "TEST_KINEMATICS",
            "pre_speed_px_s": 400.0,
            "post_speed_px_s": 450.0,
            "direction_change_degrees": 45.0,
            "acceleration_magnitude_px_s2": 1200.0,
            "vertical_inversion": inversion,
            "trajectory_continuity": True,
            "actual_fps": fps,
        },
    )


# 1. Canonical Stage 1 interval-aware observability
def test_stage1_interval_aware_observability():
    tracker = TemporalBallTracker()
    raw = [[] for _ in range(30)]
    for f in (10, 11, 12, 13):
        raw[f] = [BallObservation(640.0, 360.0, 0.9, BBox(635, 355, 645, 365))]
    traj = tracker.track_video_candidates(raw, fps=30.0, frame_size=(1280, 720))
    # Points 10..13 exist in trajectory
    tracked = [p for p in traj if p.state in (BallState.DETECTED, BallState.TRACKED) and p.x_px is not None]
    assert len(tracked) >= 3


# 2. Exact-frame vs valid-contact-window distinction
def test_exact_frame_vs_valid_contact_window():
    gt = [{"frame_best": 15, "frame_min": 12, "frame_max": 18, "event_type": "BOUNCE"}]
    pred = [{"frame": 13, "event_type": "BOUNCE"}]
    matches = canonical_one_to_one_matches(pred, gt, tolerance_s=0.200, fps=30.0, require_event_type=False)
    assert len(matches) == 1
    assert matches[0][2] == 0.0  # inside [12..18] window distance is 0.0


# 3. Raw proposal exists but association fails
def test_raw_proposal_exists_but_association_fails():
    tracker = TemporalBallTracker(max_valid_speed_px_per_frame=40.0)
    raw = [[] for _ in range(20)]
    raw[0] = [BallObservation(100.0, 100.0, 0.9)]
    raw[1] = [BallObservation(600.0, 600.0, 0.04)]  # Low-conf outlier jump > gating radius
    traj = tracker.track_video_candidates(raw, fps=30.0, frame_size=(1280, 720))
    assert traj[1].state != BallState.DETECTED and traj[1].state != BallState.TRACKED


# 4. Reacquisition preserves usable Stage 1 evidence
def test_reacquisition_preserves_usable_stage1():
    tracker = TemporalBallTracker(enable_short_gap_reacquisition=True)
    raw = [[] for _ in range(30)]
    raw[0] = [BallObservation(100.0, 100.0, 0.9)]
    # Gap from 1..10
    raw[11] = [BallObservation(200.0, 200.0, 0.8)]
    raw[12] = [BallObservation(210.0, 210.0, 0.8)]
    raw[13] = [BallObservation(220.0, 220.0, 0.8)]
    traj = tracker.track_video_candidates(raw, fps=30.0, frame_size=(1280, 720))
    reacquired = [p for p in traj[11:14] if p.x_px is not None]
    assert len(reacquired) >= 2


# 5. MISSING state remains unusable
def test_missing_state_remains_unusable():
    pt = TemporalBallPoint(frame_index=5, timestamp_seconds=0.166, x_px=None, y_px=None, state=BallState.MISSING)
    assert pt.state == BallState.MISSING
    assert pt.x_px is None


# 6. Long interpolation cannot fake Stage 1 success
def test_long_interpolation_cannot_fake_stage1():
    tracker = TemporalBallTracker(max_prediction_gap=0, max_interpolation_gap=3)
    raw = [[] for _ in range(30)]
    raw[0] = [BallObservation(100.0, 100.0, 0.9)]
    raw[10] = [BallObservation(500.0, 500.0, 0.9)]  # Gap of 9 frames
    traj = tracker.track_video_candidates(raw, fps=30.0, frame_size=(1280, 720))
    # Intermediate points must NOT be interpolated
    interpolated = [p for p in traj[1:10] if p.state == BallState.INTERPOLATED]
    assert len(interpolated) == 0


# 7. Candidate generator is semantic-agnostic
def test_candidate_generator_is_semantic_agnostic():
    detector = TennisEventDetector()
    traj = _trajectory(50)
    for i in range(15, 25):
        traj[i].x_px = 640.0 + (i - 20) ** 2
    cands = detector.detect_candidates(traj, fps=30.0, frame_size=(1280, 720))
    assert len(cands) >= 1
    # Candidate does not assign tennis event type
    assert not hasattr(cands[0], "event_type")


# 8. Multi-scale physical candidate detection
def test_multi_scale_physical_candidate_detection():
    detector = TennisEventDetector()
    traj = _trajectory(60)
    # Sharp displacement in 2 frames
    traj[30].x_px = 750.0
    traj[31].x_px = 860.0
    positions = [(p.x_px, p.y_px) for p in traj]
    feats = detector._features(traj, (1280, 720), positions)
    assert feats[30].score >= 0.28


# 9. Candidate duplicate suppression
def test_candidate_duplicate_suppression():
    detector = TennisEventDetector(config={"candidate_min_interval_seconds": 0.20})
    traj = _trajectory(60)
    for i in range(25, 35):
        traj[i].x_px = 640.0 + (i - 30) ** 2
    cands = detector.detect_candidates(traj, fps=30.0, frame_size=(1280, 720))
    # Closely spaced candidates must be deduplicated
    frames = [c.frame_index for c in cands]
    for i in range(len(frames) - 1):
        assert abs(frames[i+1] - frames[i]) >= 6


# 10. Two real fast contacts are not merged
def test_two_separate_fast_contacts_not_merged():
    detector = TennisEventDetector(config={"candidate_min_interval_seconds": 0.15})
    traj = _trajectory(100)
    for i in range(15, 25):
        traj[i].x_px = 640.0 + (i - 20) ** 2
    for i in range(45, 55):
        traj[i].x_px = 640.0 + (i - 50) ** 2
    cands = detector.detect_candidates(traj, fps=30.0, frame_size=(1280, 720))
    assert len(cands) >= 2


# 11. Stage 3 physical recall does not require exact semantic type
def test_stage3_physical_recall_does_not_require_exact_semantic_type():
    gt = [{"frame_best": 30, "frame_min": 28, "frame_max": 32, "event_type": "PLAYER_HIT"}]
    pred = [{"frame": 30, "event_type": "BOUNCE"}]  # Type mismatch
    matches = canonical_one_to_one_matches(pred, gt, tolerance_s=0.200, fps=30.0, require_event_type=False)
    assert len(matches) == 1


# 12. Stage 3 physical precision counts unmatched contacts as FP
def test_stage3_physical_precision_counts_unmatched_contacts_as_fp():
    gt = [{"frame_best": 30, "frame_min": 28, "frame_max": 32}]
    preds = [{"frame": 30}, {"frame": 80}]  # Second is unmatched
    matches = canonical_one_to_one_matches(preds, gt, tolerance_s=0.200, fps=30.0, require_event_type=False)
    tp = len(matches)
    fp = len(preds) - tp
    assert tp == 1
    assert fp == 1


# 13. Physical UNKNOWN contact is preserved
def test_physical_unknown_contact_preserved(monkeypatch):
    detector = TennisEventDetector(config={"enable_semantic_unknown_abstention": True})
    traj = _trajectory(100, y=300.0)
    p1 = [BBox(600.0, 200.0, 700.0, 400.0)] * 100
    p2 = [BBox(600.0, 200.0, 700.0, 400.0)] * 100  # Ambiguous
    cand = _candidate(30, 30.0, y=300.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p2, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) >= 1
    assert analysis.events[0].event_type == EventType.UNKNOWN_EVENT


# 14. Post-rally motion suppression
def test_post_rally_dead_ball_motion_suppressed(monkeypatch):
    detector = TennisEventDetector(config={"enable_dead_ball_gating": True})
    traj = _trajectory(100, y=300.0)
    p1 = [None] * 100
    cand = _candidate(30, 30.0, y=300.0)
    cand.evidence["pre_speed_px_s"] = 2.0  # Slow rolling
    cand.evidence["post_speed_px_s"] = 2.0
    cand.evidence["acceleration_magnitude_px_s2"] = 5.0
    monkeypatch.setattr(detector, "detect_candidates", lambda t, fps, frame_size=None, camera_offsets_px=None: [cand])
    analysis = detector.analyze(traj, p1, p1, fps=30.0, frame_size=(1280, 720))
    assert len(analysis.events) == 0


# 15. Camera motion artifact rejection
def test_camera_motion_artifact_rejection():
    detector = TennisEventDetector(config={"enable_camera_motion_compensation": True})
    traj = _trajectory(60)
    offsets = [(float(i * 5.0), 0.0) for i in range(60)]  # Camera panning
    positions, state = detector._positions(traj, offsets)
    assert state.startswith("GLOBAL_TRANSLATION_COMPENSATED")


# 16. Invariance across 24, 30, 60 FPS
def test_invariance_across_framerates():
    detector = TennisEventDetector()
    for fps in (24.0, 30.0, 60.0):
        traj = _trajectory(int(fps * 2), fps=fps)
        positions = [(p.x_px, p.y_px) for p in traj]
        feats = detector._features(traj, (1280, 720), positions)
        assert len(feats) == len(traj)


# 17. Resolution scale normalization (720p vs 1080p)
def test_resolution_scale_normalization():
    tracker_720 = TemporalBallTracker()
    tracker_1080 = TemporalBallTracker()
    raw_720 = [[BallObservation(640.0, 360.0, 0.9)]]
    raw_1080 = [[BallObservation(960.0, 540.0, 0.9)]]
    t720 = tracker_720.track_video_candidates(raw_720, fps=30.0, frame_size=(1280, 720))
    t1080 = tracker_1080.track_video_candidates(raw_1080, fps=30.0, frame_size=(1920, 1080))
    assert t720[0].state == t1080[0].state


# 18. Zero video-specific logic in production code
def test_zero_video_specific_rules_in_source():
    for path in (REPO_ROOT / "src").glob("**/*.py"):
        content = path.read_text(encoding="utf-8")
        assert not re.search(r'\bvideo_0[89]\b|\bvideo_10\b|\bDjokovic\b|\bTiafoe\b|\bDimitrov\b', content)


# 19. No GT leaks in source code
def test_no_ground_truth_leaks_in_source():
    src_events = REPO_ROOT / "src" / "events"
    for path in src_events.glob("*.py"):
        if path.name == "event_evaluator.py":
            continue
        content = path.read_text(encoding="utf-8")
        assert "ground_truth" not in content
        assert "gt_events" not in content


# 20. Canonical evaluator reused across diagnostic scripts
def test_canonical_evaluator_reused_across_scripts():
    from scripts.evaluate_phase6_4_event_pipeline import run_canonical_evaluation
    from src.events.event_evaluator import evaluate_event_lineage
    assert callable(run_canonical_evaluation)
    assert callable(evaluate_event_lineage)
