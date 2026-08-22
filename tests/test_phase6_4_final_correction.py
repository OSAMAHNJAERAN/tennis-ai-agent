"""Regression tests for the Phase 6.4 scientific-integrity correction."""

import json

from scripts.evaluate_phase6_4_cross_match import evaluate_split_metrics, one_to_one_matches
from src.events.event_detector import EventCandidate, EventType, TennisEventDetector
from src.analytics.rally_analyzer import RallyAnalyzer
from src.pipeline.phase6_pipeline import sanitize_json_value
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_types import (
    PlayerHandedness,
    ShotClassificationSource,
    ShotDirection,
    ShotEventEvidence,
    ShotType,
)
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


def _trajectory(count=400, fps=30.0, x=150.0, y=200.0):
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


def _candidate(frame, fps, *, y=200.0, post_speed=500.0, inversion=False):
    return EventCandidate(
        frame_index=frame,
        timestamp_s=frame / fps,
        score=80.0,
        ball_position_px=(150.0, y),
        trajectory_state="DETECTED",
        evidence={
            "candidate_source": "TEST_TRAJECTORY_INFLECTION",
            "pre_speed_px_s": 450.0,
            "post_speed_px_s": post_speed,
            "direction_change_degrees": 50.0,
            "acceleration_magnitude_px_s2": 1200.0,
            "curvature": 0.01,
            "vertical_inversion": inversion,
            "trajectory_continuity": True,
            "actual_fps": fps,
        },
    )


def _detect_from_candidate(monkeypatch, frame, fps, *, y=200.0, p1=True, p2=False, inversion=False):
    detector = TennisEventDetector()
    trajectory = _trajectory(fps=fps, y=y)
    trajectory[frame].y_px = y
    monkeypatch.setattr(
        detector,
        "detect_candidates",
        lambda ball_trajectory, fps: [_candidate(frame, fps, y=y, inversion=inversion)],
    )
    player_box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=0)
    boxes_1 = [player_box if p1 else None] * len(trajectory)
    boxes_2 = [player_box if p2 else None] * len(trajectory)
    return detector.detect_events(trajectory, boxes_1, boxes_2, fps=fps)


def test_first_event_is_not_automatically_serve(monkeypatch):
    events = _detect_from_candidate(monkeypatch, frame=3, fps=30.0)
    assert len(events) == 1
    assert events[0].event_type == EventType.PLAYER_1_HIT


def test_clip_beginning_mid_rally_is_not_serve(monkeypatch):
    events = _detect_from_candidate(monkeypatch, frame=0, fps=30.0)
    assert events[0].event_type == EventType.PLAYER_1_HIT


def test_long_pre_serve_interval_does_not_create_absolute_frame_rule(monkeypatch):
    events = _detect_from_candidate(monkeypatch, frame=300, fps=30.0)
    assert events[0].event_type == EventType.PLAYER_1_HIT


def test_pre_serve_low_body_ball_bounce_is_suppressed(monkeypatch):
    events = _detect_from_candidate(monkeypatch, frame=24, fps=30.0, y=295.0, inversion=True)
    assert events == []


def test_ambiguous_wrong_player_attribution_abstains(monkeypatch):
    events = _detect_from_candidate(monkeypatch, frame=50, fps=30.0, p1=True, p2=True)
    assert events == []


def test_event_semantics_are_equivalent_at_24_30_60_fps(monkeypatch):
    event_types = []
    for fps in (24.0, 30.0, 60.0):
        frame = int(fps * 1.0)
        event_types.append(_detect_from_candidate(monkeypatch, frame, fps)[0].event_type)
    assert event_types == [EventType.PLAYER_1_HIT] * 3


def test_unknown_handedness_and_orientation_abstain_safely():
    classifier = TennisShotClassifier()
    box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=0)
    ball = TemporalBallPoint(50, 2.0, 190.0, 200.0, confidence=0.9, state=BallState.DETECTED)
    shot, _, _, _, _ = classifier.classify_shot("PLAYER_1_HIT", 50, 1, box, ball)
    assert shot == ShotType.UNKNOWN


def test_explicit_left_handed_normalization_inverts_geometry():
    box = BBox(100.0, 100.0, 200.0, 300.0, confidence=0.9, class_id=0)
    ball = TemporalBallPoint(50, 2.0, 190.0, 200.0, confidence=0.9, state=BallState.DETECTED)
    right = TennisShotClassifier(
        handedness_map={1: PlayerHandedness.RIGHT_HANDED}, court_orientation_map={1: 1.0}
    )
    left = TennisShotClassifier(
        handedness_map={1: PlayerHandedness.LEFT_HANDED}, court_orientation_map={1: 1.0}
    )
    assert right.classify_shot("PLAYER_1_HIT", 50, 1, box, ball)[0] == ShotType.FOREHAND
    assert left.classify_shot("PLAYER_1_HIT", 50, 1, box, ball)[0] == ShotType.BACKHAND


def test_evaluator_duplicate_prediction_is_false_positive():
    predictions = [
        {"frame": 100, "event_type": "PLAYER_1_HIT"},
        {"frame": 101, "event_type": "PLAYER_1_HIT"},
    ]
    ground_truth = [{"frame_best": 100, "event_type": "PLAYER_HIT"}]
    matches = one_to_one_matches(predictions, ground_truth, tolerance_frames=6, require_event_type=True)
    assert len(matches) == 1


def test_evaluator_uses_actual_fps_for_tolerance_and_milliseconds():
    for fps, difference in ((24.0, 5), (30.0, 6), (60.0, 12)):
        events = [{"frame": difference, "event_type": "BOUNCE"}]
        gt_events = [{"frame_best": 0, "event_type": "BOUNCE"}]
        metrics = evaluate_split_metrics([], [], gt_events, fps=fps, event_predictions=events)
        assert metrics["event_detection"]["true_positives"] == 1
        assert metrics["event_detection"]["matching_tolerance_ms"] == 200.0


def test_diagnostic_manifest_cannot_claim_qualification():
    with open("data/benchmarks/cross_match_final_holdout/videos.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["scientific_split"] == "CROSS_MATCH_DIAGNOSTIC"
    assert manifest["qualification_evidence"] is False


def test_rally_time_bounds_use_frames_and_do_not_exceed_terminal_frame():
    shot = ShotEventEvidence(
        shot_id=1,
        match_event_id=1,
        frame_index=300,
        timestamp_s=99.0,
        player_id=1,
        shot_type=ShotType.UNKNOWN,
        shot_confidence=0.0,
        classification_source=ShotClassificationSource.ABSTENTION_UNKNOWN,
        direction=ShotDirection.UNKNOWN,
        direction_confidence=0.0,
        direction_source="GEOMETRY_DERIVED",
        bounce_frame=360,
    )
    rally = RallyAnalyzer.analyze_rallies([shot], fps=30.0)[0]
    assert rally.start_time_s == 10.0
    assert rally.end_time_s == 12.0
    assert rally.duration_s == 2.0


def test_contract_boolean_is_not_serialized_as_integer():
    assert sanitize_json_value({"is_dead_ball": False}) == {"is_dead_ball": False}
    assert isinstance(sanitize_json_value({"value": True})["value"], bool)
