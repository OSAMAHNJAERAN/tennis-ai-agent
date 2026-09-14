"""Targeted regression specifications for Phase 6.4 event recovery.

The synthetic trajectories isolate candidate generation, physical verification,
semantic mapping, and temporal filtering.  They intentionally contain no
benchmark-frame lookups or scoring-state inputs.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pytest

from src.events.event_detector import EventCandidate, EventType, TennisEventDetector
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


FPS = 30.0
FRAME_SIZE_720P = (1280, 720)


def _trajectory(
    *,
    count: int = 90,
    fps: float = FPS,
    x: float = 150.0,
    y: float = 200.0,
    state: BallState = BallState.DETECTED,
) -> list[TemporalBallPoint]:
    return [
        TemporalBallPoint(
            frame_index=frame,
            timestamp_seconds=frame / fps,
            x_px=x,
            y_px=y,
            confidence=0.90,
            state=state,
        )
        for frame in range(count)
    ]


def _candidate(
    frame: int,
    *,
    fps: float = FPS,
    x: float = 150.0,
    y: float = 200.0,
    state: BallState = BallState.DETECTED,
    direction_change: float = 60.0,
    vertical_inversion: bool = False,
) -> EventCandidate:
    return EventCandidate(
        frame_index=frame,
        timestamp_s=frame / fps,
        score=0.90,
        ball_position_px=(x, y),
        trajectory_state=state.value,
        discovery_frame_index=frame,
        discovery_timestamp_s=frame / fps,
        evidence={
            "candidate_source": ["SYNTHETIC_TEST_KINEMATICS"],
            "pre_speed_px_s": 300.0,
            "post_speed_px_s": 320.0,
            "direction_change_degrees": direction_change,
            "acceleration_magnitude_px_s2": 0.0,
            "vertical_inversion": vertical_inversion,
            # Image y grows downward: a court rebound is down -> up.
            "court_rebound": vertical_inversion,
            "trajectory_continuity": True,
            "actual_fps": fps,
        },
    )


def _box(x1: float, y1: float, x2: float, y2: float) -> BBox:
    return BBox(x1, y1, x2, y2, confidence=0.95, class_id=0)


def _timeline(count: int, box: Optional[BBox]) -> list[Optional[BBox]]:
    return [box] * count


def _analyze_candidates(
    monkeypatch: pytest.MonkeyPatch,
    candidates: Iterable[EventCandidate],
    trajectory: list[TemporalBallPoint],
    *,
    p1_box: Optional[BBox] = None,
    p2_box: Optional[BBox] = None,
    frame_size: tuple[int, int] = FRAME_SIZE_720P,
    **settings: object,
):
    config = {
        "enable_event_time_refinement": False,
        "enable_serve_semantics": False,
        **settings,
    }
    detector = TennisEventDetector(config=config)
    candidate_list = list(candidates)
    monkeypatch.setattr(detector, "detect_candidates", lambda *args, **kwargs: candidate_list)
    return detector.analyze(
        trajectory,
        _timeline(len(trajectory), p1_box),
        _timeline(len(trajectory), p2_box),
        fps=1.0 / (trajectory[1].timestamp_seconds - trajectory[0].timestamp_seconds),
        frame_size=frame_size,
    )


def _horizontal_reversal(fps: float, *, frame_size: tuple[int, int] = FRAME_SIZE_720P):
    count = int(2.0 * fps) + 1
    points = _trajectory(count=count, fps=fps)
    for point in points:
        timestamp = point.timestamp_seconds
        point.x_px = 200.0 + 220.0 * timestamp if timestamp <= 1.0 else 420.0 - 220.0 * (timestamp - 1.0)
        point.y_px = 300.0
    return points


def _nearest_candidate_time(fps: float) -> float:
    detector = TennisEventDetector(config={"candidate_min_interval_seconds": 0.08})
    candidates = detector.detect_candidates(
        _horizontal_reversal(fps), fps=fps, frame_size=FRAME_SIZE_720P
    )
    assert candidates
    return min(candidates, key=lambda item: abs(item.timestamp_s - 1.0)).timestamp_s


def test_candidate_survives_generation_but_semantic_verifier_rejects(monkeypatch):
    trajectory = _trajectory()
    candidate = _candidate(30, vertical_inversion=False)
    analysis = _analyze_candidates(monkeypatch, [candidate], trajectory)

    assert len(analysis.candidates) == 1
    assert analysis.events == []
    assert analysis.verification_traces[0]["verification_pass"] is False


def test_candidate_generation_and_final_event_collections_are_distinct():
    detector = TennisEventDetector(config={"enable_event_time_refinement": False})
    trajectory = _horizontal_reversal(FPS)
    empty_boxes = _timeline(len(trajectory), None)

    analysis = detector.analyze(
        trajectory,
        empty_boxes,
        empty_boxes,
        fps=FPS,
        frame_size=FRAME_SIZE_720P,
    )

    assert analysis.candidates
    assert analysis.events == []


def test_rejected_candidate_trace_records_stage_and_reason(monkeypatch):
    trajectory = _trajectory()
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, vertical_inversion=False)],
        trajectory,
    )

    trace = analysis.verification_traces[0]
    assert trace["rejection_stage"] == "PHYSICS_VERIFICATION"
    assert trace["rejection_reason"] == "BOUNCE_CONTACT_SIGNATURE_REJECTION"
    assert trace["stage_pass"]["raw_candidate_generator"] is True
    assert trace["stage_pass"]["final_authoritative_event"] is False
    assert trace["final_event_emitted"] is False


def test_real_hit_survives_multi_cue_verification(monkeypatch):
    trajectory = _trajectory()
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert [event.event_type for event in analysis.events] == [EventType.PLAYER_1_HIT]
    trace = analysis.verification_traces[0]
    assert trace["player_contact_cues"]["temporal_player_proximity"] is True
    assert trace["player_contact_cues"]["trajectory_change"] is True
    assert trace["player_contact_cues"]["trajectory_continuity"] is True
    assert trace["verification_pass"] is True


def test_timing_drifted_hit_is_recovered_by_local_refinement(monkeypatch):
    contact_frame = 30
    discovery_frame = 27
    trajectory = _trajectory(x=220.0)
    for point in trajectory:
        if point.frame_index <= contact_frame:
            point.x_px = 100.0 + 4.0 * point.frame_index
        else:
            point.x_px = 220.0 - 12.0 * (point.frame_index - contact_frame)
        point.y_px = 200.0

    detector = TennisEventDetector(
        config={
            "enable_event_time_refinement": True,
            "enable_serve_semantics": False,
            "event_time_refinement_window_seconds": 0.14,
        }
    )
    candidate = _candidate(
        discovery_frame,
        x=float(trajectory[discovery_frame].x_px),
        y=200.0,
    )
    monkeypatch.setattr(detector, "detect_candidates", lambda *args, **kwargs: [candidate])
    boxes = _timeline(len(trajectory), _box(180.0, 100.0, 260.0, 300.0))
    analysis = detector.analyze(
        trajectory,
        boxes,
        _timeline(len(trajectory), None),
        fps=FPS,
        frame_size=FRAME_SIZE_720P,
    )

    assert len(analysis.events) == 1
    trace = analysis.verification_traces[0]
    assert trace["discovery_frame"] == discovery_frame
    assert abs(trace["refined_frame"] - contact_frame) <= 1
    assert abs(trace["refined_frame"] - contact_frame) < abs(discovery_frame - contact_frame)
    assert analysis.events[0].timestamp_s == pytest.approx(
        analysis.events[0].frame_index / FPS
    )


def test_near_court_player_hit_is_preserved(monkeypatch):
    trajectory = _trajectory(x=205.0, y=250.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=205.0, y=250.0)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 400.0),
    )

    assert analysis.events[0].event_type == EventType.PLAYER_1_HIT
    assert analysis.events[0].player_id == 1


def test_far_court_player_hit_is_preserved(monkeypatch):
    trajectory = _trajectory(x=855.0, y=150.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=855.0, y=150.0)],
        trajectory,
        p2_box=_box(800.0, 100.0, 850.0, 220.0),
    )

    assert analysis.events[0].event_type == EventType.PLAYER_2_HIT
    assert analysis.events[0].player_id == 2


def test_bounce_far_from_players_is_preserved(monkeypatch):
    trajectory = _trajectory(x=640.0, y=500.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=640.0, y=500.0, vertical_inversion=True)],
        trajectory,
    )

    assert analysis.events[0].event_type == EventType.BOUNCE
    assert analysis.events[0].player_id is None


def test_court_like_bounce_near_player_is_not_forced_to_hit(monkeypatch):
    trajectory = _trajectory(x=150.0, y=310.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=150.0, y=310.0, vertical_inversion=True)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert analysis.events[0].event_type == EventType.BOUNCE


def test_pre_serve_ritual_bounce_is_not_serve_contact(monkeypatch):
    trajectory = _trajectory(x=150.0, y=310.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(20, x=150.0, y=310.0, vertical_inversion=True)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
        enable_serve_semantics=True,
    )

    assert analysis.events
    assert all(event.event_type != EventType.SERVE_CONTACT for event in analysis.events)
    assert analysis.events[0].event_type == EventType.BOUNCE


def test_real_serve_contact_is_preserved(monkeypatch):
    contact_frame = 30
    trajectory = _trajectory(count=70, x=150.0, y=240.0)
    for point in trajectory:
        frame = point.frame_index
        if frame < contact_frame:
            point.x_px = 150.0
            point.y_px = 260.0 - 5.0 * max(0, frame - 18)
        else:
            point.x_px = 150.0 + 35.0 * (frame - contact_frame)
            point.y_px = 195.0

    detector = TennisEventDetector(
        config={"enable_event_time_refinement": False, "enable_serve_semantics": True}
    )
    candidate = _candidate(contact_frame, x=150.0, y=195.0)
    monkeypatch.setattr(detector, "detect_candidates", lambda *args, **kwargs: [candidate])
    analysis = detector.analyze(
        trajectory,
        _timeline(len(trajectory), _box(100.0, 100.0, 200.0, 310.0)),
        _timeline(len(trajectory), None),
        fps=FPS,
        frame_size=FRAME_SIZE_720P,
    )

    assert analysis.events[0].event_type == EventType.SERVE_CONTACT
    serve = analysis.verification_traces[0]["serve_evidence"]
    assert serve["overhead_contact"] is True
    assert serve["toss_like_motion"] is True
    assert serve["post_contact_rapid_departure"] is True


def test_post_rally_low_energy_rolling_ball_is_suppressed(monkeypatch):
    trajectory = _trajectory()
    rolling = _candidate(25, direction_change=0.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [rolling],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert analysis.events == []
    assert analysis.verification_traces[0]["rejection_stage"] == "VISION_ACTIVITY_FILTER"
    assert analysis.verification_traces[0]["activity_state"] == "DEAD_BALL"


def test_dead_ball_candidate_does_not_suppress_next_rally_contact(monkeypatch):
    trajectory = _trajectory(count=100)
    rolling = _candidate(20, direction_change=0.0)
    next_contact = _candidate(50, direction_change=70.0)
    analysis = _analyze_candidates(
        monkeypatch,
        [rolling, next_contact],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert [event.frame_index for event in analysis.events] == [50]
    assert analysis.verification_traces[0]["activity_state"] == "DEAD_BALL"
    assert analysis.verification_traces[1]["final_event_emitted"] is True


def test_camera_pan_reversal_does_not_become_bounce():
    fps = FPS
    count = 61
    offsets = []
    trajectory = _trajectory(count=count, fps=fps, x=500.0, y=300.0)
    for point in trajectory:
        frame = point.frame_index
        offset_x = 10.0 * frame if frame <= 30 else 300.0 - 10.0 * (frame - 30)
        offsets.append((offset_x, 0.0))
        point.x_px = 500.0 + offset_x
        point.y_px = 300.0

    detector = TennisEventDetector()
    raw_candidates = detector.detect_candidates(
        trajectory, fps=fps, frame_size=FRAME_SIZE_720P
    )
    compensated_candidates = detector.detect_candidates(
        trajectory,
        fps=fps,
        frame_size=FRAME_SIZE_720P,
        camera_offsets_px=offsets,
    )

    assert raw_candidates
    assert compensated_candidates == []


def test_24_fps_candidate_timing_is_time_based():
    assert _nearest_candidate_time(24.0) == pytest.approx(1.0, abs=1.0 / 24.0)


def test_30_fps_candidate_timing_is_time_based():
    assert _nearest_candidate_time(30.0) == pytest.approx(1.0, abs=1.0 / 30.0)


def test_60_fps_candidate_timing_is_time_based():
    assert _nearest_candidate_time(60.0) == pytest.approx(1.0, abs=1.0 / 60.0)


def test_720p_and_1080p_player_reach_normalize_equivalently(monkeypatch):
    trajectory_720 = _trajectory(x=210.0, y=200.0)
    result_720 = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=210.0, y=200.0)],
        trajectory_720,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
        frame_size=(1280, 720),
    )

    scale = 1.5
    trajectory_1080 = _trajectory(x=210.0 * scale, y=200.0 * scale)
    result_1080 = _analyze_candidates(
        monkeypatch,
        [_candidate(30, x=210.0 * scale, y=200.0 * scale)],
        trajectory_1080,
        p1_box=_box(100.0 * scale, 100.0 * scale, 200.0 * scale, 300.0 * scale),
        frame_size=(1920, 1080),
    )

    assert result_720.events[0].event_type == result_1080.events[0].event_type
    assert result_720.verification_traces[0]["player1_distance_normalized"] == pytest.approx(
        result_1080.verification_traces[0]["player1_distance_normalized"]
    )


def test_predicted_trajectory_reduces_confidence_without_forcing_rejection(monkeypatch):
    detected_trajectory = _trajectory(state=BallState.DETECTED)
    detected = _analyze_candidates(
        monkeypatch,
        [_candidate(30, state=BallState.DETECTED)],
        detected_trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )
    predicted_trajectory = _trajectory(state=BallState.PREDICTED)
    predicted = _analyze_candidates(
        monkeypatch,
        [_candidate(30, state=BallState.PREDICTED)],
        predicted_trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert len(predicted.events) == 1
    assert predicted.events[0].confidence < detected.events[0].confidence


def test_short_interpolated_contact_is_handled_safely(monkeypatch):
    trajectory = _trajectory()
    for frame in (29, 30, 31):
        trajectory[frame].state = BallState.INTERPOLATED
        trajectory[frame].confidence = None
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, state=BallState.INTERPOLATED)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert len(analysis.events) == 1
    assert analysis.events[0].trajectory_state == BallState.INTERPOLATED.value
    assert 0.0 < analysis.events[0].confidence < 1.0


def test_missing_state_cannot_create_physical_event(monkeypatch):
    trajectory = _trajectory()
    trajectory[30].state = BallState.MISSING
    trajectory[30].confidence = None
    analysis = _analyze_candidates(
        monkeypatch,
        [_candidate(30, state=BallState.MISSING)],
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert analysis.events == []
    assert analysis.verification_traces[0]["rejection_reason"] == (
        "MISSING_STATE_CANNOT_CREATE_PHYSICAL_EVENT"
    )


def test_duplicate_candidates_are_suppressed_after_verification(monkeypatch):
    trajectory = _trajectory()
    candidates = [_candidate(30), _candidate(32)]
    analysis = _analyze_candidates(
        monkeypatch,
        candidates,
        trajectory,
        p1_box=_box(100.0, 100.0, 200.0, 300.0),
    )

    assert len(analysis.events) == 1
    rejected = [
        trace for trace in analysis.verification_traces
        if trace["rejection_stage"] == "TEMPORAL_SUPPRESSION"
    ]
    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"] == "DUPLICATE_VERIFIED_EVENT"
