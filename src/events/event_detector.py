"""Staged, auditable verification of physical tennis events.

Candidates are intentionally untyped.  Physical verification, player
attribution, tennis semantics, and temporal suppression are distinct stages.
Unavailable evidence remains ``None`` rather than receiving a fabricated value.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


class EventType(Enum):
    SERVE_CONTACT = "SERVE_CONTACT"
    PLAYER_1_HIT = "PLAYER_1_HIT"
    PLAYER_2_HIT = "PLAYER_2_HIT"
    BOUNCE = "BOUNCE"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"


class PhysicalEventType(Enum):
    PLAYER_CONTACT = "BALL_CONTACT_WITH_PLAYER"
    COURT_CONTACT = "BALL_CONTACT_WITH_COURT"
    UNKNOWN = "UNKNOWN_PHYSICAL_EVENT"


@dataclass
class TennisEvent:
    event_id: int
    event_type: EventType
    frame_index: int
    timestamp_s: float
    player_id: Optional[int]
    ball_position_px: Tuple[float, float]
    court_position_m: Optional[Tuple[float, float]]
    confidence: float
    trajectory_state: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventCandidate:
    frame_index: int
    timestamp_s: float
    score: float
    ball_position_px: Tuple[float, float]
    trajectory_state: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    discovery_frame_index: Optional[int] = None
    discovery_timestamp_s: Optional[float] = None


@dataclass
class EventDetectorSettings:
    # Independently runnable diagnostic switches.
    enable_camera_motion_compensation: bool = True
    enable_player_temporal_proximity: bool = True
    enable_velocity_discontinuity: bool = True
    enable_direction_change: bool = True
    enable_pose_support: bool = False
    enable_bounce_contact_check: bool = True
    enable_bounce_verification: bool = True
    enable_dead_ball_gating: bool = True
    enable_serve_semantics: bool = True
    enable_serve_context: bool = True
    enable_event_time_refinement: bool = True
    enable_provenance_weighting: bool = True
    enable_contact_family_stage: bool = True
    enable_player_attribution_fusion: bool = True
    enable_semantic_unknown_abstention: bool = True
    enable_camera_motion_event_features: bool = True

    # Seconds and frame-diagonal-normalized kinematics.
    feature_half_window_seconds: float = 0.067
    candidate_min_interval_seconds: float = 0.100
    candidate_score_threshold: float = 0.28
    candidate_min_normalized_speed_per_s: float = 0.005
    candidate_min_direction_change_deg: float = 12.0
    candidate_min_normalized_velocity_change_per_s: float = 0.040
    candidate_min_normalized_acceleration_per_s2: float = 0.35
    candidate_min_normalized_vertical_speed_per_s: float = 0.015
    candidate_min_continuity_ratio: float = 0.60
    max_normalized_edge_speed_per_s: float = 3.00

    # Player distance is divided by the time-local bbox height.
    temporal_proximity_window_seconds: float = 0.140
    event_time_refinement_window_seconds: float = 0.120
    player_reach_normalized: float = 1.35
    player_attribution_margin_normalized: float = 0.15
    racket_region_top_normalized: float = -0.35
    racket_region_bottom_normalized: float = 0.85

    player_contact_min_score: float = 0.38
    player_contact_min_independent_cues: int = 3
    court_contact_min_score: float = 0.36
    court_contact_min_independent_cues: int = 3
    physical_type_margin: float = 0.05
    departure_window_seconds: float = 0.120

    serve_evidence_window_seconds: float = 0.40
    serve_min_score: float = 0.58
    serve_overhead_y_normalized: float = 0.48
    serve_min_post_speed_normalized_per_s: float = 0.075
    serve_min_departure_speed_gain: float = 1.05

    final_event_min_interval_seconds: float = 0.100
    rolling_max_normalized_speed_per_s: float = 0.030
    dead_ball_min_kinematic_support: float = 0.20

    state_weights: Dict[str, float] = field(default_factory=lambda: {
        BallState.DETECTED.value: 1.00,
        BallState.TRACKED.value: 0.90,
        BallState.INTERPOLATED.value: 0.70,
        BallState.PREDICTED.value: 0.55,
        BallState.OCCLUDED.value: 0.45,
        BallState.MISSING.value: 0.00,
    })


@dataclass
class EventDetectionAnalysis:
    candidates: List[EventCandidate]
    events: List[TennisEvent]
    verification_traces: List[Dict[str, Any]]
    settings: Dict[str, Any]
    coordinate_system: Dict[str, Any]


@dataclass
class _Feature:
    frame: int
    timestamp: float
    raw_position: Optional[Tuple[float, float]]
    feature_position: Optional[Tuple[float, float]]
    state: str
    confidence: Optional[float]
    pre_velocity: Optional[Tuple[float, float]] = None
    post_velocity: Optional[Tuple[float, float]] = None
    pre_speed: Optional[float] = None
    post_speed: Optional[float] = None
    normalized_speed: Optional[float] = None
    normalized_velocity_change: Optional[float] = None
    normalized_acceleration: Optional[float] = None
    direction_change: Optional[float] = None
    normalized_curvature: Optional[float] = None
    vertical_inversion: Optional[bool] = None
    court_rebound: Optional[bool] = None
    continuity: float = 0.0
    edge_continuity: bool = False
    score: float = 0.0
    strengths: Dict[str, float] = field(default_factory=dict)


class TennisEventDetector:
    """Candidate -> physical contact -> semantic event detector."""

    _SETTING_NAMES = {item.name for item in fields(EventDetectorSettings)}

    def __init__(
        self,
        min_event_interval_frames: Optional[int] = None,
        min_event_interval_seconds: Optional[float] = None,
        player_reach_radius_px: float = 160.0,
        min_hit_deflection_deg: Optional[float] = None,
        min_bounce_curvature: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        **overrides: Any,
    ) -> None:
        supplied = dict(config or {})
        supplied.update(overrides)
        if min_event_interval_seconds is not None:
            supplied.setdefault("candidate_min_interval_seconds", float(min_event_interval_seconds))
            supplied.setdefault("final_event_min_interval_seconds", float(min_event_interval_seconds))
        elif min_event_interval_frames is not None:
            interval = float(min_event_interval_frames) / 30.0
            supplied.setdefault("candidate_min_interval_seconds", interval)
            supplied.setdefault("final_event_min_interval_seconds", interval)
        if min_hit_deflection_deg is not None:
            supplied.setdefault("candidate_min_direction_change_deg", float(min_hit_deflection_deg))

        ignored_legacy = {
            "min_event_interval", "suppression_radius", "reach_scale_factor",
            "bounce_elevation_threshold", "player_reach_radius_px", "min_bounce_curvature",
        }
        unknown = set(supplied) - self._SETTING_NAMES - ignored_legacy - {"min_hit_deflection_deg"}
        if unknown:
            raise ValueError(f"Unknown event detector settings: {sorted(unknown)}")
        self.settings = EventDetectorSettings()
        for key, value in supplied.items():
            if key in self._SETTING_NAMES:
                setattr(self.settings, key, value)
            elif key == "min_hit_deflection_deg":
                self.settings.candidate_min_direction_change_deg = float(value)
        self.player_reach_radius_px = float(player_reach_radius_px)  # legacy introspection only
        self.min_hit_deflection_deg = self.settings.candidate_min_direction_change_deg
        self.min_bounce_curvature = min_bounce_curvature
        self.last_analysis: Optional[EventDetectionAnalysis] = None

    @staticmethod
    def _clip(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _distance(point: Tuple[float, float], box: Optional[BBox]) -> float:
        if box is None:
            return float("inf")
        x = max(box.x1, min(point[0], box.x2))
        y = max(box.y1, min(point[1], box.y2))
        return math.hypot(point[0] - x, point[1] - y)

    @staticmethod
    def _frame_size(
        trajectory: Sequence[TemporalBallPoint], supplied: Optional[Tuple[int, int]]
    ) -> Tuple[int, int, str]:
        if supplied is not None:
            width, height = int(supplied[0]), int(supplied[1])
            if width <= 0 or height <= 0:
                raise ValueError("frame_size must contain positive dimensions")
            return width, height, "VIDEO_METADATA"
        xs = [float(p.x_px) for p in trajectory if p.x_px is not None]
        ys = [float(p.y_px) for p in trajectory if p.y_px is not None]
        return (
            max(1280, int(math.ceil(max(xs, default=1279.0) + 1))),
            max(720, int(math.ceil(max(ys, default=719.0) + 1))),
            "INFERRED_LOWER_BOUND_FOR_LEGACY_CALLER",
        )

    def _positions(
        self,
        trajectory: Sequence[TemporalBallPoint],
        offsets: Optional[Sequence[Optional[Tuple[float, float]]]],
    ) -> Tuple[List[Optional[Tuple[float, float]]], str]:
        raw = [
            (float(p.x_px), float(p.y_px)) if p.x_px is not None and p.y_px is not None else None
            for p in trajectory
        ]
        if not self.settings.enable_camera_motion_compensation:
            return raw, "DISABLED"
        if offsets is None:
            return raw, "UNAVAILABLE_NO_PER_FRAME_STATIC_SCENE_TRANSFORM"
        if len(offsets) != len(raw):
            raise ValueError("camera_offsets_px must align with trajectory")
        result = []
        for position, offset in zip(raw, offsets):
            if position is None or offset is None:
                result.append(position)
            else:
                result.append((position[0] - float(offset[0]), position[1] - float(offset[1])))
        return result, "GLOBAL_TRANSLATION_COMPENSATED_STATIC_SCENE_INPUT"

    @staticmethod
    def _nearest_index(timestamps: Sequence[float], target: float) -> int:
        return min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - target))

    @staticmethod
    def _velocity(
        first: Optional[Tuple[Optional[float], Optional[float]]],
        second: Optional[Tuple[Optional[float], Optional[float]]],
        first_time: float,
        second_time: float,
    ) -> Optional[Tuple[float, float]]:
        if (
            first is None or second is None
            or first[0] is None or first[1] is None
            or second[0] is None or second[1] is None
            or second_time <= first_time
        ):
            return None
        dt = second_time - first_time
        return ((float(second[0]) - float(first[0])) / dt, (float(second[1]) - float(first[1])) / dt)

    def _features(
        self,
        trajectory: Sequence[TemporalBallPoint],
        frame_size: Tuple[int, int],
        positions: Sequence[Optional[Tuple[float, float]]],
    ) -> List[_Feature]:
        timestamps = [float(p.timestamp_seconds) for p in trajectory]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("trajectory timestamps must be strictly increasing")
        diagonal = math.hypot(*frame_size)
        half_window = self.settings.feature_half_window_seconds
        result: List[_Feature] = []
        for index, point in enumerate(trajectory):
            raw = (
                (float(point.x_px), float(point.y_px))
                if point.x_px is not None and point.y_px is not None else None
            )
            feature = _Feature(
                index, timestamps[index], raw, positions[index], point.state.value,
                float(point.confidence) if point.confidence is not None else None,
            )
            if raw is None or point.state == BallState.MISSING:
                result.append(feature)
                continue
            before_i = self._nearest_index(timestamps, timestamps[index] - half_window)
            after_i = self._nearest_index(timestamps, timestamps[index] + half_window)
            feature.pre_velocity = self._velocity(
                positions[before_i], positions[index], timestamps[before_i], timestamps[index]
            )
            feature.post_velocity = self._velocity(
                positions[index], positions[after_i], timestamps[index], timestamps[after_i]
            )
            if feature.pre_velocity:
                feature.pre_speed = math.hypot(*feature.pre_velocity)
            if feature.post_velocity:
                feature.post_speed = math.hypot(*feature.post_velocity)
            speeds = [v for v in (feature.pre_speed, feature.post_speed) if v is not None]
            if speeds:
                feature.normalized_speed = sum(speeds) / len(speeds) / diagonal
            feature.edge_continuity = (
                feature.pre_speed is not None
                and feature.post_speed is not None
                and feature.pre_speed / diagonal <= self.settings.max_normalized_edge_speed_per_s
                and feature.post_speed / diagonal <= self.settings.max_normalized_edge_speed_per_s
            )
            if feature.pre_velocity and feature.post_velocity and feature.edge_continuity:
                dv = (
                    feature.post_velocity[0] - feature.pre_velocity[0],
                    feature.post_velocity[1] - feature.pre_velocity[1],
                )
                dv_mag = math.hypot(*dv)
                feature.normalized_velocity_change = dv_mag / diagonal
                dt = max(1e-9, 0.5 * (timestamps[after_i] - timestamps[before_i]))
                feature.normalized_acceleration = dv_mag / dt / diagonal
                pre_mag, post_mag = math.hypot(*feature.pre_velocity), math.hypot(*feature.post_velocity)
                if pre_mag > 1e-6 and post_mag > 1e-6:
                    cosine = max(-1.0, min(1.0, (
                        feature.pre_velocity[0] * feature.post_velocity[0]
                        + feature.pre_velocity[1] * feature.post_velocity[1]
                    ) / (pre_mag * post_mag)))
                    feature.direction_change = math.degrees(math.acos(cosine))
                    cross = abs(feature.pre_velocity[0] * dv[1] - feature.pre_velocity[1] * dv[0])
                    feature.normalized_curvature = cross * diagonal / max(pre_mag ** 3, 1e-9)
                vertical_min = self.settings.candidate_min_normalized_vertical_speed_per_s * diagonal
                feature.vertical_inversion = (
                    feature.pre_velocity[1] * feature.post_velocity[1] < 0
                    and abs(feature.pre_velocity[1]) >= vertical_min
                    and abs(feature.post_velocity[1]) >= vertical_min
                )
                feature.court_rebound = (
                    feature.pre_velocity[1] >= vertical_min
                    and feature.post_velocity[1] <= -vertical_min
                )
            local = range(
                bisect_left(timestamps, timestamps[index] - 2 * half_window),
                bisect_right(timestamps, timestamps[index] + 2 * half_window),
            )
            feature.continuity = (
                sum(positions[i] is not None for i in local) / len(local) if local else 0.0
            )
            speed_s = self._clip((feature.normalized_speed or 0.0) / self.settings.candidate_min_normalized_speed_per_s)
            direction_s = (
                self._clip((feature.direction_change or 0.0) / (2 * self.settings.candidate_min_direction_change_deg))
                if self.settings.enable_direction_change else 0.0
            )
            velocity_s = (
                self._clip((feature.normalized_velocity_change or 0.0) / (2 * self.settings.candidate_min_normalized_velocity_change_per_s))
                if self.settings.enable_velocity_discontinuity else 0.0
            )
            acceleration_s = self._clip(
                (feature.normalized_acceleration or 0.0) / (2 * self.settings.candidate_min_normalized_acceleration_per_s2)
            )
            vertical_s = 1.0 if feature.vertical_inversion else 0.0
            continuity_s = (
                self._clip(feature.continuity / self.settings.candidate_min_continuity_ratio)
                if feature.edge_continuity else 0.0
            )
            kinematic_s = max(direction_s, velocity_s, acceleration_s, vertical_s)
            feature.strengths = {
                "speed": speed_s, "direction_change": direction_s,
                "velocity_discontinuity": velocity_s, "acceleration": acceleration_s,
                "vertical_inversion": vertical_s, "continuity": continuity_s,
                "kinematic": kinematic_s,
            }
            feature.score = 0.68 * kinematic_s + 0.17 * speed_s + 0.15 * continuity_s
            result.append(feature)
        return result

    def _candidate(self, feature: _Feature, fps: float, camera_state: str, diagonal: float) -> EventCandidate:
        assert feature.raw_position is not None
        sources = [name for name, strength in feature.strengths.items() if name not in {"speed", "continuity", "kinematic"} and strength >= 0.5]
        return EventCandidate(
            frame_index=feature.frame,
            timestamp_s=feature.timestamp,
            score=feature.score,
            ball_position_px=feature.raw_position,
            trajectory_state=feature.state,
            discovery_frame_index=feature.frame,
            discovery_timestamp_s=feature.timestamp,
            evidence={
                "candidate_source": sources or ["WEAK_COMBINED_KINEMATICS"],
                "pre_velocity_px_s": list(feature.pre_velocity) if feature.pre_velocity else None,
                "post_velocity_px_s": list(feature.post_velocity) if feature.post_velocity else None,
                "pre_speed_px_s": feature.pre_speed,
                "post_speed_px_s": feature.post_speed,
                "normalized_speed_per_s": feature.normalized_speed,
                "normalized_velocity_change_per_s": feature.normalized_velocity_change,
                "normalized_acceleration_per_s2": feature.normalized_acceleration,
                "direction_change_degrees": feature.direction_change,
                "normalized_curvature": feature.normalized_curvature,
                "vertical_inversion": feature.vertical_inversion,
                "court_rebound": feature.court_rebound,
                "trajectory_continuity_ratio": feature.continuity,
                "edge_continuity": feature.edge_continuity,
                "trajectory_continuity": feature.continuity >= self.settings.candidate_min_continuity_ratio,
                "ball_confidence": feature.confidence,
                "actual_fps": float(fps),
                "frame_diagonal_px": diagonal,
                "camera_motion_state": camera_state,
                "feature_coordinate_system": "COMPENSATED_IMAGE_PLANE" if camera_state.startswith("GLOBAL_") else "RAW_IMAGE_PLANE",
                "scientific_limit": "Camera compensation does not make airborne ball coordinates metric court-plane positions.",
            },
        )

    def detect_candidates(
        self,
        ball_trajectory: List[TemporalBallPoint],
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        camera_offsets_px: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> List[EventCandidate]:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(ball_trajectory) < 3:
            return []
        width, height, _ = self._frame_size(ball_trajectory, frame_size)
        positions, camera_state = self._positions(ball_trajectory, camera_offsets_px)
        features = self._features(ball_trajectory, (width, height), positions)
        eligible = [
            item for item in features
            if item.raw_position is not None
            and item.state != BallState.MISSING.value
            and item.score >= self.settings.candidate_score_threshold
            and (item.normalized_speed or 0.0) >= self.settings.candidate_min_normalized_speed_per_s
        ]
        selected: List[_Feature] = []
        for item in sorted(
            eligible,
            # Saturated strengths still need a physically meaningful ranking;
            # otherwise an arbitrary plateau edge wins and shifts event time.
            key=lambda row: (
                -row.score,
                -(1 if row.vertical_inversion else 0),
                -(row.direction_change or 0.0),
                -(row.normalized_velocity_change or 0.0),
                -(row.normalized_acceleration or 0.0),
                row.timestamp,
            ),
        ):
            if all(abs(item.timestamp - kept.timestamp) > self.settings.candidate_min_interval_seconds for kept in selected):
                selected.append(item)
        selected.sort(key=lambda row: row.timestamp)
        diagonal = math.hypot(width, height)
        return [self._candidate(item, fps, camera_state, diagonal) for item in selected]

    def _player_features(
        self,
        frame: int,
        trajectory: Sequence[TemporalBallPoint],
        p1_boxes: Sequence[Optional[BBox]],
        p2_boxes: Sequence[Optional[BBox]],
        window_seconds: float,
    ) -> Dict[str, Any]:
        timestamp = trajectory[frame].timestamp_seconds
        timestamps = [point.timestamp_seconds for point in trajectory]
        left = bisect_left(timestamps, timestamp - window_seconds)
        right = bisect_right(timestamps, timestamp + window_seconds)
        records: Dict[int, Dict[str, Any]] = {}
        for player_id, boxes in ((1, p1_boxes), (2, p2_boxes)):
            best: Optional[Tuple[float, float, int, float, float]] = None
            for index in range(left, right):
                point = trajectory[index]
                box = boxes[index] if index < len(boxes) else None
                if point.x_px is None or point.y_px is None or box is None:
                    continue
                height = max(1.0, box.y2 - box.y1)
                distance_px = self._distance((float(point.x_px), float(point.y_px)), box)
                option = (
                    distance_px / height,
                    abs(point.timestamp_seconds - timestamp),
                    index,
                    distance_px,
                    (float(point.y_px) - box.y1) / height,
                )
                if best is None or option < best:
                    best = option
            if best is None:
                records[player_id] = {
                    "distance_normalized": None, "distance_px": None,
                    "frame_index": None, "player_scale_px": None, "normalized_y": None,
                }
            else:
                matched_box = boxes[best[2]]
                assert matched_box is not None
                records[player_id] = {
                    "distance_normalized": best[0], "distance_px": best[3],
                    "frame_index": best[2], "player_scale_px": max(1.0, matched_box.y2 - matched_box.y1),
                    "normalized_y": best[4],
                }
        available = sorted(
            (record["distance_normalized"], player_id)
            for player_id, record in records.items()
            if record["distance_normalized"] is not None
        )
        selected_player = available[0][1] if available else None
        if available:
            first = available[0][0]
            second = available[1][0] if len(available) > 1 else float("inf")
            clear = (
                first <= self.settings.player_reach_normalized
                and second - first >= self.settings.player_attribution_margin_normalized
            )
        else:
            clear = False
        return {"players": records, "selected_player": selected_player, "attribution_clear": clear}

    def _legacy_feature_fallback(self, feature: _Feature, candidate: EventCandidate, diagonal: float) -> _Feature:
        """Support persisted/monkeypatched candidates without inventing missing data."""
        evidence = candidate.evidence
        item = replace(feature, strengths=dict(feature.strengths))
        if item.pre_speed is None:
            item.pre_speed = evidence.get("pre_speed_px_s")
        if item.post_speed is None:
            item.post_speed = evidence.get("post_speed_px_s")
        if item.normalized_speed is None and (item.pre_speed is not None or item.post_speed is not None):
            speeds = [v for v in (item.pre_speed, item.post_speed) if v is not None]
            item.normalized_speed = sum(speeds) / len(speeds) / diagonal
        if item.direction_change is None:
            item.direction_change = evidence.get("direction_change_degrees")
        raw_acceleration = evidence.get("acceleration_magnitude_px_s2")
        if item.normalized_acceleration is None and raw_acceleration is not None:
            item.normalized_acceleration = float(raw_acceleration) / diagonal
        if evidence.get("vertical_inversion") is True:
            item.vertical_inversion = True
        if evidence.get("court_rebound") is True:
            item.court_rebound = True
        if item.continuity <= 0 and evidence.get("trajectory_continuity"):
            item.continuity = 1.0
        if evidence.get("edge_continuity") is True:
            item.edge_continuity = True
        if not item.strengths:
            item.strengths = {}
        item.strengths.update({
            "speed": max(item.strengths.get("speed", 0.0), self._clip((item.normalized_speed or 0.0) / self.settings.candidate_min_normalized_speed_per_s)),
            "direction_change": max(item.strengths.get("direction_change", 0.0), self._clip((item.direction_change or 0.0) / (2 * self.settings.candidate_min_direction_change_deg))),
            "velocity_discontinuity": item.strengths.get("velocity_discontinuity", 0.0),
            "acceleration": max(item.strengths.get("acceleration", 0.0), self._clip((item.normalized_acceleration or 0.0) / (2 * self.settings.candidate_min_normalized_acceleration_per_s2))),
            "vertical_inversion": max(item.strengths.get("vertical_inversion", 0.0), 1.0 if item.vertical_inversion else 0.0),
            "continuity": max(item.strengths.get("continuity", 0.0), self._clip(item.continuity / self.settings.candidate_min_continuity_ratio)),
        })
        item.strengths["kinematic"] = max(
            item.strengths.get("direction_change", 0.0),
            item.strengths.get("velocity_discontinuity", 0.0),
            item.strengths.get("acceleration", 0.0),
            item.strengths.get("vertical_inversion", 0.0),
        )
        return item

    def _refined_frame(
        self,
        candidate: EventCandidate,
        features: Sequence[_Feature],
        trajectory: Sequence[TemporalBallPoint],
        p1_boxes: Sequence[Optional[BBox]],
        p2_boxes: Sequence[Optional[BBox]],
    ) -> int:
        if not self.settings.enable_event_time_refinement:
            return candidate.frame_index
        options: List[Tuple[float, float, int, int]] = []
        timestamps = [item.timestamp for item in features]
        left = bisect_left(timestamps, candidate.timestamp_s - self.settings.event_time_refinement_window_seconds)
        right = bisect_right(timestamps, candidate.timestamp_s + self.settings.event_time_refinement_window_seconds)
        for item in features[left:right]:
            if item.raw_position is None:
                continue
            player = self._player_features(
                item.frame, trajectory, p1_boxes, p2_boxes,
                self.settings.temporal_proximity_window_seconds,
            )
            distances = [
                row["distance_normalized"] for row in player["players"].values()
                if row["distance_normalized"] is not None
            ]
            proximity = self._clip(1 - min(distances) / self.settings.player_reach_normalized) if distances else 0.0
            combined = 0.78 * item.score + 0.22 * proximity
            options.append((combined, -abs(item.timestamp - candidate.timestamp_s), -item.frame, item.frame))
        return max(options)[3] if options else candidate.frame_index

    def _departure(
        self,
        frame: int,
        player_id: Optional[int],
        trajectory: Sequence[TemporalBallPoint],
        boxes: Sequence[Optional[BBox]],
    ) -> float:
        if player_id is None:
            return 0.0
        target = trajectory[frame].timestamp_seconds + self.settings.departure_window_seconds
        index = min(range(len(trajectory)), key=lambda i: abs(trajectory[i].timestamp_seconds - target))
        point, box = trajectory[index], boxes[index] if index < len(boxes) else None
        if point.x_px is None or point.y_px is None or box is None:
            return 0.0
        distance = self._distance((point.x_px, point.y_px), box) / max(1.0, box.y2 - box.y1)
        return self._clip(distance / self.settings.player_reach_normalized)

    @staticmethod
    def _feature_dict(item: _Feature) -> Dict[str, Any]:
        return {
            "pre_velocity_px_s": list(item.pre_velocity) if item.pre_velocity else None,
            "post_velocity_px_s": list(item.post_velocity) if item.post_velocity else None,
            "pre_speed_px_s": item.pre_speed,
            "post_speed_px_s": item.post_speed,
            "normalized_speed_per_s": item.normalized_speed,
            "normalized_velocity_change_per_s": item.normalized_velocity_change,
            "normalized_acceleration_per_s2": item.normalized_acceleration,
            "direction_change_degrees": item.direction_change,
            "normalized_curvature": item.normalized_curvature,
            "vertical_inversion": item.vertical_inversion,
            "court_rebound": item.court_rebound,
            "trajectory_continuity_ratio": item.continuity,
            "edge_continuity": item.edge_continuity,
            "cue_strengths": dict(item.strengths),
        }

    def _serve_evidence(
        self,
        frame: int,
        item: _Feature,
        selected: Dict[str, Any],
        trajectory: Sequence[TemporalBallPoint],
        frame_height: int,
        frame_diagonal: float,
    ) -> Dict[str, Any]:
        normalized_y = selected.get("normalized_y")
        overhead = normalized_y is not None and normalized_y <= self.settings.serve_overhead_y_normalized
        post_normalized = (item.post_speed or 0.0) / max(frame_diagonal, 1.0)
        speed_gain = (item.post_speed or 0.0) / max(item.pre_speed or 0.0, 1e-9)
        fast = post_normalized >= self.settings.serve_min_post_speed_normalized_per_s
        timestamp = trajectory[frame].timestamp_seconds
        timestamps = [point.timestamp_seconds for point in trajectory]
        left = bisect_left(timestamps, timestamp - self.settings.serve_evidence_window_seconds)
        right = bisect_left(timestamps, timestamp)
        prior = [
            point for point in trajectory[left:right]
            if point.timestamp_seconds < timestamp
            and point.x_px is not None and point.y_px is not None
        ]
        toss = len(prior) >= 2 and (float(prior[0].y_px) - float(prior[-1].y_px)) > 0.008 * frame_height
        continuity = item.continuity >= self.settings.candidate_min_continuity_ratio
        quality = self.settings.state_weights.get(item.state, 0.0)
        score = 0.35 * (1.0 if overhead else 0.0) + 0.25 * (1.0 if fast else 0.0) + 0.25 * (1.0 if toss else 0.0) + 0.15 * quality
        return {
            "score": score,
            "overhead_contact": overhead,
            "post_contact_rapid_departure": fast,
            "toss_like_motion": toss,
            "trajectory_continuity": continuity,
            "normalized_post_speed_per_s": post_normalized,
            "departure_speed_gain": speed_gain,
        }

    def _verify(
        self,
        candidate_id: int,
        candidate: EventCandidate,
        item: _Feature,
        trajectory: Sequence[TemporalBallPoint],
        p1_boxes: Sequence[Optional[BBox]],
        p2_boxes: Sequence[Optional[BBox]],
        pose_support: Optional[Mapping[int, Mapping[int, float]]],
        camera_state: str,
        frame_height: int,
        frame_diagonal: float,
    ) -> Tuple[Optional[TennisEvent], Dict[str, Any]]:
        point = trajectory[item.frame]
        player = self._player_features(
            item.frame,
            trajectory,
            p1_boxes,
            p2_boxes,
            self.settings.temporal_proximity_window_seconds
            if self.settings.enable_player_temporal_proximity else 0.0,
        )
        selected_player = player["selected_player"]
        selected = player["players"].get(selected_player, {}) if selected_player else {}
        distance = selected.get("distance_normalized")
        normalized_y = selected.get("normalized_y")
        proximity = (
            self._clip(1.0 - distance / self.settings.player_reach_normalized)
            if distance is not None else 0.0
        )
        racket_region = (
            normalized_y is not None
            and self.settings.racket_region_top_normalized <= normalized_y <= self.settings.racket_region_bottom_normalized
        )
        strengths = item.strengths
        direction = strengths.get("direction_change", 0.0) if self.settings.enable_direction_change else 0.0
        velocity = strengths.get("velocity_discontinuity", 0.0) if self.settings.enable_velocity_discontinuity else 0.0
        kinematic = max(
            direction, velocity, strengths.get("acceleration", 0.0),
            strengths.get("vertical_inversion", 0.0),
        )
        continuity = strengths.get("continuity", 0.0)
        state_weight = self.settings.state_weights.get(item.state, 0.0)
        quality = state_weight if self.settings.enable_provenance_weighting else 1.0
        boxes = p1_boxes if selected_player == 1 else p2_boxes
        departure = self._departure(item.frame, selected_player, trajectory, boxes) if selected_player else 0.0
        pose: Optional[float] = None
        if self.settings.enable_pose_support and pose_support is not None and selected_player is not None:
            pose = pose_support.get(item.frame, {}).get(selected_player)

        player_cues = {
            "temporal_player_proximity": proximity > 0,
            "clear_player_attribution": bool(player["attribution_clear"]),
            "trajectory_change": kinematic >= 0.5,
            "trajectory_continuity": continuity >= 0.5,
            "ball_leaves_player_region": departure >= 0.25,
            "ball_state_quality": quality >= 0.55,
            "racket_region": racket_region,
            "pose_support": pose is not None and pose >= 0.5,
        }
        independent_player = sum(
            player_cues[name] for name in (
                "temporal_player_proximity", "trajectory_change", "trajectory_continuity",
                "ball_leaves_player_region", "ball_state_quality", "pose_support",
            )
        )
        racket_weight = 1.0 if racket_region else (0.5 if (normalized_y is not None and -0.5 <= normalized_y <= 1.25) else 0.0)
        player_score = (
            0.30 * proximity + 0.22 * kinematic + 0.16 * continuity
            + 0.12 * departure + 0.12 * quality + 0.08 * racket_weight
        )
        player_pass = (
            item.state != BallState.MISSING.value
            and proximity > 0.0
            and racket_region
            and player_score >= self.settings.player_contact_min_score
        )

        distances = [
            row["distance_normalized"] for row in player["players"].values()
            if row["distance_normalized"] is not None
        ]
        far_support = self._clip(min(distances) / self.settings.player_reach_normalized) if distances else 1.0
        vertical = 1.0 if item.court_rebound else (0.70 if item.vertical_inversion else 0.0)
        bounce_change = max(direction, velocity, strengths.get("acceleration", 0.0))

        bounce_cues = {
            "local_vertical_reversal": vertical >= 0.5,
            "velocity_or_direction_change": bounce_change >= 0.5,
            "trajectory_continuity": continuity >= 0.5,
            "ball_state_quality": quality >= 0.55,
            "far_from_players": far_support >= 0.5,
        }
        independent_bounce = sum(bounce_cues.values())
        bounce_score = (
            0.36 * vertical + 0.22 * bounce_change + 0.18 * continuity
            + 0.14 * quality + 0.10 * far_support
        )
        bounce_pass = (
            self.settings.enable_bounce_contact_check
            and item.state != BallState.MISSING.value
            and vertical > 0
            and bounce_score >= self.settings.court_contact_min_score
            and not (
                distance is not None and distance <= 1e-9
                and normalized_y is not None and normalized_y >= 0.90
            )
        )

        physical = PhysicalEventType.UNKNOWN
        if self.settings.enable_contact_family_stage:
            if player_pass and bounce_pass:
                if player_score >= bounce_score + self.settings.physical_type_margin:
                    physical = PhysicalEventType.PLAYER_CONTACT
                elif bounce_score >= player_score + self.settings.physical_type_margin:
                    physical = PhysicalEventType.COURT_CONTACT
                elif distance is not None and distance <= 0.65:
                    physical = PhysicalEventType.PLAYER_CONTACT
                else:
                    physical = PhysicalEventType.COURT_CONTACT
            elif player_pass:
                physical = PhysicalEventType.PLAYER_CONTACT
            elif bounce_pass:
                physical = PhysicalEventType.COURT_CONTACT
            elif (
                self.settings.enable_semantic_unknown_abstention
                and item.state != BallState.MISSING.value
                and max(player_score, bounce_score) >= 0.32
                and kinematic >= 0.40
            ):
                physical = PhysicalEventType.UNKNOWN
        else:
            if player_pass:
                physical = PhysicalEventType.PLAYER_CONTACT
            elif bounce_pass:
                physical = PhysicalEventType.COURT_CONTACT

        trace: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "discovery_frame": candidate.discovery_frame_index if candidate.discovery_frame_index is not None else candidate.frame_index,
            "discovery_timestamp_s": candidate.discovery_timestamp_s if candidate.discovery_timestamp_s is not None else candidate.timestamp_s,
            "refined_frame": item.frame,
            "refined_timestamp_s": item.timestamp,
            "candidate_score": candidate.score,
            "ball_state": item.state,
            "ball_confidence": item.confidence,
            **self._feature_dict(item),
            "player1_distance_normalized": player["players"][1]["distance_normalized"],
            "player2_distance_normalized": player["players"][2]["distance_normalized"],
            "selected_player": selected_player,
            "player_scale_px": selected.get("player_scale_px"),
            "pose_support": pose,
            "camera_motion_state": camera_state,
            "physical_event_type": physical.value,
            "player_contact_score": player_score,
            "court_contact_score": bounce_score,
            "verification_score": max(player_score, bounce_score),
            "player_contact_cues": player_cues,
            "court_contact_cues": bounce_cues,
            "stage_pass": {
                "raw_candidate_generator": True,
                "physics_verification": physical != PhysicalEventType.UNKNOWN,
                "player_attribution": (
                    physical != PhysicalEventType.UNKNOWN
                    and (
                        physical != PhysicalEventType.PLAYER_CONTACT
                        or bool(player["attribution_clear"])
                    )
                ),
                "event_type_classification": False,
                "temporal_suppression": False,
                "final_authoritative_event": False,
            },
            "candidate_event_type": "UNKNOWN_EVENT",
            "verification_pass": False,
            "rejection_stage": None,
            "rejection_reason": None,
            "activity_state": "UNKNOWN",
            "final_event_emitted": False,
        }
        if physical == PhysicalEventType.UNKNOWN:
            trace["rejection_stage"] = "PHYSICS_VERIFICATION"
            if item.state == BallState.MISSING.value:
                trace["rejection_reason"] = "MISSING_STATE_CANNOT_CREATE_PHYSICAL_EVENT"
            elif distance is not None and distance <= self.settings.player_reach_normalized:
                trace["rejection_reason"] = "PLAYER_CONTACT_MULTI_CUE_REJECTION"
            elif vertical <= 0:
                trace["rejection_reason"] = "BOUNCE_CONTACT_SIGNATURE_REJECTION"
            else:
                trace["rejection_reason"] = "AMBIGUOUS_PHYSICAL_EVENT"
            return None, trace

        if (
            self.settings.enable_dead_ball_gating
            and (item.normalized_speed or 0.0) <= self.settings.rolling_max_normalized_speed_per_s
            and kinematic < self.settings.dead_ball_min_kinematic_support
        ):
            trace.update({
                "rejection_stage": "VISION_ACTIVITY_FILTER",
                "rejection_reason": "DEAD_BALL_LOW_ENERGY_ROLL_OR_JITTER",
                "activity_state": "DEAD_BALL",
            })
            return None, trace

        player_id: Optional[int]
        serve: Optional[Dict[str, Any]] = None
        if physical == PhysicalEventType.COURT_CONTACT:
            event_type = EventType.BOUNCE
            player_id = None
            semantic_score = bounce_score
            trace["activity_state"] = "COURT_CONTACT_CANDIDATE"
        else:
            if not bool(player["attribution_clear"]):
                event_type = EventType.UNKNOWN_EVENT
                player_id = None
                semantic_score = player_score
            elif self.settings.enable_player_attribution_fusion:
                player_id = selected_player if selected_player is not None else 1
                event_type = EventType.PLAYER_1_HIT if player_id == 1 else EventType.PLAYER_2_HIT
                semantic_score = player_score
            else:
                player_id = selected_player
                event_type = EventType.PLAYER_1_HIT if player_id == 1 else EventType.PLAYER_2_HIT
                semantic_score = player_score
            if self.settings.enable_serve_semantics:
                serve = self._serve_evidence(
                    item.frame, item, selected, trajectory, frame_height, frame_diagonal
                )
                overhead = serve["overhead_contact"]
                toss = serve["toss_like_motion"]
                fast_depart = serve["post_contact_rapid_departure"]
                pre_speed_norm = (item.pre_speed or 0.0) / max(frame_diagonal, 1.0)
                is_serve = (
                    overhead
                    and (toss or pre_speed_norm <= 0.22)
                    and fast_depart
                    and serve["score"] >= 0.50
                )
                if is_serve:
                    event_type = EventType.SERVE_CONTACT
                    semantic_score = max(semantic_score, serve["score"])
            trace["activity_state"] = "PLAYER_CONTACT_CANDIDATE"

        trace["serve_evidence"] = serve
        trace["candidate_event_type"] = event_type.value
        trace["stage_pass"]["event_type_classification"] = trace["stage_pass"]["player_attribution"]
        trace["verification_pass"] = True
        confidence = self._clip(semantic_score * quality)
        assert point.x_px is not None and point.y_px is not None
        court_position = (
            (float(point.court_x_m), float(point.court_y_m))
            if point.court_x_m is not None and point.court_y_m is not None else None
        )
        evidence = dict(candidate.evidence)
        evidence.update({
            "candidate_id": candidate_id,
            "discovery_frame": trace["discovery_frame"],
            "discovery_timestamp_s": trace["discovery_timestamp_s"],
            "refined_frame": item.frame,
            "refined_timestamp_s": item.timestamp,
            "physical_event_type": physical.value,
            "verification_score": trace["verification_score"],
            "player_contact_score": player_score,
            "court_contact_score": bounce_score,
            "player1_distance_normalized": trace["player1_distance_normalized"],
            "player2_distance_normalized": trace["player2_distance_normalized"],
            "selected_player": selected_player,
            "player_scale_px": trace["player_scale_px"],
            "pose_support": pose,
            "serve_evidence": serve,
            "ball_state": item.state,
            "state_confidence_weight": quality,
            **self._feature_dict(item),
        })
        event = TennisEvent(
            event_id=0,
            event_type=event_type,
            frame_index=item.frame,
            timestamp_s=item.timestamp,
            player_id=player_id,
            ball_position_px=(float(point.x_px), float(point.y_px)),
            court_position_m=court_position,
            confidence=confidence,
            trajectory_state=item.state,
            evidence=evidence,
        )
        return event, trace

    def analyze(
        self,
        ball_trajectory: List[TemporalBallPoint],
        player1_boxes: List[Optional[BBox]],
        player2_boxes: List[Optional[BBox]],
        homography_matrix: Optional[np.ndarray] = None,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        camera_offsets_px: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
        pose_support: Optional[Mapping[int, Mapping[int, float]]] = None,
    ) -> EventDetectionAnalysis:
        del homography_matrix  # A static transform cannot compensate per-frame camera motion.
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(player1_boxes) != len(ball_trajectory) or len(player2_boxes) != len(ball_trajectory):
            raise ValueError("player timelines must align with ball trajectory")
        width, height, size_source = self._frame_size(ball_trajectory, frame_size)
        positions, camera_state = self._positions(ball_trajectory, camera_offsets_px)
        features = self._features(ball_trajectory, (width, height), positions)
        if frame_size is None and camera_offsets_px is None:
            candidates = self.detect_candidates(ball_trajectory, fps=fps)
        else:
            candidates = self.detect_candidates(
                ball_trajectory, fps=fps, frame_size=(width, height),
                camera_offsets_px=camera_offsets_px,
            )
        diagonal = math.hypot(width, height)
        provisional: List[Tuple[TennisEvent, Dict[str, Any]]] = []
        traces: List[Dict[str, Any]] = []
        for candidate_id, candidate in enumerate(candidates, 1):
            frame = self._refined_frame(candidate, features, ball_trajectory, player1_boxes, player2_boxes)
            frame = max(0, min(len(features) - 1, frame))
            item = self._legacy_feature_fallback(features[frame], candidate, diagonal)
            event, trace = self._verify(
                candidate_id, candidate, item, ball_trajectory, player1_boxes, player2_boxes,
                pose_support, camera_state, height, diagonal,
            )
            traces.append(trace)
            if event is not None:
                provisional.append((event, trace))

        selected: List[Tuple[TennisEvent, Dict[str, Any]]] = []
        for event, trace in sorted(
            provisional, key=lambda row: (-row[0].confidence, row[0].timestamp_s, row[1]["candidate_id"])
        ):
            if all(
                abs(event.timestamp_s - kept.timestamp_s) > self.settings.final_event_min_interval_seconds
                for kept, _ in selected
            ):
                selected.append((event, trace))
            else:
                trace.update({
                    "verification_pass": False,
                    "rejection_stage": "TEMPORAL_SUPPRESSION",
                    "rejection_reason": "DUPLICATE_VERIFIED_EVENT",
                })
        selected.sort(key=lambda row: row[0].timestamp_s)
        events: List[TennisEvent] = []
        for event_id, (event, trace) in enumerate(selected, 1):
            event.event_id = event_id
            trace["stage_pass"]["temporal_suppression"] = True
            trace["stage_pass"]["final_authoritative_event"] = True
            trace["final_event_emitted"] = True
            trace["final_event_id"] = event_id
            trace["final_event_type"] = event.event_type.value
            events.append(event)

        analysis = EventDetectionAnalysis(
            candidates=candidates,
            events=events,
            verification_traces=traces,
            settings=asdict(self.settings),
            coordinate_system={
                "frame_width_px": width,
                "frame_height_px": height,
                "frame_diagonal_px": diagonal,
                "frame_size_source": size_source,
                "event_feature_coordinates": "COMPENSATED_IMAGE_PLANE" if camera_state.startswith("GLOBAL_") else "RAW_IMAGE_PLANE",
                "camera_motion_state": camera_state,
                "metric_court_plane_claimed_for_airborne_ball": False,
            },
        )
        self.last_analysis = analysis
        return analysis

    def detect_events(
        self,
        ball_trajectory: List[TemporalBallPoint],
        player1_boxes: List[Optional[BBox]],
        player2_boxes: List[Optional[BBox]],
        homography_matrix: Optional[np.ndarray] = None,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        camera_offsets_px: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    ) -> List[TennisEvent]:
        analysis = self.analyze(
            ball_trajectory, player1_boxes, player2_boxes, homography_matrix, fps,
            frame_size, camera_offsets_px,
        )
        return [e for e in analysis.events if e.event_type != EventType.UNKNOWN_EVENT]
