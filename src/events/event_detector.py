import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.events.trajectory_derivatives import TrajectoryDerivativeCalculator, KinematicDerivatives

class EventType(Enum):
    SERVE_CONTACT = "SERVE_CONTACT"
    PLAYER_1_HIT = "PLAYER_1_HIT"
    PLAYER_2_HIT = "PLAYER_2_HIT"
    BOUNCE = "BOUNCE"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"

@dataclass
class TennisEvent:
    """Represents an atomic, verified physical tennis match event."""
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
    """High-recall physical candidate awaiting multi-cue semantic verification."""
    frame_index: int
    timestamp_s: float
    score: float
    ball_position_px: Tuple[float, float]
    trajectory_state: str
    evidence: Dict[str, Any] = field(default_factory=dict)

class TennisEventDetector:
    """
    Physics-Informed Temporal Tennis Event Detector.
    Discriminates court bounces from player racket hits using longitudinal court flight
    segmentation, kinematic derivatives, and player spatial proximity.
    """
    def __init__(
        self,
        min_event_interval_frames: int = 12,
        min_event_interval_seconds: Optional[float] = None,
        player_reach_radius_px: float = 160.0,
        min_hit_deflection_deg: float = 30.0,
        min_bounce_curvature: float = 0.002
    ):
        # Preserve the historical 30 FPS configuration while applying it in time,
        # not as an absolute frame-count rule, at inference time.
        self.min_event_interval_seconds = (
            float(min_event_interval_seconds)
            if min_event_interval_seconds is not None
            else float(min_event_interval_frames) / 30.0
        )
        self.player_reach_radius_px = player_reach_radius_px
        self.min_hit_deflection_deg = min_hit_deflection_deg
        self.min_bounce_curvature = min_bounce_curvature

    @staticmethod
    def _point_to_bbox_distance(pt: Tuple[float, float], bbox: Optional[BBox]) -> float:
        """Computes minimum Euclidean distance from a point to a player bounding box."""
        if bbox is None or pt[0] is None or pt[1] is None:
            return float('inf')
        px, py = pt
        closest_x = max(bbox.x1, min(px, bbox.x2))
        closest_y = max(bbox.y1, min(py, bbox.y2))
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def _local_speed(
        positions: List[Optional[Tuple[float, float]]],
        timestamps_s: List[float],
        start: int,
        end: int,
    ) -> Optional[float]:
        if start < 0 or end >= len(positions) or positions[start] is None or positions[end] is None:
            return None
        dt = timestamps_s[end] - timestamps_s[start]
        if dt <= 0:
            return None
        return math.dist(positions[start], positions[end]) / dt

    def detect_candidates(
        self,
        ball_trajectory: List[TemporalBallPoint],
        fps: float = 30.0
    ) -> List[EventCandidate]:
        """
        Generates high-recall physical candidates without assigning tennis semantics.

        ``fps`` is retained for API compatibility and validation, but all temporal
        calculations use the trajectory timestamps supplied by video ingest.
        """
        n = len(ball_trajectory)
        if n < 10:
            return []
        if fps <= 0:
            raise ValueError("fps must be positive")

        # 1. Extract Positions and Native Timestamps
        positions: List[Optional[Tuple[float, float]]] = [
            (p.x_px, p.y_px) if p.x_px is not None and p.y_px is not None else None
            for p in ball_trajectory
        ]
        timestamps_s = [p.timestamp_seconds for p in ball_trajectory]
        
        # 2. Compute Trajectory Derivatives (Velocity, Acceleration, Curvature)
        derivatives = TrajectoryDerivativeCalculator.compute_derivatives(positions, timestamps_s)

        # 3. Dynamic Candidate Frame Selection
        candidate_scores = np.zeros(n)
        for i in range(3, n - 3):
            if positions[i] is None:
                continue
            d = derivatives[i]
            speed = d.speed_px_s or 0.0
            
            # Reject stationary/rolling/jitter noise below active speed threshold
            if speed < 180.0:  # < 6 px/frame at 30 FPS
                continue
                
            dir_deg = math.degrees(d.direction_change_rad) if d.direction_change_rad is not None else 0.0
            acc = d.accel_mag_px_s2 or 0.0
            curv = d.curvature or 0.0
            
            p_prev = positions[i - 2]
            p_curr = positions[i]
            p_next = positions[i + 2]
            y_inversion = False
            if p_prev and p_next and p_curr:
                vy_in = (p_curr[1] - p_prev[1]) / 2.0
                vy_out = (p_next[1] - p_curr[1]) / 2.0
                if (vy_in > 1.5 and vy_out < -1.5) or (vy_in < -1.5 and vy_out > 1.5):
                    y_inversion = True

            score = 0.0
            if y_inversion:
                score += 40.0
            if dir_deg >= 25.0:
                score += min(50.0, dir_deg * 0.8)
            if acc >= 800.0:
                score += min(30.0, acc / 200.0)
            if curv >= 0.005 and speed > 250.0:
                score += min(25.0, curv * 2000.0)
                
            if score >= 30.0:
                candidate_scores[i] = score

        selected_frames = []
        scores_copy = candidate_scores.copy()
        while True:
            best_frame = int(np.argmax(scores_copy))
            if scores_copy[best_frame] < 30.0:
                break
            selected_frames.append(best_frame)
            best_time = timestamps_s[best_frame]
            for i, timestamp_s in enumerate(timestamps_s):
                if abs(timestamp_s - best_time) <= self.min_event_interval_seconds:
                    scores_copy[i] = 0.0
        selected_frames.sort()

        candidates: List[EventCandidate] = []
        for f in selected_frames:
            p = ball_trajectory[f]
            d = derivatives[f]
            p_prev = positions[f - 2] if f >= 2 else None
            p_next = positions[f + 2] if f + 2 < n else None
            y_inversion = False
            # Derivative maxima can land one frame beside the visual inversion;
            # verify a small timestamp-local neighborhood without changing the
            # semantic event into a benchmark-frame lookup.
            for center in range(max(2, f - 2), min(n - 2, f + 3)):
                before = positions[center - 2]
                current = positions[center]
                after = positions[center + 2]
                if before is None or current is None or after is None:
                    continue
                vy_in = current[1] - before[1]
                vy_out = after[1] - current[1]
                if (vy_in > 0 > vy_out) or (vy_in < 0 < vy_out):
                    y_inversion = True
                    break
            pre_speed = self._local_speed(positions, timestamps_s, f - 2, f)
            post_speed = self._local_speed(positions, timestamps_s, f, f + 2)
            continuity = all(
                0 <= j < n and positions[j] is not None
                for j in (f - 2, f - 1, f, f + 1, f + 2)
            )
            candidates.append(EventCandidate(
                frame_index=f,
                timestamp_s=timestamps_s[f],
                score=float(candidate_scores[f]),
                ball_position_px=(float(p.x_px), float(p.y_px)),
                trajectory_state=p.state.value,
                evidence={
                    "candidate_source": "TRAJECTORY_INFLECTION",
                    "pre_speed_px_s": pre_speed,
                    "post_speed_px_s": post_speed,
                    "direction_change_degrees": float(math.degrees(d.direction_change_rad or 0.0)),
                    "acceleration_magnitude_px_s2": float(d.accel_mag_px_s2 or 0.0),
                    "curvature": float(d.curvature or 0.0),
                    "vertical_inversion": y_inversion,
                    "trajectory_continuity": continuity,
                    "ball_confidence": p.confidence,
                    "actual_fps": float(fps),
                },
            ))
        return candidates

    def detect_events(
        self,
        ball_trajectory: List[TemporalBallPoint],
        player1_boxes: List[Optional[BBox]],
        player2_boxes: List[Optional[BBox]],
        homography_matrix: Optional[np.ndarray] = None,
        fps: float = 30.0
    ) -> List[TennisEvent]:
        """Verifies high-recall candidates into authoritative semantic events."""
        candidates = self.detect_candidates(ball_trajectory, fps=fps)
        n = len(ball_trajectory)

        events: List[TennisEvent] = []
        event_id = 1
        
        for candidate in candidates:
            f = candidate.frame_index
            # Inflection derivatives can peak beside visual contact. Align only
            # within a short time window using player-scale geometry and require
            # clear separation between the two possible players.
            alignment_options = []
            for aligned_frame, aligned_point in enumerate(ball_trajectory):
                if abs(aligned_point.timestamp_seconds - candidate.timestamp_s) > 0.2:
                    continue
                if aligned_point.x_px is None or aligned_point.y_px is None:
                    continue
                aligned_p1 = player1_boxes[aligned_frame] if aligned_frame < len(player1_boxes) else None
                aligned_p2 = player2_boxes[aligned_frame] if aligned_frame < len(player2_boxes) else None
                distances = []
                for player_id, box in ((1, aligned_p1), (2, aligned_p2)):
                    if box is None:
                        continue
                    player_h = max(1.0, box.y2 - box.y1)
                    normalized_y = (aligned_point.y_px - box.y1) / player_h
                    if not (-0.30 <= normalized_y <= 0.85):
                        continue
                    distance = self._point_to_bbox_distance((aligned_point.x_px, aligned_point.y_px), box)
                    reach = max(
                        self.player_reach_radius_px if player_id == 1 else self.player_reach_radius_px * 0.85,
                        (1.2 if player_id == 1 else 1.4) * player_h,
                    )
                    distances.append((distance / max(reach, 1.0), distance, player_id, player_h))
                if not distances:
                    continue
                distances.sort()
                best = distances[0]
                second_distance = distances[1][1] if len(distances) > 1 else float("inf")
                if best[0] <= 1.0 and second_distance - best[1] >= 0.20 * best[3]:
                    alignment_options.append((best[0], abs(aligned_frame - f), aligned_frame))
            if alignment_options:
                _, _, f = min(alignment_options)
            if f >= n:
                continue
            p = ball_trajectory[f]
            t_s = candidate.timestamp_s
            
            p1_box = player1_boxes[f] if f < len(player1_boxes) else None
            p2_box = player2_boxes[f] if f < len(player2_boxes) else None

            d_p1 = self._point_to_bbox_distance((p.x_px, p.y_px), p1_box)
            d_p2 = self._point_to_bbox_distance((p.x_px, p.y_px), p2_box)
            
            h_p1 = (p1_box.y2 - p1_box.y1) if p1_box else 180.0
            h_p2 = (p2_box.y2 - p2_box.y1) if p2_box else 100.0
            reach_p1 = max(self.player_reach_radius_px, 1.2 * h_p1)
            reach_p2 = max(self.player_reach_radius_px * 0.85, 1.4 * h_p2)

            is_near_p1 = d_p1 <= reach_p1
            is_near_p2 = d_p2 <= reach_p2
            
            direction_change = candidate.evidence["direction_change_degrees"]
            acceleration = candidate.evidence["acceleration_magnitude_px_s2"]
            vertical_inversion = candidate.evidence["vertical_inversion"]
            continuity = candidate.evidence["trajectory_continuity"]
            post_speed = candidate.evidence["post_speed_px_s"] or 0.0
            reliable_track = p.state in (BallState.DETECTED, BallState.TRACKED)

            nearest_box = p1_box if d_p1 <= d_p2 else p2_box
            nearest_player = 1 if d_p1 <= d_p2 else 2
            nearest_distance = min(d_p1, d_p2)
            nearest_reach = reach_p1 if nearest_player == 1 else reach_p2
            near_exactly_one = is_near_p1 != is_near_p2
            attribution_clear = near_exactly_one
            if is_near_p1 and is_near_p2 and nearest_box is not None:
                nearest_h = max(1.0, nearest_box.y2 - nearest_box.y1)
                attribution_clear = abs(d_p1 - d_p2) >= 0.20 * nearest_h

            # A serve requires source-independent overhead-contact evidence. It is
            # never inferred merely because this is the first or an early frame.
            overhead_contact = False
            reachable_racket_region = False
            if nearest_box is not None:
                player_h = max(1.0, nearest_box.y2 - nearest_box.y1)
                normalized_y = (p.y_px - nearest_box.y1) / player_h
                overhead_contact = normalized_y <= 0.45
                reachable_racket_region = -0.30 <= normalized_y <= 0.85

            hit_cues = sum((
                nearest_distance <= nearest_reach,
                direction_change >= self.min_hit_deflection_deg or acceleration >= 800.0,
                continuity,
                reliable_track,
                reachable_racket_region,
            ))
            serve_cues = sum((overhead_contact, post_speed >= 240.0, continuity, reliable_track))

            if attribution_clear and serve_cues >= 4 and candidate.evidence.get("serve_context_verified") is True:
                ev_type = EventType.SERVE_CONTACT
                player_id = nearest_player
                base_conf = 0.92
            elif attribution_clear and reachable_racket_region and hit_cues >= 4:
                ev_type = EventType.PLAYER_1_HIT if nearest_player == 1 else EventType.PLAYER_2_HIT
                player_id = nearest_player
                base_conf = 0.88
            elif not is_near_p1 and not is_near_p2 and vertical_inversion and continuity and reliable_track:
                ev_type = EventType.BOUNCE
                player_id = None
                base_conf = 0.86
            else:
                # Retain the candidate through ``detect_candidates`` for audit,
                # but abstain from creating an authoritative semantic event.
                continue

            state_weights = {
                BallState.DETECTED: 1.0,
                BallState.TRACKED: 0.92,
                BallState.PREDICTED: 0.75,
                BallState.INTERPOLATED: 0.60,
                BallState.OCCLUDED: 0.50,
                BallState.MISSING: 0.10
            }
            conf_mod = state_weights.get(p.state, 0.7)
            final_conf = float(np.clip(base_conf * conf_mod, 0.1, 0.98))
            
            court_pos = (float(p.court_x_m), float(p.court_y_m)) if p.court_x_m is not None and p.court_y_m is not None else None
            
            evidence = dict(candidate.evidence)
            evidence.update({
                "player1_distance_px": float(round(d_p1, 1)) if d_p1 != float('inf') else None,
                "player2_distance_px": float(round(d_p2, 1)) if d_p2 != float('inf') else None,
                "player1_reach_px": float(round(reach_p1, 1)),
                "player2_reach_px": float(round(reach_p2, 1)),
                "overhead_contact": overhead_contact,
                "reachable_racket_region": reachable_racket_region,
                "verification_cue_count": serve_cues if ev_type == EventType.SERVE_CONTACT else hit_cues,
                "verification_status": "VERIFIED_AUTHORITATIVE",
                "ball_state": p.state.value,
            })

            events.append(TennisEvent(
                event_id=event_id,
                event_type=ev_type,
                frame_index=f,
                timestamp_s=t_s,
                player_id=player_id,
                ball_position_px=(float(p.x_px), float(p.y_px)) if (p.x_px is not None and p.y_px is not None) else (0.0, 0.0),
                court_position_m=court_pos,
                confidence=final_conf,
                trajectory_state=p.state.value,
                evidence=evidence
            ))
            event_id += 1

        return events
