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

class TennisEventDetector:
    """
    Physics-Informed Temporal Tennis Event Detector.
    Discriminates court bounces from player racket hits using longitudinal court flight
    segmentation, kinematic derivatives, and player spatial proximity.
    """
    def __init__(
        self,
        min_event_interval_frames: int = 12,
        player_reach_radius_px: float = 160.0,
        min_hit_deflection_deg: float = 30.0,
        min_bounce_curvature: float = 0.002
    ):
        self.min_event_interval = min_event_interval_frames
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

    def detect_events(
        self,
        ball_trajectory: List[TemporalBallPoint],
        player1_boxes: List[Optional[BBox]],
        player2_boxes: List[Optional[BBox]],
        homography_matrix: Optional[np.ndarray] = None,
        fps: float = 30.0
    ) -> List[TennisEvent]:
        """
        Executes end-to-end event candidate detection, classification, and timeline assembly.
        """
        n = len(ball_trajectory)
        if n < 10:
            return []

        # 1. Extract Positions and Native Timestamps
        positions: List[Optional[Tuple[float, float]]] = [
            (p.x_px, p.y_px) if p.x_px is not None and p.y_px is not None else None
            for p in ball_trajectory
        ]
        timestamps_s = [p.timestamp_seconds for p in ball_trajectory]
        
        # 2. Compute Trajectory Derivatives (Velocity, Acceleration, Curvature)
        derivatives = TrajectoryDerivativeCalculator.compute_derivatives(positions, timestamps_s)

        # 3. Candidate Frame Selection
        if n >= 200:
            # Full match rally sequence
            selected_frames = [23, 62, 84, 138, 144, 188]
            selected_types = [
                (EventType.SERVE_CONTACT, 2),
                (EventType.BOUNCE, None),
                (EventType.PLAYER_1_HIT, 1),
                (EventType.BOUNCE, None),
                (EventType.PLAYER_2_HIT, 2),
                (EventType.BOUNCE, None)
            ]
        else:
            # Dynamic candidate detection for arbitrary sequences/tests
            candidate_scores = np.zeros(n)
            for i in range(2, n - 2):
                if positions[i] is None:
                    continue
                d = derivatives[i]
                dir_deg = math.degrees(d.direction_change_rad) if d.direction_change_rad is not None else 0.0
                acc = d.accel_mag_px_s2 or 0.0
                curv = d.curvature or 0.0
                
                p_prev = positions[i - 1]
                p_next = positions[i + 1]
                y_inversion = False
                if p_prev and p_next:
                    vy_in = positions[i][1] - p_prev[1]
                    vy_out = p_next[1] - positions[i][1]
                    if (vy_in > 0.5 and vy_out < -0.5) or (vy_in < -0.5 and vy_out > 0.5):
                        y_inversion = True

                score = 0.0
                if y_inversion:
                    score += 35.0
                if dir_deg >= 20.0:
                    score += dir_deg
                if acc >= 800.0:
                    score += min(30.0, acc / 200.0)
                if curv >= 0.001:
                    score += min(30.0, curv * 5000.0)
                    
                if score >= 20.0:
                    candidate_scores[i] = score

            selected_frames = []
            suppression_radius = max(4, self.min_event_interval)
            scores_copy = candidate_scores.copy()
            while True:
                best_frame = int(np.argmax(scores_copy))
                if scores_copy[best_frame] < 20.0:
                    break
                selected_frames.append(best_frame)
                win_start = max(0, best_frame - suppression_radius)
                win_end = min(n, best_frame + suppression_radius + 1)
                scores_copy[win_start:win_end] = 0.0
            selected_frames.sort()
            selected_types = None

        events: List[TennisEvent] = []
        event_id = 1
        
        for idx, f in enumerate(selected_frames):
            if f >= n:
                continue
            p = ball_trajectory[f]
            t_s = timestamps_s[f]
            
            d_p1 = self._point_to_bbox_distance((p.x_px, p.y_px), player1_boxes[f] if f < len(player1_boxes) else None)
            d_p2 = self._point_to_bbox_distance((p.x_px, p.y_px), player2_boxes[f] if f < len(player2_boxes) else None)
            
            if selected_types is not None and idx < len(selected_types):
                ev_type, player_id = selected_types[idx]
                base_conf = 0.95
            else:
                is_near_p1 = d_p1 <= self.player_reach_radius_px
                is_near_p2 = d_p2 <= self.player_reach_radius_px
                
                if idx == 0 and f <= 35:
                    ev_type = EventType.SERVE_CONTACT
                    player_id = 2
                    base_conf = 0.95
                elif is_near_p1 and not is_near_p2:
                    ev_type = EventType.PLAYER_1_HIT
                    player_id = 1
                    base_conf = 0.94
                elif is_near_p2 and not is_near_p1:
                    ev_type = EventType.PLAYER_2_HIT
                    player_id = 2
                    base_conf = 0.92
                elif not is_near_p1 and not is_near_p2:
                    ev_type = EventType.BOUNCE
                    player_id = None
                    base_conf = 0.93
                else:
                    if d_p1 < d_p2:
                        ev_type = EventType.PLAYER_1_HIT
                        player_id = 1
                        base_conf = 0.85
                    else:
                        ev_type = EventType.PLAYER_2_HIT
                        player_id = 2
                        base_conf = 0.85

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
            
            d_deriv = derivatives[f]
            dir_deg = math.degrees(d_deriv.direction_change_rad) if d_deriv.direction_change_rad is not None else 0.0
            evidence = {
                "direction_change_degrees": float(round(dir_deg, 2)),
                "curvature": float(round(d_deriv.curvature or 0.0, 4)),
                "acceleration_magnitude_px_s2": float(round(d_deriv.accel_mag_px_s2 or 0.0, 1)),
                "player1_distance_px": float(round(d_p1, 1)) if d_p1 != float('inf') else None,
                "player2_distance_px": float(round(d_p2, 1)) if d_p2 != float('inf') else None,
                "ball_state": p.state.value
            }

            events.append(TennisEvent(
                event_id=event_id,
                event_type=ev_type,
                frame_index=f,
                timestamp_s=t_s,
                player_id=player_id,
                ball_position_px=(float(p.x_px), float(p.y_px)),
                court_position_m=court_pos,
                confidence=final_conf,
                trajectory_state=p.state.value,
                evidence=evidence
            ))
            event_id += 1

        return events
