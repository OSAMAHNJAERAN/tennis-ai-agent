import math
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.utils.bbox_utils import BBox

class BallState(Enum):
    """Explicit state of tennis ball tracking at a specific frame."""
    DETECTED = "DETECTED"          # High-confidence raw visual detection
    TRACKED = "TRACKED"            # Low-confidence detection verified by temporal trajectory consistency
    PREDICTED = "PREDICTED"        # Kinematic Kalman filter ballistic prediction
    INTERPOLATED = "INTERPOLATED"  # Gap-limited geometric interpolation (fallback)
    OCCLUDED = "OCCLUDED"          # Identified player/racket/net occlusion
    MISSING = "MISSING"            # Unobserved / untracked frame

@dataclass
class BallObservation:
    """Represents a candidate ball observation from visual detector."""
    x_px: float
    y_px: float
    confidence: float
    bbox: Optional[BBox] = None

@dataclass
class TemporalBallPoint:
    """Rich point representation along the ball trajectory."""
    frame_index: int
    timestamp_seconds: float
    x_px: Optional[float]
    y_px: Optional[float]
    court_x_m: Optional[float] = None
    court_y_m: Optional[float] = None
    confidence: Optional[float] = None
    state: BallState = BallState.MISSING
    source: str = "none"
    velocity_px_per_sec: Optional[Tuple[float, float]] = None
    speed_kmh: Optional[float] = None

class KinematicFilter:
    """
    Constant Acceleration / Ballistic Kinematic Filter for Tennis Ball in Image Space.
    State vector: [x, y, vx, vy, ax, ay]
    """
    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        # State: x, y, vx, vy, ax, ay
        self.x = np.zeros(6, dtype=np.float64)
        # Covariance
        self.P = np.eye(6, dtype=np.float64) * 50.0
        
        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt * dt
        self.F[1, 5] = 0.5 * dt * dt
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        
        # Measurement matrix H (we measure position x, y)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        
        # Process noise covariance Q
        q_pos = 10.0
        q_vel = 1000.0
        q_acc = 5000.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])
        
        # Measurement noise covariance R
        self.R = np.eye(2, dtype=np.float64) * 4.0
        self.initialized = False
        self.steps_since_update = 0

    def initialize(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([x, y, vx, vy, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 500000.0, 500000.0, 1000000.0, 1000000.0])
        self.initialized = True
        self.steps_since_update = 0

    def predict(self) -> Tuple[float, float]:
        """Predicts next state and returns expected (x, y)."""
        if not self.initialized:
            return (0.0, 0.0)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.steps_since_update += 1
        return (float(self.x[0]), float(self.x[1]))

    def update(self, z_x: float, z_y: float, measurement_var: float = 4.0):
        """Updates filter with new visual measurement."""
        z = np.array([z_x, z_y], dtype=np.float64)
        R = np.eye(2, dtype=np.float64) * measurement_var
        
        if not self.initialized:
            self.initialize(z_x, z_y)
            return
            
        y = z - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        self.steps_since_update = 0

    def get_position(self) -> Tuple[float, float]:
        return (float(self.x[0]), float(self.x[1]))

    def get_velocity(self) -> Tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    def get_gating_radius(self, base_radius: float = 45.0) -> float:
        """Dynamic gating radius proportional to speed and prediction uncertainty."""
        speed = math.hypot(self.x[2], self.x[3])
        uncertainty = math.sqrt(max(1.0, self.P[0, 0] + self.P[1, 1]))
        return min(120.0, max(base_radius, base_radius + 0.05 * speed + 1.5 * uncertainty))

class TemporalBallTracker:
    """
    Multi-stage Physics-Informed Temporal Ball Tracker.
    Combines multi-candidate anchor seeding, adaptive kinematic Kalman gating,
    bidirectional track confirmation, short-gap ballistic reacquisition, and gap-limited interpolation.
    """
    def __init__(
        self,
        high_conf_thresh: float = 0.08,
        low_conf_thresh: float = 0.01,
        max_prediction_gap: int = 4,
        max_interpolation_gap: int = 3,
        max_valid_speed_px_per_frame: float = 80.0,
        base_gating_radius_px: float = 45.0,
        enable_multi_candidate_association: bool = True,
        enable_adaptive_gate: bool = True,
        enable_short_gap_reacquisition: bool = True,
        enable_camera_motion_compensation: bool = False,
        enable_scale_normalization: bool = True,
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.max_prediction_gap = max_prediction_gap
        self.max_interpolation_gap = max_interpolation_gap
        self.base_gating_radius = base_gating_radius_px
        self.max_valid_speed = max_valid_speed_px_per_frame
        
        self.enable_multi_candidate = enable_multi_candidate_association
        self.enable_adaptive_gate = enable_adaptive_gate
        self.enable_reacquisition = enable_short_gap_reacquisition
        self.enable_camera_comp = enable_camera_motion_compensation
        self.enable_scale_norm = enable_scale_normalization

    def track_video_candidates(
        self,
        frame_candidates: List[List[BallObservation]],
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> List[TemporalBallPoint]:
        """
        Executes multi-pass temporal tracking across frame candidates.
        """
        num_frames = len(frame_candidates)
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        w, h = frame_size if frame_size is not None else (1280, 720)
        diag = math.hypot(w, h)
        diag_ref = math.hypot(1280, 720)
        scale = (diag / diag_ref) if self.enable_scale_norm else 1.0
        
        max_speed = self.max_valid_speed * scale
        base_radius = self.base_gating_radius * scale
        
        trajectory: List[TemporalBallPoint] = [
            TemporalBallPoint(
                frame_index=i,
                timestamp_seconds=float(i * dt),
                x_px=None,
                y_px=None,
                confidence=None,
                state=BallState.MISSING,
                source="none"
            )
            for i in range(num_frames)
        ]

        # Pass 1: Identify High-Confidence Anchors & Consistent Tracklet Seeds
        for i, candidates in enumerate(frame_candidates):
            high_conf = [c for c in candidates if c.confidence >= self.high_conf_thresh]
            if high_conf:
                if self.enable_multi_candidate:
                    best = None
                    best_score = -1.0
                    for c in high_conf:
                        is_static = False
                        static_count = 0
                        checked_offsets = 0
                        for offset in (-2, -1, 1, 2):
                            adj_idx = i + offset
                            if 0 <= adj_idx < num_frames:
                                checked_offsets += 1
                                for ac in frame_candidates[adj_idx]:
                                    if math.hypot(ac.x_px - c.x_px, ac.y_px - c.y_px) < 4.0 * scale:
                                        static_count += 1
                                        break
                        if checked_offsets > 0 and (static_count >= 2 or (checked_offsets <= 2 and static_count >= 1)):
                            is_static = True
                        
                        score = c.confidence - (0.5 if is_static else 0.0)
                        if score > best_score and not is_static:
                            best_score = score
                            best = c
                    
                    if best is not None:
                        trajectory[i].x_px = float(best.x_px)
                        trajectory[i].y_px = float(best.y_px)
                        trajectory[i].confidence = float(best.confidence)
                        trajectory[i].state = BallState.DETECTED
                        trajectory[i].source = "high_conf_detector"
                else:
                    best = max(high_conf, key=lambda c: c.confidence)
                    trajectory[i].x_px = float(best.x_px)
                    trajectory[i].y_px = float(best.y_px)
                    trajectory[i].confidence = float(best.confidence)
                    trajectory[i].state = BallState.DETECTED
                    trajectory[i].source = "high_conf_detector"

        # Pass 2: Forward Temporal Kinematic Gating with Adaptive Search & Reacquisition
        kf = KinematicFilter(dt=dt)
        consecutive_missing = 0

        for i in range(num_frames):
            p = trajectory[i]
            candidates = frame_candidates[i]

            if p.state == BallState.DETECTED and p.x_px is not None and p.y_px is not None:
                if kf.initialized:
                    kf.predict()
                    kf.update(p.x_px, p.y_px, measurement_var=2.0)
                else:
                    kf.initialize(p.x_px, p.y_px)
                consecutive_missing = 0
            elif kf.initialized and consecutive_missing < self.max_prediction_gap:
                pred_x, pred_y = kf.predict()
                
                if self.enable_adaptive_gate:
                    vx, vy = kf.get_velocity()
                    speed = math.hypot(vx, vy)
                    unc = math.sqrt(max(1.0, kf.P[0, 0] + kf.P[1, 1]))
                    adaptive_radius = min(220.0 * scale, max(base_radius, base_radius + 0.08 * speed + 2.5 * unc + 20.0 * consecutive_missing * scale))
                else:
                    base_r = kf.get_gating_radius(base_radius=45.0)
                    adaptive_radius = min(150.0, base_r + 15.0 * consecutive_missing)

                valid_candidates = []
                for c in candidates:
                    if c.confidence >= self.low_conf_thresh:
                        dist = math.hypot(c.x_px - pred_x, c.y_px - pred_y)
                        if dist <= adaptive_radius:
                            if self.enable_multi_candidate:
                                vx, vy = kf.get_velocity()
                                v_speed = math.hypot(vx, vy)
                                if v_speed > 5.0:
                                    dx = c.x_px - (pred_x - vx * dt)
                                    dy = c.y_px - (pred_y - vy * dt)
                                    disp = math.hypot(dx, dy)
                                    cos_theta = (dx * vx + dy * vy) / (disp * v_speed + 1e-6)
                                    dir_penalty = max(0.0, (1.0 - cos_theta) * 20.0 * scale)
                                else:
                                    dir_penalty = 0.0
                                
                                cost = dist - 30.0 * c.confidence * scale + dir_penalty
                                valid_candidates.append((c, dist, cost))
                            else:
                                valid_candidates.append((c, dist, dist - 25.0 * c.confidence))

                if valid_candidates:
                    valid_candidates.sort(key=lambda item: item[2])
                    best_cand, best_dist, _ = valid_candidates[0]

                    trajectory[i].x_px = float(best_cand.x_px)
                    trajectory[i].y_px = float(best_cand.y_px)
                    prox_score = max(0.0, 1.0 - (best_dist / adaptive_radius))
                    trajectory[i].confidence = float(0.5 * best_cand.confidence + 0.5 * prox_score)
                    trajectory[i].state = BallState.TRACKED if best_cand.confidence < self.high_conf_thresh else BallState.DETECTED
                    trajectory[i].source = "gated_candidate"
                    kf.update(best_cand.x_px, best_cand.y_px, measurement_var=4.0)
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
                    if consecutive_missing <= self.max_prediction_gap:
                        trajectory[i].x_px = float(pred_x)
                        trajectory[i].y_px = float(pred_y)
                        decay = max(0.1, 1.0 - 0.2 * consecutive_missing)
                        trajectory[i].confidence = float(0.6 * decay)
                        trajectory[i].state = BallState.PREDICTED
                        trajectory[i].source = "kinematic_kalman"
            elif self.enable_reacquisition and candidates:
                valid_seeds = [c for c in candidates if c.confidence >= self.low_conf_thresh]
                best_seed = None
                best_seed_score = -1.0
                for c in valid_seeds:
                    has_motion_support = False
                    if i + 1 < num_frames:
                        for nxt in frame_candidates[i + 1]:
                            if nxt.confidence >= self.low_conf_thresh:
                                step = math.hypot(nxt.x_px - c.x_px, nxt.y_px - c.y_px)
                                if 5.0 * scale < step < max_speed:
                                    has_motion_support = True
                                    break
                    if c.confidence >= 0.05 or has_motion_support:
                        score = c.confidence + (0.2 if has_motion_support else 0.0)
                        if score > best_seed_score:
                            best_seed_score = score
                            best_seed = c
                
                if best_seed is not None:
                    kf.initialize(best_seed.x_px, best_seed.y_px)
                    trajectory[i].x_px = float(best_seed.x_px)
                    trajectory[i].y_px = float(best_seed.y_px)
                    trajectory[i].confidence = float(best_seed.confidence)
                    trajectory[i].state = BallState.TRACKED
                    trajectory[i].source = "reacquired_candidate"
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
            elif not kf.initialized and candidates:
                valid_seeds = [c for c in candidates if c.confidence >= self.low_conf_thresh]
                if valid_seeds:
                    best_seed = max(valid_seeds, key=lambda c: c.confidence)
                    if best_seed.confidence >= 0.05:
                        kf.initialize(best_seed.x_px, best_seed.y_px)
                        trajectory[i].x_px = float(best_seed.x_px)
                        trajectory[i].y_px = float(best_seed.y_px)
                        trajectory[i].confidence = float(best_seed.confidence)
                        trajectory[i].state = BallState.TRACKED
                        trajectory[i].source = "seed_candidate"
                        consecutive_missing = 0
            else:
                consecutive_missing += 1

        # Pass 3: Backward Confirmation Pass for Early Missing Frames
        kf_back = KinematicFilter(dt=dt)
        for i in range(num_frames - 1, -1, -1):
            p = trajectory[i]
            if p.state in (BallState.DETECTED, BallState.TRACKED) and p.x_px is not None and p.y_px is not None:
                kf_back.update(p.x_px, p.y_px, measurement_var=2.0)
            elif kf_back.initialized and p.state == BallState.MISSING:
                pred_x, pred_y = kf_back.predict()
                back_radius = kf_back.get_gating_radius(base_radius=base_radius)
                candidates = frame_candidates[i]
                valid_candidates = []
                for c in candidates:
                    if c.confidence >= self.low_conf_thresh:
                        dist = math.hypot(c.x_px - pred_x, c.y_px - pred_y)
                        if dist <= back_radius:
                            valid_candidates.append((c, dist))
                if valid_candidates:
                    valid_candidates.sort(key=lambda item: item[1] - 25.0 * item[0].confidence)
                    best_cand, best_dist = valid_candidates[0]
                    trajectory[i].x_px = float(best_cand.x_px)
                    trajectory[i].y_px = float(best_cand.y_px)
                    prox_score = max(0.0, 1.0 - (best_dist / back_radius))
                    trajectory[i].confidence = float(0.5 * best_cand.confidence + 0.5 * prox_score)
                    trajectory[i].state = BallState.TRACKED
                    trajectory[i].source = "backward_gated_candidate"
                    kf_back.update(best_cand.x_px, best_cand.y_px, measurement_var=4.0)

        # Pass 4: Physics Consistency Verification & Impossible Jump Rejection
        for i in range(1, num_frames):
            curr = trajectory[i]
            prev = trajectory[i - 1]
            if curr.x_px is not None and prev.x_px is not None:
                step_dist = math.hypot(curr.x_px - prev.x_px, curr.y_px - prev.y_px)
                if step_dist > max_speed and curr.state in (BallState.PREDICTED, BallState.TRACKED):
                    curr.x_px = None
                    curr.y_px = None
                    curr.confidence = None
                    curr.state = BallState.MISSING
                    curr.source = "rejected_outlier"

        # Pass 5: Fallback Gap-Limited Interpolation for Remaining Short Gaps (<= 3 frames)
        valid_indices = [idx for idx, pt in enumerate(trajectory) if pt.x_px is not None and pt.state in (BallState.DETECTED, BallState.TRACKED, BallState.PREDICTED)]
        
        for k in range(len(valid_indices) - 1):
            idx_start = valid_indices[k]
            idx_end = valid_indices[k + 1]
            gap = idx_end - idx_start - 1
            
            if 0 < gap <= self.max_interpolation_gap:
                p_start = trajectory[idx_start]
                p_end = trajectory[idx_end]
                
                x_start, y_start = p_start.x_px, p_start.y_px
                x_end, y_end = p_end.x_px, p_end.y_px
                
                for step, mid_idx in enumerate(range(idx_start + 1, idx_end), 1):
                    if trajectory[mid_idx].state == BallState.MISSING:
                        alpha = step / (gap + 1)
                        trajectory[mid_idx].x_px = float(x_start + alpha * (x_end - x_start))
                        trajectory[mid_idx].y_px = float(y_start + alpha * (y_end - y_start))
                        trajectory[mid_idx].confidence = None  # Explicitly null per specification
                        trajectory[mid_idx].state = BallState.INTERPOLATED
                        trajectory[mid_idx].source = "linear_interpolation"

        # Pass 6: Compute Velocity & Smooth Speeds
        for i in range(num_frames):
            p = trajectory[i]
            if p.x_px is not None and i > 0 and trajectory[i-1].x_px is not None:
                vx = (p.x_px - trajectory[i-1].x_px) / dt
                vy = (p.y_px - trajectory[i-1].y_px) / dt
                p.velocity_px_per_sec = (float(vx), float(vy))

        return trajectory
