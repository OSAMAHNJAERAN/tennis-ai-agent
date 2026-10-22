import math
import operator
from collections.abc import Mapping, Sequence
from fractions import Fraction
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

    def predict(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """Predicts next state and returns expected (x, y)."""
        if not self.initialized:
            return (0.0, 0.0)
        # Omitting dt preserves the established fixed-F/Q calculation exactly.
        # The opt-in source-clock path advances by measured elapsed time and
        # scales the existing per-reference-step process noise by elapsed time.
        transition, process_noise = self.F, self.Q
        if dt is not None:
            if not math.isfinite(dt) or dt <= 0 or not math.isfinite(dt * dt):
                raise ValueError('Source-clock filter delta must be finite, positive and numerically representable')
            transition = np.eye(6, dtype=np.float64)
            transition[0, 2] = transition[1, 3] = dt
            transition[0, 4] = transition[1, 5] = .5 * dt * dt
            transition[2, 4] = transition[3, 5] = dt
            process_noise = self.Q * (dt / self.dt)
        self.x = transition @ self.x
        self.P = transition @ self.P @ transition.T + process_noise
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

class _FrameBoundsCandidates(Sequence):
    """Filter requested frames without retaining a second whole-video list."""

    def __init__(self, frames, width, height):
        self.frames, self.width, self.height = frames, width, height

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return [candidate for candidate in self.frames[index]
                if 0.0 <= candidate.x_px < self.width and 0.0 <= candidate.y_px < self.height]


class _TrajectoryWindow(Sequence):
    """Relative writable indices into appended output; no segment copy/cache."""

    def __init__(self, output, indices):
        self.output, self.indices = output, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        selected = self.indices[index]
        return _TrajectoryWindow(self.output, selected) if isinstance(selected, range) else self.output[selected]

    def __setitem__(self, index, point):
        if isinstance(index, slice):
            raise TypeError('Write back one temporal point by index')
        self.output[self.indices[operator.index(index)]] = point


class _SourceClock:
    """Read exact source tokens lazily; invalid edges never invent cadence.

    Playback timestamps are deliberately not consumed here. A row with valid
    source time and a false incoming transition can initialize a new segment.
    """

    def __init__(self, records, frame_count, fps):
        if len(records) != frame_count:
            raise ValueError('Source timing must contain exactly one record per candidate frame')
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Source-clock tracking requires a finite positive reference FPS')
        self.records = records
        self.reference_step = 1 / Fraction(str(fps))
        self.reference_seconds = float(self.reference_step)
        if not math.isfinite(self.reference_seconds * self.reference_seconds):
            raise ValueError('Source-clock reference step is too large for the image-space filter')

    def at(self, index):
        row = self.records[index]
        if not isinstance(row, Mapping):
            return None
        pts, base, frame = row.get('source_pts'), row.get('source_time_base'), row.get('source_frame_index')
        if (isinstance(pts, bool) or not isinstance(pts, (str, int))
                or isinstance(base, bool) or not isinstance(base, (str, int, Fraction))
                or type(frame) is not int or frame < 0):
            return None
        try:
            ticks, time_base = int(pts), Fraction(base)
            if ticks == 9223372036854775807 or time_base <= 0:
                return None
            exact = ticks * time_base
            if not math.isfinite(float(exact)):
                return None
            return exact
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None

    def edge(self, index):
        if index <= 0:
            return None
        previous, current = self.records[index - 1], self.records[index]
        if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
            return None
        if current.get('measurement_transition_allowed') is not True:
            return None
        segment = current.get('segment_id')
        if not isinstance(segment, str) or not segment.strip() or segment != previous.get('segment_id'):
            return None
        before, after = self.at(index - 1), self.at(index)
        if before is None or after is None:
            return None
        if current['source_frame_index'] != previous['source_frame_index'] + 1:
            return None
        delta = after - before
        try:
            seconds = float(delta)
        except OverflowError:
            return None
        if delta <= 0 or not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(seconds * seconds):
            return None
        return delta


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
        max_prediction_gap: int = 6,
        max_interpolation_gap: int = 6,
        max_valid_speed_px_per_frame: float = 90.0,
        base_gating_radius_px: float = 45.0,
        enable_multi_candidate_association: bool = True,
        enable_adaptive_gate: bool = True,
        enable_short_gap_reacquisition: bool = True,
        enable_camera_motion_compensation: bool = False,
        enable_scale_normalization: bool = True,
        enable_frame_bounds_filter: bool = True,
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
        self.enable_frame_bounds_filter = enable_frame_bounds_filter

    def track_video_candidates(
        self,
        frame_candidates: Sequence[Sequence[BallObservation]],
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        *, output=None, source_timing=None,
    ) -> Sequence[TemporalBallPoint]:
        """
        Executes multi-pass temporal tracking across frame candidates.

        Without output, return the established list. A supplied mutable indexed
        sink needs len/getitem/setitem/append; append one point per input frame
        after its existing prefix, process only that new range, and return the
        same sink. Stored reads may be snapshots: every mutation is written
        back explicitly. Frame indices/timestamps are local to this input;
        callers combining camera intervals must remap their appended range.
        This offline result sink is not a resumable filter checkpoint.

        source_timing opts into exact source-PTS elapsed-time association. It is
        a lazy sequence of existing interval mappings (source_pts/time_base,
        source_frame_index, segment_id, measurement_transition_allowed). The
        resulting timestamps still use the normalized playback clock. This
        distinct offline policy uses causal forward association, no backward
        confirmation or future-supported seeding, then explicitly labeled
        source-time interpolation. It does not qualify live or physical speed.
        """
        if source_timing is not None:
            return self._track_source_timed_candidates(
                frame_candidates, source_timing, fps, frame_size, output=output)
        num_frames = len(frame_candidates)
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        
        w, h = frame_size if frame_size is not None else (1280, 720)
        diag = math.hypot(w, h)
        diag_ref = math.hypot(1280, 720)
        scale = (diag / diag_ref) if self.enable_scale_norm else 1.0
        
        max_speed = self.max_valid_speed * scale
        base_radius = self.base_gating_radius * scale

        if self.enable_frame_bounds_filter:
            frame_candidates = _FrameBoundsCandidates(frame_candidates, w, h)
        
        output = [] if output is None else output
        output_start = len(output)
        for i in range(num_frames):
            output.append(TemporalBallPoint(
                frame_index=i,
                timestamp_seconds=float(i * dt),
                x_px=None,
                y_px=None,
                confidence=None,
                state=BallState.MISSING,
                source="none"
            ))
        trajectory = _TrajectoryWindow(output, range(output_start, output_start + num_frames))

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
                        point = trajectory[i]
                        point.x_px = float(best.x_px)
                        point.y_px = float(best.y_px)
                        point.confidence = float(best.confidence)
                        point.state = BallState.DETECTED
                        point.source = "high_conf_detector"
                        trajectory[i] = point
                else:
                    best = max(high_conf, key=lambda c: c.confidence)
                    point = trajectory[i]
                    point.x_px = float(best.x_px)
                    point.y_px = float(best.y_px)
                    point.confidence = float(best.confidence)
                    point.state = BallState.DETECTED
                    point.source = "high_conf_detector"
                    trajectory[i] = point

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

                    p.x_px = float(best_cand.x_px)
                    p.y_px = float(best_cand.y_px)
                    prox_score = max(0.0, 1.0 - (best_dist / adaptive_radius))
                    p.confidence = float(0.5 * best_cand.confidence + 0.5 * prox_score)
                    p.state = BallState.TRACKED if best_cand.confidence < self.high_conf_thresh else BallState.DETECTED
                    p.source = "gated_candidate"
                    trajectory[i] = p
                    kf.update(best_cand.x_px, best_cand.y_px, measurement_var=4.0)
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
                    if consecutive_missing <= self.max_prediction_gap:
                        p.x_px = float(pred_x)
                        p.y_px = float(pred_y)
                        decay = max(0.1, 1.0 - 0.2 * consecutive_missing)
                        p.confidence = float(0.6 * decay)
                        p.state = BallState.PREDICTED
                        p.source = "kinematic_kalman"
                        trajectory[i] = p
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
                    p.x_px = float(best_seed.x_px)
                    p.y_px = float(best_seed.y_px)
                    p.confidence = float(best_seed.confidence)
                    p.state = BallState.TRACKED
                    p.source = "reacquired_candidate"
                    trajectory[i] = p
                    consecutive_missing = 0
                else:
                    consecutive_missing += 1
            elif not kf.initialized and candidates:
                valid_seeds = [c for c in candidates if c.confidence >= self.low_conf_thresh]
                if valid_seeds:
                    best_seed = max(valid_seeds, key=lambda c: c.confidence)
                    if best_seed.confidence >= 0.05:
                        kf.initialize(best_seed.x_px, best_seed.y_px)
                        p.x_px = float(best_seed.x_px)
                        p.y_px = float(best_seed.y_px)
                        p.confidence = float(best_seed.confidence)
                        p.state = BallState.TRACKED
                        p.source = "seed_candidate"
                        trajectory[i] = p
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
                    p.x_px = float(best_cand.x_px)
                    p.y_px = float(best_cand.y_px)
                    prox_score = max(0.0, 1.0 - (best_dist / back_radius))
                    p.confidence = float(0.5 * best_cand.confidence + 0.5 * prox_score)
                    p.state = BallState.TRACKED
                    p.source = "backward_gated_candidate"
                    trajectory[i] = p
                    kf_back.update(best_cand.x_px, best_cand.y_px, measurement_var=4.0)

        # Pass 4: Frame Bounds & Physics Consistency Verification
        if self.enable_frame_bounds_filter:
            for i, point in enumerate(trajectory):
                if (
                    point.x_px is not None
                    and point.y_px is not None
                    and not (0.0 <= point.x_px < w and 0.0 <= point.y_px < h)
                ):
                    point.x_px = None
                    point.y_px = None
                    point.confidence = None
                    point.state = BallState.MISSING
                    point.source = "rejected_out_of_frame"
                    trajectory[i] = point

        # Impossible Jump Rejection
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
                    trajectory[i] = curr

        # Pass 5: Fallback Gap-Limited Interpolation for Remaining Short Gaps (<= 3 frames)
        idx_start = None
        p_start = None
        for idx_end, p_end in enumerate(trajectory):
            if p_end.x_px is None or p_end.state not in (BallState.DETECTED, BallState.TRACKED, BallState.PREDICTED):
                continue
            if idx_start is None:
                idx_start, p_start = idx_end, p_end
                continue
            gap = idx_end - idx_start - 1
            
            if 0 < gap <= self.max_interpolation_gap:
                x_start, y_start = p_start.x_px, p_start.y_px
                x_end, y_end = p_end.x_px, p_end.y_px
                gap_speed = math.hypot(x_end - x_start, y_end - y_start) / (gap + 1)
                
                if gap_speed <= max_speed:
                    for step, mid_idx in enumerate(range(idx_start + 1, idx_end), 1):
                        middle = trajectory[mid_idx]
                        if middle.state == BallState.MISSING:
                            alpha = step / (gap + 1)
                            middle.x_px = float(x_start + alpha * (x_end - x_start))
                            middle.y_px = float(y_start + alpha * (y_end - y_start))
                            middle.confidence = None  # Explicitly null per specification
                            middle.state = BallState.INTERPOLATED
                            middle.source = "linear_interpolation"
                            trajectory[mid_idx] = middle
            idx_start, p_start = idx_end, p_end

        # Pass 6: Compute Velocity & Smooth Speeds
        for i in range(num_frames):
            p = trajectory[i]
            if p.x_px is not None and i > 0 and trajectory[i-1].x_px is not None:
                vx = (p.x_px - trajectory[i-1].x_px) / dt
                vy = (p.y_px - trajectory[i-1].y_px) / dt
                p.velocity_px_per_sec = (float(vx), float(vy))
                trajectory[i] = p

        return output

    def _track_source_timed_candidates(self, frame_candidates, source_timing, fps, frame_size, *, output):
        """Offline source-clock policy, isolated from the retained legacy math.

        Frame settings are reference-cadence limits: predictions cover at most
        max_prediction_gap/fps seconds; a later observation can reconnect at
        most (max_prediction_gap+1)/fps seconds after its last observation.
        Interpolation retains its missing-frame count and additionally limits
        anchor span to (max_interpolation_gap+1)/fps seconds. Every intervening
        source edge must be eligible. No impact state is inferred here.
        """
        count = len(frame_candidates)
        clock = _SourceClock(source_timing, count, fps)
        for name in ('max_prediction_gap', 'max_interpolation_gap'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError('Source-clock gap limits must be nonnegative integer frame counts')
        width, height = frame_size if frame_size is not None else (1280, 720)
        scale = math.hypot(width, height) / math.hypot(1280, 720) if self.enable_scale_norm else 1.
        max_speed = self.max_valid_speed * scale / clock.reference_seconds
        base_radius = self.base_gating_radius * scale
        prediction_span = self.max_prediction_gap * clock.reference_step
        observation_span = (self.max_prediction_gap + 1) * clock.reference_step
        interpolation_span = (self.max_interpolation_gap + 1) * clock.reference_step
        if (not math.isfinite(max_speed) or max_speed < 0 or not math.isfinite(base_radius) or base_radius <= 0
                or not math.isfinite(float(observation_span) * float(observation_span))):
            raise ValueError('Source-clock motion limits must be finite and nonnegative')

        def candidates_at(index):
            # Nonfinite candidates must not poison the filter even when the
            # caller deliberately disables the optional frame-bounds filter.
            return [candidate for candidate in frame_candidates[index]
                    if math.isfinite(candidate.x_px) and math.isfinite(candidate.y_px)
                    and math.isfinite(candidate.confidence)
                    and (not self.enable_frame_bounds_filter
                         or (0 <= candidate.x_px < width and 0 <= candidate.y_px < height))]

        def static_in_eligible_past(index, candidate):
            if not self.enable_multi_candidate or index < 2:
                return False
            if (clock.edge(index) is None or clock.edge(index - 1) is None
                    or clock.at(index) - clock.at(index - 2) > 2 * clock.reference_step):
                return False
            # The timed policy never consults a future frame to suppress an
            # earlier observation. Two eligible past matches are required.
            return all(any(math.hypot(other.x_px - candidate.x_px, other.y_px - candidate.y_px) < 4 * scale
                           for other in candidates_at(prior)) for prior in (index - 2, index - 1))

        output = [] if output is None else output
        output_start = len(output)
        for index in range(count):
            output.append(TemporalBallPoint(index, index * clock.reference_seconds, None, None,
                                             source='none_source_pts_forward_only'))
        trajectory = _TrajectoryWindow(output, range(output_start, output_start + count))
        kf = KinematicFilter(clock.reference_seconds)
        last_observation = None
        had_observation = False

        for index in range(count):
            point, now, edge = trajectory[index], clock.at(index), clock.edge(index)
            reset_reason = ''
            if index == 0 or edge is None:
                kf = KinematicFilter(clock.reference_seconds)
                last_observation = None
                reset_reason = '_transition_reset'
            if now is None:
                reset_reason = '_invalid_source_timing'
                kf = KinematicFilter(clock.reference_seconds)
                last_observation = None
            if last_observation is not None:
                span = now - clock.at(last_observation)
                if (index - last_observation - 1 > self.max_prediction_gap or span > observation_span):
                    kf = KinematicFilter(clock.reference_seconds)
                    last_observation = None
                    reset_reason = '_filter_expired'
            candidates = [candidate for candidate in candidates_at(index)
                          if candidate.confidence >= self.low_conf_thresh
                          and not static_in_eligible_past(index, candidate)]
            strong = [candidate for candidate in candidates if candidate.confidence >= self.high_conf_thresh]

            chosen, chosen_source, predicted, radius = None, '', None, None
            if kf.initialized:
                # Expire before forming F(dt), including when a late anchor is
                # present: never update a stale filter at a later observation.
                span = now - clock.at(last_observation)
                if candidates or (index - last_observation <= self.max_prediction_gap and span <= prediction_span):
                    elapsed = float(edge)
                    predicted = kf.predict(dt=elapsed)
                    ratio = elapsed / clock.reference_seconds
                    uncertainty = math.sqrt(max(1., kf.P[0, 0] + kf.P[1, 1]))
                    velocity = math.hypot(*kf.get_velocity())
                    if self.enable_adaptive_gate:
                        radius = min(220 * scale * ratio,
                                     base_radius * ratio + .08 * velocity * ratio
                                     + 2.5 * uncertainty * math.sqrt(ratio)
                                     + 20 * (index - last_observation - 1) * scale * ratio)
                    else:
                        radius = base_radius * ratio
                    previous = trajectory[last_observation]
                    previous_output = trajectory[index - 1]
                    valid = [candidate for candidate in candidates
                             if math.hypot(candidate.x_px - previous.x_px, candidate.y_px - previous.y_px)
                             <= max_speed * float(span)
                             and (previous_output.x_px is None
                                  or math.hypot(candidate.x_px - previous_output.x_px,
                                                candidate.y_px - previous_output.y_px) <= max_speed * elapsed)
                             and math.hypot(candidate.x_px - predicted[0], candidate.y_px - predicted[1]) <= radius]
                    if valid:
                        chosen = min(valid, key=lambda candidate:
                                     math.hypot(candidate.x_px - predicted[0], candidate.y_px - predicted[1])
                                     - 30 * candidate.confidence * scale)
                        chosen_source = 'gated_candidate'
                    elif strong:
                        # A detector score cannot establish temporal identity.
                        # Keep a strong raw observation but break its linkage.
                        chosen = max(strong, key=lambda candidate: candidate.confidence)
                        chosen_source = 'high_conf_detector'
                        reset_reason = '_jump_reset'
                        kf = KinematicFilter(clock.reference_seconds)
                        last_observation = None
                else:
                    kf = KinematicFilter(clock.reference_seconds)
                    last_observation = None
                    reset_reason = '_filter_expired'

            if chosen is None and not kf.initialized:
                if strong:
                    chosen = max(strong, key=lambda candidate: candidate.confidence)
                    chosen_source = 'high_conf_detector'
                else:
                    # Reacquisition uses current visual support only. The
                    # legacy future-neighbor seed and backward pass are absent.
                    seeds = [candidate for candidate in candidates if candidate.confidence >= .05]
                    if seeds and (self.enable_reacquisition or not had_observation):
                        chosen = max(seeds, key=lambda candidate: candidate.confidence)
                        chosen_source = 'reacquired_candidate' if self.enable_reacquisition else 'seed_candidate'

            if chosen is not None:
                point.x_px, point.y_px = float(chosen.x_px), float(chosen.y_px)
                point.confidence = float(chosen.confidence)
                point.state = BallState.DETECTED if chosen.confidence >= self.high_conf_thresh else BallState.TRACKED
                point.source = chosen_source + '_source_pts_forward_only' + reset_reason
                if now is not None:
                    if kf.initialized:
                        kf.update(point.x_px, point.y_px, measurement_var=2. if point.state is BallState.DETECTED else 4.)
                    else:
                        kf.initialize(point.x_px, point.y_px)
                    last_observation = index
                    had_observation = True
            elif (predicted is not None and last_observation is not None
                  and index - last_observation <= self.max_prediction_gap
                  and now - clock.at(last_observation) <= prediction_span):
                predicted_inside = (not self.enable_frame_bounds_filter
                                    or (0 <= predicted[0] < width and 0 <= predicted[1] < height))
                previous_output = trajectory[index - 1]
                predicted_jump_allowed = (previous_output.x_px is None
                                          or math.hypot(predicted[0] - previous_output.x_px,
                                                        predicted[1] - previous_output.y_px) <= max_speed * float(edge))
                if predicted_inside and predicted_jump_allowed and all(math.isfinite(value) for value in predicted):
                    point.x_px, point.y_px = map(float, predicted)
                    point.confidence = float(.6 * max(.1, 1 - .2 * (index - last_observation)))
                    point.state = BallState.PREDICTED
                    point.source = 'kinematic_kalman_source_pts_forward_only'
                elif not predicted_jump_allowed:
                    point.source = 'rejected_source_pts_prediction_jump'
                    kf = KinematicFilter(clock.reference_seconds)
                    last_observation = None
            trajectory[index] = point

        # This is explicitly offline interpolation, not a causal live result.
        left_index = left = left_time = None
        for index in range(count):
            point, now = trajectory[index], clock.at(index)
            if (clock.edge(index) is None or '_jump_reset' in point.source or now is None
                    or (left_index is not None
                        and (index - left_index - 1 > self.max_interpolation_gap
                             or now - left_time > interpolation_span))):
                left_index = left = left_time = None
            if point.x_px is None:
                continue
            if left_index is not None:
                gap, span = index - left_index - 1, now - left_time
                if (0 < gap <= self.max_interpolation_gap and span > 0
                        and math.hypot(point.x_px - left.x_px, point.y_px - left.y_px) <= max_speed * float(span)):
                    for middle_index in range(left_index + 1, index):
                        middle = trajectory[middle_index]
                        if middle.state is BallState.MISSING:
                            alpha = float((clock.at(middle_index) - left_time) / span)
                            middle.x_px = float(left.x_px + alpha * (point.x_px - left.x_px))
                            middle.y_px = float(left.y_px + alpha * (point.y_px - left.y_px))
                            middle.confidence = None
                            middle.state = BallState.INTERPOLATED
                            middle.source = 'linear_interpolation_source_pts_offline'
                            trajectory[middle_index] = middle
                    point.source += '_offline_interpolation_endpoint'
                    trajectory[index] = point
            left_index, left, left_time = index, point, now

        for index in range(1, count):
            point, previous, edge = trajectory[index], trajectory[index - 1], clock.edge(index)
            disconnected = ('_jump_reset' in point.source or '_transition_reset' in point.source
                            or ('_filter_expired' in point.source and '_offline_interpolation_endpoint' not in point.source))
            if edge is not None and not disconnected and point.x_px is not None and previous.x_px is not None:
                seconds = float(edge)  # exact subtraction happened inside edge()
                point.velocity_px_per_sec = (float((point.x_px - previous.x_px) / seconds),
                                             float((point.y_px - previous.y_px) / seconds))
                trajectory[index] = point
        return output
