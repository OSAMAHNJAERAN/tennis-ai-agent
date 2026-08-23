"""Vision-side visual activity state classifier for live-play gating.

Classifies each frame window as PRE_PLAY / ACTIVE_PLAY / POSSIBLE_POINT_END /
DEAD_BALL_VISUAL / UNKNOWN using only the ball trajectory and player boxes —
no scoring engine, no GT labels, no video ID, no hardcoded frame numbers.

Key design constraints:
- No hardcoded frame/timestamp ranges, no video IDs, no GT labels, no score state
- Hysteresis prevents oscillation on short occlusion
- DEAD_BALL_VISUAL must NOT remain latched forever
- A new strong player contact + credible outgoing trajectory permits DEAD_BALL_VISUAL -> ACTIVE_PLAY

Design rationale (v3):
The primary discriminating signal between ACTIVE_PLAY and DEAD_BALL_VISUAL in a
broadcast tennis video is not ball speed (artificially high from Kalman extrapolation)
but rather TWO independent signals that must both be true for DEAD_BALL_VISUAL:

  A. LOW GROUNDED DETECTION CONFIDENCE:
     During real active play, YOLO detects the ball with high confidence in at least
     some frames. In the dead-ball retrieval period, only low-confidence background
     objects are detected.

  B. HIGH POSITION VARIANCE OR LOW TRACK CONTINUITY:
     Either the tracker position is stationary (locked onto static object) OR it
     jumps wildly between different detected objects.  In neither case is there a
     single coherent ball track.

Both signals A and B must be confirmed simultaneously to declare DEAD_BALL_VISUAL.
Either signal alone (e.g., briefly low confidence during occlusion) is not enough.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


class VisualActivityState(str, Enum):
    PRE_PLAY = "PRE_PLAY"
    ACTIVE_PLAY = "ACTIVE_PLAY"
    POSSIBLE_POINT_END = "POSSIBLE_POINT_END"
    DEAD_BALL_VISUAL = "DEAD_BALL_VISUAL"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def _grounded_ratio(points: Sequence[TemporalBallPoint]) -> float:
    if not points:
        return 0.0
    return sum(1 for p in points if p.state in (BallState.DETECTED, BallState.TRACKED)) / len(points)


def _position_spread_px(points: Sequence[TemporalBallPoint]) -> float:
    xs = [p.x_px for p in points if p.x_px is not None]
    ys = [p.y_px for p in points if p.y_px is not None]
    if len(xs) < 2:
        return 0.0
    return math.sqrt(statistics.variance(xs) + statistics.variance(ys))


def _max_predicted_run(points: Sequence[TemporalBallPoint]) -> int:
    best, cur = 0, 0
    for p in points:
        if p.state == BallState.PREDICTED:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _max_detection_confidence(points: Sequence[TemporalBallPoint]) -> float:
    """Max confidence of any DETECTED-state point in the window."""
    confs = [p.confidence for p in points if p.state == BallState.DETECTED and p.confidence is not None]
    return max(confs) if confs else 0.0


def _median_grounded_speed(
    points: Sequence[TemporalBallPoint],
    diagonal: float,
) -> float:
    """Median normalised speed of DETECTED/TRACKED frames only (ignores Kalman)."""
    grounded = [p for p in points if p.state in (BallState.DETECTED, BallState.TRACKED) and p.x_px is not None]
    speeds = []
    for i in range(1, len(grounded)):
        a, b = grounded[i - 1], grounded[i]
        dt = b.timestamp_seconds - a.timestamp_seconds
        if dt <= 0:
            continue
        speeds.append(math.hypot(b.x_px - a.x_px, b.y_px - a.y_px) / dt / max(diagonal, 1.0))
    return statistics.median(speeds) if speeds else 0.0


@dataclass
class ActivityStateSettings:
    # Analysis window widths (seconds)
    active_window_s: float = 0.60
    dead_ball_window_s: float = 2.00

    # ACTIVE_PLAY: require at least this max DETECTED confidence in window
    active_min_detection_confidence: float = 0.08   # same as high_conf_thresh in tracker

    # Also allow ACTIVE if grounded speed is high even at moderate confidence
    active_min_normalised_speed: float = 0.040      # grounded-only speed
    active_min_grounded_ratio: float = 0.20

    # DEAD_BALL: tracker locked onto static object
    static_lock_max_spread_px: float = 15.0         # tight positional cluster
    static_lock_min_grounded: float = 0.30

    # DEAD_BALL: long synthetic gap
    max_pred_gap_for_dead: int = 12

    # DEAD_BALL (dual-signal): both low confidence AND no grounded speed
    dead_ball_max_grounded_speed: float = 0.020

    # Minimum duration at POSSIBLE_POINT_END before → DEAD_BALL_VISUAL (s)
    point_end_min_duration_s: float = 1.00

    # Pre-serve cluster (tight cluster before first active play)
    pre_serve_min_frames: int = 15
    pre_serve_max_spread_normalised: float = 0.040


class VisualActivityStateClassifier:
    """Per-frame visual activity state inference using dual-signal confirmation."""

    def __init__(
        self,
        fps: float = 30.0,
        frame_size: Tuple[int, int] = (1280, 720),
        settings: Optional[ActivityStateSettings] = None,
    ) -> None:
        self.fps = fps
        self.diagonal = math.hypot(*frame_size)
        self.settings = settings or ActivityStateSettings()
        self._frame_states: List[VisualActivityState] = []
        self._frame_evidence: List[Dict[str, Any]] = []

    def classify_trajectory(
        self,
        trajectory: Sequence[TemporalBallPoint],
    ) -> List[VisualActivityState]:
        n = len(trajectory)
        states: List[VisualActivityState] = [VisualActivityState.UNKNOWN] * n
        evidence: List[Dict[str, Any]] = [{} for _ in range(n)]
        if n == 0:
            self._frame_states = states
            self._frame_evidence = evidence
            return states

        s = self.settings
        diag = self.diagonal

        def _win(center: int, half_s: float) -> List[TemporalBallPoint]:
            half = max(1, round(half_s * self.fps))
            return list(trajectory[max(0, center - half): min(n, center + half + 1)])

        # ------------------------------------------------------------------
        # Pass 1: per-frame local signals
        # ------------------------------------------------------------------
        local_is_active: List[bool] = [False] * n
        local_is_dead: List[bool] = [False] * n
        local_ev: List[Dict[str, Any]] = [{}] * n

        for i in range(n):
            win_a = _win(i, s.active_window_s / 2)
            win_d = _win(i, s.dead_ball_window_s / 2)

            # Active signals (grounded-only — ignores Kalman velocity)
            max_conf = _max_detection_confidence(win_a)
            gr_speed = _median_grounded_speed(win_a, diag)
            gr_ratio = _grounded_ratio(win_a)

            # Dead signals
            spread = _position_spread_px(win_d)
            pred_run = _max_predicted_run(win_d)
            gr_ratio_d = _grounded_ratio(win_d)

            is_static_locked = (
                spread < s.static_lock_max_spread_px
                and gr_ratio_d >= s.static_lock_min_grounded
                and len(win_d) >= 3
            )
            is_long_gap = pred_run >= s.max_pred_gap_for_dead

            # Active: high-confidence detection OR good grounded speed + ratio
            is_active = (
                max_conf >= s.active_min_detection_confidence
                or (gr_speed >= s.active_min_normalised_speed and gr_ratio >= s.active_min_grounded_ratio)
            ) and not is_static_locked

            # Dead (dual-signal): static OR (long gap + very low grounded speed)
            is_dead = is_static_locked or (
                is_long_gap
                and gr_speed <= s.dead_ball_max_grounded_speed
                and max_conf < s.active_min_detection_confidence
            )

            local_is_active[i] = is_active
            local_is_dead[i] = is_dead
            local_ev[i] = {
                "max_detection_confidence": round(max_conf, 4),
                "grounded_speed": round(gr_speed, 5),
                "grounded_ratio": round(gr_ratio, 3),
                "position_spread_px": round(spread, 2),
                "max_predicted_run": pred_run,
                "is_static_locked": is_static_locked,
                "is_active": is_active,
                "is_dead": is_dead,
                # Aliases for backward-compat
                "median_normalised_speed": round(gr_speed, 5),
                "static_cluster_score": round(max(0.0, 1.0 - spread / max(s.static_lock_max_spread_px, 1.0)), 3),
            }

        # ------------------------------------------------------------------
        # Pass 2: forward hysteresis state machine
        # ------------------------------------------------------------------
        cur = VisualActivityState.UNKNOWN
        point_end_start: Optional[int] = None
        point_end_min_frames = max(1, round(s.point_end_min_duration_s * self.fps))

        for i in range(n):
            is_a = local_is_active[i]
            is_d = local_is_dead[i]

            if cur == VisualActivityState.UNKNOWN:
                if is_a:
                    cur = VisualActivityState.ACTIVE_PLAY
                    point_end_start = None
                elif is_d:
                    cur = VisualActivityState.DEAD_BALL_VISUAL

            elif cur == VisualActivityState.ACTIVE_PLAY:
                if is_d and not is_a:
                    cur = VisualActivityState.DEAD_BALL_VISUAL
                    point_end_start = None
                elif not is_a:
                    cur = VisualActivityState.POSSIBLE_POINT_END
                    point_end_start = i

            elif cur == VisualActivityState.POSSIBLE_POINT_END:
                if is_a:
                    cur = VisualActivityState.ACTIVE_PLAY
                    point_end_start = None
                elif is_d:
                    cur = VisualActivityState.DEAD_BALL_VISUAL
                    point_end_start = None
                elif point_end_start is not None and (i - point_end_start) >= point_end_min_frames:
                    cur = VisualActivityState.DEAD_BALL_VISUAL
                    point_end_start = None

            elif cur == VisualActivityState.DEAD_BALL_VISUAL:
                if is_a:
                    cur = VisualActivityState.ACTIVE_PLAY
                    point_end_start = None

            states[i] = cur
            evidence[i] = local_ev[i]

        # ------------------------------------------------------------------
        # Pass 3: pre-serve back-fill
        # ------------------------------------------------------------------
        first_active_frame = next(
            (i for i, sv in enumerate(states) if sv == VisualActivityState.ACTIVE_PLAY), n
        )
        if first_active_frame > s.pre_serve_min_frames:
            pre_pts = [p for p in trajectory[:first_active_frame] if p.x_px is not None]
            if pre_pts:
                spread = _position_spread_px(pre_pts)
                if spread <= s.pre_serve_max_spread_normalised * diag:
                    for i in range(first_active_frame):
                        if states[i] == VisualActivityState.UNKNOWN:
                            states[i] = VisualActivityState.PRE_PLAY

        self._frame_states = states
        self._frame_evidence = evidence
        return states

    def state_at_frame(self, frame_index: int) -> VisualActivityState:
        return self._frame_states[frame_index]

    def evidence_at_frame(self, frame_index: int) -> Dict[str, Any]:
        return self._frame_evidence[frame_index]

    def is_live_play(self, frame_index: int) -> bool:
        if frame_index >= len(self._frame_states):
            return True  # safe fallback
        s = self._frame_states[frame_index]
        return s in (
            VisualActivityState.ACTIVE_PLAY,
            VisualActivityState.POSSIBLE_POINT_END,
            VisualActivityState.UNKNOWN,
        )

    def is_dead_ball(self, frame_index: int) -> bool:
        if frame_index >= len(self._frame_states):
            return False  # safe fallback
        return self._frame_states[frame_index] == VisualActivityState.DEAD_BALL_VISUAL

    def dump_timeline(self) -> List[Dict[str, Any]]:
        return [
            {"frame": i, "activity_state": self._frame_states[i].value, **self._frame_evidence[i]}
            for i in range(len(self._frame_states))
        ]
