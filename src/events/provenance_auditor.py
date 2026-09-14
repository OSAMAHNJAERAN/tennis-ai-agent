"""Trajectory provenance auditor for Stage 2/3 candidates.

For every verified physical contact, analyses the surrounding trajectory
window to classify the dominant evidence quality:

    DIRECT_DOMINANT     — mostly DETECTED frames
    TRACKED_DOMINANT    — mostly TRACKED frames
    INTERPOLATED_DOMINANT — mostly INTERPOLATED frames
    PREDICTED_DOMINANT  — mostly PREDICTED frames (weak evidence)
    MIXED               — no single state dominates

Also detects:
  - Long synthetic gaps (≥ N consecutive PREDICTED frames)
  - Kalman velocity runaways (normalised speed > threshold in PREDICTED state)
  - Static-cluster lock (tracker captured by scoreboard / court logo)

Both of the latter two are strong signals that the contact is a Kalman
artefact rather than a genuine physical interaction.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


# ---------------------------------------------------------------------------
# Dominance enum labels
# ---------------------------------------------------------------------------

PROVENANCE_DIRECT_DOMINANT = "DIRECT_DOMINANT"
PROVENANCE_TRACKED_DOMINANT = "TRACKED_DOMINANT"
PROVENANCE_INTERPOLATED_DOMINANT = "INTERPOLATED_DOMINANT"
PROVENANCE_PREDICTED_DOMINANT = "PREDICTED_DOMINANT"
PROVENANCE_MISSING_DOMINANT = "MISSING_DOMINANT"
PROVENANCE_MIXED = "MIXED"


@dataclass
class ProvenanceAudit:
    """Provenance audit result for a single candidate or contact."""
    frame: int
    window_frames: int

    # State counts
    n_detected: int
    n_tracked: int
    n_interpolated: int
    n_predicted: int
    n_missing: int

    # Dominance
    dominance: str       # one of the PROVENANCE_* constants

    # Derived flags
    max_consecutive_predicted: int
    kalman_runaway_detected: bool
    static_cluster_locked: bool
    tracker_jumping_between_objects: bool
    position_spread_px: float
    median_normalised_speed: float
    dominant_state_fraction: float

    # Trust score 0-1 (higher = more grounded evidence)
    provenance_trust_score: float

    # Reason for low trust (if any)
    low_trust_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "window_frames": self.window_frames,
            "n_detected": self.n_detected,
            "n_tracked": self.n_tracked,
            "n_interpolated": self.n_interpolated,
            "n_predicted": self.n_predicted,
            "n_missing": self.n_missing,
            "dominance": self.dominance,
            "max_consecutive_predicted": self.max_consecutive_predicted,
            "kalman_runaway_detected": self.kalman_runaway_detected,
            "static_cluster_locked": self.static_cluster_locked,
            "tracker_jumping_between_objects": self.tracker_jumping_between_objects,
            "position_spread_px": round(self.position_spread_px, 2),
            "median_normalised_speed": round(self.median_normalised_speed, 5),
            "dominant_state_fraction": round(self.dominant_state_fraction, 3),
            "provenance_trust_score": round(self.provenance_trust_score, 3),
            "low_trust_reason": self.low_trust_reason,
        }



# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------

class ProvenanceAuditor:
    """Computes :class:`ProvenanceAudit` for a trajectory frame window.

    Args:
        fps:          Video frame rate.
        frame_size:   (width, height) in pixels.
        window_s:     Half-window in seconds around each candidate frame.
        max_pred_gap_for_rejection: Consecutive PREDICTED frames above which
                       the contact is treated as a Kalman artefact.
        kalman_speed_threshold:  Normalised speed (diagonal/s) above which
                       inter-frame motion in PREDICTED state is a runaway.
        static_spread_px:  Position spread below which the tracker is
                       assumed to be locked onto a static object.
    """

    def __init__(
        self,
        fps: float = 30.0,
        frame_size: Tuple[int, int] = (1280, 720),
        window_s: float = 0.33,
        max_pred_gap_for_rejection: int = 8,
        kalman_speed_threshold: float = 5.0,
        static_spread_px: float = 25.0,
        max_high_spread_px: float = 100.0,
    ) -> None:
        self.fps = fps
        self.diagonal = math.hypot(*frame_size)
        self.half_frames = max(2, round(window_s * fps))
        self.max_pred_gap = max_pred_gap_for_rejection
        self.kalman_thresh = kalman_speed_threshold
        self.static_spread = static_spread_px
        self.max_high_spread = max_high_spread_px


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit(
        self,
        frame: int,
        trajectory: Sequence[TemporalBallPoint],
    ) -> ProvenanceAudit:
        """Return a ``ProvenanceAudit`` for *frame* in *trajectory*."""
        n = len(trajectory)
        lo = max(0, frame - self.half_frames)
        hi = min(n, frame + self.half_frames + 1)
        window = list(trajectory[lo:hi])

        # State counts
        cnt: Counter = Counter()
        for pt in window:
            state = pt.state
            if hasattr(state, "value"):
                state = state.value
            cnt[state] += 1

        n_det = cnt.get(BallState.DETECTED.value, 0)
        n_trk = cnt.get(BallState.TRACKED.value, 0)
        n_itp = cnt.get(BallState.INTERPOLATED.value, 0)
        n_prd = cnt.get(BallState.PREDICTED.value, 0)
        n_mis = cnt.get(BallState.MISSING.value, 0)
        total = len(window)

        # Dominant state
        state_scores = {
            PROVENANCE_DIRECT_DOMINANT: n_det,
            PROVENANCE_TRACKED_DOMINANT: n_trk,
            PROVENANCE_INTERPOLATED_DOMINANT: n_itp,
            PROVENANCE_PREDICTED_DOMINANT: n_prd,
            PROVENANCE_MISSING_DOMINANT: n_mis,
        }
        if total == 0:
            dominance = PROVENANCE_MIXED
            dom_frac = 0.0
        else:
            best_label = max(state_scores, key=state_scores.get)  # type: ignore[arg-type]
            dom_count = state_scores[best_label]
            dom_frac = dom_count / total
            dominance = best_label if dom_frac >= 0.45 else PROVENANCE_MIXED

        # Max consecutive predicted
        max_pred_run = 0
        cur_run = 0
        for pt in window:
            st = pt.state if not hasattr(pt.state, "value") else pt.state.value  # noqa: E501 (keep consistent)
            if st == BallState.PREDICTED.value:
                cur_run += 1
                max_pred_run = max(max_pred_run, cur_run)
            else:
                cur_run = 0

        # Kalman runaway
        runaway = False
        for i in range(1, len(window)):
            a, b = window[i - 1], window[i]
            if b.state not in (BallState.PREDICTED, BallState.INTERPOLATED):
                continue
            if a.x_px is None or b.x_px is None:
                continue
            dt = b.timestamp_seconds - a.timestamp_seconds
            if dt <= 0:
                continue
            spd = math.hypot(b.x_px - a.x_px, b.y_px - a.y_px) / dt / max(self.diagonal, 1.0)
            if spd > self.kalman_thresh:
                runaway = True
                break

        # Position spread
        xs = [pt.x_px for pt in window if pt.x_px is not None]
        ys = [pt.y_px for pt in window if pt.y_px is not None]
        if len(xs) >= 2:
            spread = math.sqrt(
                statistics.variance(xs) + statistics.variance(ys)
            )
        else:
            spread = 0.0

        static_locked = spread < self.static_spread and len(xs) >= 3
        # High-spread: tracker jumping between different objects
        high_spread = spread >= self.max_high_spread

        # Median normalised speed
        speeds = []
        for i in range(1, len(window)):
            a, b = window[i - 1], window[i]
            if a.x_px is None or b.x_px is None:
                continue
            dt = b.timestamp_seconds - a.timestamp_seconds
            if dt <= 0:
                continue
            speeds.append(
                math.hypot(b.x_px - a.x_px, b.y_px - a.y_px) / dt / max(self.diagonal, 1.0)
            )
        med_spd = statistics.median(speeds) if speeds else 0.0

        # Trust score
        grounded_ratio = (n_det + n_trk) / max(total, 1)
        base_trust = (
            1.00 * (n_det / max(total, 1))
            + 0.90 * (n_trk / max(total, 1))
            + 0.70 * (n_itp / max(total, 1))
            + 0.55 * (n_prd / max(total, 1))
        )
        trust = base_trust
        if runaway:
            trust *= 0.30
        if static_locked:
            trust *= 0.20
        # Note: high_spread penalty removed — position spread is high for real
        # active-play events (ball in flight) so cannot be used as rejection signal here.
        if max_pred_run >= self.max_pred_gap:
            trust *= 0.40
        trust = max(0.0, min(1.0, trust))


        # Low trust reason
        low_reason: Optional[str] = None
        if trust < 0.50:
            if static_locked:
                low_reason = "TRACKER_LOCKED_STATIC_OBJECT"
            elif high_spread:
                low_reason = "TRACKER_JUMPING_BETWEEN_OBJECTS"
            elif runaway:
                low_reason = "KALMAN_VELOCITY_EXPLOSION"
            elif max_pred_run >= self.max_pred_gap:
                low_reason = "LONG_SYNTHETIC_GAP"
            elif dominance == PROVENANCE_PREDICTED_DOMINANT:
                low_reason = "PREDICTED_DOMINANT_NO_GROUNDED_EVIDENCE"
            else:
                low_reason = "INSUFFICIENT_GROUNDED_EVIDENCE"

        return ProvenanceAudit(
            frame=frame,
            window_frames=total,
            n_detected=n_det,
            n_tracked=n_trk,
            n_interpolated=n_itp,
            n_predicted=n_prd,
            n_missing=n_mis,
            dominance=dominance,
            max_consecutive_predicted=max_pred_run,
            kalman_runaway_detected=runaway,
            static_cluster_locked=static_locked,
            tracker_jumping_between_objects=high_spread,
            position_spread_px=spread,
            median_normalised_speed=med_spd,
            dominant_state_fraction=dom_frac,
            provenance_trust_score=trust,
            low_trust_reason=low_reason,
        )


    def audit_all(
        self,
        frames: Sequence[int],
        trajectory: Sequence[TemporalBallPoint],
    ) -> List[ProvenanceAudit]:
        """Batch audit for a list of candidate frame indices."""
        return [self.audit(f, trajectory) for f in frames]
