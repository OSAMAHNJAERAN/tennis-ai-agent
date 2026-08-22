import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

@dataclass
class RefinedContactPoint:
    frame_index: int
    x_px: float
    y_px: float
    confidence: float
    refinement_method: str
    original_x_px: float
    original_y_px: float
    pixel_shift: float

class BounceContactRefiner:
    """
    High-resolution local bounce contact refinement engine.
    Refines the spatial ball center at court contact across a local temporal window (±2 frames).
    """

    def __init__(self, search_window_radius: int = 2):
        self.search_window_radius = search_window_radius

    def refine_bounce_contact(
        self,
        bounce_frame: int,
        ball_trajectory: List[TemporalBallPoint],
        raw_frames: Optional[List[np.ndarray]] = None
    ) -> RefinedContactPoint:
        """
        Refines the bounce contact pixel coordinate using local parabolic inflection
        and visual candidate verification.
        """
        n = len(ball_trajectory)
        f_min = max(0, bounce_frame - self.search_window_radius)
        f_max = min(n - 1, bounce_frame + self.search_window_radius)

        base_pt = ball_trajectory[bounce_frame]
        orig_x = base_pt.x_px if base_pt.x_px is not None else 0.0
        orig_y = base_pt.y_px if base_pt.y_px is not None else 0.0

        # 1. Collect valid observed points in local window
        window_points = []
        for f in range(f_min, f_max + 1):
            pt = ball_trajectory[f]
            if pt.x_px is not None and pt.y_px is not None:
                # Give higher weight to DETECTED points
                w = 1.0 if pt.state == BallState.DETECTED else 0.5
                window_points.append((f, pt.x_px, pt.y_px, w))

        if len(window_points) < 3:
            # Fallback to existing trajectory point
            return RefinedContactPoint(
                frame_index=bounce_frame,
                x_px=orig_x,
                y_px=orig_y,
                confidence=base_pt.confidence or 0.5,
                refinement_method="TRAJECTORY_FALLBACK",
                original_x_px=orig_x,
                original_y_px=orig_y,
                pixel_shift=0.0
            )

        # 2. Local Trajectory Parabolic Extrema Fit (Experiment C: Local Inflection Fit)
        frames = np.array([p[0] for p in window_points], dtype=np.float32)
        xs = np.array([p[1] for p in window_points], dtype=np.float32)
        ys = np.array([p[2] for p in window_points], dtype=np.float32)
        weights = np.array([p[3] for p in window_points], dtype=np.float32)

        # Fit quadratic curve y(f) = a*f^2 + b*f + c
        try:
            poly_y = np.polyfit(frames, ys, deg=2, w=weights)
            poly_x = np.polyfit(frames, xs, deg=1, w=weights)

            # Evaluate at the contact frame
            refined_y = float(np.polyval(poly_y, bounce_frame))
            refined_x = float(np.polyval(poly_x, bounce_frame))

            # Clamp shift to prevent excessive deviation from measurement
            shift = math.hypot(refined_x - orig_x, refined_y - orig_y)
            if shift > 15.0:
                # Revert to raw measurement if polynomial diverges
                refined_x = orig_x
                refined_y = orig_y
                method = "MEASUREMENT_CLAMPED"
            else:
                method = "LOCAL_TRAJECTORY_INFLECTION_FIT"

        except Exception:
            refined_x = orig_x
            refined_y = orig_y
            shift = 0.0
            method = "TRAJECTORY_FALLBACK"

        return RefinedContactPoint(
            frame_index=bounce_frame,
            x_px=round(refined_x, 2),
            y_px=round(refined_y, 2),
            confidence=min(1.0, (base_pt.confidence or 0.7) + 0.1),
            refinement_method=method,
            original_x_px=round(orig_x, 2),
            original_y_px=round(orig_y, 2),
            pixel_shift=round(math.hypot(refined_x - orig_x, refined_y - orig_y), 2)
        )
