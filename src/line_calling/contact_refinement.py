import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

@dataclass
class RefinedContactPoint:
    frame_index: int
    sub_frame_time: float
    timestamp_s: float
    x_px: float
    y_px: float
    confidence: float
    refinement_method: str
    original_x_px: float
    original_y_px: float
    pixel_shift: float
    timing_uncertainty_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "sub_frame_time": round(self.sub_frame_time, 3),
            "timestamp_s": round(self.timestamp_s, 4),
            "x_px": round(self.x_px, 2),
            "y_px": round(self.y_px, 2),
            "confidence": round(self.confidence, 3),
            "refinement_method": self.refinement_method,
            "original_x_px": round(self.original_x_px, 2),
            "original_y_px": round(self.original_y_px, 2),
            "pixel_shift": round(self.pixel_shift, 2),
            "timing_uncertainty_ms": round(self.timing_uncertainty_ms, 1)
        }

class BounceContactRefiner:
    """
    Impulsive collision bounce contact refinement engine.
    Uses Piecewise Pre-Impact vs Post-Impact trajectory intersection to locate
    the physical velocity change-point without imposing smooth zero-velocity constraints.
    """

    def __init__(self, window_radius: int = 3, fps: float = 30.0):
        self.window_radius = window_radius
        self.fps = fps
        self.frame_interval_ms = (1.0 / fps) * 1000.0

    def refine_bounce_contact(
        self,
        bounce_frame: int,
        ball_trajectory: List[TemporalBallPoint],
        raw_frames: Optional[List[np.ndarray]] = None
    ) -> RefinedContactPoint:
        """
        Primary refinement method: Attempts piecewise impact change-point fitting,
        falling back to local quadratic inflection if pre/post points are insufficient.
        """
        n = len(ball_trajectory)
        base_pt = ball_trajectory[bounce_frame]
        orig_x = base_pt.x_px if base_pt.x_px is not None else 0.0
        orig_y = base_pt.y_px if base_pt.y_px is not None else 0.0
        base_time_s = base_pt.timestamp_seconds

        # 1. Collect pre-impact and post-impact points
        f_min = max(0, bounce_frame - self.window_radius)
        f_max = min(n - 1, bounce_frame + self.window_radius)

        pre_pts = [ball_trajectory[f] for f in range(f_min, bounce_frame) if ball_trajectory[f].x_px is not None and ball_trajectory[f].y_px is not None]
        post_pts = [ball_trajectory[f] for f in range(bounce_frame + 1, f_max + 1) if ball_trajectory[f].x_px is not None and ball_trajectory[f].y_px is not None]

        # 2. Try Piecewise Trajectory Intersection
        if len(pre_pts) >= 2 and len(post_pts) >= 2:
            try:
                pre_f = np.array([p.frame_index for p in pre_pts], dtype=np.float32)
                pre_x = np.array([p.x_px for p in pre_pts], dtype=np.float32)
                pre_y = np.array([p.y_px for p in pre_pts], dtype=np.float32)

                post_f = np.array([p.frame_index for p in post_pts], dtype=np.float32)
                post_x = np.array([p.x_px for p in post_pts], dtype=np.float32)
                post_y = np.array([p.y_px for p in post_pts], dtype=np.float32)

                # Linear ballistic slopes
                m_pre_y, c_pre_y = np.polyfit(pre_f, pre_y, 1)
                m_post_y, c_post_y = np.polyfit(post_f, post_y, 1)

                m_pre_x, c_pre_x = np.polyfit(pre_f, pre_x, 1)
                m_post_x, c_post_x = np.polyfit(post_f, post_x, 1)

                # Check for slope opposition / velocity change
                if abs(m_pre_y - m_post_y) > 0.5:
                    t_inter = (c_post_y - c_pre_y) / (m_pre_y - m_post_y)
                    
                    # Sanity check: intersection must lie near bounce frame
                    if abs(t_inter - bounce_frame) <= 2.0:
                        y_inter = m_pre_y * t_inter + c_pre_y
                        x_inter = 0.5 * ((m_pre_x * t_inter + c_pre_x) + (m_post_x * t_inter + c_post_x))
                        shift = math.hypot(x_inter - orig_x, y_inter - orig_y)
                        
                        if shift <= 20.0:
                            return RefinedContactPoint(
                                frame_index=bounce_frame,
                                sub_frame_time=float(t_inter),
                                timestamp_s=float(t_inter / self.fps),
                                x_px=float(x_inter),
                                y_px=float(y_inter),
                                confidence=min(0.98, (base_pt.confidence or 0.7) + 0.15),
                                refinement_method="PIECEWISE_IMPACT_INTERSECTION",
                                original_x_px=orig_x,
                                original_y_px=orig_y,
                                pixel_shift=round(shift, 2),
                                timing_uncertainty_ms=round(self.frame_interval_ms * 0.5, 1)
                            )
            except Exception:
                pass

        # 3. Fallback: Local Trajectory Parabolic Inflection Fit
        window_pts = [p for p in ball_trajectory[f_min:f_max + 1] if p.x_px is not None and p.y_px is not None]
        if len(window_pts) >= 3:
            try:
                w_f = np.array([p.frame_index for p in window_pts], dtype=np.float32)
                w_x = np.array([p.x_px for p in window_pts], dtype=np.float32)
                w_y = np.array([p.y_px for p in window_pts], dtype=np.float32)
                
                poly_y = np.polyfit(w_f, w_y, 2)
                poly_x = np.polyfit(w_f, w_x, 1)

                ref_y = float(np.polyval(poly_y, bounce_frame))
                ref_x = float(np.polyval(poly_x, bounce_frame))
                shift = math.hypot(ref_x - orig_x, ref_y - orig_y)

                if shift <= 15.0:
                    return RefinedContactPoint(
                        frame_index=bounce_frame,
                        sub_frame_time=float(bounce_frame),
                        timestamp_s=base_time_s,
                        x_px=ref_x,
                        y_px=ref_y,
                        confidence=min(0.95, (base_pt.confidence or 0.7) + 0.05),
                        refinement_method="LOCAL_TRAJECTORY_INFLECTION_FIT",
                        original_x_px=orig_x,
                        original_y_px=orig_y,
                        pixel_shift=round(shift, 2),
                        timing_uncertainty_ms=round(self.frame_interval_ms * 1.0, 1)
                    )
            except Exception:
                pass

        # 4. Fallback to raw discrete measurement
        return RefinedContactPoint(
            frame_index=bounce_frame,
            sub_frame_time=float(bounce_frame),
            timestamp_s=base_time_s,
            x_px=orig_x,
            y_px=orig_y,
            confidence=base_pt.confidence or 0.5,
            refinement_method="RAW_DISCRETE_FRAME",
            original_x_px=orig_x,
            original_y_px=orig_y,
            pixel_shift=0.0,
            timing_uncertainty_ms=round(self.frame_interval_ms * 1.0, 1)
        )
