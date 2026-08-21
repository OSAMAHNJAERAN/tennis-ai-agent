from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import pandas as pd
from src.utils.bbox_utils import BBox

class BallPointState(Enum):
    """State of the ball detection at a specific frame."""
    DETECTED = "DETECTED"
    INTERPOLATED = "INTERPOLATED"
    PREDICTED = "PREDICTED"
    OCCLUDED = "OCCLUDED"
    MISSING = "MISSING"

@dataclass
class BallPoint:
    """Represents a ball position at a specific frame."""
    frame_index: int
    timestamp_seconds: float
    x_px: Optional[float]
    y_px: Optional[float]
    confidence: float
    state: BallPointState
    model_source: str = "yolo"

class BallTracker:
    """Tracks ball trajectory with state management."""
    
    def __init__(self, max_interpolation_gap: int = 5):
        self.max_interpolation_gap = max_interpolation_gap

    def create_trajectory(self, detections: List[Optional[BBox]], fps: float) -> List[BallPoint]:
        """
        Creates a trajectory from per-frame detections.
        
        Args:
            detections: List of ball bounding boxes per frame.
            fps: Video frames per second.
            
        Returns:
            List of BallPoint objects representing the trajectory.
        """
        trajectory = []
        for i, det in enumerate(detections):
            ts = i / fps if fps > 0 else 0.0
            if det is not None:
                cx = (det.x1 + det.x2) / 2
                cy = (det.y1 + det.y2) / 2
                trajectory.append(BallPoint(i, ts, cx, cy, det.confidence, BallPointState.DETECTED))
            else:
                trajectory.append(BallPoint(i, ts, None, None, 0.0, BallPointState.MISSING))
        return trajectory

    def interpolate_trajectory(self, trajectory: List[BallPoint]) -> List[BallPoint]:
        """
        Interpolates missing trajectory points using pandas.
        
        Args:
            trajectory: Original trajectory with potentially missing points.
            
        Returns:
            Trajectory with interpolated points.
        """
        if not trajectory:
            return trajectory

        # Find valid detection indices
        valid_indices = [i for i, p in enumerate(trajectory) if p.state == BallPointState.DETECTED and p.x_px is not None and p.y_px is not None]
        if len(valid_indices) < 2:
            return trajectory

        # Interpolate only between pairs of detections with gap <= max_interpolation_gap
        for k in range(len(valid_indices) - 1):
            idx_start = valid_indices[k]
            idx_end = valid_indices[k + 1]
            gap = idx_end - idx_start - 1
            
            if 0 < gap <= self.max_interpolation_gap:
                p_start = trajectory[idx_start]
                p_end = trajectory[idx_end]
                
                x_start, y_start = p_start.x_px, p_start.y_px
                x_end, y_end = p_end.x_px, p_end.y_px
                
                for step, idx in enumerate(range(idx_start + 1, idx_end), 1):
                    alpha = step / (gap + 1)
                    interp_x = x_start + alpha * (x_end - x_start)
                    interp_y = y_start + alpha * (y_end - y_start)
                    
                    trajectory[idx].x_px = float(interp_x)
                    trajectory[idx].y_px = float(interp_y)
                    trajectory[idx].state = BallPointState.INTERPOLATED

        return trajectory

    def get_ball_positions(self, trajectory: List[BallPoint]) -> List[Optional[Tuple[float, float]]]:
        """
        Extracts (x, y) positions for DETECTED and INTERPOLATED points.
        
        Args:
            trajectory: The ball trajectory.
            
        Returns:
            List of (x, y) tuples or None for MISSING points.
        """
        positions = []
        for p in trajectory:
            if p.state in (BallPointState.DETECTED, BallPointState.INTERPOLATED) and p.x_px is not None and p.y_px is not None:
                positions.append((p.x_px, p.y_px))
            else:
                positions.append(None)
        return positions
