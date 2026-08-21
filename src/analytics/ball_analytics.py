import numpy as np
from typing import List, Optional, Tuple
from src.tracking.ball_tracker import BallPoint, BallPointState

def calculate_ball_speed(trajectory: List[BallPoint], court_positions: List[Optional[Tuple[float, float]]], window: int = 3) -> List[Optional[float]]:
    """
    Calculates ball speed in m/s using court coordinates.
    Only uses DETECTED and INTERPOLATED points.
    
    Args:
        trajectory: List of BallPoint instances.
        court_positions: Corresponding court coordinates (x, y).
        window: Temporal window for smoothing.
        
    Returns:
        List of ball speeds in m/s.
    """
    speeds = [None] * len(trajectory)
    for i in range(len(trajectory)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(trajectory) - 1, i + window // 2)
        
        p_start = trajectory[start_idx]
        p_end = trajectory[end_idx]
        
        valid_states = (BallPointState.DETECTED, BallPointState.INTERPOLATED)
        
        if p_start.state in valid_states and p_end.state in valid_states:
            c_start = court_positions[start_idx]
            c_end = court_positions[end_idx]
            
            if c_start is not None and c_end is not None:
                d = np.linalg.norm(np.array(c_end) - np.array(c_start))
                t_diff = p_end.timestamp_seconds - p_start.timestamp_seconds
                if t_diff > 0:
                    speeds[i] = d / t_diff
    return speeds

def detect_shot_frames(trajectory: List[BallPoint], min_direction_change_frames: int = 10) -> List[int]:
    """
    Detect frames where ball changes direction (shot/hit events).
    
    Args:
        trajectory: List of BallPoint instances.
        min_direction_change_frames: Minimum frame gap for direction change check.
        
    Returns:
        List of frame indices where a shot was detected.
    """
    shots = []
    # Basic direction change heuristic on x-axis (tennis shots usually change x direction)
    valid_points = [(i, p) for i, p in enumerate(trajectory) if p.state in (BallPointState.DETECTED, BallPointState.INTERPOLATED)]
    
    for j in range(1, len(valid_points) - 1):
        idx_prev, p_prev = valid_points[j-1]
        idx_curr, p_curr = valid_points[j]
        idx_next, p_next = valid_points[j+1]
        
        if p_prev.x_px is not None and p_curr.x_px is not None and p_next.x_px is not None:
            vx_prev = p_curr.x_px - p_prev.x_px
            vx_next = p_next.x_px - p_curr.x_px
            
            # If x velocity changes sign, might be a hit
            if vx_prev * vx_next < 0:
                shots.append(p_curr.frame_index)
                
    return shots
