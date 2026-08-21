import numpy as np
from typing import List, Optional, Tuple

def calculate_distance(positions: List[Optional[Tuple[float, float]]]) -> float:
    """
    Calculates total distance in meters.
    Outliers (>5m between consecutive frames) are rejected.
    
    Args:
        positions: List of (x, y) court positions.
        
    Returns:
        Total distance in meters.
    """
    total_dist = 0.0
    prev_pos = None
    for pos in positions:
        if pos is not None:
            if prev_pos is not None:
                d = np.linalg.norm(np.array(pos) - np.array(prev_pos))
                if d <= 5.0:  # Outlier rejection
                    total_dist += d
            prev_pos = pos
    return total_dist

def calculate_speed(positions: List[Optional[Tuple[float, float]]], timestamps: List[float], window: int = 5) -> List[Optional[float]]:
    """
    Calculates speed in m/s with temporal smoothing.
    
    Args:
        positions: List of (x, y) court positions.
        timestamps: List of timestamps in seconds.
        window: Temporal window size for smoothing.
        
    Returns:
        List of speeds in m/s.
    """
    speeds = [None] * len(positions)
    for i in range(len(positions)):
        start_idx = max(0, i - window // 2)
        end_idx = min(len(positions) - 1, i + window // 2)
        
        pos_start = positions[start_idx]
        pos_end = positions[end_idx]
        t_start = timestamps[start_idx]
        t_end = timestamps[end_idx]
        
        if pos_start is not None and pos_end is not None and t_end > t_start:
            d = np.linalg.norm(np.array(pos_end) - np.array(pos_start))
            if d <= 5.0 * (t_end - t_start) * 10:  # Scaled outlier rejection
                speeds[i] = d / (t_end - t_start)
    return speeds

def calculate_cumulative_distance(positions: List[Optional[Tuple[float, float]]]) -> List[float]:
    """
    Calculates running total distance.
    
    Args:
        positions: List of (x, y) court positions.
        
    Returns:
        List of cumulative distances.
    """
    cum_dist = []
    total_dist = 0.0
    prev_pos = None
    for pos in positions:
        if pos is not None:
            if prev_pos is not None:
                d = np.linalg.norm(np.array(pos) - np.array(prev_pos))
                if d <= 5.0:
                    total_dist += d
            prev_pos = pos
        cum_dist.append(total_dist)
    return cum_dist
