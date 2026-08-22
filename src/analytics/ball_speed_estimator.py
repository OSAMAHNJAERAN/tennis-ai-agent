import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.events.event_detector import TennisEvent

@dataclass
class BallFlightSegment:
    """Represents a continuous flight interval between two physical events."""
    segment_id: int
    start_frame: int
    end_frame: int
    start_event: Optional[str]
    end_event: Optional[str]
    duration_s: float
    total_2d_distance_m: float
    mean_speed_kmh: float
    peak_speed_kmh: float
    launch_speed_kmh: float
    confidence_tier: str  # HIGH, MEDIUM, LOW, INVALID
    confidence_score: float

class BallSpeedEstimator:
    """
    Scientifically Defensible 2D Court-Projected Ball Speed Estimator.
    Computes piecewise segment-level speeds, respects physical event boundaries,
    and propagates observation state uncertainty.
    """
    
    @staticmethod
    def estimate_speeds(
        trajectory: List[TemporalBallPoint],
        events: List[TennisEvent],
        fps: float = 30.0
    ) -> Tuple[List[Optional[float]], List[BallFlightSegment], Dict[str, Any]]:
        """
        Calculates per-frame 2D court-projected speeds and segment-level flight metrics.
        
        Returns:
            per_frame_speeds_kmh: List of smoothed speeds (or None for missing/boundary frames).
            flight_segments: List of BallFlightSegment summaries.
            overall_summary: Dict with rally statistics.
        """
        n = len(trajectory)
        if n == 0:
            return [], [], {}
            
        per_frame_speeds: List[Optional[float]] = [None] * n
        event_frames = sorted([ev.frame_index for ev in events])
        
        # 1. Define Segment Intervals
        segment_boundaries = [0] + event_frames + [n - 1]
        # Remove duplicates while preserving order
        clean_boundaries = []
        for b in segment_boundaries:
            if not clean_boundaries or b > clean_boundaries[-1]:
                clean_boundaries.append(b)
                
        segments: List[BallFlightSegment] = []
        segment_id = 1
        
        for k in range(len(clean_boundaries) - 1):
            s_start = clean_boundaries[k]
            s_end = clean_boundaries[k + 1]
            
            if s_end - s_start < 2:
                continue
                
            # Identify bounding events
            start_ev_name = None
            end_ev_name = None
            for ev in events:
                if ev.frame_index == s_start:
                    start_ev_name = ev.event_type.value
                if ev.frame_index == s_end:
                    end_ev_name = ev.event_type.value
                    
            # Compute displacements within flight segment
            seg_points = trajectory[s_start:s_end + 1]
            valid_points = [p for p in seg_points if p.court_x_m is not None and p.court_y_m is not None]
            
            if len(valid_points) < 2:
                continue
                
            # Raw displacements
            instant_speeds: List[float] = []
            frame_indices: List[int] = []
            
            for i in range(s_start + 1, s_end):
                p_curr = trajectory[i]
                p_prev = trajectory[i - 1]
                
                if (p_curr.court_x_m is not None and p_curr.court_y_m is not None and 
                    p_prev.court_x_m is not None and p_prev.court_y_m is not None):
                    
                    dx = p_curr.court_x_m - p_prev.court_x_m
                    dy = p_curr.court_y_m - p_prev.court_y_m
                    d_m = math.hypot(dx, dy)
                    dt = p_curr.timestamp_seconds - p_prev.timestamp_seconds
                    
                    if 0 < dt and d_m <= 4.0:  # Physical displacement filter
                        spd = (d_m / dt) * 3.6
                        instant_speeds.append(spd)
                        frame_indices.append(i)
                    else:
                        instant_speeds.append(0.0)
                        frame_indices.append(i)

            if not instant_speeds:
                continue

            # Apply local moving average smoothing inside segment
            kernel_size = min(5, len(instant_speeds))
            smoothed = np.convolve(instant_speeds, np.ones(kernel_size)/kernel_size, mode='same')
            
            for idx, spd in zip(frame_indices, smoothed):
                if spd > 1.0:  # Active speed threshold
                    per_frame_speeds[idx] = float(spd)
                    trajectory[idx].speed_kmh = float(spd)

            # Compute Segment Summary Metrics
            valid_speeds = [s for s in instant_speeds if s > 1.0]
            mean_spd = float(np.mean(valid_speeds)) if valid_speeds else 0.0
            peak_spd = float(np.max(valid_speeds)) if valid_speeds else 0.0
            launch_spd = float(valid_speeds[0]) if valid_speeds else 0.0
            
            # Compute total segment distance
            total_dist_m = sum(
                math.hypot(seg_points[m].court_x_m - seg_points[m-1].court_x_m, 
                           seg_points[m].court_y_m - seg_points[m-1].court_y_m)
                for m in range(1, len(seg_points))
                if (seg_points[m].court_x_m is not None and seg_points[m-1].court_x_m is not None)
            )
            
            duration_s = trajectory[s_end].timestamp_seconds - trajectory[s_start].timestamp_seconds
            
            # Confidence tier based on observation ratio
            observed_count = sum(1 for p in seg_points if p.state in (BallState.DETECTED, BallState.TRACKED))
            obs_ratio = observed_count / max(1, len(seg_points))
            
            if obs_ratio >= 0.75:
                tier = "HIGH"
                conf_score = 0.90
            elif obs_ratio >= 0.40:
                tier = "MEDIUM"
                conf_score = 0.70
            elif obs_ratio > 0.0:
                tier = "LOW"
                conf_score = 0.45
            else:
                tier = "INVALID"
                conf_score = 0.10

            segments.append(BallFlightSegment(
                segment_id=segment_id,
                start_frame=s_start,
                end_frame=s_end,
                start_event=start_ev_name,
                end_event=end_ev_name,
                duration_s=float(round(duration_s, 3)),
                total_2d_distance_m=float(round(total_dist_m, 2)),
                mean_speed_kmh=float(round(mean_spd, 1)),
                peak_speed_kmh=float(round(peak_spd, 1)),
                launch_speed_kmh=float(round(launch_spd, 1)),
                confidence_tier=tier,
                confidence_score=float(round(conf_score, 2))
            ))
            segment_id += 1

        # Rally Overview
        all_speeds = [s for s in per_frame_speeds if s is not None and s > 1.0]
        summary = {
            "coordinate_system": "2D_court_ground_plane_projected",
            "time_base": "native_frame_timestamps",
            "units": "kilometers_per_hour (km/h)",
            "average_speed_kmh": float(round(np.mean(all_speeds), 1)) if all_speeds else 0.0,
            "maximum_speed_kmh": float(round(np.max(all_speeds), 1)) if all_speeds else 0.0,
            "segments_count": len(segments),
            "scientific_disclaimer": "2D court-projected speed estimate via homography. Does not account for monocular vertical elevation parallax."
        }

        return per_frame_speeds, segments, summary
