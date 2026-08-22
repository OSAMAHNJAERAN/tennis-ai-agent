import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class KinematicDerivatives:
    """Kinematic derivative quantities for a single trajectory point."""
    frame_index: int
    timestamp_s: float
    x_px: Optional[float]
    y_px: Optional[float]
    vx_px_s: Optional[float] = None
    vy_px_s: Optional[float] = None
    speed_px_s: Optional[float] = None
    ax_px_s2: Optional[float] = None
    ay_px_s2: Optional[float] = None
    accel_mag_px_s2: Optional[float] = None
    curvature: Optional[float] = None
    direction_change_rad: Optional[float] = None

class TrajectoryDerivativeCalculator:
    """
    Computes smooth 1st and 2nd numerical derivatives and differential geometry
    quantities (velocity, acceleration, curvature, angular deflection) across native timestamps.
    """
    
    @staticmethod
    def compute_derivatives(
        positions: List[Optional[Tuple[float, float]]],
        timestamps_s: List[float]
    ) -> List[KinematicDerivatives]:
        """
        Computes velocity, acceleration, and curvature for a sequence of points.
        
        Args:
            positions: List of (x, y) pixel coordinates or None.
            timestamps_s: Corresponding native timestamp in seconds for each frame.
            
        Returns:
            List of KinematicDerivatives objects.
        """
        n = len(positions)
        derivatives: List[KinematicDerivatives] = []
        
        # Initialize containers
        for i in range(n):
            pt = positions[i]
            x = float(pt[0]) if pt is not None else None
            y = float(pt[1]) if pt is not None else None
            derivatives.append(KinematicDerivatives(
                frame_index=i,
                timestamp_s=timestamps_s[i],
                x_px=x,
                y_px=y
            ))
            
        # 1. First Derivatives: Velocity
        for i in range(1, n - 1):
            p_prev = positions[i - 1]
            p_next = positions[i + 1]
            t_prev = timestamps_s[i - 1]
            t_next = timestamps_s[i + 1]
            dt = t_next - t_prev
            
            if p_prev is not None and p_next is not None and dt > 0:
                vx = (p_next[0] - p_prev[0]) / dt
                vy = (p_next[1] - p_prev[1]) / dt
                derivatives[i].vx_px_s = float(vx)
                derivatives[i].vy_px_s = float(vy)
                derivatives[i].speed_px_s = float(math.hypot(vx, vy))
            elif p_prev is not None and positions[i] is not None:
                dt_step = timestamps_s[i] - t_prev
                if dt_step > 0:
                    vx = (positions[i][0] - p_prev[0]) / dt_step
                    vy = (positions[i][1] - p_prev[1]) / dt_step
                    derivatives[i].vx_px_s = float(vx)
                    derivatives[i].vy_px_s = float(vy)
                    derivatives[i].speed_px_s = float(math.hypot(vx, vy))
                    
        # 2. Second Derivatives & Curvature
        for i in range(1, n - 1):
            d_prev = derivatives[i - 1]
            d_next = derivatives[i + 1]
            t_prev = timestamps_s[i - 1]
            t_next = timestamps_s[i + 1]
            dt = t_next - t_prev
            
            if d_prev.vx_px_s is not None and d_next.vx_px_s is not None and dt > 0:
                ax = (d_next.vx_px_s - d_prev.vx_px_s) / dt
                ay = (d_next.vy_px_s - d_prev.vy_px_s) / dt
                accel_mag = math.hypot(ax, ay)
                derivatives[i].ax_px_s2 = float(ax)
                derivatives[i].ay_px_s2 = float(ay)
                derivatives[i].accel_mag_px_s2 = float(accel_mag)
                
                # Curvature kappa = |vx * ay - vy * ax| / (vx^2 + vy^2)^(1.5)
                vx = derivatives[i].vx_px_s or 0.0
                vy = derivatives[i].vy_px_s or 0.0
                v_sq = vx * vx + vy * vy
                if v_sq > 1e-4:
                    num = abs(vx * ay - vy * ax)
                    den = math.pow(v_sq, 1.5)
                    derivatives[i].curvature = float(num / den)
                else:
                    derivatives[i].curvature = 0.0
                    
                # Angular direction change between forward and backward velocity vectors
                v1 = (d_prev.vx_px_s, d_prev.vy_px_s)
                v2 = (d_next.vx_px_s, d_next.vy_px_s)
                mag1 = math.hypot(v1[0], v1[1])
                mag2 = math.hypot(v2[0], v2[1])
                if mag1 > 1e-3 and mag2 > 1e-3:
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    derivatives[i].direction_change_rad = float(math.acos(cos_theta))
                    
        return derivatives
