import math
from typing import Tuple, Optional
from src.shot_analysis.shot_types import ShotDirection

class ShotDirectionClassifier:
    """
    Geometry-derived Shot Direction Classifier.
    Computes CROSS_COURT, DOWN_THE_LINE, or MIDDLE in canonical metric court coordinates.
    """
    CENTER_X = 5.485 # Midpoint of 10.97m court

    @classmethod
    def classify_direction(
        cls,
        start_pos_m: Optional[Tuple[float, float]],
        end_pos_m: Optional[Tuple[float, float]],
        min_depth_displacement_m: float = 4.0
    ) -> Tuple[ShotDirection, float, str]:
        """
        Classifies shot direction between impact and landing points.
        
        Args:
            start_pos_m: (X, Y) of stroke impact in meters.
            end_pos_m: (X, Y) of ball bounce landing in meters.
            min_depth_displacement_m: Minimum forward depth required to evaluate direction.
            
        Returns:
            (ShotDirection, confidence, reason)
        """
        if start_pos_m is None or end_pos_m is None:
            return ShotDirection.UNKNOWN, 0.0, "Missing start or end court coordinates."

        x0, y0 = start_pos_m
        x1, y1 = end_pos_m

        dy = abs(y1 - y0)
        dx = x1 - x0

        if dy < min_depth_displacement_m:
            return ShotDirection.UNKNOWN, 0.4, f"Insufficient depth displacement ({dy:.2f}m < {min_depth_displacement_m}m)."

        # Determine hitter side: Left (X < 5.485) or Right (X >= 5.485)
        hitter_on_left = (x0 < cls.CENTER_X)
        landing_on_left = (x1 < cls.CENTER_X)

        # Check for Middle landing
        if 4.20 <= x1 <= 6.77:
            conf = 0.90
            return ShotDirection.MIDDLE, conf, f"Ball landed in central corridor (X={x1:.2f}m)."

        # Check Cross-court vs Down-the-line
        if hitter_on_left:
            # Struck from left side
            if not landing_on_left and dx > 1.5:
                conf = min(0.98, 0.70 + abs(dx) * 0.08)
                return ShotDirection.CROSS_COURT, conf, f"Struck from Left (X={x0:.2f}m) across center to Right (X={x1:.2f}m, dx={dx:+.2f}m)."
            else:
                conf = min(0.98, 0.70 + max(0, 2.0 - abs(dx)) * 0.10)
                return ShotDirection.DOWN_THE_LINE, conf, f"Struck from Left (X={x0:.2f}m) parallel down Left sideline (X={x1:.2f}m, dx={dx:+.2f}m)."
        else:
            # Struck from right side
            if landing_on_left and dx < -1.5:
                conf = min(0.98, 0.70 + abs(dx) * 0.08)
                return ShotDirection.CROSS_COURT, conf, f"Struck from Right (X={x0:.2f}m) across center to Left (X={x1:.2f}m, dx={dx:+.2f}m)."
            else:
                conf = min(0.98, 0.70 + max(0, 2.0 - abs(dx)) * 0.10)
                return ShotDirection.DOWN_THE_LINE, conf, f"Struck from Right (X={x0:.2f}m) parallel down Right sideline (X={x1:.2f}m, dx={dx:+.2f}m)."
