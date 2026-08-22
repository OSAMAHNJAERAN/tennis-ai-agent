import math
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

class CourtLineType(str, Enum):
    LEFT_SIDELINE = "LEFT_SIDELINE"
    RIGHT_SIDELINE = "RIGHT_SIDELINE"
    NEAR_BASELINE = "NEAR_BASELINE"
    FAR_BASELINE = "FAR_BASELINE"
    NEAR_SERVICE_LINE = "NEAR_SERVICE_LINE"
    FAR_SERVICE_LINE = "FAR_SERVICE_LINE"
    CENTER_SERVICE_LINE = "CENTER_SERVICE_LINE"
    NONE = "NONE"

class ServiceBoxType(str, Enum):
    NEAR_DEUCE = "NEAR_DEUCE"  # Receiver's right side / server's cross-court from far deuce
    NEAR_AD = "NEAR_AD"        # Receiver's left side / server's cross-court from far ad
    FAR_DEUCE = "FAR_DEUCE"    # Top-left box from camera perspective
    FAR_AD = "FAR_AD"          # Top-right box from camera perspective

@dataclass
class BoundaryEvaluationResult:
    signed_center_distance_m: float
    signed_center_distance_cm: float
    effective_ball_radius_cm: float
    ball_edge_margin_cm: float
    is_center_inside: bool
    is_footprint_inside_or_touching: bool
    nearest_line: CourtLineType
    target_region_name: str

class CourtLineGeometry:
    """
    Authoritative ITF canonical singles and service-box boundary geometry in metric space.
    Origin (0,0) is at top-left outer doubles corner.
    X-axis points right (width: 0.0 to 10.97m).
    Y-axis points down (length: 0.0 to 23.77m).
    """

    # ITF Physical Constants (meters)
    BALL_RADIUS_M: float = 0.0335   # 3.35 cm nominal radius (ITF Type 2)
    BALL_RADIUS_CM: float = 3.35
    
    # Singles Court Boundaries
    SINGLES_LEFT_X: float = 1.37
    SINGLES_RIGHT_X: float = 9.60
    FAR_BASELINE_Y: float = 0.00
    NEAR_BASELINE_Y: float = 23.77
    NET_Y: float = 11.885
    
    # Service Lines
    FAR_SERVICE_LINE_Y: float = 5.485
    NEAR_SERVICE_LINE_Y: float = 18.285
    CENTER_SERVICE_X: float = 5.485

    # Service Box Metric Polygons [X_min, X_max, Y_min, Y_max]
    SERVICE_BOX_BOUNDS = {
        ServiceBoxType.NEAR_DEUCE: (1.37, 5.485, 11.885, 18.285),
        ServiceBoxType.NEAR_AD: (5.485, 9.60, 11.885, 18.285),
        ServiceBoxType.FAR_DEUCE: (1.37, 5.485, 5.485, 11.885),
        ServiceBoxType.FAR_AD: (5.485, 9.60, 5.485, 11.885)
    }

    @classmethod
    def evaluate_singles_rally_boundary(
        cls,
        court_x_m: float,
        court_y_m: float,
        effective_radius_m: Optional[float] = None
    ) -> BoundaryEvaluationResult:
        """
        Evaluates a bounce contact point against canonical singles rally boundaries.
        Returns signed distances to nearest boundary lines and ball footprint margin.
        """
        r_m = effective_radius_m if effective_radius_m is not None else cls.BALL_RADIUS_M
        r_cm = r_m * 100.0

        d_left = court_x_m - cls.SINGLES_LEFT_X
        d_right = cls.SINGLES_RIGHT_X - court_x_m
        d_far = court_y_m - cls.FAR_BASELINE_Y
        d_near = cls.NEAR_BASELINE_Y - court_y_m

        lines = [
            (d_left, CourtLineType.LEFT_SIDELINE),
            (d_right, CourtLineType.RIGHT_SIDELINE),
            (d_far, CourtLineType.FAR_BASELINE),
            (d_near, CourtLineType.NEAR_BASELINE)
        ]

        # Signed distance is minimum of distances to all 4 edges
        min_d, nearest_line = min(lines, key=lambda item: item[0])
        min_d_cm = min_d * 100.0
        edge_margin_cm = min_d_cm + r_cm

        return BoundaryEvaluationResult(
            signed_center_distance_m=round(min_d, 4),
            signed_center_distance_cm=round(min_d_cm, 2),
            effective_ball_radius_cm=round(r_cm, 2),
            ball_edge_margin_cm=round(edge_margin_cm, 2),
            is_center_inside=(min_d >= 0.0),
            is_footprint_inside_or_touching=(edge_margin_cm >= 0.0),
            nearest_line=nearest_line,
            target_region_name="SINGLES_COURT"
        )

    @classmethod
    def evaluate_service_box_boundary(
        cls,
        court_x_m: float,
        court_y_m: float,
        target_box: ServiceBoxType,
        effective_radius_m: Optional[float] = None
    ) -> BoundaryEvaluationResult:
        """
        Evaluates a serve bounce contact point against the legal target service box.
        """
        r_m = effective_radius_m if effective_radius_m is not None else cls.BALL_RADIUS_M
        r_cm = r_m * 100.0

        x_min, x_max, y_min, y_max = cls.SERVICE_BOX_BOUNDS[target_box]

        d_left = court_x_m - x_min
        d_right = x_max - court_x_m
        d_top = court_y_m - y_min
        d_bottom = y_max - court_y_m

        # Determine line types based on box
        if target_box in (ServiceBoxType.NEAR_DEUCE, ServiceBoxType.FAR_DEUCE):
            left_type = CourtLineType.LEFT_SIDELINE
            right_type = CourtLineType.CENTER_SERVICE_LINE
        else:
            left_type = CourtLineType.CENTER_SERVICE_LINE
            right_type = CourtLineType.RIGHT_SIDELINE

        if target_box in (ServiceBoxType.FAR_DEUCE, ServiceBoxType.FAR_AD):
            top_type = CourtLineType.FAR_SERVICE_LINE
            bottom_type = CourtLineType.NONE  # Net
        else:
            top_type = CourtLineType.NONE      # Net
            bottom_type = CourtLineType.NEAR_SERVICE_LINE

        candidates = [
            (d_left, left_type),
            (d_right, right_type),
            (d_top, top_type),
            (d_bottom, bottom_type)
        ]

        min_d, nearest_line = min(candidates, key=lambda item: item[0])
        min_d_cm = min_d * 100.0
        edge_margin_cm = min_d_cm + r_cm

        return BoundaryEvaluationResult(
            signed_center_distance_m=round(min_d, 4),
            signed_center_distance_cm=round(min_d_cm, 2),
            effective_ball_radius_cm=round(r_cm, 2),
            ball_edge_margin_cm=round(edge_margin_cm, 2),
            is_center_inside=(min_d >= 0.0),
            is_footprint_inside_or_touching=(edge_margin_cm >= 0.0),
            nearest_line=nearest_line,
            target_region_name=target_box.value
        )

    @classmethod
    def get_spatial_uncertainty_tier(cls, court_y_m: float) -> Tuple[str, float]:
        """
        Determines the spatial court tier and baseline geometric uncertainty (meters)
        based on calibrated camera distance and grazing-angle homography sensitivity.
        """
        if court_y_m >= cls.NET_Y:
            # Near Court (Closer to camera, high pixel density, low perspective error)
            return ("NEAR", 0.008)  # 0.8 cm base uncertainty
        elif court_y_m >= cls.FAR_SERVICE_LINE_Y:
            # Mid Court (Service line to net)
            return ("MID", 0.025)   # 2.5 cm base uncertainty
        else:
            # Far Court (Far baseline to service line, steep perspective foreshortening)
            return ("FAR", 0.350)   # 35.0 cm base uncertainty
