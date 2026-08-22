import math
from enum import Enum
from dataclasses import dataclass, asdict
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

class ContactPatchModelType(str, Enum):
    POINT_CONTACT = "POINT_CONTACT"                 # Contact center point only (r_c = 0.0 cm)
    EMPIRICAL_PATCH = "EMPIRICAL_PATCH"             # Nominal dynamic impact patch (r_c = 1.25 cm)
    DYNAMIC_IMPACT = "DYNAMIC_IMPACT"               # Velocity-scaled dynamic patch (r_c = 0.8 - 2.5 cm)
    ABSTENTION_GEOMETRIC = "ABSTENTION_GEOMETRIC"   # Conservative zero-assumption geometric gating

@dataclass
class CourtLineStrip:
    line_type: CourtLineType
    outer_edge_m: float
    inner_edge_m: float
    centerline_m: float
    width_m: float
    width_uncertainty_m: float
    polygon_bounds_m: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_type": self.line_type.value,
            "outer_edge_m": round(self.outer_edge_m, 4),
            "inner_edge_m": round(self.inner_edge_m, 4),
            "centerline_m": round(self.centerline_m, 4),
            "width_cm": round(self.width_m * 100.0, 2),
            "width_uncertainty_cm": round(self.width_uncertainty_m * 100.0, 2),
            "polygon_bounds_m": [round(v, 4) for v in self.polygon_bounds_m]
        }

@dataclass
class BoundaryEvaluationResult:
    signed_center_distance_m: float
    signed_center_distance_cm: float
    contact_patch_model: str
    contact_patch_radius_cm: float
    ball_edge_margin_cm: float
    is_center_inside: bool
    is_center_on_painted_line: bool
    is_contact_inside_or_touching: bool
    nearest_line: CourtLineType
    target_region_name: str
    line_strip: Optional[Dict[str, Any]] = None

class CourtLineGeometry:
    """
    ITF Rule 1 & Rule 12 canonical singles and service-box boundary geometry in metric space.
    All court measurements are made to the OUTSIDE of the lines.
    Lines are modeled as finite-width 2D strips.
    Origin (0,0) is at top-left outer doubles corner.
    X-axis points right (0.0 to 10.97m).
    Y-axis points down (0.0 to 23.77m).
    """

    # Physical Constants (ITF Rules of Tennis 2026)
    BALL_RADIUS_M: float = 0.0335        # 3.35 cm nominal ball radius (diameter 6.70 cm)
    BALL_RADIUS_CM: float = 3.35
    DEFAULT_CONTACT_PATCH_RADIUS_M: float = 0.0125  # 1.25 cm empirical contact radius (Cross 1999)
    DEFAULT_CONTACT_PATCH_RADIUS_CM: float = 1.25

    # Line Widths (ITF Rule 1: lines 2.5-5.0 cm wide, baselines up to 10.0 cm)
    DEFAULT_LINE_WIDTH_M: float = 0.05       # 5.0 cm for sidelines and service lines
    DEFAULT_BASELINE_WIDTH_M: float = 0.10   # 10.0 cm for baselines
    CENTER_SERVICE_LINE_WIDTH_M: float = 0.05 # 5.0 cm center service line

    # Singles Outer Legal Court Boundaries (Measured to Outside of Lines)
    SINGLES_LEFT_X: float = 1.37
    SINGLES_RIGHT_X: float = 9.60
    FAR_BASELINE_Y: float = 0.00
    NEAR_BASELINE_Y: float = 23.77
    NET_Y: float = 11.885

    # Service Outer Legal Boundaries (Measured to Outside of Lines)
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
    def get_line_strip(
        cls,
        line_type: CourtLineType,
        custom_widths: Optional[Dict[str, float]] = None
    ) -> CourtLineStrip:
        """
        Returns the explicit 2D bounding strip for a court line respecting the
        ITF Rule 1 outside-of-line measurement convention.
        """
        widths = custom_widths or {}
        w_side = widths.get("sideline_m", cls.DEFAULT_LINE_WIDTH_M)
        w_base = widths.get("baseline_m", cls.DEFAULT_BASELINE_WIDTH_M)
        w_serv = widths.get("service_line_m", cls.DEFAULT_LINE_WIDTH_M)
        w_center = widths.get("center_line_m", cls.CENTER_SERVICE_LINE_WIDTH_M)

        if line_type == CourtLineType.LEFT_SIDELINE:
            outer = cls.SINGLES_LEFT_X
            inner = round(outer + w_side, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer + (w_side / 2.0), 4),
                width_m=w_side,
                width_uncertainty_m=0.0125,
                polygon_bounds_m=(outer, inner, cls.FAR_BASELINE_Y, cls.NEAR_BASELINE_Y)
            )

        elif line_type == CourtLineType.RIGHT_SIDELINE:
            outer = cls.SINGLES_RIGHT_X
            inner = round(outer - w_side, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer - (w_side / 2.0), 4),
                width_m=w_side,
                width_uncertainty_m=0.0125,
                polygon_bounds_m=(inner, outer, cls.FAR_BASELINE_Y, cls.NEAR_BASELINE_Y)
            )

        elif line_type == CourtLineType.FAR_BASELINE:
            outer = cls.FAR_BASELINE_Y
            inner = round(outer + w_base, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer + (w_base / 2.0), 4),
                width_m=w_base,
                width_uncertainty_m=0.025,
                polygon_bounds_m=(cls.SINGLES_LEFT_X, cls.SINGLES_RIGHT_X, outer, inner)
            )

        elif line_type == CourtLineType.NEAR_BASELINE:
            outer = cls.NEAR_BASELINE_Y
            inner = round(outer - w_base, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer - (w_base / 2.0), 4),
                width_m=w_base,
                width_uncertainty_m=0.025,
                polygon_bounds_m=(cls.SINGLES_LEFT_X, cls.SINGLES_RIGHT_X, inner, outer)
            )

        elif line_type == CourtLineType.FAR_SERVICE_LINE:
            outer = cls.FAR_SERVICE_LINE_Y
            inner = round(outer + w_serv, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer + (w_serv / 2.0), 4),
                width_m=w_serv,
                width_uncertainty_m=0.0125,
                polygon_bounds_m=(cls.SINGLES_LEFT_X, cls.SINGLES_RIGHT_X, outer, inner)
            )

        elif line_type == CourtLineType.NEAR_SERVICE_LINE:
            outer = cls.NEAR_SERVICE_LINE_Y
            inner = round(outer - w_serv, 4)
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=outer,
                inner_edge_m=inner,
                centerline_m=round(outer - (w_serv / 2.0), 4),
                width_m=w_serv,
                width_uncertainty_m=0.0125,
                polygon_bounds_m=(cls.SINGLES_LEFT_X, cls.SINGLES_RIGHT_X, inner, outer)
            )

        elif line_type == CourtLineType.CENTER_SERVICE_LINE:
            half_w = w_center / 2.0
            return CourtLineStrip(
                line_type=line_type,
                outer_edge_m=round(cls.CENTER_SERVICE_X + half_w, 4),
                inner_edge_m=round(cls.CENTER_SERVICE_X - half_w, 4),
                centerline_m=cls.CENTER_SERVICE_X,
                width_m=w_center,
                width_uncertainty_m=0.0125,
                polygon_bounds_m=(round(cls.CENTER_SERVICE_X - half_w, 4), round(cls.CENTER_SERVICE_X + half_w, 4), cls.FAR_SERVICE_LINE_Y, cls.NEAR_SERVICE_LINE_Y)
            )

        else:
            return CourtLineStrip(
                line_type=CourtLineType.NONE,
                outer_edge_m=0.0,
                inner_edge_m=0.0,
                centerline_m=0.0,
                width_m=0.0,
                width_uncertainty_m=0.0,
                polygon_bounds_m=(0.0, 0.0, 0.0, 0.0)
            )

    @classmethod
    def evaluate_singles_rally_boundary(
        cls,
        court_x_m: float,
        court_y_m: float,
        contact_patch_radius_m: Optional[float] = None,
        contact_model: ContactPatchModelType = ContactPatchModelType.EMPIRICAL_PATCH
    ) -> BoundaryEvaluationResult:
        """
        Evaluates a bounce contact point against canonical singles rally boundaries
        with finite line widths and physical contact patch modeling.
        """
        if contact_model == ContactPatchModelType.POINT_CONTACT:
            r_m = 0.0
        elif contact_patch_radius_m is not None:
            r_m = contact_patch_radius_m
        else:
            r_m = cls.DEFAULT_CONTACT_PATCH_RADIUS_M
        r_cm = r_m * 100.0

        # Distances to outer legal perimeter boundaries
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

        min_d, nearest_line = min(lines, key=lambda item: item[0])
        min_d_cm = min_d * 100.0

        # Check if contact center is on the painted line strip
        strip = cls.get_line_strip(nearest_line)
        x_min, x_max, y_min, y_max = strip.polygon_bounds_m
        is_on_strip = (x_min <= court_x_m <= x_max) and (y_min <= court_y_m <= y_max)

        # Contact edge margin to legal outer boundary
        edge_margin_cm = min_d_cm + r_cm

        return BoundaryEvaluationResult(
            signed_center_distance_m=round(min_d, 4),
            signed_center_distance_cm=round(min_d_cm, 2),
            contact_patch_model=contact_model.value,
            contact_patch_radius_cm=round(r_cm, 2),
            ball_edge_margin_cm=round(edge_margin_cm, 2),
            is_center_inside=(min_d >= 0.0),
            is_center_on_painted_line=is_on_strip,
            is_contact_inside_or_touching=(edge_margin_cm >= 0.0 or is_on_strip),
            nearest_line=nearest_line,
            target_region_name="SINGLES_COURT",
            line_strip=strip.to_dict()
        )

    @classmethod
    def evaluate_service_box_boundary(
        cls,
        court_x_m: float,
        court_y_m: float,
        target_box: ServiceBoxType,
        contact_patch_radius_m: Optional[float] = None,
        contact_model: ContactPatchModelType = ContactPatchModelType.EMPIRICAL_PATCH
    ) -> BoundaryEvaluationResult:
        """
        Evaluates a serve bounce contact point against the legal target service box
        with finite line widths and physical contact patch modeling.
        """
        if contact_model == ContactPatchModelType.POINT_CONTACT:
            r_m = 0.0
        elif contact_patch_radius_m is not None:
            r_m = contact_patch_radius_m
        else:
            r_m = cls.DEFAULT_CONTACT_PATCH_RADIUS_M
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

        strip = cls.get_line_strip(nearest_line) if nearest_line != CourtLineType.NONE else None
        is_on_strip = False
        if strip:
            sx_min, sx_max, sy_min, sy_max = strip.polygon_bounds_m
            is_on_strip = (sx_min <= court_x_m <= sx_max) and (sy_min <= court_y_m <= sy_max)

        edge_margin_cm = min_d_cm + r_cm

        return BoundaryEvaluationResult(
            signed_center_distance_m=round(min_d, 4),
            signed_center_distance_cm=round(min_d_cm, 2),
            contact_patch_model=contact_model.value,
            contact_patch_radius_cm=round(r_cm, 2),
            ball_edge_margin_cm=round(edge_margin_cm, 2),
            is_center_inside=(min_d >= 0.0),
            is_center_on_painted_line=is_on_strip,
            is_contact_inside_or_touching=(edge_margin_cm >= 0.0 or is_on_strip),
            nearest_line=nearest_line,
            target_region_name=target_box.value,
            line_strip=strip.to_dict() if strip else None
        )

    @classmethod
    def get_spatial_uncertainty_tier(cls, court_y_m: float) -> Tuple[str, float]:
        """
        Determines the spatial court tier and baseline geometric uncertainty (meters)
        based on calibrated camera distance and grazing-angle homography sensitivity.
        """
        if court_y_m >= cls.NET_Y:
            return ("NEAR", 0.008)  # 0.8 cm base uncertainty
        elif court_y_m >= cls.FAR_SERVICE_LINE_Y:
            return ("MID", 0.025)   # 2.5 cm base uncertainty
        else:
            return ("FAR", 0.350)   # 35.0 cm base uncertainty
