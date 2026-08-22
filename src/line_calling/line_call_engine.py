import math
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from src.line_calling.line_geometry import (
    CourtLineGeometry,
    CourtLineType,
    ServiceBoxType,
    ContactPatchModelType,
    BoundaryEvaluationResult,
    CourtLineStrip
)
from src.line_calling.contact_refinement import BounceContactRefiner, RefinedContactPoint
from src.court.homography import transform_point
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint

class LineCallDecision(str, Enum):
    IN = "IN"
    OUT = "OUT"
    SERVE_IN = "SERVE_IN"
    SERVE_FAULT = "SERVE_FAULT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNKNOWN = "UNKNOWN"

class LineCallContext(str, Enum):
    RALLY = "RALLY"
    SERVE = "SERVE"

@dataclass
class LineCallEvidence:
    event_id: int
    bounce_frame: int
    decision: LineCallDecision
    decision_context: LineCallContext
    nearest_line: CourtLineType
    bounce_position_px: Tuple[float, float]
    bounce_position_m: Tuple[float, float]
    tracker_state: str
    center_signed_distance_cm: float
    contact_patch_model: str
    contact_patch_radius_cm: float
    ball_edge_margin_cm: float
    position_uncertainty_cm: float
    spatial_tier: str
    confidence: float
    reason: str
    refinement_method: str
    line_strip_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['decision'] = self.decision.value
        d['decision_context'] = self.decision_context.value
        d['nearest_line'] = self.nearest_line.value
        return d

class TennisLineCallEngine:
    """
    Confidence-aware assisted tennis IN/OUT line-call decision engine.
    Integrates finite-width painted line geometry, empirical contact patch modeling,
    piecewise collision trajectory refinement, and depth-dependent spatial uncertainty.
    """

    def __init__(
        self,
        refiner: Optional[BounceContactRefiner] = None,
        contact_patch_model: ContactPatchModelType = ContactPatchModelType.EMPIRICAL_PATCH,
        contact_patch_radius_cm: float = 1.25,
        uncertainty_safety_factor: float = 1.5,
        line_width_uncertainty_cm: float = 1.25
    ):
        self.refiner = refiner or BounceContactRefiner()
        self.contact_patch_model = contact_patch_model
        self.contact_patch_radius_cm = contact_patch_radius_cm
        self.uncertainty_safety_factor = uncertainty_safety_factor
        self.line_width_uncertainty_cm = line_width_uncertainty_cm

    def evaluate_bounce(
        self,
        event_id: int,
        bounce_frame: int,
        ball_trajectory: List[TemporalBallPoint],
        homography_matrix: np.ndarray,
        context: LineCallContext = LineCallContext.RALLY,
        target_service_box: Optional[ServiceBoxType] = None,
        raw_frames: Optional[List[np.ndarray]] = None
    ) -> LineCallEvidence:
        """
        Evaluates a verified bounce event and returns structured line call evidence.
        """
        base_pt = ball_trajectory[bounce_frame]
        tracker_st = base_pt.state.value if base_pt.state else "UNKNOWN"

        # 1. Contact Point Refinement (Piecewise Impact / Change-Point)
        refined = self.refiner.refine_bounce_contact(bounce_frame, ball_trajectory, raw_frames)
        bounce_px = (refined.x_px, refined.y_px)

        # 2. Metric Projection
        court_m = transform_point(bounce_px, homography_matrix)
        court_x_m, court_y_m = float(court_m[0]), float(court_m[1])

        # 3. Spatial Uncertainty Estimation
        spatial_tier, base_unc_m = CourtLineGeometry.get_spatial_uncertainty_tier(court_y_m)
        
        # Tracking state uncertainty multiplier
        if tracker_st == BallState.DETECTED.value:
            state_multiplier = 1.0
        elif tracker_st == BallState.TRACKED.value:
            state_multiplier = 1.5
        elif tracker_st == BallState.INTERPOLATED.value:
            state_multiplier = 3.0
        else: # PREDICTED or UNKNOWN
            state_multiplier = 20.0

        geom_unc_cm = (base_unc_m * state_multiplier) * 100.0
        total_unc_cm = round(math.hypot(geom_unc_cm, self.line_width_uncertainty_cm), 2)
        safety_margin_cm = self.uncertainty_safety_factor * total_unc_cm

        # 4. Mandatory Safety Gate: PREDICTED State Abstention
        if tracker_st == BallState.PREDICTED.value:
            return LineCallEvidence(
                event_id=event_id,
                bounce_frame=bounce_frame,
                decision=LineCallDecision.REVIEW_REQUIRED,
                decision_context=context,
                nearest_line=CourtLineType.NONE,
                bounce_position_px=bounce_px,
                bounce_position_m=(round(court_x_m, 3), round(court_y_m, 3)),
                tracker_state=tracker_st,
                center_signed_distance_cm=0.0,
                contact_patch_model=self.contact_patch_model.value,
                contact_patch_radius_cm=self.contact_patch_radius_cm,
                ball_edge_margin_cm=0.0,
                position_uncertainty_cm=total_unc_cm,
                spatial_tier=spatial_tier,
                confidence=0.20,
                reason="Bounce position derived from PREDICTED Kalman state without optical measurement; automatic call is unsafe.",
                refinement_method=refined.refinement_method,
                line_strip_info=None
            )

        # 5. Boundary Evaluation based on Shot Context and Finite Line Strips
        r_m = self.contact_patch_radius_cm / 100.0
        if context == LineCallContext.SERVE:
            box = target_service_box or ServiceBoxType.NEAR_DEUCE
            eval_res = CourtLineGeometry.evaluate_service_box_boundary(
                court_x_m=court_x_m,
                court_y_m=court_y_m,
                target_box=box,
                contact_patch_radius_m=r_m,
                contact_model=self.contact_patch_model
            )
        else:
            eval_res = CourtLineGeometry.evaluate_singles_rally_boundary(
                court_x_m=court_x_m,
                court_y_m=court_y_m,
                contact_patch_radius_m=r_m,
                contact_model=self.contact_patch_model
            )

        margin_cm = eval_res.ball_edge_margin_cm
        nearest_line = eval_res.nearest_line

        # 6. Geometric Abstention Model Gate (Model D)
        if (self.contact_patch_model == ContactPatchModelType.ABSTENTION_GEOMETRIC and
            eval_res.signed_center_distance_cm < 0 and abs(eval_res.signed_center_distance_cm) <= 2.5):
            return LineCallEvidence(
                event_id=event_id,
                bounce_frame=bounce_frame,
                decision=LineCallDecision.REVIEW_REQUIRED,
                decision_context=context,
                nearest_line=nearest_line,
                bounce_position_px=bounce_px,
                bounce_position_m=(round(court_x_m, 3), round(court_y_m, 3)),
                tracker_state=tracker_st,
                center_signed_distance_cm=eval_res.signed_center_distance_cm,
                contact_patch_model=self.contact_patch_model.value,
                contact_patch_radius_cm=self.contact_patch_radius_cm,
                ball_edge_margin_cm=margin_cm,
                position_uncertainty_cm=total_unc_cm,
                spatial_tier=spatial_tier,
                confidence=0.50,
                reason=f"Ball center is outside legal outer edge ({eval_res.signed_center_distance_cm:+.1f} cm) within deformation limit (< 2.5 cm); automatic call is unsafe.",
                refinement_method=refined.refinement_method,
                line_strip_info=eval_res.line_strip
            )

        # 7. Decision Classification with Uncertainty Propagation
        if margin_cm >= safety_margin_cm or eval_res.is_center_on_painted_line:
            # Definite IN beyond uncertainty envelope or directly on painted strip
            decision = LineCallDecision.SERVE_IN if context == LineCallContext.SERVE else LineCallDecision.IN
            conf = min(0.98, max(0.70, 1.0 - (total_unc_cm / (abs(margin_cm) + 1e-3))))
            reason = f"Ball contact patch is inside legal {eval_res.target_region_name} by {margin_cm:.1f} cm beyond uncertainty (±{total_unc_cm:.1f} cm)."
            
        elif margin_cm <= -safety_margin_cm:
            # Definite OUT beyond uncertainty envelope
            decision = LineCallDecision.SERVE_FAULT if context == LineCallContext.SERVE else LineCallDecision.OUT
            conf = min(0.98, max(0.70, 1.0 - (total_unc_cm / (abs(margin_cm) + 1e-3))))
            reason = f"Ball contact patch is outside legal {eval_res.target_region_name} by {abs(margin_cm):.1f} cm past {nearest_line.value} beyond uncertainty (±{total_unc_cm:.1f} cm)."
            
        else:
            # Boundary line intersects uncertainty envelope -> Safe Abstention
            decision = LineCallDecision.REVIEW_REQUIRED
            conf = 0.50
            reason = f"Ball edge margin ({margin_cm:+.1f} cm from {nearest_line.value}) is within spatial uncertainty envelope (±{total_unc_cm:.1f} cm); automatic call is unsafe."

        return LineCallEvidence(
            event_id=event_id,
            bounce_frame=bounce_frame,
            decision=decision,
            decision_context=context,
            nearest_line=nearest_line,
            bounce_position_px=bounce_px,
            bounce_position_m=(round(court_x_m, 3), round(court_y_m, 3)),
            tracker_state=tracker_st,
            center_signed_distance_cm=eval_res.signed_center_distance_cm,
            contact_patch_model=self.contact_patch_model.value,
            contact_patch_radius_cm=self.contact_patch_radius_cm,
            ball_edge_margin_cm=margin_cm,
            position_uncertainty_cm=total_unc_cm,
            spatial_tier=spatial_tier,
            confidence=round(conf, 3),
            reason=reason,
            refinement_method=refined.refinement_method,
            line_strip_info=eval_res.line_strip
        )
