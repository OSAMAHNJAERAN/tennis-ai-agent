"""Image-space calibration with explicit units and nullable ground projections.

Fit residuals measure consistency with supplied landmarks, not independent court
accuracy. A homography applies only to points on the court plane; mapping an
airborne ball intersects its viewing ray with that plane, not its real position.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CourtCalibration:
    image_to_court: Optional[np.ndarray]
    inlier_mask: Tuple[bool, ...] = ()
    reprojection_error_px: Optional[float] = None
    reprojection_error_m: Optional[float] = None
    inlier_reprojection_error_px: Optional[float] = None
    rejection_reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.image_to_court is not None and self.rejection_reason is None

    @property
    def inlier_count(self) -> int:
        return sum(self.inlier_mask)

    def project_ground_point(self, point) -> Optional[Tuple[float, float]]:
        if not self.is_valid:
            return None
        xy = np.asarray(point, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return None
        projected = self.image_to_court @ np.append(xy, 1.)
        if not np.isfinite(projected).all() or abs(projected[2]) < 1e-10:
            return None
        return tuple(float(v) for v in projected[:2] / projected[2])

    def to_dict(self):
        return {
            "schema_version": "2.0",
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "reprojection_error_px": self.reprojection_error_px,
            "reprojection_error_m": self.reprojection_error_m,
            "inlier_reprojection_error_px": self.inlier_reprojection_error_px,
            "inlier_count": self.inlier_count,
            "correspondence_count": len(self.inlier_mask),
            "inlier_mask": list(self.inlier_mask),
            "homography_matrix": self.image_to_court.tolist() if self.is_valid else None,
            "fit_method": "COURT_TO_IMAGE_RANSAC_THEN_INVERSE",
            "validity_scope": "SUPPLIED_LANDMARK_FIT_AT_CALIBRATION_FRAME",
            "independent_accuracy_validated": False,
        }


def calibrate_court(image_points, court_points, *, ransac_threshold_px=3.0,
                    min_inliers=6, min_inlier_ratio=.75) -> CourtCalibration:
    """Estimate with a pixel RANSAC threshold; reject degenerate/weak support.

    Six inliers require evidence beyond the four-point exact-fit minimum.
    The default thresholds are engineering checks, not calibrated probabilities.
    """
    if not np.isfinite(ransac_threshold_px) or ransac_threshold_px <= 0:
        raise ValueError("ransac_threshold_px must be positive and finite")
    if min_inliers < 4 or not 0 < min_inlier_ratio <= 1:
        raise ValueError("Require at least four inliers and a ratio in (0, 1]")
    image = np.asarray(image_points, dtype=np.float64)
    court = np.asarray(court_points, dtype=np.float64)
    if (image.ndim != 2 or image.shape[1:] != (2,) or image.shape != court.shape
            or len(image) < min_inliers):
        return CourtCalibration(None, rejection_reason="INSUFFICIENT_CORRESPONDENCES")
    if not np.isfinite(image).all() or not np.isfinite(court).all():
        return CourtCalibration(None, rejection_reason="NONFINITE_LANDMARKS")
    if any(np.linalg.matrix_rank(points - points.mean(axis=0)) < 2 for points in (image, court)):
        return CourtCalibration(None, rejection_reason="DEGENERATE_LANDMARKS")
    # OpenCV's threshold uses destination units. Fit into pixels, then invert.
    try:
        court_to_image, mask = cv2.findHomography(court, image, cv2.RANSAC, ransac_threshold_px)
        if court_to_image is None or mask is None:
            return CourtCalibration(None, rejection_reason="HOMOGRAPHY_FIT_FAILED")
        inverse = np.linalg.inv(court_to_image)
        if not np.isfinite(inverse).all():
            return CourtCalibration(None, rejection_reason="NONFINITE_HOMOGRAPHY")
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < min_inliers or inliers.mean() < min_inlier_ratio:
            return CourtCalibration(None, tuple(bool(v) for v in inliers),
                                    rejection_reason="INSUFFICIENT_INLIER_SUPPORT")

        def project(points, matrix):
            homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
            if np.any(np.abs(homogeneous[:, 2]) < 1e-10):
                raise ValueError("Projection intersects horizon")
            return homogeneous[:, :2] / homogeneous[:, 2:]

        errors_px = np.linalg.norm(project(court, court_to_image) - image, axis=1)
        errors_m = np.linalg.norm(project(image, inverse) - court, axis=1)
        if not np.isfinite(errors_px).all() or not np.isfinite(errors_m).all():
            raise ValueError("Nonfinite residuals")
        inlier_error = float(errors_px[inliers].mean())
        reason = "EXCESSIVE_INLIER_RESIDUAL" if inlier_error > ransac_threshold_px else None
        return CourtCalibration(inverse if reason is None else None,
                                tuple(bool(v) for v in inliers), float(errors_px.mean()),
                                float(errors_m.mean()), inlier_error, reason)
    except (cv2.error, np.linalg.LinAlgError, ValueError):
        return CourtCalibration(None, rejection_reason="DEGENERATE_HOMOGRAPHY")
