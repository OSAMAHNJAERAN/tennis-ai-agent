"""Calibration must reject unsupported geometry and report residual units."""

import cv2
import numpy as np
import pytest

from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry


def correspondences():
    court = TennisCourtGeometry.get_canonical_keypoints().astype(float)
    matrix = np.array([[60., 8., 300.], [0., 30., 120.], [0., .012, 1.]])
    image = cv2.perspectiveTransform(court[None], matrix)[0]
    return court, image


def test_calibration_recovers_projective_mapping_and_reports_both_units():
    court, image = correspondences()
    result = calibrate_court(image, court)
    assert result.is_valid
    assert result.inlier_count == len(court)
    assert result.reprojection_error_px < .001
    assert result.reprojection_error_m < .001
    assert result.project_ground_point(tuple(image[5])) == pytest.approx(court[5], abs=.001)
    assert result.to_dict()["homography_matrix"] is not None


def test_image_space_ransac_rejects_a_wrong_landmark():
    court, image = correspondences()
    image[3] += [110, -80]
    result = calibrate_court(image, court)
    assert result.is_valid
    assert result.inlier_count == len(court) - 1
    assert result.inlier_reprojection_error_px < .001
    assert result.reprojection_error_px > 5
    assert not result.inlier_mask[3]


@pytest.mark.parametrize("kind", ["collinear", "nonfinite", "insufficient", "disagreement"])
def test_invalid_calibration_cannot_produce_metric_positions(kind):
    court, image = correspondences()
    if kind == "collinear":
        image[:, 1] = 100
    elif kind == "nonfinite":
        image[0, 0] = np.nan
    elif kind == "insufficient":
        court, image = court[:3], image[:3]
    else:
        image = np.random.default_rng(82).uniform(0, 1000, image.shape)
    result = calibrate_court(image, court)
    assert not result.is_valid
    assert result.project_ground_point((123, 456)) is None
    assert result.to_dict()["homography_matrix"] is None
    assert result.rejection_reason


def test_nonfinite_point_is_unknown_even_with_valid_calibration():
    court, image = correspondences()
    result = calibrate_court(image, court)
    assert result.project_ground_point((float("inf"), 1)) is None
