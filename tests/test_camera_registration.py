import cv2
import numpy as np
import pytest

from src.court.calibration import CourtCalibration
from src.court.camera_registration import CourtCameraRegistration
from src.utils.bbox_utils import BBox


def fixture():
    rng = np.random.default_rng(7)
    image = rng.integers(20, 220, (360, 640), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .5)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), np.array([[40, 40], [600, 40], [600, 320], [40, 320]], np.float32)


def test_camera_translation_does_not_become_player_ground_motion():
    image, points = fixture()
    initial = CourtCalibration(np.diag([.02, .02, 1.]))
    tracker = CourtCameraRegistration(initial, points)
    assert tracker.update(image).calibration.is_valid
    translation = np.array([[1., 0, 18], [0, 1., -8], [0, 0, 1.]])
    moved = cv2.warpPerspective(image, translation, (640, 360))
    result = tracker.update(moved)
    assert result.calibration.is_valid, result.audit
    assert result.audit['max_landmark_displacement_source_px'] == pytest.approx(np.hypot(18, 8), abs=.5)
    assert result.calibration.project_ground_point((218, 192)) == pytest.approx((4, 4), abs=.03)
    assert initial.project_ground_point((218, 192)) != pytest.approx((4, 4), abs=.03)


def test_small_zoom_and_projective_motion_preserve_court_coordinates():
    image, points = fixture()
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    tracker.update(image)
    transform = np.array([[1.02, .008, -8], [.003, 1.015, -4], [.00001, -.00001, 1.]])
    result = tracker.update(cv2.warpPerspective(image, transform, (640, 360)))
    assert result.calibration.is_valid, result.audit
    moved = cv2.perspectiveTransform(np.array([[[320., 200.]]], np.float32), transform)[0, 0]
    assert result.calibration.project_ground_point(moved) == pytest.approx((320, 200), abs=.3)


def test_scene_loss_latches_instead_of_reusing_stale_calibration():
    image, points = fixture()
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    tracker.update(image)
    lost = tracker.update(np.zeros_like(image))
    assert not lost.calibration.is_valid
    assert lost.audit['reinitialization_required']
    assert tracker.update(image).calibration.image_to_court is None


def test_resolution_change_and_featureless_anchor_abstain():
    image, points = fixture()
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    tracker.update(image)
    assert tracker.update(image[:180]).audit['rejection_reason'] == 'SOURCE_RESOLUTION_CHANGED'
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    blank = np.zeros_like(image)
    assert tracker.update(blank).calibration.is_valid  # only the supplied first-frame fit
    assert tracker.update(blank).audit['rejection_reason'] == 'INSUFFICIENT_ANCHOR_FEATURES'


def test_moving_foreground_person_does_not_define_the_camera_transform():
    image, points = fixture()
    first = image.copy()
    first[90:270, 180:420] = np.random.default_rng(20).integers(0, 255, (180, 240, 3), np.uint8)
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    tracker.update(first, [BBox(180, 90, 420, 270)])
    translation = np.array([[1., 0, 12], [0, 1, 4], [0, 0, 1.]])
    moved = cv2.warpPerspective(image, translation, (640, 360))
    moved[110:290, 280:520] = first[90:270, 180:420]
    result = tracker.update(moved, [BBox(280, 110, 520, 290)])
    assert result.calibration.is_valid, result.audit
    assert result.calibration.project_ground_point((112, 104)) == pytest.approx((100, 100), abs=.2)


def test_features_in_one_tiny_patch_cannot_certify_the_entire_court():
    image, points = fixture()
    clustered = np.zeros_like(image)
    clustered[150:205, 300:355] = image[150:205, 300:355]
    tracker = CourtCameraRegistration(CourtCalibration(np.eye(3)), points)
    tracker.update(clustered)
    result = tracker.update(clustered)
    assert not result.calibration.is_valid
    assert result.audit['rejection_reason'] == 'CAMERA_FEATURES_TOO_LOCALIZED'
