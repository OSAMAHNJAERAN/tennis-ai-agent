import numpy as np

from src.court.calibration import calibrate_court, CourtCalibration
from src.court.court_geometry import TennisCourtGeometry
from src.tracking.court_player_tracker import CourtPlayerTracker
from src.utils.bbox_utils import BBox


def calibration():
    points = TennisCourtGeometry.get_canonical_keypoints()
    return calibrate_court(points * 30 + 100, points)


def box(x, y, identity):
    cx, foot = x * 30 + 100, y * 30 + 100
    return BBox(cx - 12, foot - 60, cx + 12, foot, track_id=identity, confidence=.9)


def test_staff_near_corner_does_not_displace_a_player_behind_baseline():
    near, far, staff = box(5, 26, 1), box(5, -2, 2), box(-5, -6, 3)
    result = CourtPlayerTracker(calibration()).select([[staff, far, near] for _ in range(12)], fps=30)
    assert all(b.track_id == 1 for b in result[1])
    assert all(b.track_id == 2 for b in result[2])


def test_one_detection_is_never_assigned_to_both_players():
    candidate = box(5, 11.885, 7)
    result = CourtPlayerTracker(calibration()).select([[candidate] for _ in range(5)], fps=30)
    for first, second in zip(result[1], result[2]):
        assert first is None or second is None


def test_short_occlusion_reacquires_new_id_without_filling_the_gap():
    frames = [[box(5, 25, 1), box(5, -2, 2)], [box(5, -2, 2)],
              [box(5.1, 25, 9), box(5, -2, 2)]]
    result = CourtPlayerTracker(calibration()).select(frames, fps=30)
    assert result[1][0].track_id == 1
    assert result[1][1] is None
    assert result[1][2].track_id == 9


def test_short_gap_does_not_jump_to_distant_spectator():
    frames = [[box(5, 25, 1)], [box(0, 25, 9)]]
    result = CourtPlayerTracker(calibration()).select(frames, fps=30)
    assert result[1][0] is not None
    assert result[1][1] is None


def test_invalid_geometry_abstains():
    invalid = calibrate_court(np.zeros((14, 2)), TennisCourtGeometry.get_canonical_keypoints())
    assert CourtPlayerTracker(invalid).select([[box(5, 20, 1)]], fps=30) == {1: [None], 2: [None]}


def test_camera_translation_uses_frame_specific_geometry_for_player_association():
    first = box(5, 25, 1)
    moved = box(5, 25, 1)
    moved.x1 += 200
    moved.x2 += 200
    initial = calibration()
    translation_inverse = np.array([[1., 0, -200], [0, 1, 0], [0, 0, 1]])
    corrected = CourtCalibration(initial.image_to_court @ translation_inverse)
    # Static coordinates see an implausible jump; dynamic geometry keeps the role.
    frames = [[first], [moved]]
    assert CourtPlayerTracker(initial).select(frames, 30)[1][1] is None
    result = CourtPlayerTracker(initial).select(frames, 30, [initial, corrected])
    assert result[1][1] is moved


def test_camera_loss_does_not_export_stale_player_positions():
    initial = calibration()
    lost = CourtCalibration(None, rejection_reason='CAMERA_LOST')
    player = box(5, 25, 1)
    result = CourtPlayerTracker(initial).select([[player], [player]], 30, [initial, lost])
    assert result[1] == [player, None]
