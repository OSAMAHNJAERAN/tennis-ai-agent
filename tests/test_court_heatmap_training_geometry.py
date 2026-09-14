import cv2
import numpy as np
import pytest

from scripts.train.finetune_court_heatmap import landmarks_with_center, make_targets, transform_points


def test_landmarks_follow_warped_marked_pixels():
    image = np.zeros((360, 640), np.uint8)
    points = np.array([[200, 100], [400, 200]], dtype=float)
    for x, y in points.astype(int):
        image[y, x] = 255
    matrix = np.array([[0, -1, 450], [1, 0, -100], [0, 0, 1]], dtype=float)
    warped = cv2.warpPerspective(image, matrix, (640, 360), flags=cv2.INTER_NEAREST)
    mapped = transform_points(points, matrix).astype(int)
    assert np.count_nonzero(warped) == 2
    for x, y in mapped:
        assert warped[y, x] == 255


def test_center_uses_projective_diagonal_intersection():
    corners = np.array([[100, 40], [400, 40], [100, 300], [400, 300]], dtype=float)
    points = np.vstack((corners, np.tile([[250, 170]], (10, 1))))
    matrix = np.array([[1, .1, 30], [.03, 1, -4], [.001, -.0002, 1]])
    transformed = transform_points(points, matrix)
    center_after = landmarks_with_center(transformed)[-1]
    center_before = transform_points([[250, 170]], matrix)[0]
    np.testing.assert_allclose(center_after, center_before, atol=1e-9)
    assert np.linalg.norm(center_after - transformed[:4].mean(axis=0)) > 1


def test_outside_targets_absent_and_edge_target_not_discarded():
    points = np.tile([[200.9, 100.2]], (15, 1))
    points[0] = [-.1, 100]
    points[1] = [640, 100]
    points[2] = [0, 0]
    target = make_targets(points)
    assert not target[0].any() and not target[1].any()
    assert target[2, 0, 0] == 1
    assert target[3, 100, 200] == 1
    np.testing.assert_allclose(target[3, 100, 201], np.exp(-1 / (2 * (111 / 6) ** 2)), rtol=1e-6)


def test_projective_horizon_is_rejected():
    with pytest.raises(ValueError, match='depth'):
        transform_points([[10, 10]], [[1, 0, 0], [0, 1, 0], [0, 0, -1]])
