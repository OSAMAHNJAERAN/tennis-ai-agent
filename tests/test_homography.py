import pytest
import numpy as np
from src.court.homography import (
    compute_homography,
    transform_points,
    transform_point,
    inverse_transform_point,
    calculate_reprojection_error,
    validate_homography
)

def test_identity_homography():
    pts = np.array([[0,0], [10,0], [10,20], [0,20]], dtype=np.float32)
    H, error = compute_homography(pts, pts)
    assert H is not None
    assert np.allclose(H / H[2, 2], np.eye(3), atol=1e-3)
    assert error < 1e-3

def test_known_transform():
    pts_src = np.array([[0,0], [10,0], [10,20], [0,20]], dtype=np.float32)
    pts_dst = np.array([[0,0], [20,0], [20,40], [0,40]], dtype=np.float32)
    H, error = compute_homography(pts_src, pts_dst)
    assert H is not None
    pts_transformed = transform_points(pts_src, H)
    assert np.allclose(pts_transformed, pts_dst, atol=1e-3)

def test_reprojection_error():
    pts_src = np.array([[0,0], [10,0], [10,20], [0,20]], dtype=np.float32)
    pts_dst = np.array([[0,0], [20,0], [20,40], [0,40]], dtype=np.float32)
    H, error = compute_homography(pts_src, pts_dst)
    err = calculate_reprojection_error(H, pts_src, pts_dst)
    assert err < 1e-3

def test_invalid_homography_rejected():
    # Less than 4 points raises ValueError
    pts_src = np.array([[0,0], [1,0], [2,0]], dtype=np.float32)
    pts_dst = np.array([[0,0], [1,0], [2,0]], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_homography(pts_src, pts_dst)

def test_transform_point_roundtrip():
    pt = (5.0, 10.0)
    pts_src = np.array([[0,0], [10,0], [10,20], [0,20]], dtype=np.float32)
    pts_dst = np.array([[0,0], [20,0], [20,40], [0,40]], dtype=np.float32)
    H, _ = compute_homography(pts_src, pts_dst)
    pt_mapped = transform_point(pt, H)
    pt_unmapped = inverse_transform_point(pt_mapped, H)
    assert np.isclose(pt[0], pt_unmapped[0], atol=1e-3)
    assert np.isclose(pt[1], pt_unmapped[1], atol=1e-3)

def test_validate_homography():
    pts_src = np.array([[0,0], [10,0], [10,20], [0,20]], dtype=np.float32)
    pts_dst = np.array([[0,0], [20,0], [20,40], [0,40]], dtype=np.float32)
    H, _ = compute_homography(pts_src, pts_dst)
    assert validate_homography(H, pts_src, pts_dst, max_error=5.0) is True
