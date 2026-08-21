import cv2
import numpy as np
from typing import Tuple, Optional

def calculate_reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """Calculate the mean reprojection error of the homography matrix."""
    if H is None or len(src) == 0 or len(dst) == 0:
        return float('inf')
        
    src_transformed = transform_points(src, H)
    errors = np.linalg.norm(dst - src_transformed, axis=1)
    return float(np.mean(errors))

def compute_homography(image_points: np.ndarray, court_points: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    Computes homography from image points to canonical court points.
    Uses RANSAC for robust estimation.
    Returns the Homography matrix and the reprojection error.
    """
    if len(image_points) < 4 or len(court_points) < 4:
        raise ValueError("At least 4 point correspondences are required to compute homography.")
        
    # Find homography using RANSAC
    H, mask = cv2.findHomography(image_points, court_points, cv2.RANSAC, 5.0)
    
    if H is not None:
        error = calculate_reprojection_error(H, image_points, court_points)
    else:
        error = float('inf')
        
    return H, error

def transform_point(point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Projects a single point using the homography matrix."""
    if H is None:
        return point
        
    p = np.array([point[0], point[1], 1.0])
    p_transformed = H @ p
    
    if p_transformed[2] == 0:
        return (0.0, 0.0)
        
    p_transformed /= p_transformed[2]
    return (float(p_transformed[0]), float(p_transformed[1]))

def transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Batch transforms multiple points using the homography matrix."""
    if H is None or len(points) == 0:
        return points
        
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply homography
    transformed_h = (H @ points_h.T).T
    
    # Convert back to Cartesian coordinates safely
    with np.errstate(divide='ignore', invalid='ignore'):
        transformed = transformed_h[:, :2] / transformed_h[:, 2:]
        
    # Replace NaNs or infs with 0s if they happen (e.g., points at infinity)
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return transformed

def inverse_transform_point(court_point: Tuple[float, float], H: np.ndarray) -> Tuple[float, float]:
    """Projects a point from court coordinates back to image coordinates."""
    if H is None:
        return court_point
        
    try:
        H_inv = np.linalg.inv(H)
        return transform_point(court_point, H_inv)
    except np.linalg.LinAlgError:
        # Handle singular matrix
        return court_point

def validate_homography(H: np.ndarray, image_points: np.ndarray, court_points: np.ndarray, max_error: float = 10.0) -> bool:
    """
    Validates if the computed homography is geometrically plausible 
    and has an acceptable reprojection error.
    """
    if H is None:
        return False
        
    # Check determinant to ensure it's not a degenerate transformation
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return False
        
    error = calculate_reprojection_error(H, image_points, court_points)
    return error <= max_error
