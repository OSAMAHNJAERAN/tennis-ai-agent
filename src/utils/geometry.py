import math
import cv2
import numpy as np
from typing import Tuple

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate standard Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """Calculate the shortest distance from a point to a line segment."""
    px, py = point
    sx, sy = line_start
    ex, ey = line_end
    
    line_mag = math.hypot(ex - sx, ey - sy)
    if line_mag == 0.0:
        return euclidean_distance(point, line_start)
        
    u = ((px - sx) * (ex - sx) + (py - sy) * (ey - sy)) / (line_mag ** 2)
    
    if u < 0.0:
        return euclidean_distance(point, line_start)
    elif u > 1.0:
        return euclidean_distance(point, line_end)
    else:
        ix = sx + u * (ex - sx)
        iy = sy + u * (ey - sy)
        return euclidean_distance(point, (ix, iy))

def is_point_inside_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a given polygon using cv2.pointPolygonTest.
    Returns True if strictly inside or on the edge.
    """
    # measureDist=False returns +1 for inside, 0 for on edge, -1 for outside
    result = cv2.pointPolygonTest(polygon.astype(np.float32), point, measureDist=False)
    return result >= 0
