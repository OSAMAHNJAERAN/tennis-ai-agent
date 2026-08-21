import math
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    track_id: Optional[int] = None

def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_center(bbox: BBox) -> Tuple[float, float]:
    """Get the center point of the bounding box."""
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)

def get_foot_position(bbox: BBox) -> Tuple[float, float]:
    """Get the bottom center point of the bounding box, representing player foot position."""
    return ((bbox.x1 + bbox.x2) / 2.0, float(bbox.y2))

def get_bbox_area(bbox: BBox) -> float:
    """Calculate the area of the bounding box."""
    width = max(0.0, bbox.x2 - bbox.x1)
    height = max(0.0, bbox.y2 - bbox.y1)
    return width * height

def bbox_distance(bbox1: BBox, bbox2: BBox) -> float:
    """Calculate the distance between the centers of two bounding boxes."""
    c1 = get_center(bbox1)
    c2 = get_center(bbox2)
    return point_distance(c1, c2)
