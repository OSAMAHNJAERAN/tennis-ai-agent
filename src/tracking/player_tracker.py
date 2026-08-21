import cv2
import numpy as np
from typing import List, Dict
from src.utils.bbox_utils import BBox

class PlayerTracker:
    """Player tracking and filtering utility."""

    @staticmethod
    def choose_players(detections: List[List[BBox]], court_keypoints: np.ndarray) -> Dict[int, List[BBox]]:
        """
        Selects the 2 court players across all frames using foot position distance to court keypoints.
        
        Args:
            detections: List of tracked BBox detections per frame.
            court_keypoints: Keypoints defining the court.
            
        Returns:
            Dictionary mapping player_id (1 or 2) to list of BBox per frame.
        """
        # Placeholder implementation for selecting the 2 players
        # Should aggregate foot distances to court keypoints and assign IDs
        player_dict = {1: [], 2: []}
        return player_dict

    @staticmethod
    def filter_non_players(detections: List[BBox], court_polygon: np.ndarray) -> List[BBox]:
        """
        Removes spectators outside the court area.
        
        Args:
            detections: List of BBox detections in a frame.
            court_polygon: Polygon defining the active court area.
            
        Returns:
            Filtered list of BBox detections inside the court.
        """
        filtered = []
        for bbox in detections:
            # Foot position (bbox bottom-center)
            foot_x = (bbox.x1 + bbox.x2) / 2
            foot_y = bbox.y2
            point = (float(foot_x), float(foot_y))
            
            # Check if point is inside or on the edge of the polygon
            if cv2.pointPolygonTest(court_polygon, point, False) >= 0:
                filtered.append(bbox)
        return filtered
