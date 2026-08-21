import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.utils.bbox_utils import BBox

class PlayerTracker:
    """Player tracking and filtering utility."""

    @staticmethod
    def choose_players(detections: List[List[BBox]], court_keypoints: np.ndarray) -> Dict[int, List[Optional[BBox]]]:
        """
        Selects the 2 court players across all frames using foot position distance to court keypoints.
        Identifies one near-court player (Player 1) and one far-court player (Player 2).
        
        Args:
            detections: List of tracked BBox detections per frame.
            court_keypoints: (14, 2) array of detected court keypoints in image coordinates.
            
        Returns:
            Dictionary mapping player_id (1 and 2) to list of Optional[BBox] per frame.
        """
        num_frames = len(detections)
        if num_frames == 0:
            return {1: [], 2: []}

        # Net vertical position roughly divides near vs far court
        # Keypoints 8, 9, 10, 11, 12, 13 are service lines; court keypoint 0..3 are baseline corners
        # Calculate mean y coordinate of all court keypoints to approximate net/midcourt level
        court_mid_y = float(np.mean(court_keypoints[:, 1]))

        # Accumulate presence and average distance to court for each track_id
        track_stats = {}  # track_id -> {'count': int, 'mean_dist': float, 'mean_y': float, 'bboxes': dict}
        
        for f_idx, frame_dets in enumerate(detections):
            for bbox in frame_dets:
                if bbox.track_id is None:
                    continue
                tid = bbox.track_id
                foot_x = (bbox.x1 + bbox.x2) / 2.0
                foot_y = float(bbox.y2)
                
                # Distance to nearest court keypoint
                dists = np.linalg.norm(court_keypoints - np.array([foot_x, foot_y]), axis=1)
                min_dist = float(np.min(dists))
                
                if tid not in track_stats:
                    track_stats[tid] = {'count': 0, 'total_dist': 0.0, 'total_y': 0.0, 'bboxes': {}}
                track_stats[tid]['count'] += 1
                track_stats[tid]['total_dist'] += min_dist
                track_stats[tid]['total_y'] += foot_y
                track_stats[tid]['bboxes'][f_idx] = bbox

        if not track_stats:
            # Fallback if no tracks available: take largest bounding boxes in upper and lower halves
            return {1: [None] * num_frames, 2: [None] * num_frames}

        # Calculate average distance and y for each track
        candidates = []
        for tid, stats in track_stats.items():
            if stats['count'] >= max(3, num_frames * 0.05):  # Must appear in at least 5% of frames
                avg_dist = stats['total_dist'] / stats['count']
                avg_y = stats['total_y'] / stats['count']
                candidates.append((tid, avg_dist, avg_y, stats['count']))

        if not candidates:
            # Take any tracks available
            candidates = [(tid, stats['total_dist'] / stats['count'], stats['total_y'] / stats['count'], stats['count']) 
                          for tid, stats in track_stats.items()]

        # Separate into near court (y > court_mid_y) and far court (y <= court_mid_y)
        near_candidates = [c for c in candidates if c[2] > court_mid_y]
        far_candidates = [c for c in candidates if c[2] <= court_mid_y]

        # Player 1 = Near player (closest to court in near half)
        # Player 2 = Far player (closest to court in far half)
        p1_track_id = None
        p2_track_id = None

        if near_candidates:
            # Sort by distance to court (ascending), then presence count (descending)
            near_candidates.sort(key=lambda x: (x[1], -x[3]))
            p1_track_id = near_candidates[0][0]
            
        if far_candidates:
            far_candidates.sort(key=lambda x: (x[1], -x[3]))
            p2_track_id = far_candidates[0][0]

        # If both ended up on one side or one side empty, pick the top 2 overall closest tracks
        if p1_track_id is None or p2_track_id is None or p1_track_id == p2_track_id:
            candidates.sort(key=lambda x: (x[1], -x[3]))
            if len(candidates) >= 2:
                # Assign the lower one (larger y) to p1, upper to p2
                c1, c2 = candidates[0], candidates[1]
                if c1[2] > c2[2]:
                    p1_track_id, p2_track_id = c1[0], c2[0]
                else:
                    p1_track_id, p2_track_id = c2[0], c1[0]
            elif len(candidates) == 1:
                p1_track_id = candidates[0][0]

        # Build per-frame lists
        player1_boxes = [track_stats[p1_track_id]['bboxes'].get(i, None) if p1_track_id in track_stats else None for i in range(num_frames)]
        player2_boxes = [track_stats[p2_track_id]['bboxes'].get(i, None) if p2_track_id in track_stats else None for i in range(num_frames)]

        return {1: player1_boxes, 2: player2_boxes}

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
