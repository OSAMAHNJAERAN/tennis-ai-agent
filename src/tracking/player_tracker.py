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

        # Sort candidates by proximity to court lines/keypoints (closest to court surface first)
        candidates.sort(key=lambda x: (x[1], -x[3]))

        if len(candidates) >= 2:
            top_two = [candidates[0], candidates[1]]
            # Sort top two by vertical y-position: larger avg_y is Near Player (P1), smaller avg_y is Far Player (P2)
            top_two.sort(key=lambda x: x[2], reverse=True)
            p1_track_id = top_two[0][0]
            p2_track_id = top_two[1][0]
        elif len(candidates) == 1:
            p1_track_id = candidates[0][0]
            p2_track_id = None
        else:
            p1_track_id = None
            p2_track_id = None

        # Build per-frame lists with active track linking and reacquisition fallback
        import math
        player1_boxes: List[Optional[BBox]] = [None] * num_frames
        player2_boxes: List[Optional[BBox]] = [None] * num_frames

        last_p1_box = None
        last_p2_box = None

        for i in range(num_frames):
            frame_dets = detections[i]
            
            # Primary choice: from assigned track IDs
            box_p1 = track_stats[p1_track_id]['bboxes'].get(i) if p1_track_id in track_stats else None
            box_p2 = track_stats[p2_track_id]['bboxes'].get(i) if p2_track_id in track_stats else None
            
            # Dynamic separator between near and far court
            if last_p1_box is not None and last_p2_box is not None:
                dynamic_mid_y = (last_p1_box.y1 + last_p2_box.y2) / 2.0
            else:
                dynamic_mid_y = court_mid_y + 40.0

            # Reacquisition / Track ID switch recovery for Near Court (Player 1)
            if box_p1 is None and frame_dets:
                near_cands = [d for d in frame_dets if ((d.y1 + d.y2) / 2.0 > dynamic_mid_y or d.y2 > dynamic_mid_y)]
                if near_cands:
                    if last_p1_box is not None:
                        near_cands.sort(key=lambda b: (
                            math.hypot((b.x1+b.x2)/2 - (last_p1_box.x1+last_p1_box.x2)/2, (b.y1+b.y2)/2 - (last_p1_box.y1+last_p1_box.y2)/2)
                            - 50.0 * b.confidence
                        ))
                    else:
                        near_cands.sort(key=lambda b: -(b.y2 + 50.0 * b.confidence))
                    box_p1 = near_cands[0]

            # Reacquisition / Track ID switch recovery for Far Court (Player 2)
            if box_p2 is None and frame_dets:
                far_cands = [d for d in frame_dets if ((d.y1 + d.y2) / 2.0 <= dynamic_mid_y or d.y1 <= dynamic_mid_y)]
                if far_cands:
                    if last_p2_box is not None:
                        far_cands.sort(key=lambda b: (
                            math.hypot((b.x1+b.x2)/2 - (last_p2_box.x1+last_p2_box.x2)/2, (b.y1+b.y2)/2 - (last_p2_box.y1+last_p2_box.y2)/2)
                            - 50.0 * b.confidence
                        ))
                    else:
                        far_cands.sort(key=lambda b: (b.y1 - 50.0 * b.confidence))
                    box_p2 = far_cands[0]

            if box_p1 is not None:
                last_p1_box = box_p1
            if box_p2 is not None:
                last_p2_box = box_p2

            player1_boxes[i] = box_p1
            player2_boxes[i] = box_p2

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
