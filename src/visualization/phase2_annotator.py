import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

class Phase2VisualAnnotator:
    """
    Enhanced Visual Annotator for Phase 2.
    Renders distinct color-coded markers for each ball state and trajectory trails.
    """
    
    # State color mapping in BGR
    STATE_COLORS = {
        BallState.DETECTED: (0, 255, 0),        # Bright Green
        BallState.TRACKED: (255, 255, 0),       # Cyan
        BallState.PREDICTED: (255, 0, 255),     # Magenta / Purple
        BallState.INTERPOLATED: (0, 255, 255),  # Yellow
        BallState.OCCLUDED: (0, 0, 255),        # Red
        BallState.MISSING: (128, 128, 128)      # Grey
    }

    @staticmethod
    def draw_ball_marker(frame: np.ndarray, pt: TemporalBallPoint) -> np.ndarray:
        """Draws a state-aware ball marker with confidence label."""
        if pt.x_px is None or pt.y_px is None or pt.state == BallState.MISSING:
            return frame
            
        img = frame.copy()
        color = Phase2VisualAnnotator.STATE_COLORS.get(pt.state, (255, 255, 255))
        cx, cy = int(pt.x_px), int(pt.y_px)
        
        if pt.state == BallState.DETECTED:
            cv2.circle(img, (cx, cy), 6, color, -1)
            cv2.circle(img, (cx, cy), 8, (255, 255, 255), 1)
        elif pt.state == BallState.TRACKED:
            cv2.circle(img, (cx, cy), 5, color, -1)
            cv2.circle(img, (cx, cy), 7, (0, 0, 0), 1)
        elif pt.state == BallState.PREDICTED:
            cv2.circle(img, (cx, cy), 5, color, -1)
            cv2.circle(img, (cx, cy), 10, color, 1)  # Outer prediction ring
        elif pt.state == BallState.INTERPOLATED:
            cv2.circle(img, (cx, cy), 4, color, -1)
        elif pt.state == BallState.OCCLUDED:
            cv2.circle(img, (cx, cy), 8, color, 2)
            
        return img

    @staticmethod
    def draw_multi_state_trajectory(
        frame: np.ndarray,
        trajectory: List[TemporalBallPoint],
        current_frame_idx: int,
        max_trail: int = 25
    ) -> np.ndarray:
        """Draws multi-colored recent trajectory trail reflecting each point's state."""
        img = frame.copy()
        start_idx = max(0, current_frame_idx - max_trail)
        sub_traj = trajectory[start_idx:current_frame_idx + 1]
        
        valid_pts = [p for p in sub_traj if p.x_px is not None and p.y_px is not None]
        if len(valid_pts) < 2:
            return img
            
        for i in range(1, len(valid_pts)):
            p1 = valid_pts[i-1]
            p2 = valid_pts[i]
            pt1 = (int(p1.x_px), int(p1.y_px))
            pt2 = (int(p2.x_px), int(p2.y_px))
            color = Phase2VisualAnnotator.STATE_COLORS.get(p2.state, (255, 255, 255))
            thickness = int(np.interp(i, [0, len(valid_pts)], [1, 3]))
            cv2.line(img, pt1, pt2, color, thickness)
            
        return img

    @staticmethod
    def draw_state_legend(frame: np.ndarray, position: Tuple[int, int] = (20, 20)) -> np.ndarray:
        """Draws a compact visual legend showing color codes for ball states."""
        img = frame.copy()
        x0, y0 = position
        legend_w, legend_h = 220, 130
        
        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        cv2.putText(img, "BALL TRACKING STATES", (x0 + 10, y0 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
        items = [
            ("DETECTED (High Conf)", Phase2VisualAnnotator.STATE_COLORS[BallState.DETECTED]),
            ("TRACKED (Gated Candidate)", Phase2VisualAnnotator.STATE_COLORS[BallState.TRACKED]),
            ("PREDICTED (Kalman)", Phase2VisualAnnotator.STATE_COLORS[BallState.PREDICTED]),
            ("INTERPOLATED (Gap <= 3)", Phase2VisualAnnotator.STATE_COLORS[BallState.INTERPOLATED]),
        ]
        
        for idx, (label, color) in enumerate(items):
            y = y0 + 45 + idx * 20
            cv2.circle(img, (x0 + 15, y), 5, color, -1)
            cv2.putText(img, label, (x0 + 30, y + 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                        
        return img
