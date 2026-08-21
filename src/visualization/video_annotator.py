import numpy as np
import cv2
from typing import Optional, List, Dict

class VideoAnnotator:
    """
    Video frame annotation for players, ball, court keypoints, etc.
    """
    
    @staticmethod
    def draw_player_boxes(frame: np.ndarray, player_bboxes: dict, colors: dict) -> np.ndarray:
        img = frame.copy()
        for track_id, bbox in player_bboxes.items():
            color = colors.get(track_id, (255, 255, 255))
            # Assuming bbox is an object with x1, y1, x2, y2 attributes or properties
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"ID: {track_id}", (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # draw foot position (bbox bottom-center)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)
            cv2.circle(img, (foot_x, foot_y), 4, (0, 0, 255), -1)
        return img

    @staticmethod
    def draw_ball(frame: np.ndarray, ball_position: Optional[tuple], state) -> np.ndarray:
        # Assuming state has name property like BallPointState.DETECTED
        state_name = getattr(state, "name", str(state))
        if not ball_position or state_name in ("MISSING", "OCCLUDED"):
            return frame
            
        img = frame.copy()
        color = (0, 255, 0) if state_name == "DETECTED" else (0, 255, 255) # Green / Yellow
        cv2.circle(img, (int(ball_position[0]), int(ball_position[1])), 5, color, -1)
        return img

    @staticmethod
    def draw_court_keypoints(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        img = frame.copy()
        for i, kp in enumerate(keypoints):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return img

    @staticmethod
    def draw_ball_trajectory(frame: np.ndarray, recent_positions: List[tuple], max_trail: int = 20) -> np.ndarray:
        if len(recent_positions) < 2:
            return frame
        img = frame.copy()
        trail = recent_positions[-max_trail:]
        for i in range(1, len(trail)):
            pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            thickness = int(np.interp(i, [0, len(trail)], [1, 3]))
            cv2.line(img, pt1, pt2, (255, 0, 255), thickness)
        return img

    @staticmethod
    def compose_frame(frame: np.ndarray, mini_court_image: np.ndarray, stats_text: dict) -> np.ndarray:
        img = frame.copy()
        
        # Place mini_court in top-right corner
        h, w = mini_court_image.shape[:2]
        frame_h, frame_w = img.shape[:2]
        
        if frame_h > h and frame_w > w:
            img[0:h, frame_w-w:frame_w] = mini_court_image
            
        return img
