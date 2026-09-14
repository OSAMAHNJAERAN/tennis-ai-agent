import math
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO

from src.utils.bbox_utils import BBox
from src.shot_analysis.shot_types import PlayerHandedness

class PoseFeatureExtractor:
    """
    Temporal Window Crop-Based YOLO11-Pose Feature Extractor.
    Efficiently processes targeted player crops around hit event frames without running full-frame inference.
    """
    
    def __init__(self, model_path: str = "yolo11n-pose.pt", device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model: Optional[YOLO] = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Warning: Failed to load YOLO11-Pose from {self.model_path}: {e}")
            self.model = None

    def extract_hit_window_features(
        self,
        frames: List[np.ndarray],
        player_boxes: List[Optional[BBox]],
        hit_frame: int,
        player_id: int,
        handedness: PlayerHandedness = PlayerHandedness.UNKNOWN_HANDEDNESS,
        court_orientation_sign: Optional[float] = None,
        window_half_size: int = 3,
        crop_margin: float = 0.65
    ) -> Dict[str, Any]:
        """
        Extracts temporal pose features around a hit contact frame.
        """
        if self.model is None or not frames:
            return {"valid": False, "reason": "Pose model unavailable or no frames."}
        if handedness == PlayerHandedness.UNKNOWN_HANDEDNESS:
            return {"valid": False, "reason": "Player handedness is unknown."}
        if court_orientation_sign not in (-1.0, 1.0):
            return {"valid": False, "reason": "Player body/court orientation is unknown."}

        total_frames = len(frames)
        start_f = max(0, hit_frame - window_half_size)
        end_f = min(total_frames - 1, hit_frame + window_half_size)

        court_side_sign = court_orientation_sign
        handedness_sign = 1.0 if handedness == PlayerHandedness.RIGHT_HANDED else (-1.0 if handedness == PlayerHandedness.LEFT_HANDED else 1.0)

        wrist_displacements = []
        pose_confidences = []
        shoulder_angles = []

        import torch
        with torch.no_grad():
            for f_idx in range(start_f, end_f + 1):
                bbox = player_boxes[f_idx]
                if bbox is None:
                    continue

                frame = frames[f_idx]
                fh, fw = frame.shape[:2]

                # Expand crop with margin
                bw = bbox.x2 - bbox.x1
                bh = bbox.y2 - bbox.y1
                cx = (bbox.x1 + bbox.x2) / 2.0
                cy = (bbox.y1 + bbox.y2) / 2.0

                x1_c = max(0, int(cx - (bw * (1.0 + crop_margin)) / 2.0))
                x2_c = min(fw, int(cx + (bw * (1.0 + crop_margin)) / 2.0))
                y1_c = max(0, int(cy - (bh * (1.0 + crop_margin)) / 2.0))
                y2_c = min(fh, int(cy + (bh * (1.0 + crop_margin)) / 2.0))

                if x2_c - x1_c < 20 or y2_c - y1_c < 20:
                    continue

                crop = frame[y1_c:y2_c, x1_c:x2_c]

                try:
                    results = self.model.predict(crop, verbose=False, device=self.device)
                    if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
                        continue

                    kpts_data = results[0].keypoints.data
                    if len(kpts_data) == 0:
                        continue

                    # Take the most confident person detection in the crop
                    kpts = kpts_data[0].cpu().numpy() # Shape: (17, 3) -> (x, y, conf)
                    if kpts.shape[0] < 17:
                        continue

                    # Keypoints (COCO):
                    # 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
                    l_sh, r_sh = kpts[5], kpts[6]
                    l_wr, r_wr = kpts[9], kpts[10]

                    # Mean confidence of upper body joints
                    arm_conf = float(np.mean([l_sh[2], r_sh[2], l_wr[2], r_wr[2]]))
                    pose_confidences.append(arm_conf)

                    # Shoulder width
                    sh_width = max(10.0, np.linalg.norm(r_sh[:2] - l_sh[:2]))

                    # Shoulder tilt angle
                    d_sh = r_sh[:2] - l_sh[:2]
                    sh_angle = math.atan2(d_sh[1], d_sh[0])
                    shoulder_angles.append(sh_angle)

                    # Dominant vs Non-dominant wrist displacement relative to shoulder midpoint
                    sh_mid_x = (l_sh[0] + r_sh[0]) / 2.0
                    if handedness == PlayerHandedness.RIGHT_HANDED:
                        if r_wr[2] < 0.25:
                            continue
                        dom_wrist_dx = (r_wr[0] - sh_mid_x) / sh_width
                    else:
                        if l_wr[2] < 0.25:
                            continue
                        dom_wrist_dx = (l_wr[0] - sh_mid_x) / sh_width

                    # Invert for court side and handedness: Positive = Forehand side, Negative = Backhand side
                    normalized_dx = court_side_sign * handedness_sign * dom_wrist_dx
                    wrist_displacements.append(normalized_dx)

                except Exception:
                    continue

        if not wrist_displacements:
            return {"valid": False, "reason": "No valid pose detections in hit window."}

        mean_displacement = float(np.mean(wrist_displacements))
        mean_conf = float(np.mean(pose_confidences)) if pose_confidences else 0.0

        return {
            "valid": True,
            "normalized_wrist_displacement": mean_displacement,
            "mean_pose_confidence": mean_conf,
            "sample_count": len(wrist_displacements),
            "shoulder_angle_mean": float(np.mean(shoulder_angles)) if shoulder_angles else 0.0
        }
