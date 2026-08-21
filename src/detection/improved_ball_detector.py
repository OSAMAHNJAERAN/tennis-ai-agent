from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import BallObservation

class ImprovedBallDetector:
    """
    Advanced Single-Frame and Candidate Proposal Ball Detector.
    Supports high-resolution inference, confidence tuning, and multi-candidate extraction.
    """
    def __init__(
        self,
        model_path: str = "models/yolo5_last.pt",
        imgsz: int = 1024,
        high_conf: float = 0.20,
        low_conf: float = 0.02,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.high_conf = high_conf
        self.low_conf = low_conf
        self.device = device
        
        # Load YOLO model
        self.model = YOLO(model_path)
        if device != "auto":
            self.model.to(device)

    def extract_candidates(self, frame: np.ndarray) -> List[BallObservation]:
        """
        Extracts all ball candidate proposals with conf >= low_conf.
        """
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.low_conf,
            verbose=False
        )[0]
        
        candidates = []
        if len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            xyxys = results.boxes.xyxy.cpu().numpy()
            
            for conf, xyxy in zip(confs, xyxys):
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=float(conf), class_id=0)
                candidates.append(BallObservation(
                    x_px=cx,
                    y_px=cy,
                    confidence=float(conf),
                    bbox=bbox
                ))
                
        return candidates

    def detect_best(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Standard single-frame detection returning best box >= high_conf.
        """
        candidates = self.extract_candidates(frame)
        high_conf_cands = [c for c in candidates if c.confidence >= self.high_conf]
        if not high_conf_cands:
            return None
            
        best = max(high_conf_cands, key=lambda c: c.confidence)
        return best.bbox
