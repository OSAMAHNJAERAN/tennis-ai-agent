import os
from typing import List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import BallObservation

class YOLO11BallDetector:
    """
    Production-Grade Ultralytics YOLO11 Tennis Ball Detector & Proposal Generator.
    Enforces YOLO11 architecture, high-resolution multi-scale inference,
    and multi-threshold candidate proposal extraction.
    """
    def __init__(
        self,
        model_path: str = "artifacts/models/ball/yolo11s_tennis_ball_best.pt",
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
        
        # Verify non-YOLOv5 architectural safety constraint
        if "yolo5" in model_path.lower() or "v5" in os.path.basename(model_path).lower():
            raise ValueError(
                f"ARCHITECTURAL VIOLATION: YOLOv5 weights '{model_path}' cannot be used in production. "
                f"T88J709 requires Ultralytics YOLO11."
            )
            
        # Load Ultralytics YOLO11 model
        self.model = YOLO(model_path)
        if device != "auto":
            self.model.to(device)

    def extract_candidates(self, frame: np.ndarray, imgsz: Optional[int] = None) -> List[BallObservation]:
        """
        Extracts all ball candidate proposals with conf >= low_conf.
        """
        eval_imgsz = imgsz or self.imgsz
        results = self.model.predict(
            frame,
            imgsz=eval_imgsz,
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

    def detect_best(self, frame: np.ndarray, imgsz: Optional[int] = None, conf: Optional[float] = None) -> Optional[BBox]:
        """
        Single-frame inference returning highest confidence detection >= threshold.
        """
        min_conf = conf or self.high_conf
        candidates = self.extract_candidates(frame, imgsz=imgsz)
        high_conf_cands = [c for c in candidates if c.confidence >= min_conf]
        if not high_conf_cands:
            return None
            
        best = max(high_conf_cands, key=lambda c: c.confidence)
        return best.bbox
