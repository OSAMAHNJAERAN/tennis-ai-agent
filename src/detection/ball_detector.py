import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from src.utils.bbox_utils import BBox

class BallDetector:
    """Ball detector using YOLO model."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.15, device: str = 'auto'):
        """Initialize the YOLO model for ball detection."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        if device != 'auto':
            self.model.to(device)

    def detect(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Detect the most likely ball in a frame.
        
        Args:
            frame: Input image/frame.
            
        Returns:
            BBox of the best ball detection, or None if none found.
        """
        detections = self.detect_all(frame)
        if not detections:
            return None
        return max(detections, key=lambda b: b.confidence)

    def detect_all(self, frame: np.ndarray) -> List[BBox]:
        """
        Detect all ball candidates in a frame.
        
        Args:
            frame: Input image/frame.
            
        Returns:
            List of all ball bounding boxes above confidence threshold.
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                bboxes.append(BBox(x1, y1, x2, y2, confidence=conf, class_id=cls))
        return bboxes
