import numpy as np
from typing import List
from ultralytics import YOLO
from src.utils.bbox_utils import BBox

class PlayerDetector:
    """Player detector using YOLO model."""
    
    def __init__(self, model_path: str = 'yolo11x.pt', device: str = 'auto'):
        """Initialize the YOLO model for player detection."""
        self.model = YOLO(model_path)
        if device != 'auto':
            self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input image/frame.
            
        Returns:
            List of detected bounding boxes for persons.
        """
        results = self.model(frame, classes=[0], verbose=False)
        bboxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].item()
                bboxes.append(BBox(x1, y1, x2, y2, confidence=conf, class_id=0))
        return bboxes

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[BBox]:
        """
        Detect and track persons in a frame using ByteTrack.
        
        Args:
            frame: Input image/frame.
            persist: Whether to persist tracks between frames.
            
        Returns:
            List of tracked bounding boxes for persons.
        """
        results = self.model.track(frame, classes=[0], persist=persist, tracker="bytetrack.yaml", verbose=False)
        bboxes = []
        for result in results:
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i in range(len(track_ids)):
                    x1, y1, x2, y2 = boxes[i]
                    bboxes.append(BBox(x1, y1, x2, y2, track_id=int(track_ids[i]), confidence=confs[i], class_id=0))
        return bboxes
