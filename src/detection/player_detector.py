import numpy as np
from typing import List
import torch
from ultralytics import YOLO
from src.utils.bbox_utils import BBox

class PlayerDetector:
    """Player detector using YOLO model."""
    
    def __init__(self, model_path: str = 'yolo11x.pt', device: str = 'auto',
                 confidence: float = 0.25, tracker: str = 'bytetrack.yaml'):
        """Initialize the YOLO model for player detection."""
        if not 0 <= confidence <= 1:
            raise ValueError('Player detection confidence must be in [0, 1]')
        if not isinstance(tracker, str) or not tracker:
            raise ValueError('A player tracker configuration is required')
        self.confidence = confidence
        self.tracker = tracker
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
        with torch.no_grad():
            results = self.model(frame, classes=[0], conf=self.confidence, verbose=False)
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
            Actual person boxes with nullable tracker IDs. An absent tracker ID
            does not erase the detection or establish a person's identity.
        """
        with torch.no_grad():
            results = self.model.track(frame, classes=[0], conf=self.confidence, persist=persist,
                                       tracker=self.tracker, verbose=False)
        bboxes = []
        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                identity = track_ids[i] if track_ids is not None and i < len(track_ids) else None
                track_id = (int(identity) if identity is not None and np.isfinite(identity)
                            and identity >= 0 and int(identity) == identity else None)
                bboxes.append(BBox(x1, y1, x2, y2, track_id=track_id,
                                   confidence=confs[i], class_id=0))
        return bboxes
