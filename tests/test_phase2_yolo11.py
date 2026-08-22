import pytest
import numpy as np
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.temporal_ball_tracker import (
    TemporalBallTracker,
    KinematicFilter,
    BallState,
    BallObservation
)

def test_yolov5_weights_rejected_in_production():
    """Verify architectural safeguard: YOLOv5 cannot be loaded as production detector."""
    with pytest.raises(ValueError, match="ARCHITECTURAL VIOLATION"):
        YOLO11BallDetector(model_path="models/yolo5_last.pt")

def test_yolo11_detector_initialization():
    """Verify YOLO11 detector initialization."""
    detector = YOLO11BallDetector(
        model_path="yolo11n.pt",
        imgsz=640,
        high_conf=0.20,
        low_conf=0.02
    )
    assert detector.model is not None
    assert detector.imgsz == 640

def test_yolo11_extract_candidates_dummy_frame():
    """Verify candidate extraction structure."""
    detector = YOLO11BallDetector(model_path="yolo11n.pt", imgsz=640, low_conf=0.01)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    candidates = detector.extract_candidates(dummy_frame)
    assert isinstance(candidates, list)
    for c in candidates:
        assert isinstance(c, BallObservation)
        assert 0.0 <= c.confidence <= 1.0

def test_yolo11_temporal_integration():
    """Verify temporal Kalman tracker works seamlessly with candidate observations."""
    tracker = TemporalBallTracker(high_conf_thresh=0.15, low_conf_thresh=0.02, max_prediction_gap=4)
    
    # 5 frames sequence
    candidates = [
        [BallObservation(x_px=100.0, y_px=200.0, confidence=0.85)],
        [BallObservation(x_px=110.0, y_px=205.0, confidence=0.05)],  # tracked
        [],                                                           # predicted
        [BallObservation(x_px=130.0, y_px=215.0, confidence=0.90)],  # detected
        []
    ]
    
    traj = tracker.track_video_candidates(candidates, fps=30.0)
    assert len(traj) == 5
    assert traj[0].state == BallState.DETECTED
    assert traj[1].state == BallState.TRACKED
    assert traj[2].state == BallState.PREDICTED
    assert traj[3].state == BallState.DETECTED
