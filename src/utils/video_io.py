import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    codec: str

def get_video_metadata(path: str) -> VideoMetadata:
    """Extract metadata without reading all frames. Uses actual FPS."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0  # Fallback just in case
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    try:
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        codec = "Unknown"
        
    duration = float(frame_count) / fps if fps > 0 else 0.0
    
    cap.release()
    return VideoMetadata(
        fps=fps, 
        frame_count=frame_count, 
        width=width, 
        height=height, 
        duration_seconds=duration, 
        codec=codec
    )

def read_video(path: str) -> Tuple[List[np.ndarray], VideoMetadata]:
    """Reads all frames from a video and returns them alongside metadata."""
    metadata = get_video_metadata(path)
    cap = cv2.VideoCapture(path)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    return frames, metadata

def save_video(frames: List[np.ndarray], path: str, fps: float) -> None:
    """Saves frames to a video file using actual FPS and H.264 codec."""
    if not frames:
        return
        
    height, width = frames[0].shape[:2]
    # Use H.264 codec (avc1) for maximum compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
