import pytest
import numpy as np
from src.court.court_geometry import TennisCourtGeometry

@pytest.fixture
def sample_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # add some white rectangles
    frame[100:200, 100:200] = 255
    return frame

@pytest.fixture
def court_geometry():
    return TennisCourtGeometry()

@pytest.fixture
def sample_keypoints():
    # 14x2 keypoints
    return np.array([
        [0.0, 0.0], [10.97, 0.0],
        [0.0, 23.77], [10.97, 23.77],
        [1.37, 0.0], [9.6, 0.0],
        [1.37, 23.77], [9.6, 23.77],
        [1.37, 5.485], [9.6, 5.485],
        [1.37, 18.285], [9.6, 18.285],
        [5.485, 5.485], [5.485, 18.285]
    ], dtype=np.float32)
