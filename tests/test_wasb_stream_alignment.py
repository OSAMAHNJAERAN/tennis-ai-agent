import numpy as np
import pytest

from src.detection.wasb_ball_detector import WASBBallDetector


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 7])
def test_overlapping_windows_preserve_frame_alignment_and_boundaries(length):
    model = WASBBallDetector.__new__(WASBBallDetector)

    def heatmaps(frames):
        return np.array([np.full((2, 2), frame) for frame in frames]), (2, 2)

    model.predict_heatmaps = heatmaps
    model.decode_heatmaps = lambda maps, shape: [float(image.mean()) for image in maps]
    assert list(model.predict_stream(iter(range(length)))) == list(range(length))
