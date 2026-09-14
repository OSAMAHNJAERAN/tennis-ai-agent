import cv2
import numpy as np
import pytest

from src.utils.video_frame_sequence import VideoFrameSequence


def test_random_access_iteration_and_bounded_cache(tmp_path):
    path = tmp_path / "numbered.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    assert writer.isOpened()
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
    writer.release()
    with VideoFrameSequence(str(path), cache_size=3) as frames:
        assert len(frames) == 30
        assert frames.metadata.fps == pytest.approx(25)
        for index in [22, 0, 15, 14, 29, 1, 1]:
            assert frames[index].mean() == pytest.approx(index * 7, abs=6)
        assert len(frames._cache) <= 3
        for index, frame in enumerate(frames):
            assert frame.mean() == pytest.approx(index * 7, abs=6)
        assert frames[-1].mean() == pytest.approx(29 * 7, abs=6)
        assert len(frames[1:4]) == 3
        assert len(frames._cache) <= 3
        with pytest.raises(IndexError):
            frames[30]
    assert not frames._cap.isOpened()
    assert not frames._cache


def test_invalid_cache_size_rejected_without_opening_video():
    with pytest.raises(ValueError):
        VideoFrameSequence("missing.mp4", cache_size=0)
