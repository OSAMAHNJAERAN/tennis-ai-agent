import pytest
import os
import cv2
import numpy as np
from src.utils.video_io import read_video, save_video, get_video_metadata

def test_video_metadata_extraction(tmp_path):
    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (192, 108))
    for _ in range(10):
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    meta = get_video_metadata(video_path)
    assert np.isclose(meta.fps, 30.0, atol=1.0)
    assert meta.width == 192
    assert meta.height == 108
    assert meta.frame_count == 10

def test_fps_not_hardcoded(tmp_path):
    video_path = str(tmp_path / "synthetic25.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 25.0, (192, 108))
    for _ in range(5):
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    meta = get_video_metadata(video_path)
    assert np.isclose(meta.fps, 25.0, atol=1.0)
    assert meta.fps != 24.0

def test_save_and_read_video(tmp_path):
    out_path = str(tmp_path / "test_out.mp4")
    frames_in = [np.zeros((108, 192, 3), dtype=np.uint8) for _ in range(15)]
    save_video(frames_in, out_path, fps=30.0)
    
    frames_out, meta = read_video(out_path)
    assert len(frames_out) == 15
    assert meta.frame_count == 15


@pytest.mark.parametrize('reported_fps', [0., float('nan'), float('inf')])
def test_unknown_frame_rate_is_not_replaced_with_assumed_timing(monkeypatch, reported_fps):
    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, field):
            return reported_fps if field == cv2.CAP_PROP_FPS else 10

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: capture)
    with pytest.raises(ValueError, match='frame rate'):
        get_video_metadata('unknown.mp4')
    assert capture.released
