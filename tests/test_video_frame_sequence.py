import cv2
import numpy as np
import pytest

from src.utils.video_frame_sequence import VideoFrameSequence
from src.utils import video_frame_sequence as video_frames
from types import SimpleNamespace


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


def test_actual_sequential_suffix_matches_continuous_decode_after_random_access(tmp_path):
    path = tmp_path / 'suffix.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
    assert writer.isOpened()
    for index in range(17):
        writer.write(np.full((48, 64, 3), index * 11, dtype=np.uint8))
    writer.release()
    with VideoFrameSequence(str(path), cache_size=2) as frames:
        frames[14]
        original_position = frames._next_index
        continuous = [frame.copy() for frame in frames.iter_decoded()]
        suffix = list(frames.iter_decoded(7))
        assert len(suffix) == 10
        assert all(np.array_equal(left, right) for left, right in zip(continuous[7:], suffix))
        assert frames._next_index == original_position and len(frames._cache) <= 2
        assert list(frames.iter_decoded(17)) == []


class Capture:
    def __init__(self, *, fail_at=None, resize_at=None, opened=True, release_error=None):
        self.index, self.reads, self.releases = 0, [], 0
        self.fail_at, self.resize_at, self.opened = fail_at, resize_at, opened
        self.release_error = release_error

    def isOpened(self):
        return self.opened

    def read(self):
        index = self.index
        self.index += 1
        self.reads.append(index)
        if index == self.fail_at:
            return False, None
        size = (3, 4, 3) if index == self.resize_at else (4, 4, 3)
        return True, np.full(size, index, dtype=np.uint8)

    def set(self, *args):
        pytest.fail('Sequential suffix attempted approximate frame seek')

    def release(self):
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error


def sequence_with_capture(monkeypatch, capture):
    frames = object.__new__(VideoFrameSequence)
    frames.metadata = SimpleNamespace(frame_count=8, width=4, height=4)
    frames._path = 'synthetic-decoder-only'
    opened = []
    monkeypatch.setattr(video_frames.cv2, 'VideoCapture', lambda path: opened.append(path) or capture)
    return frames, opened


def test_suffix_discards_prefix_and_interrupt_closes_without_another_read(monkeypatch):
    capture = Capture()
    frames, opened = sequence_with_capture(monkeypatch, capture)
    rows = frames.iter_decoded(3)
    first = next(rows)
    assert first[0, 0, 0] == 3 and capture.reads == [0, 1, 2, 3]
    rows.close()
    assert capture.releases == 1 and capture.reads == [0, 1, 2, 3]
    assert opened == ['synthetic-decoder-only']


@pytest.mark.parametrize('start', [True, False, np.bool_(True), np.bool_(False), 1., None, '3', -1, 9])
def test_invalid_suffix_never_opens_decoder(monkeypatch, start):
    capture = Capture()
    frames, opened = sequence_with_capture(monkeypatch, capture)
    with pytest.raises(ValueError):
        list(frames.iter_decoded(start))
    assert not opened and not capture.reads


@pytest.mark.parametrize('problem,match', [('open', 'open independent'),
                                         ('short', 'frame 2'), ('shape', 'resolution changed')])
def test_prefix_decode_errors_release_and_never_reach_consumer(monkeypatch, problem, match):
    capture = Capture(opened=problem != 'open', fail_at=2 if problem == 'short' else None,
                      resize_at=2 if problem == 'shape' else None)
    frames, _ = sequence_with_capture(monkeypatch, capture)
    rows = frames.iter_decoded(5)
    with pytest.raises((ValueError, RuntimeError), match=match):
        next(rows)
    assert capture.releases == 1


def test_decode_failure_keeps_its_cause_when_release_also_fails(monkeypatch):
    capture = Capture(fail_at=2, release_error=ValueError('injected release failure'))
    frames, _ = sequence_with_capture(monkeypatch, capture)
    with pytest.raises(RuntimeError, match='frame 2') as caught:
        next(frames.iter_decoded(5))
    assert capture.releases == 1
    assert caught.value.__notes__ == ['Sequential decoder cleanup also failed: injected release failure']


def test_iterator_close_surfaces_its_decoder_release_failure(monkeypatch):
    failure = ValueError('injected release failure')
    capture = Capture(release_error=failure)
    frames, _ = sequence_with_capture(monkeypatch, capture)
    rows = frames.iter_decoded()
    next(rows)
    with pytest.raises(ValueError) as caught:
        rows.close()
    assert caught.value is failure and capture.releases == 1
