"""Bounded decoded-frame storage for multipass offline video analysis."""

from collections import OrderedDict
from collections.abc import Sequence
import operator

import cv2
import numpy as np

from src.utils.video_io import get_video_metadata


class VideoFrameSequence(Sequence):
    """Read-only frames with sequential decode and bounded random-access cache.

    Frame indices use the decoder's constant-rate frame ordering. This class
    does not establish presentation timestamps for variable-frame-rate footage.
    Consumers must not modify cached arrays in place.
    """

    def __init__(self, path: str, cache_size: int = 8):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.metadata = get_video_metadata(path)
        if self.metadata.frame_count <= 0:
            raise ValueError("Video must report a positive frame count")
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"Cannot decode video: {path}")
        self._cache = OrderedDict()
        self._path = path
        self._cache_size = cache_size
        self._next_index = 0

    def __len__(self):
        return self.metadata.frame_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        if index != self._next_index:
            if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(f"Decoder could not seek to frame {index}")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Decoder failed at frame {index}; output would be incomplete")
        if frame.shape[:2] != (self.metadata.height, self.metadata.width):
            raise ValueError("Video resolution changed; split into separately calibrated segments")
        self._next_index = index + 1
        self._cache[index] = frame
        self._cache.move_to_end(index)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frame

    def iter_decoded(self, start_index=0):
        """Decode from frame zero, yielding the requested normalized suffix.

        A new capture avoids approximate random seeking. Prefix frames are
        decoded and discarded, never yielded to an inference consumer. This
        preserves decoder ordering on a sealed constant-rate source; it does
        not recover tracker state or establish original variable-rate PTS.
        Closing or interrupting this iterator releases its independent handle.
        """
        if isinstance(start_index, (bool, np.bool_)):
            raise ValueError('Sequential start must be an integer frame boundary')
        try:
            start_index = operator.index(start_index)
        except TypeError as error:
            raise ValueError('Sequential start must be an integer frame boundary') from error
        if not 0 <= start_index <= len(self):
            raise ValueError('Sequential start is outside the normalized video')
        if start_index == len(self):
            return
        capture = cv2.VideoCapture(self._path)
        try:
            if not capture.isOpened():
                raise RuntimeError('Unable to open independent sequential video decoder')
            for index in range(len(self)):
                ret, frame = capture.read()
                if not ret or frame is None:
                    raise RuntimeError(f'Sequential decoder failed at frame {index}; output would be incomplete')
                if frame.shape[:2] != (self.metadata.height, self.metadata.width):
                    raise ValueError('Video resolution changed; split into separately calibrated segments')
                if index >= start_index:
                    yield frame
        except GeneratorExit:
            # close() must surface a release failure to its owning consumer.
            capture.release()
            raise
        except BaseException as error:
            try:
                capture.release()
            except BaseException as cleanup_error:
                error.add_note('Sequential decoder cleanup also failed: ' + str(cleanup_error))
            raise
        else:
            capture.release()

    def close(self):
        self._cache.clear()
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
