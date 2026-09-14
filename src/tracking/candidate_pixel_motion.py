"""Causal local pixel-change evidence, not a ball classifier or camera tracker."""

from collections import deque
import math

import cv2
import numpy as np


class CandidatePixelMotion:
    def __init__(self, lag_seconds=.1, reference_size=(960, 540), patch_radius=3):
        if (not math.isfinite(lag_seconds) or lag_seconds <= 0
                or not math.isfinite(patch_radius) or int(patch_radius) != patch_radius or patch_radius < 1
                or any(int(value) != value or value <= 0 for value in reference_size)):
            raise ValueError('Invalid pixel-motion settings')
        self.lag_seconds = lag_seconds
        self.reference_size = tuple(map(int, reference_size))
        self.patch_radius = int(patch_radius)
        self.history = deque()
        self.previous_timestamp = None
        self.source_shape = None

    def measure(self, frame, candidates, timestamp):
        if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
                or not math.isfinite(timestamp)
                or (self.previous_timestamp is not None and timestamp <= self.previous_timestamp)):
            raise ValueError('Require uint8 BGR frames and increasing finite timestamps')
        if self.source_shape is not None and frame.shape != self.source_shape:
            raise ValueError('Geometry change requires a new segment')
        self.source_shape = frame.shape
        if self.previous_timestamp is not None and timestamp - self.previous_timestamp > max(.25, 2 * self.lag_seconds):
            self.history.clear()
        gray = cv2.cvtColor(cv2.resize(frame, self.reference_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        target_time = timestamp - self.lag_seconds
        while len(self.history) > 1 and self.history[1][0] <= target_time + 1e-9:
            self.history.popleft()
        reference = self.history[0][1] if self.history and self.history[0][0] <= target_time + 1e-9 else None
        scores = []
        difference = cv2.absdiff(gray, reference) if reference is not None else None
        height, width = frame.shape[:2]
        for candidate in candidates:
            if (not math.isfinite(candidate.x_px) or not math.isfinite(candidate.y_px)
                    or not 0 <= candidate.x_px < width or not 0 <= candidate.y_px < height):
                raise ValueError('Candidate must be a finite point inside the source frame')
            if difference is None:
                scores.append(None)
                continue
            x = min(self.reference_size[0] - 1, round(candidate.x_px * self.reference_size[0] / width))
            y = min(self.reference_size[1] - 1, round(candidate.y_px * self.reference_size[1] / height))
            radius = self.patch_radius
            patch = difference[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1].ravel()
            count = min(3, len(patch))
            scores.append(float(np.partition(patch, len(patch) - count)[-count:].mean()))
        self.history.append((timestamp, gray))
        self.previous_timestamp = timestamp
        return scores

    def filter(self, frame, candidates, timestamp, minimum_score=12.):
        if not math.isfinite(minimum_score) or not 0 <= minimum_score <= 255:
            raise ValueError('Pixel-change threshold must lie in [0,255]')
        scores = self.measure(frame, candidates, timestamp)
        kept, evidence = [], []
        for candidate, score in zip(candidates, scores):
            accepted = score is None or score >= minimum_score
            if accepted:
                kept.append(candidate)
            evidence.append({'x_px': candidate.x_px, 'y_px': candidate.y_px,
                             'confidence': candidate.confidence, 'pixel_motion_score': score,
                             'history_available': score is not None, 'accepted': accepted})
        return kept, evidence
