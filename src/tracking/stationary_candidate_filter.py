"""Experimental causal rejection of persistent image-stationary ball candidates.

This is a distractor hypothesis, not object recognition. It can reject a real
stationary ball and therefore requires explicit recall/absence benchmarking.
Coordinates and radii are normalized to a 512 x 288 reference image.
"""

from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class _Spot:
    x: float
    y: float
    hits: deque = field(default_factory=deque)


class StationaryCandidateFilter:
    def __init__(self, radius_reference_px=1.5, minimum_seconds=.5, window_seconds=2.,
                 minimum_presence=.35):
        if (not all(math.isfinite(v) for v in (radius_reference_px, minimum_seconds, window_seconds, minimum_presence))
                or radius_reference_px <= 0 or minimum_seconds <= 0 or window_seconds < minimum_seconds
                or not 0 < minimum_presence <= 1):
            raise ValueError('Invalid stationary candidate filter settings')
        self.radius = radius_reference_px
        self.minimum_seconds = minimum_seconds
        self.window_seconds = window_seconds
        self.minimum_presence = minimum_presence
        self.spots = []
        self.previous_timestamp = None

    def filter(self, candidates, timestamp, fps, frame_size):
        width, height = frame_size
        if (not math.isfinite(timestamp) or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0
                or (self.previous_timestamp is not None and timestamp <= self.previous_timestamp)):
            raise ValueError('Require valid image size, FPS and increasing timestamps')
        if self.previous_timestamp is not None and timestamp - self.previous_timestamp > self.window_seconds:
            self.spots.clear()
        self.previous_timestamp = timestamp
        for spot in self.spots:
            while spot.hits and timestamp - spot.hits[0] > self.window_seconds:
                spot.hits.popleft()
        self.spots = [spot for spot in self.spots if spot.hits]
        kept, rejected, assigned = [], [], set()
        for candidate in candidates:
            x, y = candidate.x_px * 512 / width, candidate.y_px * 288 / height
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            options = [(math.hypot(x - spot.x, y - spot.y), index) for index, spot in enumerate(self.spots)
                       if index not in assigned]
            distance, match = min(options, default=(math.inf, -1))
            if distance > self.radius:
                match = len(self.spots)
                self.spots.append(_Spot(x, y))
            spot = self.spots[match]
            assigned.add(match)
            spot.hits.append(timestamp)
            span = timestamp - spot.hits[0]
            # Require both persistence and repeated observations. No filled hits.
            stationary = (span >= self.minimum_seconds
                          and len(spot.hits) >= max(3, self.minimum_seconds * fps)
                          and len(spot.hits) / (span * fps + 1) >= self.minimum_presence)
            (rejected if stationary else kept).append(candidate)
        return kept, rejected
