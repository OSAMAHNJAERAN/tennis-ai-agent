"""Experimental learned verification of observed WASB candidates.

Consumes local current/past RGB appearance plus detector confidence and rank.
It never invents candidate positions, identities, contacts or physical speeds.
"""

import math

import cv2
import numpy as np
import torch
from torch import nn


def labeled_frame_candidates(detector, frames, index):
    """Exactly the real overlapping windows contributing to one labeled frame."""
    if not 0 <= index < len(frames):
        raise ValueError('Labeled frame is out of range')
    if len(frames) < 3:
        return detector.predict_triplet(frames[:])[index]
    heatmaps, shape = [], None
    for start in range(max(0, index - 2), min(index, len(frames) - 3) + 1):
        predicted, shape = detector.predict_heatmaps(frames[start:start + 3])
        heatmaps.append(predicted[index - start])
    return detector.decode_heatmaps([np.mean(heatmaps, axis=0)], shape)[0]


def reference_rgb(frame):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR image')
    return cv2.cvtColor(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def candidate_patch(current_rgb, past_rgb, candidate, frame_size, side=32):
    width, height = frame_size
    if current_rgb.shape != (540, 960, 3) or past_rgb.shape != current_rgb.shape:
        raise ValueError('Verifier requires aligned 960x540 RGB references')
    if (width <= 0 or height <= 0 or side != 32
            or not all(math.isfinite(v) for v in (candidate.x_px, candidate.y_px, candidate.confidence))
            or not 0 <= candidate.x_px < width or not 0 <= candidate.y_px < height):
        raise ValueError('Invalid candidate or frame dimensions')
    x = round(candidate.x_px * 960 / width)
    y = round(candidate.y_px * 540 / height)
    patches = []
    for frame in (current_rgb, past_rgb):
        padded = cv2.copyMakeBorder(frame, side // 2, side // 2, side // 2, side // 2, cv2.BORDER_REFLECT_101)
        patch = padded[y:y + side, x:x + side]
        patches.append(patch.transpose(2, 0, 1))
    return np.concatenate(patches, axis=0)


def candidate_features(candidate, rank):
    if rank < 0 or not math.isfinite(candidate.confidence):
        raise ValueError('Invalid candidate rank/confidence')
    return np.array([candidate.confidence, 1. / (rank + 1)], dtype=np.float32)


class BallCandidateVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 * 4 * 4 + 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, patches, features):
        return self.head(torch.cat((self.appearance(patches), features), dim=1)).squeeze(1)
