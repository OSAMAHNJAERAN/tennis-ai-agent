"""Frozen-encoder research features and a head scoring real candidates only."""
import math
import cv2
import numpy as np
import torch
from torch import nn

ENCODER_SHA256 = '11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
APPEARANCE_DIM = 6144


def context_patch(rgb, point, size):
    if rgb.shape != (540, 960, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected reference RGB uint8')
    if len(size) != 2 or any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Invalid geometry')
    if len(point) != 2 or any(not math.isfinite(v) or not 0 <= v < limit for v, limit in zip(point, size)):
        raise ValueError('Invalid observed point')
    x, y = round(point[0]*960/size[0]), round(point[1]*540/size[1])
    padded = cv2.copyMakeBorder(rgb, 32, 32, 32, 32, cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(padded[y:y+64, x:x+64].transpose(2, 0, 1))


class FeatureBallVerifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalization = nn.LayerNorm(APPEARANCE_DIM)
        self.head = nn.Sequential(nn.Linear(APPEARANCE_DIM+2, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 1))

    def forward(self, appearance, metadata):
        return self.head(torch.cat([self.normalization(appearance), metadata], dim=1)).squeeze(1)
