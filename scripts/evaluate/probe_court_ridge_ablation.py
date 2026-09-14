"""Image-supported line proposals, without landmark identity or court claims."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np


def line_evidence(image, segment, mode="baseline"):
    """Contrast on both sides of a bright ridge; support fractions stay explicit."""
    a, b = np.asarray(segment, dtype=float).reshape(2, 2)
    length = np.linalg.norm(b - a)
    if length < 30:
        return None
    tangent = (b - a) / length
    normal = np.array([-tangent[1], tangent[0]])
    # LSD gives edges: sample both candidate ridge centers3px away.
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    base = a + np.linspace(.05, .95, 80)[:, None] * (b - a)
    for shift in [-3, -1.5, 0, 1.5, 3]:
        centers = base + normal * shift
        locations = centers[:, None, :] + normal * np.array([-8, -6, -1, 0, 1, 6, 8])[None, :, None]
        valid = ((locations[:, :, 0] >= 0) & (locations[:, :, 0] < image.shape[1]-1) &
                 (locations[:, :, 1] >= 0) & (locations[:, :, 1] < image.shape[0]-1)).all(axis=1)
        if valid.sum() < 40:
            continue
        locations = locations[valid].astype(np.float32)
        values = cv2.remap(gray, locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        saturation = cv2.remap(hsv[:, :, 1], locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        ridge = values[:, 2:5].max(axis=1)
        sides = np.maximum(values[:, :2].mean(axis=1), values[:, 5:].mean(axis=1))
        side_difference = np.abs(values[:, :2].mean(axis=1) - values[:, 5:].mean(axis=1))
        bright = (ridge - sides > 10) & (ridge > 100)
        white = saturation[:, 2:5].min(axis=1) < 110
        consistent_sides = side_difference < 45
        if mode == 'baseline':
            support = bright & white & consistent_sides
        elif mode == 'no_white':
            support = bright & consistent_sides
        elif mode == 'no_sides':
            support = bright & white
        elif mode == 'ridge_only':
            support = bright
        else:
            raise ValueError('Unknown photometric ablation')
        results.append({'support_fraction': float(support.mean()), 'bright_fraction': float(bright.mean()),
                        'white_fraction': float(white.mean()), 'similar_sides_fraction': float(consistent_sides.mean()),
                        'length': float(length), 'center_shift': shift,
                        'segment': np.r_[a + normal * shift, b + normal * shift].tolist()})
    return max(results, key=lambda row: row['support_fraction']) if results else None


def propose(image, mode="baseline"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    rows = []
    if lines is not None:
        for line in lines[:, 0]:
            row = line_evidence(image, line, mode)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: row['length'] * row['support_fraction'], reverse=True)
    return rows

