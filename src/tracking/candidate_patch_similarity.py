"""Photometric patch persistence evidence; not ball identity or physical motion."""

import math

import numpy as np


def normalized_patch_similarity(current, previous):
    """Zero-mean correlation, invariant to uniform brightness and positive gain.

    Flat patches provide no texture evidence and deliberately return unknown.
    This score can also be high for a genuinely stationary or faint ball.
    """
    if current.shape != previous.shape or current.ndim != 2 or current.size < 9:
        raise ValueError('Require equal two-dimensional patches with at least nine pixels')
    first, second = current.astype(np.float64), previous.astype(np.float64)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('Nonfinite image patch')
    first -= first.mean()
    second -= second.mean()
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm <= 1e-9:
        return None
    return max(-1., min(1., float(np.sum(first * second)) / norm))


def shifted_patch_similarity(current, previous, point, radius=5, shift=2):
    """Best actual previous-patch match within a small translation neighborhood.

    The current point is never moved. Shift search only allows comparison with
    nearby texture in the previous image; it does not output a tracked position.
    """
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError('Reference images must have equal grayscale geometry')
    if not all(math.isfinite(value) for value in point) or radius < 1 or shift < 0:
        raise ValueError('Invalid point or patch bounds')
    x, y = map(round, point)
    height, width = current.shape
    margin = radius + shift
    if not margin <= x < width - margin or not margin <= y < height - margin:
        return None
    patch = current[y-radius:y+radius+1, x-radius:x+radius+1]
    scores = []
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            other = previous[y+dy-radius:y+dy+radius+1, x+dx-radius:x+dx+radius+1]
            value = normalized_patch_similarity(patch, other)
            if value is not None:
                scores.append(value)
    return max(scores) if scores else None


def aligned_local_residual(current, previous, point, radius=5, shift=2, center_radius=3):
    """Central change after translation alignment and outer-ring photometric fit.

    Excluding the center from gain/offset fitting protects tiny local changes.
    No supported fit means unknown. This does not classify the changed object.
    """
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError('Require equal grayscale images')
    if (len(point) != 2 or not all(math.isfinite(v) for v in point)
            or any(not isinstance(v, int) for v in (radius, shift, center_radius))
            or not 0 <= center_radius < radius or shift < 0):
        raise ValueError('Invalid patch bounds or point')
    x, y = map(round, point)
    height, width = current.shape
    margin = radius + shift
    if not margin <= x < width-margin or not margin <= y < height-margin:
        return None
    patch = current[y-radius:y+radius+1, x-radius:x+radius+1].astype(float)
    best_score, best = -math.inf, None
    for dy in range(-shift,shift+1):
        for dx in range(-shift,shift+1):
            other = previous[y+dy-radius:y+dy+radius+1, x+dx-radius:x+dx+radius+1].astype(float)
            score = normalized_patch_similarity(patch,other)
            if score is not None and score > best_score:
                best_score, best = score, other
    if best is None:
        return None
    lo, hi = radius-center_radius, radius+center_radius+1
    mask = np.ones(patch.shape,dtype=bool)
    mask[lo:hi,lo:hi] = False
    source, target = best[mask], patch[mask]
    centered = source-source.mean()
    energy = float(centered @ centered)
    if energy <= 1e-9:
        return None
    gain = float(centered @ (target-target.mean()) / energy)
    if not .5 <= gain <= 2.:
        return None
    offset = float(target.mean()-gain*source.mean())
    residual = np.abs(patch[lo:hi,lo:hi]-(gain*best[lo:hi,lo:hi]+offset)).ravel()
    count = min(3,len(residual))
    return float(np.partition(residual,len(residual)-count)[-count:].mean())
