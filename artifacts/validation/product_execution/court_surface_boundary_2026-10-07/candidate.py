"""Frozen, research-only surface-boundary evidence; no runtime court admission.

The original passing LSD record has priority. A previously rejected record may
gain evidence only at a consistently oriented boundary of the actual connected
surface mask. Constants implement this directory's pre-execution protocol.
"""
from __future__ import annotations

import cv2
import numpy as np

from scripts.evaluate.probe_court_line_segments import line_evidence as baseline_evidence
from scripts.evaluate.probe_court_surface_support import surface_candidates


def _image(image):
    if (not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or min(image.shape[:2]) < 2):
        raise ValueError("Require a nonempty uint8 BGR image")


def _mask(image, mask):
    if mask is None:
        return None
    if (not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]
            or mask.dtype.kind not in "buif" or not np.isfinite(mask).all()
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError("Require an aligned, finite binary connected-surface mask")
    return mask.astype(np.uint8, copy=False)


def select_surface(image):
    """Return the unchanged first eligible component and raw-order hull quad.

    Selection sees source pixels alone. The component's connected mask is
    retained; a quadrilateral is never rasterized as a substitute surface.
    """
    _image(image)
    for component in surface_candidates(image):
        hull = np.asarray(component["hull"], dtype=np.float32).reshape(-1, 1, 2)
        approximation = cv2.approxPolyDP(hull, .01 * cv2.arcLength(hull, True), True)
        if len(approximation) == 4:
            return {"component": component, "quadrilateral": approximation[:, 0].astype(float)}
    return None


def _conditional_evidence(image, segment, mask, gray, hsv):
    a, b = np.asarray(segment, dtype=float).reshape(2, 2)
    length = np.linalg.norm(b - a)
    normal = np.array([-(b-a)[1], (b-a)[0]]) / length
    base = a + np.linspace(.05, .95, 80)[:, None] * (b - a)
    results = []
    for shift in [-3, -1.5, 0, 1.5, 3]:
        centers = base + normal * shift
        locations = centers[:, None, :] + normal * np.array([-8, -6, -1, 0, 1, 6, 8])[None, :, None]
        valid = ((locations[:, :, 0] >= 0) & (locations[:, :, 0] < image.shape[1]-1)
                 & (locations[:, :, 1] >= 0) & (locations[:, :, 1] < image.shape[0]-1)).all(axis=1)
        if valid.sum() < 40:
            continue
        locations = locations[valid].astype(np.float32)
        values = cv2.remap(gray, locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        saturation = cv2.remap(hsv[:, :, 1], locations[:, :, 0], locations[:, :, 1], cv2.INTER_LINEAR)
        occupancy = cv2.remap(mask, locations[:, [0, 1, 5, 6], 0],
                             locations[:, [0, 1, 5, 6], 1], cv2.INTER_NEAREST)
        negative_inside = (occupancy[:, :2] == 1).all(axis=1) & (occupancy[:, 2:] == 0).all(axis=1)
        positive_inside = (occupancy[:, 2:] == 1).all(axis=1) & (occupancy[:, :2] == 0).all(axis=1)
        negative_fraction = float(negative_inside.mean())
        positive_fraction = float(positive_inside.mean())
        if negative_fraction >= .8:
            boundary, side = negative_inside, "NEGATIVE_NORMAL"
        elif positive_fraction >= .8:
            boundary, side = positive_inside, "POSITIVE_NORMAL"
        else:
            boundary, side = np.zeros(len(values), dtype=bool), None
        ridge = values[:, 2:5].max(axis=1)
        sides = np.maximum(values[:, :2].mean(axis=1), values[:, 5:].mean(axis=1))
        side_difference = np.abs(values[:, :2].mean(axis=1) - values[:, 5:].mean(axis=1))
        bright = (ridge - sides > 10) & (ridge > 100)
        white = saturation[:, 2:5].min(axis=1) < 110
        consistent_sides = side_difference < 45
        support = bright & white & (consistent_sides | boundary)
        results.append({
            "support_fraction": float(support.mean()),
            "bright_fraction": float(bright.mean()),
            "white_fraction": float(white.mean()),
            "similar_sides_fraction": float(consistent_sides.mean()),
            "length": float(length), "center_shift": shift,
            "segment": np.r_[a + normal * shift, b + normal * shift].tolist(),
            "boundary_negative_inside_fraction": negative_fraction,
            "boundary_positive_inside_fraction": positive_fraction,
            "boundary_pattern_fraction": max(negative_fraction, positive_fraction),
            "boundary_inside_side": side,
            "boundary_gate_pass": side is not None,
            "boundary_replacement_sample_count": int((bright & white & boundary & ~consistent_sides).sum()),
            "valid_sample_count": int(valid.sum()),
        })
    return max(results, key=lambda row: row["support_fraction"]) if results else None


def _evidence(image, segment, mask, gray=None, hsv=None):
    points = np.asarray(segment, dtype=float)
    if points.size != 4 or not np.isfinite(points).all():
        raise ValueError("Require two finite segment endpoints")
    points = points.reshape(2, 2)
    original = baseline_evidence(image, points)
    if original is None:
        return None
    baseline_record = dict(original)
    if original["support_fraction"] >= .5:
        return {**original, "evidence_origin": "BASELINE", "baseline_record": baseline_record}
    if mask is not None:
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        recovered = _conditional_evidence(image, points, mask, gray, hsv)
        if recovered is not None and recovered["support_fraction"] >= .5:
            return {**recovered, "evidence_origin": "SURFACE_BOUNDARY_RECOVERY",
                    "baseline_record": baseline_record}
    return {**original, "evidence_origin": "BASELINE_REJECTED", "baseline_record": baseline_record}


def line_evidence(image, segment, mask=None):
    """Expose the selected evidence, including an unchanged rejected baseline."""
    _image(image)
    return _evidence(image, segment, _mask(image, mask))


def propose(image, mask=None):
    """Retain exact baseline passes plus conditional recovery; stable LSD ties."""
    _image(image)
    mask = _mask(image, mask)
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray_u8)[0]
    gray = gray_u8.astype(np.float32) if mask is not None else None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32) if mask is not None else None
    rows = []
    if lines is not None:
        for index, segment in enumerate(lines[:, 0]):
            row = _evidence(image, segment, mask, gray, hsv)
            if row is not None and row["support_fraction"] >= .5:
                rows.append({**row, "lsd_index": index})
    rows.sort(key=lambda row: row["length"] * row["support_fraction"], reverse=True)
    return rows
