"""Frozen offline rejection of brief detours between actual moving observations."""
import math
import numpy as np


def reject_detours(points, fps, size):
    """Return a simultaneous rejection mask and evidence, never new coordinates."""
    if not math.isfinite(fps) or fps <= 0 or len(size) != 2:
        raise ValueError('Invalid frame geometry or rate')
    if any(not math.isfinite(v) or v <= 0 for v in size):
        raise ValueError('Invalid frame geometry')
    scale = np.array([512 / size[0], 288 / size[1]])
    values = np.full((len(points), 2), np.nan)
    for index, point in enumerate(points):
        if point is not None:
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError('Non-finite selected point')
            values[index] = np.asarray(point) * scale
    rejected, intervals = set(), []
    maximum_frames = math.floor(.10 * fps + 1e-9)
    for start in range(2, len(points) - 2):
        for length in range(1, maximum_frames + 1):
            end = start + length
            if end + 1 >= len(points):
                break
            indices = np.array([start - 2, start - 1, end, end + 1])
            anchors = values[indices]
            if not np.isfinite(anchors).all():
                continue
            if min(np.linalg.norm(anchors[1] - anchors[0]),
                   np.linalg.norm(anchors[3] - anchors[2])) < 2:
                continue
            times = (indices - start) / fps
            design = np.column_stack([times, np.ones(4)])
            fit = np.linalg.lstsq(design, anchors, rcond=None)[0]
            error = np.linalg.norm(design @ fit - anchors, axis=1)
            if error.max() > 4:
                continue
            interior_indices = np.arange(start, end)
            available = np.isfinite(values[start:end]).all(axis=1)
            interior_indices = interior_indices[available]
            if not len(interior_indices):
                continue
            interior_design = np.column_stack([(interior_indices - start) / fps,
                                               np.ones(len(interior_indices))])
            residuals = np.linalg.norm(values[interior_indices] - interior_design @ fit, axis=1)
            if residuals.min() < 20:
                continue
            rejected.update(map(int, interior_indices))
            intervals.append({'start': start, 'end_exclusive': end,
                              'anchor_frames': indices.tolist(),
                              'maximum_anchor_error_reference_px': float(error.max()),
                              'minimum_detour_error_reference_px': float(residuals.min()),
                              'rejected_frames': interior_indices.tolist()})
    return rejected, intervals
