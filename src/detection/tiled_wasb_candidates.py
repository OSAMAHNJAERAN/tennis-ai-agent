"""Experimental higher-detail temporal proposals from overlapping image crops."""

import math
from collections import deque

import numpy as np

from src.tracking.temporal_ball_tracker import BallObservation


def predict_tiled_stream(detector, frames, frame_size, fraction=.6):
    """Full-frame plus four cropped streams, synchronized with bounded buffers.

    Each view uses the detector's unchanged real-window temporal averaging.
    The five generators advance together, sharing one source decode pass.
    """
    width, height = frame_size
    boxes = [(0, 0, width, height), *overlapping_tiles(width, height, fraction)]
    source = iter(frames)
    buffers = [deque() for _ in boxes]

    def view(index):
        left, top, right, bottom = boxes[index]
        while True:
            if not buffers[index]:
                try:
                    frame = next(source)
                except StopIteration:
                    return
                if frame.shape[:2] != (height, width):
                    raise ValueError('Source frame geometry changed')
                for buffer in buffers:
                    buffer.append(frame)
            yield buffers[index].popleft()[top:bottom, left:right]

    streams = [detector.predict_stream(view(index)) for index in range(len(boxes))]
    for groups in zip(*streams, strict=True):
        candidates = []
        for (left, top, _, _), group in zip(boxes, groups):
            candidates.extend(BallObservation(item.x_px + left, item.y_px + top, item.confidence)
                              for item in group)
        yield merge_by_confidence(candidates, frame_size)


def overlapping_tiles(width, height, fraction=.6):
    if (not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 2
            or not math.isfinite(fraction) or not .5 <= fraction < 1):
        raise ValueError('Require valid dimensions and crop fraction in [.5,1)')
    tw, th = max(1, round(width * fraction)), max(1, round(height * fraction))
    return [(x, y, x + tw, y + th) for y in (0, height - th) for x in (0, width - tw)]


def sparse_tile_candidates(detector, frames, frame_index, fraction=.6):
    """Predict each tile with exactly the real windows covering this frame.

    This is sparse evaluation inference. It does not use label coordinates,
    retain temporal tracker state, or establish full-video inference throughput.
    """
    if not 0 <= frame_index < len(frames):
        raise ValueError('Requested frame is outside the source sequence')
    height, width = frames[frame_index].shape[:2]
    tiles = overlapping_tiles(width, height, fraction)
    # Load the at-most-five source frames once, avoiding repeated decoder seeks.
    starts = list(range(max(0, frame_index - 2), min(frame_index, len(frames) - 3) + 1)) if len(frames) >= 3 else []
    needed = sorted({index for start in starts for index in range(start, start + 3)}) if starts else list(range(len(frames)))
    context = {index: frames[index] for index in needed}
    output = []
    for tile_index, (left, top, right, bottom) in enumerate(tiles):
        if starts:
            heatmaps = []
            for start in starts:
                values, shape = detector.predict_heatmaps([context[index][top:bottom, left:right] for index in range(start, start + 3)])
                heatmaps.append(values[frame_index - start])
            candidates = detector.decode_heatmaps([np.mean(heatmaps, axis=0)], shape)[0]
        else:
            candidates = detector.predict_triplet([context[index][top:bottom, left:right] for index in needed])[frame_index]
        for rank, candidate in enumerate(candidates):
            output.append({'observation': BallObservation(candidate.x_px + left, candidate.y_px + top, candidate.confidence),
                           'tile_index': tile_index, 'tile_box': [left, top, right, bottom], 'tile_mass_rank': rank})
    return output


def merge_by_confidence(candidates, frame_size, duplicate_radius_reference_px=4.):
    """Greedy cross-view duplicate removal; confidence is not calibrated across crops."""
    width, height = frame_size
    if min(width, height) <= 0 or not math.isfinite(duplicate_radius_reference_px) or duplicate_radius_reference_px <= 0:
        raise ValueError('Invalid merge geometry or radius')
    if any(not all(math.isfinite(v) for v in (item.x_px, item.y_px, item.confidence)) for item in candidates):
        raise ValueError('Nonfinite candidate')
    selected = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if not any(math.hypot((candidate.x_px - previous.x_px) * 512 / width,
                              (candidate.y_px - previous.y_px) * 288 / height) <= duplicate_radius_reference_px
                   for previous in selected):
            selected.append(candidate)
    return selected
