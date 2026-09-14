"""Offline recovery of real weak candidates inside short, strong-anchor gaps.

Linear interpolation is a search guide only. Every recovered output is an
actual detector candidate; this module never emits an interpolated position.
False anchors can still cause false recovery. This is not event or 3D evidence.
"""

import math


def recover_anchored_gaps(strong, weak_candidates, fps, frame_size, max_span_seconds=.2,
                          max_deviation_reference_px=6., minimum_anchor_confidence=.5,
                          minimum_speed_reference_px_s=20., maximum_speed_reference_px_s=1200.):
    width, height = frame_size
    if (len(strong) != len(weak_candidates)
            or not all(math.isfinite(v) and v > 0 for v in
                       (fps, width, height, max_span_seconds, max_deviation_reference_px,
                        minimum_speed_reference_px_s, maximum_speed_reference_px_s))
            or not 0 <= minimum_anchor_confidence <= 1
            or minimum_speed_reference_px_s > maximum_speed_reference_px_s):
        raise ValueError('Invalid aligned candidate sequences or recovery settings')
    def point(candidate):
        values = candidate.x_px * 512 / width, candidate.y_px * 288 / height
        if not all(math.isfinite(value) for value in (*values, candidate.confidence)):
            raise ValueError('Candidate coordinates and confidence must be finite')
        return values
    output, audit, previous = list(strong), [], None
    for right, anchor in enumerate(strong):
        if anchor is None:
            continue
        endpoint = point(anchor)
        if previous is not None and right - previous > 1:
            left_anchor = strong[previous]
            span = (right - previous) / fps
            origin = point(left_anchor)
            velocity = math.dist(origin, endpoint) / span
            if (span <= max_span_seconds and min(left_anchor.confidence, anchor.confidence) >= minimum_anchor_confidence
                    and minimum_speed_reference_px_s <= velocity <= maximum_speed_reference_px_s):
                for index in range(previous + 1, right):
                    fraction = (index - previous) / (right - previous)
                    guide = tuple(a + fraction * (b - a) for a, b in zip(origin, endpoint))
                    candidates = [(math.dist(point(candidate), guide), rank, candidate)
                                  for rank, candidate in enumerate(weak_candidates[index])]
                    if candidates:
                        error, _, candidate = min(candidates, key=lambda item: (item[0], item[1]))
                        if error <= max_deviation_reference_px:
                            output[index] = candidate
                            audit.append({'frame': index, 'left_anchor_frame': previous,
                                          'right_anchor_frame': right, 'lookahead_frames': right - index,
                                          'guide_deviation_reference_px': error,
                                          'source': 'OBSERVED_WEAK_CANDIDATE; NOT_INTERPOLATED_POSITION'})
        previous = right
    return output, audit
