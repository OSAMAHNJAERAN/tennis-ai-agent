"""Experimental offline person selection from camera-compensated image travel.

Motion is supporting image evidence, not proof of active participation. This
candidate deliberately has no court roles, physical metrics, identity merging,
top-two ranking or gap filling. It is not connected to the product defaults.
"""
from collections import defaultdict
import math
from statistics import median


def _project(matrix, point):
    if (not isinstance(matrix, list) or len(matrix) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                   for row in matrix for value in row)):
        return None
    x, y = point
    scale = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(scale) <= 1e-8:
        return None
    result = [(matrix[axis][0] * x + matrix[axis][1] * y + matrix[axis][2]) / scale for axis in (0, 1)]
    return result if all(math.isfinite(value) for value in result) else None


def select_motion_supported_people(detections, fps, camera_samples, *, minimum_observed_seconds=.5,
                                   minimum_motion_extent_heights=.6, minimum_camera_samples=4,
                                   minimum_motion_span_seconds=.5):
    """Return actual observations of qualifying moving tracks, abstaining at >2."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    if len(camera_samples) != len(detections):
        raise ValueError('Camera samples must align with person observations')
    tracks = defaultdict(list)
    observations = defaultdict(list)
    valid_rows, duplicated_frames = [], set()
    for index, people in enumerate(detections):
        seen = set()
        current_rows = []
        for person in people:
            if not isinstance(person, dict):
                continue
            identity, box = person.get('sourceTrackId'), person.get('bbox')
            confidence = person.get('confidence')
            if type(identity) is int and identity in seen:
                duplicated_frames.add(index)
            valid = (type(identity) is int and identity >= 0 and identity not in seen
                     and isinstance(box, list) and len(box) == 4
                     and all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in box)
                     and box[0] < box[2] and box[1] < box[3]
                     and isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                     and math.isfinite(confidence) and 0 < confidence <= 1)
            if not valid:
                continue
            seen.add(identity)
            current_rows.append(person)
            observations[identity].append(index)
            camera = camera_samples[index]
            if not isinstance(camera, dict) or camera.get('isValid') is not True or camera.get('frame_index') != index:
                continue
            timestamp = camera.get('timestamp_seconds')
            if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or abs(timestamp - index / fps) > .01:
                continue
            foot = _project(camera.get('frameToAnchor'), [(box[0] + box[2]) / 2, box[3]])
            top = _project(camera.get('frameToAnchor'), [(box[0] + box[2]) / 2, box[1]])
            if foot is None or top is None or math.dist(foot, top) <= 1:
                continue
            tracks[identity].append((index, foot, math.dist(foot, top)))
        valid_rows.append(current_rows)
    statistics = {}
    for identity, observed in observations.items():
        support = tracks[identity]
        extent = (math.hypot(max(row[1][0] for row in support) - min(row[1][0] for row in support),
                             max(row[1][1] for row in support) - min(row[1][1] for row in support))
                  / median(row[2] for row in support)) if support else None
        span = (support[-1][0] - support[0][0]) / fps if support else 0
        eligible = (len(observed) / fps >= minimum_observed_seconds and len(support) >= minimum_camera_samples
                    and span >= minimum_motion_span_seconds and extent is not None
                    and extent >= minimum_motion_extent_heights)
        statistics[identity] = {'observedFrames': len(observed), 'cameraSupportedSamples': len(support),
                                'motionSpanSeconds': span, 'motionExtentBodyHeights': extent,
                                'eligibleMovingPersonCandidate': eligible,
                                'identityScope': 'SOURCE_TRACK_FRAGMENT_NOT_VERIFIED_PERSON_ID'}
    selected, ambiguity = [], []
    for index, people in enumerate(valid_rows):
        candidates = [person for person in people if statistics.get(person.get('sourceTrackId'), {}).get('eligibleMovingPersonCandidate')]
        ambiguous = index in duplicated_frames or len(candidates) > 2
        selected.append([] if ambiguous else [dict(person) for person in candidates])
        if ambiguous:
            ambiguity.append(index)
    return selected, {'statistics': statistics, 'ambiguousFrames': ambiguity,
                      'selectionScope': 'OFFLINE_MOVING_PERSON_CANDIDATE_NOT_VERIFIED_ACTIVE_PARTICIPANT',
                      'qualificationEvidence': False, 'identityMerging': False, 'gapFilling': False}
