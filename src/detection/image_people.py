"""Preserve observed person boxes without turning them into active-player IDs.

Court roles are annotations of the exact boxes selected by CourtPlayerTracker.
An unselected person may be a participant, spectator, official or false positive.
No missing observation is filled, and tracker IDs are not verified person IDs.
"""

import math


IDENTITY_SCOPE = 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID'
PARTICIPANT_SEMANTICS = ('Person detector proposals may include spectators, officials and false positives. '
                         'Court-role candidates are not independently verified active participants.')


def _finite(value):
    try:
        return float(value) if not isinstance(value, bool) and math.isfinite(float(value)) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _track_id(value):
    identity = _finite(value)
    return int(identity) if identity is not None and identity >= 0 and int(identity) == identity else None


def image_people_metadata():
    """Describe raw person observations independently of their frame records."""
    return {'schema_version': '1.0', 'coordinateSystem': 'normalized_video_pixels',
            'identityScope': IDENTITY_SCOPE, 'participantSemantics': PARTICIPANT_SEMANTICS,
            'accuracyValidated': False}


def image_people_records(detections, selected, *, fps, width, height):
    """Validate alignment eagerly, then yield one observed frame at a time."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    if width <= 0 or height <= 0:
        raise ValueError('Video dimensions must be positive')
    count = len(detections)
    if any(len(selected.get(role, [])) != count for role in (1, 2)):
        raise ValueError('Court-role observations must align with person detection frames')
    return _image_people_records(detections, selected, fps=fps, width=width, height=height)


def _image_people_records(detections, selected, *, fps, width, height):
    source_indices = getattr(selected, 'selected_detection_indices', None)
    for index, proposals in enumerate(detections):
        people = []
        frame_indices = source_indices[index] if source_indices is not None else None
        for proposal_index, box in enumerate(proposals):
            bounds = [_finite(value) for value in (box.x1, box.y1, box.x2, box.y2)]
            confidence = _finite(box.confidence)
            if (any(value is None for value in bounds) or confidence is None or not 0 < confidence <= 1
                    or not 0 <= bounds[0] < bounds[2] <= width
                    or not 0 <= bounds[1] < bounds[3] <= height):
                continue
            # Disk snapshots have distinct Python objects. Bind to the exact
            # original proposal index, preserving even equal boxes/duplicate IDs.
            role = next((role for role in (1, 2) if
                         (frame_indices[role] == proposal_index if frame_indices is not None
                          else selected[role][index] is box)), None)
            people.append({'bbox': bounds, 'confidence': confidence, 'sourceTrackId': _track_id(box.track_id),
                           'courtRole': role,
                           'participantStatus': 'COURT_ROLE_CANDIDATE' if role is not None else 'UNKNOWN'})
        yield {'frame_index': index, 'timestamp_seconds': index / fps, 'people': people}


def image_people_artifact(detections, selected, *, fps, width, height):
    """Export every valid actual box, independently of court calibration success."""
    return {**image_people_metadata(),
            'frames': list(image_people_records(detections, selected, fps=fps, width=width, height=height))}
