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


def image_people_artifact(detections, selected, *, fps, width, height):
    """Export every valid actual box, independently of court calibration success."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be positive and finite')
    if width <= 0 or height <= 0:
        raise ValueError('Video dimensions must be positive')
    count = len(detections)
    if any(len(selected.get(role, [])) != count for role in (1, 2)):
        raise ValueError('Court-role observations must align with person detection frames')
    frames = []
    for index, proposals in enumerate(detections):
        people = []
        for box in proposals:
            bounds = [_finite(value) for value in (box.x1, box.y1, box.x2, box.y2)]
            confidence = _finite(box.confidence)
            if (any(value is None for value in bounds) or confidence is None or not 0 < confidence <= 1
                    or not 0 <= bounds[0] < bounds[2] <= width
                    or not 0 <= bounds[1] < bounds[3] <= height):
                continue
            # Selection currently returns an actual input BBox object. Never
            # attach a role by track-ID alone or by guessing the top two boxes.
            role = next((role for role in (1, 2) if selected[role][index] is box), None)
            people.append({'bbox': bounds, 'confidence': confidence, 'sourceTrackId': _track_id(box.track_id),
                           'courtRole': role,
                           'participantStatus': 'COURT_ROLE_CANDIDATE' if role is not None else 'UNKNOWN'})
        frames.append({'frame_index': index, 'timestamp_seconds': index / fps, 'people': people})
    return {'schema_version': '1.0', 'coordinateSystem': 'normalized_video_pixels',
            'identityScope': IDENTITY_SCOPE, 'participantSemantics': PARTICIPANT_SEMANTICS,
            'accuracyValidated': False, 'frames': frames}
