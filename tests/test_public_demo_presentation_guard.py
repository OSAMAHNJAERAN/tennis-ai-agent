"""Presentation refresh must not erase valid evidence or invent ball pixels."""
import copy

import pytest

from scripts.run.refresh_public_demo_annotations import validate_evidence


def evidence():
    frames = [{'frame_index': 0, 'court_registration_valid': False,
               'player_1': None, 'player_2': None,
               'ball': {'state': 'DETECTED', 'position_px': [30, 40]}}]
    detections = {'metadata': {'width': 100, 'height': 100}, 'frames': frames}
    result = {'summary': {'frames': 1, 'capabilities': {}},
              'detections': {'frames': copy.deepcopy(frames)}, 'events': []}
    return detections, result


@pytest.mark.parametrize('capability', ['court', 'players', 'pose', 'racket', 'events', 'lineCalls', 'scoring'])
def test_refresh_cannot_omit_valid_optional_evidence(capability):
    detections, result = evidence()
    result['summary']['capabilities'][capability] = True
    with pytest.raises(ValueError, match='omit valid'):
        validate_evidence(detections, result)


def test_refresh_cannot_change_published_ball_evidence():
    detections, result = evidence()
    result['detections']['frames'][0]['ball']['position_px'] = [31, 40]
    with pytest.raises(ValueError, match='disagree'):
        validate_evidence(detections, result)


@pytest.mark.parametrize('position,state', [([100, 40], 'DETECTED'), ([30, 40], 'MISSING')])
def test_refresh_rejects_outside_or_unavailable_ball_pixels(position, state):
    detections, result = evidence()
    detections['frames'][0]['ball'] = {'position_px': position, 'state': state}
    result['detections']['frames'] = copy.deepcopy(detections['frames'])
    with pytest.raises(ValueError):
        validate_evidence(detections, result)
