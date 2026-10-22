"""Configured inference settings must reach the actual detector calls."""

import numpy as np
import pytest
from types import SimpleNamespace
import torch
from src.detection import player_detector


def test_player_settings_are_forwarded_without_loading_weights(monkeypatch):
    calls = []
    class Model:
        def __init__(self, _path):
            pass
        def __call__(self, _frame, **kwargs):
            calls.append(kwargs)
            return []
        def track(self, _frame, **kwargs):
            calls.append(kwargs)
            return []
    monkeypatch.setattr(player_detector, 'YOLO', Model)
    detector = player_detector.PlayerDetector('local.pt', confidence=.4, tracker='botsort.yaml')
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    detector.detect(frame)
    detector.detect_and_track(frame, persist=False)
    assert calls[0]['conf'] == .4
    assert calls[1]['tracker'] == 'botsort.yaml' and calls[1]['conf'] == .4 and calls[1]['persist'] is False
    with pytest.raises(ValueError):
        player_detector.PlayerDetector('local.pt', confidence=1.1)


@pytest.mark.parametrize('identities,expected', [(None, [None, None]), ([7., float('nan')], [7, None])])
def test_person_observations_survive_absent_or_invalid_tracker_ids(monkeypatch, identities, expected):
    boxes = SimpleNamespace(xyxy=torch.tensor([[1., 2., 8., 18.], [9., 3., 17., 19.]]),
                            conf=torch.tensor([.8, .7]), id=torch.tensor(identities) if identities is not None else None)
    class Model:
        def __init__(self, _path):
            pass
        def track(self, _frame, **_kwargs):
            return [SimpleNamespace(boxes=boxes)]
    monkeypatch.setattr(player_detector, 'YOLO', Model)
    detector = player_detector.PlayerDetector('unused-local.pt')
    observations = detector.detect_and_track(np.zeros((20, 20, 3), dtype=np.uint8))
    assert len(observations) == 2
    assert [box.track_id for box in observations] == expected
    assert [box.x1 for box in observations] == [1., 9.]
    assert [box.confidence for box in observations] == pytest.approx([.8, .7])
