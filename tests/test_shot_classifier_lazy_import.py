"""The classifier's software branches do not initialize a pose model runtime."""

from __future__ import annotations

import builtins
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest


@contextmanager
def _without_model_imports():
    original = builtins.__import__
    attempts = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            attempts.append(name)
            raise AssertionError("Classifier attempted a model runtime import: " + name)
        return original(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield attempts
    finally:
        builtins.__import__ = original
        assert attempts == []


def test_classifier_import_has_no_model_dependency_in_fresh_process():
    # Use a fresh interpreter so a previous test's module cache cannot hide an
    # eager import. Both the import attempt and loaded modules are inspected.
    program = '''
import builtins
import json
import sys

attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Unexpected model import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from src.shot_analysis.shot_classifier import TennisShotClassifier
classifier = TennisShotClassifier()
serve = classifier.classify_shot('SERVE_CONTACT', 0, 1, None, None)
dead = classifier.classify_shot('PLAYER_1_HIT', 1, 1, None, None, is_dead_ball=True)
unknown = classifier.classify_shot('PLAYER_1_HIT', 2, 1, None, None)
print(json.dumps({
    'attempts': attempts,
    'loaded_model_modules': sorted(name for name in sys.modules
        if name.split('.')[0] in {'torch', 'ultralytics'}),
    'extractor_imported': 'src.shot_analysis.pose_feature_extractor' in sys.modules,
    'shots': [result[0].value for result in (serve, dead, unknown)],
    'annotation': TennisShotClassifier.__init__.__annotations__['pose_extractor'],
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result == {
        "attempts": [],
        "loaded_model_modules": [],
        "extractor_imported": False,
        "shots": ["SERVE", "UNKNOWN", "UNKNOWN"],
        "annotation": "Optional[PoseFeatureExtractor]",
    }


class _PoseProvider:
    def __init__(self, displacement=0.6):
        self.calls = []
        self.displacement = displacement

    def extract_hit_window_features(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "valid": True,
            "mean_pose_confidence": 0.8,
            "normalized_wrist_displacement": self.displacement,
        }


@pytest.mark.parametrize("displacement,expected", [(0.6, "FOREHAND"), (-0.6, "BACKHAND")])
def test_configured_pose_provider_is_called_with_original_inputs(displacement, expected):
    with _without_model_imports():
        import numpy as np
        from src.shot_analysis.shot_classifier import TennisShotClassifier
        from src.shot_analysis.shot_types import PlayerHandedness
        from src.utils.bbox_utils import BBox

        provider = _PoseProvider(displacement)
        classifier = TennisShotClassifier(
            provider,
            handedness_map={2: PlayerHandedness.LEFT_HANDED},
            court_orientation_map={2: -1.0},
        )
        frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
        boxes = [BBox(10.25, 20.5, 30.75, 40.125, confidence=0.9, track_id=7)]
        result = classifier.classify_shot(
            "PLAYER_2_HIT", 0, 2, boxes[0], None,
            frames=frames, all_player_boxes=boxes,
        )

        assert result[0].value == expected
        assert result[1] == 0.8
        assert result[2].value == "YOLO11_POSE_TEMPORAL"
        assert len(provider.calls) == 1
        received = provider.calls[0]
        assert set(received) == {
            "frames", "player_boxes", "hit_frame", "player_id",
            "handedness", "court_orientation_sign",
        }
        assert received["frames"] is frames
        assert received["player_boxes"] is boxes
        assert received["hit_frame"] == 0
        assert received["player_id"] == 2
        assert received["handedness"] is PlayerHandedness.LEFT_HANDED
        assert received["court_orientation_sign"] == -1.0


@pytest.mark.parametrize(
    "event_type,dead,player_id,handedness,orientation,expected,confidence",
    [
        ("SERVE_CONTACT", False, 1, True, True, "SERVE", 0.95),
        ("SERVE_CONTACT", True, 1, True, True, "UNKNOWN", 0.0),
        ("PLAYER_1_HIT", True, 1, True, True, "UNKNOWN", 0.0),
        ("PLAYER_1_HIT", False, None, True, True, "UNKNOWN", 0.0),
        ("PLAYER_1_HIT", False, 1, False, True, "UNKNOWN", 0.5),
        ("PLAYER_1_HIT", False, 1, True, False, "UNKNOWN", 0.5),
        ("PLAYER_1_HIT", False, 1, False, False, "UNKNOWN", 0.5),
    ],
)
def test_early_branches_preserve_abstention_without_pose(
    event_type, dead, player_id, handedness, orientation, expected, confidence,
):
    with _without_model_imports():
        from src.shot_analysis.shot_classifier import TennisShotClassifier
        from src.shot_analysis.shot_types import PlayerHandedness

        provider = _PoseProvider()
        classifier = TennisShotClassifier(
            provider,
            handedness_map={1: PlayerHandedness.RIGHT_HANDED} if handedness else {},
            court_orientation_map={1: 1.0} if orientation else {},
        )
        result = classifier.classify_shot(
            event_type, 0, player_id, None, None,
            frames=[object()], all_player_boxes=[None], is_dead_ball=dead,
        )
        assert result[0].value == expected
        assert result[1] == confidence
        assert provider.calls == []


def test_pose_provider_failures_are_not_silently_converted_to_unknown():
    with _without_model_imports():
        from src.shot_analysis.shot_classifier import TennisShotClassifier
        from src.shot_analysis.shot_types import PlayerHandedness

        class UnavailablePoseProvider:
            def extract_hit_window_features(self, **kwargs):
                raise RuntimeError("Selected pose computation requires admission")

        classifier = TennisShotClassifier(
            UnavailablePoseProvider(),
            handedness_map={1: PlayerHandedness.RIGHT_HANDED},
            court_orientation_map={1: 1.0},
        )
        with pytest.raises(RuntimeError, match="requires admission"):
            classifier.classify_shot(
                "PLAYER_1_HIT", 0, 1, None, None,
                frames=[object()], all_player_boxes=[None],
            )
