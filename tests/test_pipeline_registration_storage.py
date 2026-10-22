"""Exercise the actual pipeline's retained registration record path without inference."""
from contextlib import closing
import sqlite3
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import src.pipeline.phase6_pipeline as pipeline_module
from src.court.calibration import CourtCalibration
from src.court.camera_registration import RegistrationResult
from src.tracking.court_player_tracker import CourtPlayerTracker
from tests.test_pipeline_candidate_storage import BallPassStopped, pipeline_with_person_observations


@pytest.mark.parametrize('enabled', [False, True])
def test_registration_records_are_bounded_and_aligned_before_ball_pass(tmp_path, monkeypatch, enabled):
    pipeline, people = pipeline_with_person_observations('stop')
    pipeline.config['court_detection']['camera_registration']['enabled'] = enabled
    anchor = CourtCalibration(np.eye(3), (True,) * 14, .125, .25, .0625)
    monkeypatch.setattr(pipeline_module, 'calibrate_court', lambda *args: anchor)
    retained = SimpleNamespace(live_calibrations=0, live_transforms=0, selected=False)
    expected = []

    class Registration:
        def __init__(self, *args, **kwargs):
            self.index = 0

        def update(self, frame, detections):
            index = self.index
            self.index += 1
            valid = index != 2
            matrix = np.array([[.125, -.0, index / 64], [0., .25, -.125], [0., 0., 1.]], dtype=np.float64)
            calibration = CourtCalibration(matrix if valid else None, (True, False, True),
                                           index / 16, index / 32, index / 64,
                                           None if valid else 'CAMERA_CUT')
            transform = np.array([[1., -.0, index / 8], [0., 1., .25], [0., 0., 1.]], dtype=np.float32) if valid else None
            expected.append((calibration.to_dict(), None if transform is None else transform.copy()))
            retained.live_calibrations += 1
            weakref.finalize(calibration, release_calibration)
            if transform is not None:
                retained.live_transforms += 1
                weakref.finalize(transform, release_transform)
            return RegistrationResult(calibration, transform, {'frame_index': index, 'is_valid': valid})

    def release_calibration():
        retained.live_calibrations -= 1

    def release_transform():
        retained.live_transforms -= 1

    monkeypatch.setattr(pipeline_module, 'CourtCameraRegistration', Registration)
    select = CourtPlayerTracker.select

    def select_checked(tracker, detections, fps, calibrations=None):
        # A full Python list is the retained allocation this integration removes.
        assert not isinstance(calibrations, list)
        assert len(calibrations) == len(detections) == 4
        snapshots = list(calibrations)
        for index, calibration in enumerate(snapshots):
            assert calibration.to_dict() == (expected[index][0] if enabled else anchor.to_dict())
        result = select(tracker, detections, fps, calibrations=calibrations)
        reference = select(tracker, detections, fps, calibrations=snapshots)
        assert all(left is right for role in (1, 2) for left, right in zip(result[role], reference[role]))
        retained.selected = True
        return result

    monkeypatch.setattr(CourtPlayerTracker, 'select', select_checked)
    def ball_stream(frames):
        assert retained.selected
        assert retained.live_calibrations == 0
        assert retained.live_transforms == 0
        people.observed_before_ball = people.live
        raise BallPassStopped('Stopped before any actual ball inference')
        yield  # Keep the same detector-generator lifecycle as the real path.

    pipeline.ball_detector.predict_stream = ball_stream
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(BallPassStopped):
        pipeline._run_with_frames('unused.mp4', tmp_path, frames, SimpleNamespace(fps=60))
    assert people.observed_before_ball is not None
    assert people.observed_before_ball < people.detected
    assert not (tmp_path / 'camera_registration.json').exists()
    assert not (tmp_path / 'image_people.json').exists()
    assert not list(tmp_path.glob('.*.tmp'))
    paths = list(tmp_path.glob('frame_registration_*.sqlite3'))
    assert len(paths) == (1 if enabled else 0)
    if enabled:
        with closing(sqlite3.connect(paths[0])) as connection:
            assert connection.execute('SELECT COUNT(*) FROM registration_frames').fetchone()[0] == 4
