"""Exercise actual pipeline stage ordering and storage with inference stubbed."""
from contextlib import closing
import sqlite3
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from src.pipeline.phase6_pipeline import Phase6Pipeline
from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.bbox_utils import BBox
from src.utils.ball_candidate_sequence import BallCandidateSequence


class BallPassStopped(RuntimeError):
    pass


def pipeline_with_person_observations(ball_mode):
    pipeline = Phase6Pipeline.__new__(Phase6Pipeline)
    pipeline.config = {'court_detection': {'camera_registration': {'enabled': False}},
                       'ball_detection': {'temporal_step': 1}, 'event_detection': {'authoritative_enabled': False}}
    pipeline.ball_backend = 'wasb'
    pipeline.line_call_engine = SimpleNamespace(refiner=None)
    pipeline.court_detector = SimpleNamespace(predict=lambda frame: np.zeros((14, 2)))
    lifetime = SimpleNamespace(live=0, detected=0, observed_before_ball=None, ball_producer_closed=False)

    def release():
        lifetime.live -= 1

    def detect(frame, persist):
        assert persist is True
        values = [BBox(1 + index * 15, 2, 12 + index * 15, 55, confidence=.8, track_id=index)
                  for index in range(4)]
        for value in values:
            lifetime.live += 1
            lifetime.detected += 1
            weakref.finalize(value, release)
        return values

    def ball_stream(frames):
        # Calibration abstention selects no boxes. Only the person export had
        # needed these objects after selection; it must finish before this pass.
        lifetime.observed_before_ball = lifetime.live
        assert lifetime.detected == 4 * len(frames)
        assert lifetime.live == 0
        try:
            if ball_mode == 'stop':
                raise BallPassStopped('Stopped before any actual ball inference')
            for index in range(len(frames) - 1):
                yield [BallObservation(20 + index, 30, .8)]
        finally:
            lifetime.ball_producer_closed = True
            if ball_mode == 'close_error':
                raise RuntimeError('Injected producer cleanup failure')

    pipeline.player_detector = SimpleNamespace(detect_and_track=detect)
    pipeline.ball_detector = SimpleNamespace(predict_stream=ball_stream)
    return pipeline, lifetime


def assert_closed_diagnostic_spool(directory, rows):
    paths = list(directory.glob('ball_candidates_*.sqlite3'))
    assert len(paths) == 1
    with closing(sqlite3.connect(paths[0])) as connection:
        assert connection.execute('SELECT COUNT(*) FROM candidate_frames').fetchone()[0] == rows
    # A completed image artifact must not publish when a later stage fails.
    assert not (directory / 'image_people.json').exists()
    assert not (directory / 'camera_registration.json').exists()
    assert not list(directory.glob('.*.tmp'))


def test_person_records_released_before_ball_inference_and_failed_exports_unpublished(tmp_path):
    pipeline, lifetime = pipeline_with_person_observations('stop')
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(BallPassStopped, match='before any actual ball inference'):
        pipeline._run_with_frames('unused.mp4', tmp_path, frames, SimpleNamespace(fps=60))
    assert lifetime.observed_before_ball == 0 and lifetime.live == 0
    assert_closed_diagnostic_spool(tmp_path, rows=0)


def test_incomplete_candidate_stream_fails_before_tracking_or_publication(tmp_path):
    pipeline, lifetime = pipeline_with_person_observations('short')
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match='candidate frames did not align'):
        pipeline._run_with_frames('unused.mp4', tmp_path, frames, SimpleNamespace(fps=60))
    assert lifetime.observed_before_ball == 0 and lifetime.live == 0
    assert_closed_diagnostic_spool(tmp_path, rows=3)


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_candidate_storage_failure_closes_original_producer_and_preserves_error(tmp_path, monkeypatch, cleanup_fails):
    pipeline, lifetime = pipeline_with_person_observations('close_error' if cleanup_fails else 'short')
    append = BallCandidateSequence.append

    def fail_second_append(storage, candidates):
        if len(storage) == 1:
            raise OSError('Injected candidate storage failure')
        append(storage, candidates)

    monkeypatch.setattr(BallCandidateSequence, 'append', fail_second_append)
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(4)]
    with pytest.raises(OSError, match='Injected candidate storage failure') as failure:
        pipeline._run_with_frames('unused.mp4', tmp_path, frames, SimpleNamespace(fps=60))
    # Inspect before dropping the traceback: cleanup must not depend on GC.
    assert lifetime.ball_producer_closed
    if cleanup_fails:
        assert any('Injected producer cleanup failure' in note for note in failure.value.__notes__)
    assert_closed_diagnostic_spool(tmp_path, rows=1)
