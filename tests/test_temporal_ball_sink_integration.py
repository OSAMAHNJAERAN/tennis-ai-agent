"""Typed disk output agrees with the retained offline list calculations."""
from collections.abc import Sequence
from dataclasses import asdict
import sqlite3
import weakref

import pytest

from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallPoint, TemporalBallTracker
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence, _decode_point


def observation(x, confidence=.9, y=30.):
    return BallObservation(float(x), float(y), confidence)


CASES = [
    pytest.param([[observation(10)], [observation(20, .2)], [], [], [], [observation(50)], [], [], [], []],
                 dict(max_prediction_gap=1, max_interpolation_gap=3),
                 {'high_conf_detector', 'gated_candidate', 'kinematic_kalman', 'linear_interpolation', 'none'},
                 (320, 240), id='detections-low-confidence-occluded-input-prediction-interpolation-missing'),
    pytest.param([[observation(55)], [observation(59)], []], dict(max_prediction_gap=2),
                 {'high_conf_detector', 'rejected_out_of_frame'}, (60, 60), id='predicted-frame-bounds-rejection'),
    pytest.param([[observation(10)], [observation(70, .2)], [observation(80)], [observation(90)]],
                 dict(max_valid_speed_px_per_frame=20., base_gating_radius_px=100., max_interpolation_gap=0),
                 {'high_conf_detector', 'rejected_outlier'}, (320, 240), id='tracked-impossible-jump-rejection'),
    pytest.param([[observation(40, .02)], [observation(45, .02)], [observation(50)]],
                 dict(enable_short_gap_reacquisition=False),
                 {'backward_gated_candidate', 'high_conf_detector'}, (320, 240), id='backward-confirmation'),
    pytest.param([[], [observation(40, .1)], [], [observation(50)]], {},
                 {'none', 'reacquired_candidate', 'kinematic_kalman', 'high_conf_detector'},
                 (320, 240), id='low-confidence-reacquisition'),
    pytest.param([[], [observation(40, .1)], [], [observation(50)]],
                 dict(enable_short_gap_reacquisition=False),
                 {'none', 'seed_candidate', 'kinematic_kalman', 'high_conf_detector'},
                 (320, 240), id='low-confidence-seeding'),
]


@pytest.mark.parametrize('candidates,options,expected_sources,frame_size', CASES)
def test_complete_mixed_state_list_and_disk_results_agree(tmp_path, candidates, options, expected_sources, frame_size):
    settings = dict(high_conf_thresh=.5, low_conf_thresh=.01,
                    enable_multi_candidate_association=False, enable_scale_normalization=False)
    settings.update(options)
    expected = TemporalBallTracker(**settings).track_video_candidates(candidates, fps=25., frame_size=frame_size)
    assert type(expected) is list
    assert {point.source for point in expected} == expected_sources
    assert any(point.velocity_px_per_sec is not None for point in expected)
    if 'rejected_outlier' in expected_sources:
        assert expected[1].state is BallState.MISSING
        assert expected[1].velocity_px_per_sec is expected[2].velocity_px_per_sec is None
        assert expected[3].velocity_px_per_sec == (250., 0.)
    with TemporalBallPointSequence(tmp_path) as sink:
        result = TemporalBallTracker(**settings).track_video_candidates(candidates, fps=25., frame_size=frame_size, output=sink)
        assert result is sink
        assert [asdict(point) for point in sink] == [asdict(point) for point in expected]
    assert not list(tmp_path.glob('temporal_ball_points_*.sqlite3'))


def test_short_gap_keeps_exact_legacy_interpolation_confidence_and_nominal_velocity(tmp_path):
    candidates = [[observation(10)], [], [observation(30)]]
    tracker = TemporalBallTracker(high_conf_thresh=.5, max_prediction_gap=0, max_interpolation_gap=1,
                                  enable_multi_candidate_association=False, enable_scale_normalization=False)
    expected = [TemporalBallPoint(0, 0., 10., 30., confidence=.9, state=BallState.DETECTED, source='high_conf_detector'),
                TemporalBallPoint(1, .04, 20., 30., confidence=None, state=BallState.INTERPOLATED,
                                  source='linear_interpolation', velocity_px_per_sec=(250., 0.)),
                TemporalBallPoint(2, .08, 30., 30., confidence=.9, state=BallState.DETECTED,
                                  source='high_conf_detector', velocity_px_per_sec=(250., 0.))]
    assert tracker.track_video_candidates(candidates, fps=25., frame_size=(320, 240)) == expected
    with TemporalBallPointSequence(tmp_path) as sink:
        tracker.track_video_candidates(candidates, fps=25., frame_size=(320, 240), output=sink)
        assert list(sink) == expected


def test_existing_sink_prefix_including_occlusion_is_never_reprocessed(tmp_path):
    prefix = TemporalBallPoint(900, 36., None, None, court_x_m=1., court_y_m=2., confidence=.1,
                               state=BallState.OCCLUDED, source='reviewed_occlusion',
                               velocity_px_per_sec=(4., 5.), speed_kmh=10.)
    candidates = [[observation(10)], [], [observation(30)]]
    tracker = TemporalBallTracker(enable_multi_candidate_association=False, max_prediction_gap=0)
    expected = tracker.track_video_candidates(candidates, fps=25.)
    ordinary = [prefix]
    assert tracker.track_video_candidates(candidates, fps=25., output=ordinary) is ordinary
    assert ordinary == [prefix, *expected]
    with TemporalBallPointSequence(tmp_path) as sink:
        sink.append(prefix)
        assert tracker.track_video_candidates(candidates, fps=25., output=sink) is sink
        assert sink[0] == prefix
        assert list(sink[1:]) == expected


def test_candidate_source_failure_retains_closed_diagnostic_scratch_and_accepted_writebacks(tmp_path):
    class BrokenCandidates(Sequence):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            if index == 1:
                raise RuntimeError('candidate source interrupted')
            if not 0 <= index < 3:
                raise IndexError(index)
            return [observation(10 + index)]

    sink = TemporalBallPointSequence(tmp_path)
    prefix = TemporalBallPoint(900, 36., None, None, state=BallState.OCCLUDED, source='prior')
    with pytest.raises(RuntimeError, match='candidate source interrupted'):
        with sink:
            sink.append(prefix)
            TemporalBallTracker(enable_multi_candidate_association=False).track_video_candidates(
                BrokenCandidates(), fps=25., output=sink)
    assert sink.path.is_file()
    with pytest.raises(RuntimeError, match='not open'):
        sink[0]
    with sqlite3.connect(sink.path) as connection:
        stored = [_decode_point(row[0]) for row in connection.execute('SELECT payload FROM temporal_points ORDER BY position')]
    assert len(stored) == 4
    assert stored[0] == prefix
    assert stored[1].state is BallState.DETECTED
    assert stored[1].x_px == 10.
    assert stored[2].state is BallState.MISSING


def test_disk_tracking_retains_bounded_number_of_point_snapshots(tmp_path, monkeypatch):
    from src.tracking import temporal_ball_tracker as tracker_module
    from src.utils import temporal_ball_point_sequence as storage_module

    live = peak = 0

    def released():
        nonlocal live
        live -= 1

    class ObservedPoint(TemporalBallPoint):
        def __init__(self, *args, **kwargs):
            nonlocal live, peak
            super().__init__(*args, **kwargs)
            live += 1
            peak = max(peak, live)
            weakref.finalize(self, released)

    class Candidates(Sequence):
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            if not 0 <= index < len(self):
                raise IndexError(index)
            return [observation(20 + index * .01)]

    monkeypatch.setattr(tracker_module, 'TemporalBallPoint', ObservedPoint)
    monkeypatch.setattr(storage_module, 'TemporalBallPoint', ObservedPoint)
    with TemporalBallPointSequence(tmp_path) as sink:
        TemporalBallTracker(enable_multi_candidate_association=False).track_video_candidates(
            Candidates(), fps=60., frame_size=(1920, 1080), output=sink)
        assert len(sink) == 1000
    assert peak < 32
    assert live == 0


def test_interval_sink_preserves_global_indices_source_provenance_and_normalized_math(tmp_path):
    from src.pipeline.phase6_pipeline import track_candidate_intervals, trajectory_export_records, _CandidateWindow

    candidates = [[observation(10 + index * 10)] for index in range(5)]
    intervals = [{0: dict(measurement_transition_allowed=value, timestamp_seconds=index / 25,
                         source_timestamp_seconds=100 + index * .05, source_pts=str(2000 + index),
                         source_time_base='1/20', source_frame_index=index, segment_id=f's{index // 2}')}
                 for index, value in enumerate((False, True, False, True, False))]
    tracker = TemporalBallTracker(enable_multi_candidate_association=False)
    expected = track_candidate_intervals(tracker, candidates, intervals, 25., (320, 240))
    assert [point.frame_index for point in expected] == list(range(5))
    assert [point.timestamp_seconds for point in expected] == [index / 25 for index in range(5)]
    assert [point.velocity_px_per_sec for point in expected] == [None, (250., 0.), None, (250., 0.), None]
    with TemporalBallPointSequence(tmp_path) as sink:
        assert track_candidate_intervals(tracker, candidates, intervals, 25., (320, 240), output=sink) is sink
        assert list(sink) == expected
        records = list(trajectory_export_records(sink, intervals))
        assert [record['source_pts'] for record in records] == [str(2000 + index) for index in range(5)]
        assert records[1]['timestamp_seconds'] == .04
        assert records[1]['source_timestamp_seconds'] == 100.05
    window = _CandidateWindow(candidates, 1, 5)
    assert not isinstance(window[::-1][1::2], list)
    assert list(window[::-1][1::2]) == [candidates[3], candidates[1]]
