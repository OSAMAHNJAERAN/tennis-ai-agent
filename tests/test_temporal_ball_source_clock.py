"""Source-clock tracker regressions; nominal legacy callers remain separate."""
from collections.abc import Sequence
from dataclasses import asdict
from fractions import Fraction
import math
import weakref

import numpy as np
import pytest

from src.api.media_timing import frame_mapping
from src.pipeline.phase6_pipeline import track_candidate_intervals
from src.tracking.temporal_ball_tracker import BallObservation, BallState, KinematicFilter, TemporalBallPoint, TemporalBallTracker
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence


def source_rows(times, *, time_base='1/1000000', offset=0):
    result = []
    for index, seconds in enumerate(times):
        ticks = Fraction(str(seconds)) / Fraction(time_base)
        assert ticks.denominator == 1
        result.append(dict(source_pts=str(offset + int(ticks)), source_time_base=time_base,
                           source_frame_index=index, segment_id='camera-0',
                           measurement_transition_allowed=index > 0))
    return result


def candidates(positions, confidence=.9):
    return [[] if value is None else [BallObservation(float(value), 30., confidence)] for value in positions]


def tracker(**options):
    settings = dict(high_conf_thresh=.5, low_conf_thresh=.01, enable_multi_candidate_association=False,
                    enable_scale_normalization=False)
    settings.update(options)
    return TemporalBallTracker(**settings)


def timed(positions, times, **options):
    return tracker(**options).track_video_candidates(candidates(positions), fps=25., frame_size=(1920, 1080),
                                                     source_timing=source_rows(times))


def test_native_adapter_uses_eligible_source_delta_not_normalized_replay_delta(tmp_path):
    import json

    stats = tmp_path / 'stats.txt'
    stats.write_text('0 0 0 1/100 0 1/25\n1 1 5 1/100 1 1/25\n', encoding='utf-8')
    mapping = tmp_path / 'mapping.jsonl'
    frame_mapping(stats, mapping, 25, source_frame_count=2)
    timing = [json.loads(line) for line in mapping.read_text(encoding='utf-8').splitlines()]
    assert timing[1]['validMeasurementTransitionFromPrevious'] is True
    frames = [{0: {**row, 'measurement_transition_allowed': row['validMeasurementTransitionFromPrevious'],
                    'segment_id': 'camera-0'}} for row in timing]
    candidates = [[BallObservation(10., 30., .9)], [BallObservation(20., 30., .9)]]
    tracker = TemporalBallTracker(enable_multi_candidate_association=False, enable_scale_normalization=False)
    points = track_candidate_intervals(tracker, candidates, frames, 25., (320, 240), use_source_timing=True)
    assert [point.timestamp_seconds for point in points] == [0., .04]
    assert points[1].velocity_px_per_sec == (200., 0.)


def test_varying_source_intervals_preserve_constant_image_velocity_and_replay_clock():
    points = timed([0, 10, 50], [0, .01, .05])
    assert [point.x_px for point in points] == [0., 10., 50.]
    assert [point.velocity_px_per_sec for point in points] == [None, (1000., 0.), (1000., 0.)]
    assert [point.timestamp_seconds for point in points] == [0., .04, .08]
    assert all('source_pts_forward_only' in point.source for point in points)
    assert all(point.speed_kmh is None for point in points)


def test_interpolation_uses_source_time_fraction_instead_of_frame_ordinal():
    points = timed([0, None, 100], [0, .01, .1], max_prediction_gap=0, max_interpolation_gap=2)
    assert points[1].x_px == 10.
    assert points[1].state is BallState.INTERPOLATED
    assert points[1].confidence is None
    assert points[1].source == 'linear_interpolation_source_pts_offline'
    assert [point.velocity_px_per_sec for point in points] == [None, (1000., 0.), (1000., 0.)]


def test_same_low_confidence_displacement_obeys_elapsed_time_scaled_jump_limit():
    options = dict(max_valid_speed_px_per_frame=20., base_gating_radius_px=1000.,
                   enable_adaptive_gate=False, max_interpolation_gap=0)
    inputs = [[BallObservation(0., 30., .9)], [BallObservation(50., 30., .2)]]
    fast = tracker(**options).track_video_candidates(inputs, fps=25., source_timing=source_rows([0, .01]))
    slower = tracker(**options).track_video_candidates(inputs, fps=25., source_timing=source_rows([0, .1]))
    assert fast[1].state is BallState.PREDICTED
    assert fast[1].x_px == 0.
    assert slower[1].state is BallState.TRACKED
    assert slower[1].x_px == 50.
    assert slower[1].velocity_px_per_sec == (500., 0.)


def test_high_confidence_jump_retains_raw_observation_but_breaks_temporal_identity():
    options = dict(max_valid_speed_px_per_frame=20., base_gating_radius_px=1000., enable_adaptive_gate=False)
    fast = timed([0, 50], [0, .01], **options)
    slower = timed([0, 50], [0, .1], **options)
    assert fast[1].state is BallState.DETECTED
    assert fast[1].x_px == 50.
    assert '_jump_reset' in fast[1].source
    assert fast[1].velocity_px_per_sec is None
    assert slower[1].velocity_px_per_sec == (500., 0.)


def test_large_elapsed_gap_expires_before_later_anchor_and_restarts_cleanly():
    points = timed([0, 10, 30, 40], [0, .04, 1, 1.04], max_interpolation_gap=0)
    assert points[1].velocity_px_per_sec == (250., 0.)
    assert points[2].velocity_px_per_sec is None
    assert '_filter_expired' in points[2].source
    assert points[3].velocity_px_per_sec == (250., 0.)


def test_missing_frame_count_limit_applies_even_when_total_elapsed_time_is_small():
    points = timed([0, None, None, 30], [0, .01, .02, .03], max_prediction_gap=1, max_interpolation_gap=0)
    assert [point.state for point in points] == [BallState.DETECTED, BallState.PREDICTED,
                                                BallState.MISSING, BallState.DETECTED]
    assert points[3].velocity_px_per_sec is None


@pytest.mark.parametrize('elapsed,prediction_gap,expected', [(.04, 1, BallState.PREDICTED),
                                                           (.040001, 1, BallState.MISSING),
                                                           (.01, 0, BallState.MISSING)])
def test_prediction_requires_both_exact_elapsed_and_count_budget(elapsed, prediction_gap, expected):
    points = timed([0, None], [0, elapsed], max_prediction_gap=prediction_gap, max_interpolation_gap=0)
    assert points[1].state is expected


@pytest.mark.parametrize('elapsed,expected', [(.08, BallState.INTERPOLATED), (.080001, BallState.MISSING)])
def test_interpolation_exact_elapsed_anchor_span_boundary(elapsed, expected):
    points = timed([0, None, 100], [0, .01, elapsed], max_prediction_gap=0, max_interpolation_gap=1)
    assert points[1].state is expected
    if expected is BallState.INTERPOLATED:
        assert points[1].x_px == 12.5


@pytest.mark.parametrize('field,value', [
    ('source_pts', None), ('source_pts', 'invalid'), ('source_pts', True), ('source_pts', 0.04),
    ('source_pts', '0'), ('source_pts', '-1'), ('source_pts', '9223372036854775807'),
    ('source_time_base', None), ('source_time_base', '0'), ('source_time_base', '-1/1000000'),
    ('source_time_base', '1/0'), ('source_time_base', True),
    ('source_frame_index', None), ('source_frame_index', True), ('source_frame_index', 0), ('source_frame_index', 5),
    ('measurement_transition_allowed', False), ('measurement_transition_allowed', 1), ('segment_id', 'camera-cut'),
    ('segment_id', None), ('segment_id', ''),
])
def test_invalid_or_cut_middle_edge_never_bridges_prediction_interpolation_or_velocity(field, value):
    rows = source_rows([0, .04, .08])
    rows[1][field] = value
    points = tracker(max_prediction_gap=0).track_video_candidates(candidates([0, None, 20]), fps=25., source_timing=rows)
    assert points[1].state is BallState.MISSING
    assert points[1].x_px is None
    assert points[2].velocity_px_per_sec is None


def test_invalid_source_frame_may_retain_raw_detection_but_cannot_seed_predictions():
    rows = source_rows([0, .04, .08])
    rows[0]['source_pts'] = None
    points = tracker().track_video_candidates(candidates([10, None, None]), fps=25., source_timing=rows)
    assert points[0].state is BallState.DETECTED
    assert 'invalid_source_timing' in points[0].source
    assert [point.x_px for point in points] == [10., None, None]


def test_underflowed_float_elapsed_time_is_a_break_without_nominal_fallback():
    rows = source_rows([0, 0])
    rows[0].update(source_pts='0', source_time_base='1/' + '1' + '0' * 400)
    rows[1].update(source_pts='1', source_time_base=rows[0]['source_time_base'])
    points = tracker().track_video_candidates(candidates([0, 1]), fps=25., source_timing=rows)
    assert points[1].velocity_px_per_sec is None
    assert '_transition_reset' in points[1].source


@pytest.mark.parametrize('offset', [10 ** 18, -(10 ** 18)])
def test_exact_subtraction_preserves_small_delta_at_large_absolute_pts(offset):
    rows = source_rows([0, .001, .005], time_base='1/1000', offset=offset)
    assert float(int(rows[1]['source_pts']) * Fraction(rows[1]['source_time_base'])) == float(
        int(rows[0]['source_pts']) * Fraction(rows[0]['source_time_base']))
    points = tracker().track_video_candidates(candidates([0, 1, 5]), fps=1000., source_timing=rows)
    assert [point.velocity_px_per_sec for point in points] == [None, (1000., 0.), (1000., 0.)]


def test_changed_source_timebase_uses_exact_common_time_and_ignores_lossy_float_field():
    rows = source_rows([1, 1.01, 1.05])
    rows[0].update(source_pts='1000', source_time_base='1/1000')
    rows[1].update(source_pts='101', source_time_base='1/100')
    rows[2].update(source_pts='105', source_time_base='1/100')
    for row in rows:
        row['source_timestamp_seconds'] = math.nan
    points = tracker().track_video_candidates(candidates([0, 10, 50]), fps=25., source_timing=rows)
    assert [point.velocity_px_per_sec for point in points] == [None, (1000., 0.), (1000., 0.)]


def test_source_filter_prediction_advances_position_and_covariance_by_measured_delta():
    filter = KinematicFilter(.04)
    filter.initialize(0., 0., vx=100., vy=200.)
    covariance = filter.P.copy()
    transition = np.eye(6)
    transition[0, 2] = transition[1, 3] = .01
    transition[0, 4] = transition[1, 5] = .5 * .01 ** 2
    transition[2, 4] = transition[3, 5] = .01
    expected_covariance = transition @ covariance @ transition.T + filter.Q * .25
    assert filter.predict(dt=.01) == (1., 2.)
    np.testing.assert_array_equal(filter.P, expected_covariance)
    assert filter.predict(dt=.04) == (5., 10.)


def test_nominal_filter_prediction_retains_fixed_transition_and_process_noise():
    filter = KinematicFilter(.04)
    filter.initialize(0., 0., vx=100.)
    expected_covariance = filter.F @ filter.P @ filter.F.T + filter.Q
    assert filter.predict() == (4., 0.)
    np.testing.assert_array_equal(filter.P, expected_covariance)


def test_timed_forward_association_does_not_use_future_support_or_backward_confirmation():
    inputs = [[BallObservation(10., 30., .02)], [BallObservation(30., 30., .02)], [BallObservation(50., 30., .9)]]
    points = tracker(max_interpolation_gap=0).track_video_candidates(inputs, fps=25., source_timing=source_rows([0, .04, .08]))
    assert [point.state for point in points] == [BallState.MISSING, BallState.MISSING, BallState.DETECTED]
    assert all('backward' not in point.source for point in points)


def test_reacquisition_after_stale_prediction_cannot_create_adjacent_velocity_spike():
    points = timed([10, None, None, None, None, None, 70], [0, .04, .08, .12, .16, .2, .24],
                   max_valid_speed_px_per_frame=10., base_gating_radius_px=1000., enable_adaptive_gate=False,
                   max_interpolation_gap=0)
    assert points[-1].x_px == 70.
    assert points[-1].state is BallState.DETECTED
    assert '_jump_reset' in points[-1].source
    assert points[-1].velocity_px_per_sec is None


def test_disabled_reacquisition_keeps_initial_low_confidence_seed_but_not_later_reseed():
    inputs = [[BallObservation(10., 30., .1)], [], [BallObservation(30., 30., .1)]]
    points = tracker(enable_short_gap_reacquisition=False, max_prediction_gap=0, max_interpolation_gap=0).track_video_candidates(
        inputs, fps=25., source_timing=source_rows([0, .04, .08]))
    assert points[0].state is BallState.TRACKED
    assert 'seed_candidate' in points[0].source
    assert points[2].state is BallState.MISSING


def test_future_static_candidates_cannot_remove_an_earlier_observation():
    settings = dict(enable_multi_candidate_association=True, max_interpolation_gap=0)
    original = timed([10, 10, 10], [0, .04, .08], **settings)
    changed_future = timed([10, 20, 30], [0, .04, .08], **settings)
    assert original[0] == changed_future[0]
    assert original[0].state is BallState.DETECTED


@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_candidate_cannot_poison_timed_filter_with_bounds_disabled(value):
    inputs = [[BallObservation(value, 30., .9)], [BallObservation(20., 30., .9)], []]
    points = tracker(enable_frame_bounds_filter=False).track_video_candidates(inputs, fps=25., source_timing=source_rows([0, .04, .08]))
    assert points[0].state is BallState.MISSING
    assert points[1].x_px == 20.
    assert points[2].x_px == 20.


def test_timing_length_mismatch_does_not_modify_existing_output_prefix():
    prefix = TemporalBallPoint(100, 4., None, None, source='prior')
    output = [prefix]
    with pytest.raises(ValueError, match='exactly one'):
        tracker().track_video_candidates(candidates([0, 10]), source_timing=source_rows([0]), output=output)
    assert output == [prefix]


def test_empty_and_single_frame_timed_contract():
    assert tracker().track_video_candidates([], source_timing=[]) == []
    point = timed([10], [-1])[0]
    assert point.x_px == 10.
    assert point.velocity_px_per_sec is None


def test_timed_list_and_disk_writebacks_preserve_prefix_and_all_point_fields(tmp_path):
    prefix = TemporalBallPoint(900, 36., None, None, court_x_m=1., court_y_m=2., confidence=.1,
                               state=BallState.OCCLUDED, source='reviewed_occlusion',
                               velocity_px_per_sec=(4., 5.), speed_kmh=10.)
    inputs, rows = candidates([0, None, 100]), source_rows([0, .01, .1])
    options = dict(max_prediction_gap=0, max_interpolation_gap=2)
    expected = tracker(**options).track_video_candidates(inputs, fps=25., source_timing=rows)
    with TemporalBallPointSequence(tmp_path) as sink:
        sink.append(prefix)
        assert tracker(**options).track_video_candidates(inputs, fps=25., source_timing=rows, output=sink) is sink
        assert sink[0] == prefix
        assert [asdict(point) for point in sink[1:]] == [asdict(point) for point in expected]
    assert not list(tmp_path.glob('temporal_ball_points_*.sqlite3'))


def test_lazy_timing_reads_do_not_materialize_frames_or_retain_source_records(tmp_path):
    live = peak = reads = 0

    def release():
        nonlocal live
        live -= 1

    class Row(dict):
        pass

    class Rows(Sequence):
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            nonlocal live, peak, reads
            assert not isinstance(index, slice), 'Source timing must remain a lazy indexed sequence'
            if not 0 <= index < len(self):
                raise IndexError(index)
            reads += 1
            row = Row(source_pts=str(index), source_time_base='1/25', source_frame_index=index,
                      segment_id='camera-0', measurement_transition_allowed=index > 0)
            live += 1
            peak = max(peak, live)
            weakref.finalize(row, release)
            return row

    class Candidates(Sequence):
        def __len__(self):
            return 1000

        def __getitem__(self, index):
            assert not isinstance(index, slice)
            if not 0 <= index < len(self):
                raise IndexError(index)
            return [BallObservation(10. + index * .01, 30., .9)]

    with TemporalBallPointSequence(tmp_path) as sink:
        tracker().track_video_candidates(Candidates(), fps=25., source_timing=Rows(), output=sink)
        assert len(sink) == 1000
        assert sink[-1].velocity_px_per_sec == pytest.approx((.25, 0.))
    assert peak < 16
    assert live == 0
    assert reads < 40000


def test_adapter_without_timing_retains_original_tracker_keyword_contract():
    class LegacyTracker:
        def track_video_candidates(self, inputs, fps, frame_size, *, output):
            for index, _ in enumerate(inputs):
                output.append(TemporalBallPoint(index, index / fps, None, None))
            return output

    rows = [{0: dict(measurement_transition_allowed=index > 0, timestamp_seconds=index / 25)} for index in range(3)]
    points = track_candidate_intervals(LegacyTracker(), [[], [], []], rows, 25., (320, 240))
    assert [point.frame_index for point in points] == [0, 1, 2]


def test_opted_adapter_keeps_exact_source_provenance_and_camera_reset_with_disk_sink(tmp_path):
    from src.pipeline.phase6_pipeline import trajectory_export_records, _CandidateSourceTimingWindow

    rows = source_rows([100, 100.05, 100.1, 100.15])
    for index, row in enumerate(rows):
        row.update(timestamp_seconds=index / 25, source_timestamp_seconds=100 + index * .05)
        row['segment_id'] = 'first' if index < 2 else 'second'
        row['measurement_transition_allowed'] = index in (1, 3)
    frames = [{0: row} for row in rows]
    with TemporalBallPointSequence(tmp_path) as sink:
        result = track_candidate_intervals(tracker(), candidates([0, 10, 20, 30]), frames, 25., (320, 240),
                                           output=sink, use_source_timing=True)
        assert result is sink
        assert [point.velocity_px_per_sec for point in sink] == [None, (200., 0.), None, (200., 0.)]
        exported = list(trajectory_export_records(sink, frames))
        assert exported[1]['source_pts'] == rows[1]['source_pts']
        assert exported[1]['timestamp_seconds'] == .04
        assert exported[1]['source_timestamp_seconds'] == 100.05
    window = _CandidateSourceTimingWindow(frames, 0, 4)
    assert not isinstance(window[::-1][1::2], list)
    assert list(window[::-1][1::2]) == [rows[2], rows[0]]
