"""Actual owned causal presentation/report behavior from synthetic native rules.

No checkpoint, video inference, calibration qualification or live system runs.
"""
import copy
from dataclasses import replace
import gc
import json
import re
import time
import tracemalloc

import pytest

from src.api.result_storage import DiskSeries, ResultStorage
from src.scoring.corrected_forecast_presentation import build_corrected_forecast_presentation
from tests.test_corrected_exports import fresh_forecast
from tests.test_corrected_forecasts import (
    build_completed_points_base, fresh_shots, republish_rule_source,
)
from tests.test_corrected_rule_rebuild import build_sealed_base, unchanged
from tests.test_corrected_shot_rebuild import build_shot_base, no_model_imports


OBSERVATIONS = []


def test_actual_double_fault_prefix_is_copied_into_result_owner_and_survives_forecast_close(tmp_path):
    fixture = build_shot_base(tmp_path)
    with ResultStorage(fixture.scratch) as storage:
        with fresh_forecast(fixture) as (rebuilt, shots, forecast):
            value = build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
            assert isinstance(value['history'], DiskSeries) and value['history']._storage is storage
            assert value['history'] is not forecast.history
            assert value['status'] == 'UNQUALIFIED' and value['calibrationStatus'] == 'not_calibrated'
            assert value['independentAccuracyValidated'] is False and value['humanReviewDependent'] is True
            assert value['analysisId'] == 'fixture' and value['analysisRevision'] == 1
            assert value['sourceSha256'] == fixture.source_sha
            assert value['baseGeneration'] == fixture.snapshot.generation
            assert value['historyPolicy'] == 'INITIAL_AND_BEFORE_AFTER_EVERY_ACCEPTED_EVENT'
            assert value['current'] == forecast.current == value['history'][-1]['forecast']
            assert len(value['history']) == 7
            before, after, dead_after = value['history'][3], value['history'][4], value['history'][6]
            assert before['forecast']['sample_count'] == 0
            assert after['forecast']['sample_count'] == dead_after['forecast']['sample_count'] == 1
            assert before['uses_points_through_event_id'] == 1
            assert after['uses_points_through_event_id'] == 2
            assert before['event_uid'] == rebuilt.owner.event(2)['uid']
            assert before['source_pts'] == str(10**18+400) and before['source_time_base'] == '1/3000'
            assert after['forecast']['serve_estimates'][0]['probability'] == .4
            expected = list(forecast.history)
        assert list(value['history']) == expected
        saved = json.loads(''.join(storage.json_chunks({'nativeForecast': value})))
        assert saved['nativeForecast']['history'] == expected
        storage.set_root({'nativeForecast': value})
        assert storage.root()['nativeForecast']['current'] == value['current']
        owned = value['history']
    with pytest.raises(RuntimeError):
        owned[0]
    unchanged(fixture)


@pytest.mark.parametrize('scenario', ['authority_off', 'reviewed_let', 'tiebreak7', 'tiebreak10', 'terminal'])
def test_actual_native_boundary_and_terminal_states_keep_their_original_causal_meaning(tmp_path, scenario):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.scoring.match_state import CompletedSetScore
    from src.scoring.match_state_codec import decode_match_state, encode_match_state
    fixture = (republish_rule_source(build_sealed_base(tmp_path, reviewed_let=True))
               if scenario == 'reviewed_let' else build_shot_base(tmp_path))
    if scenario == 'authority_off':
        fixture = republish_rule_source(fixture, authority=False)
    request = copy.deepcopy(fixture.request)
    fields = request['initial_state_amendment']['replacement_initial_state']['fields']
    if scenario.startswith('tiebreak'):
        target = 7 if scenario == 'tiebreak7' else 10
        fields.update(games_p1=6, games_p2=6, points_p1=3, points_p2=2,
            tie_break_active=True, tie_break_points_p1=target-1, tie_break_points_p2=target-2,
            tie_break_points_to_win=target, initial_server_id=1 if target == 7 else 2)
    elif scenario == 'terminal':
        initial = decode_match_state(request['initial_state_amendment']['replacement_initial_state'])
        initial.sets_won_p2, initial.current_set = 1, 2
        initial.games_p2, initial.points_p2 = 5, 3
        initial.completed_sets = [CompletedSetScore(1, 0, 6, winner_id=2)]
        request['initial_state_amendment']['replacement_initial_state'] = encode_match_state(initial)
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            with ResultStorage(fixture.scratch) as storage:
                value = build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
                assert list(value['history']) == list(forecast.history)
                if scenario == 'authority_off':
                    assert len(value['history']) == 1
                    assert value['current']['status'] == 'unavailable'
                    assert value['current']['probabilities'] is None
                elif scenario == 'terminal':
                    assert value['current']['status'] == 'confirmed_result'
                    assert value['current']['winner_id'] == 2 and value['current']['probabilities'] is None
                    from src.api.reporting import native_forecast_report_chunks
                    html = ''.join(native_forecast_report_chunks({'nativeForecast': value}))
                    assert 'Confirmed match result: Player 2' in html and 'separate from a forecast' in html
                elif scenario == 'reviewed_let':
                    let = next(row for row in value['history'] if row['event_id'] == 3 and row['phase'] == 'AFTER_EVENT')
                    assert let['forecast']['sample_count'] == 0
                    assert value['current']['sample_count'] == 1
                else:
                    assert value['current']['sample_count'] == 1
                    assert value['current']['status'] == 'uncalibrated'
    unchanged(fixture)


@pytest.mark.parametrize('damage', [
    'phase', 'source_pts', 'source_base', 'source_frame', 'event_uid', 'frontier',
    'frame', 'normalized_clock', 'future_sample_count', 'future_probabilities',
])
def test_self_consistent_native_row_damage_cannot_enter_the_presentation(tmp_path, damage):
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        original = forecast.history[3]
        row = copy.deepcopy(original)
        if damage == 'phase': row['phase'] = 'AFTER_EVENT'
        elif damage == 'source_pts': row['source_pts'] = str(int(row['source_pts'])+1)
        elif damage == 'source_base': row['source_time_base'] = '1/6000'
        elif damage == 'source_frame': row['source_frame_index'] += 1
        elif damage == 'event_uid': row['event_uid'] = 'event_'+'b'*64
        elif damage == 'frontier': row['uses_points_through_event_id'] = 2
        elif damage == 'frame': row['frame_index'] = 5
        elif damage == 'normalized_clock': row['timestamp_seconds'] += .01
        elif damage == 'future_sample_count': row['forecast']['sample_count'] = 1
        else: row['forecast']['probabilities'] = copy.deepcopy(forecast.history[4]['forecast']['probabilities'])
        forecast.owner._execute('UPDATE history SET payload=? WHERE position=4',
                                (json.dumps(row, allow_nan=False),))
        try:
            with ResultStorage(fixture.scratch) as storage:
                with pytest.raises(ValueError, match='native causal history'):
                    build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
        finally:
            forecast.owner._execute('UPDATE history SET payload=? WHERE position=4',
                                    (json.dumps(original, allow_nan=False),))
    unchanged(fixture)


def test_closed_forecast_foreign_history_and_closed_result_storage_are_rejected(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.utils.forecast_sequence import ForecastSequence
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        with ResultStorage(fixture.scratch) as storage:
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                with ForecastSequence(fixture.scratch) as foreign:
                    foreign.history.append(forecast.history[0])
                    foreign.seal()
                    with pytest.raises(ValueError, match='bound forecast|owners|generation'):
                        build_corrected_forecast_presentation(rebuilt, shots,
                            replace(forecast, _history=foreign.history), storage)
                closed = ResultStorage(fixture.scratch)
                with closed:
                    pass
                with pytest.raises(RuntimeError, match='not open'):
                    build_corrected_forecast_presentation(rebuilt, shots, forecast, closed)
            with pytest.raises(RuntimeError):
                build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
    unchanged(fixture)


def test_report_streams_actual_native_history_with_separate_actual_result_and_escaping(tmp_path):
    from src.api.reporting import readable_report_chunks, native_forecast_report_chunks
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        with ResultStorage(fixture.scratch) as storage:
            value = build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
            value['analysisId'] = '<analysis & "source">'
            value['modelSpec']['prior']['source'] = '<script>NOT_EXECUTABLE</script>'
            value['current']['reason'] = '<untrusted & "reason">'
            result = {'nativeForecast': value, 'summary': {'id': 'fixture', 'label': 'Synthetic report',
                'analyzedAt': '2026-10-18', 'frames': 12, 'fps': 30., 'durationSeconds': .4}}
            html = ''.join(readable_report_chunks(result))
            assert 'Statistical forecasts · uncalibrated' in html
            assert 'no qualified match-winner prediction' in html
            assert '&lt;analysis &amp; &quot;source&quot;&gt;' in html
            assert '&lt;script&gt;NOT_EXECUTABLE&lt;/script&gt;' in html and '<script>' not in html
            assert '&lt;untrusted &amp; &quot;reason&quot;&gt;' in html
            assert str(10**18+400) in html and 'INITIAL' in html and 'BEFORE_EVENT' in html
            assert html.count('(uncalibrated)') == 7
            terminal = copy.copy(value)
            terminal['current'] = {**value['current'], 'status': 'confirmed_result',
                'winner_id': 2, 'probabilities': None}
            terminal_html = ''.join(native_forecast_report_chunks({'nativeForecast': terminal}))
            assert 'Confirmed match result: Player 2' in terminal_html
            assert 'separate from a forecast' in terminal_html
    unchanged(fixture)


class GeneratedHistory(DiskSeries):
    """No resident rows or slicing, but the actual report's owned-series protocol."""
    def __init__(self, count, missing):
        self.count, self.missing, self.reads = count, missing, 0
    def __len__(self):
        return self.count
    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError('A report cannot slice growing forecast history')
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        missing = index in self.missing
        return {'timestamp_seconds': index/30., 'forecast': {
            'status': 'unavailable' if missing else 'uncalibrated',
            'probabilities': None if missing else {'match': {'player_1': .6, 'player_2': .4}}}}
    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def test_forecast_curve_bounds_display_bins_and_preserves_unavailable_gaps():
    from src.api.reporting import _native_forecast_curve
    history = GeneratedHistory(1201, {650})
    svg = _native_forecast_curve(history)
    curves = re.findall(r'<path d="([ML0-9., ]+)" fill="none" stroke="#26785d"', svg)
    assert len(curves) == 2  # A missing prefix is a gap, never an interpolated segment.
    assert sum(len(re.findall(r'[ML]', path)) for path in curves) <= 600
    assert history.reads <= 1203
    assert 'at most 600 display bins' in svg and 'uncalibrated' in svg
    empty = _native_forecast_curve(GeneratedHistory(1201, set(range(1201))))
    assert 'curve unavailable' in empty and '<svg' not in empty


def test_private_correction_report_escapes_review_metadata_and_preserves_authorization_limits():
    from src.api.reporting import readable_report_chunks
    correction = {'schemaVersion': '1.0', 'status': 'PRIVATE_PREPARED', 'analysisRevision': 1,
        'baseGeneration': 'b'*64, 'sourceSha256': 'a'*64, 'correctedAt': '2026-10-18T10:00:00Z',
        'review': {'correctionId': '<correction & "id">', 'reviewerId': '<local reviewer>',
            'reason': '<script>NOT_EXECUTABLE</script>', 'evidenceRef': 'source & review',
            'inputAmendmentCount': 1, 'initialStateAmended': True, 'humanReviewDependent': True},
        'authorizationStatus': 'PRIVATE_LOCAL_REVIEW_NOT_AUTHENTICATED',
        'independentAccuracyValidated': False, 'published': False}
    result = {'analysisCorrection': correction, 'summary': {'id': 'fixture', 'label': 'Synthetic review',
        'analyzedAt': '2026-10-18', 'frames': 1, 'fps': 30., 'durationSeconds': 1/30.}}
    html = ''.join(readable_report_chunks(result))
    assert 'Private correction review' in html
    assert 'not established authenticated owner authorization' in html
    assert 'PRIVATE_LOCAL_REVIEW_NOT_AUTHENTICATED' in html
    assert '&lt;correction &amp; &quot;id&quot;&gt;' in html and '&lt;local reviewer&gt;' in html
    assert '&lt;script&gt;NOT_EXECUTABLE&lt;/script&gt;' in html and '<script>' not in html
    assert 'source &amp; review' in html and 'Independent accuracy validated</th><td>False' in html
    assert 'Published</th><td>False' in html


def test_owned_presentation_and_report_bound_python_retention_as_completed_points_grow(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.api.reporting import native_forecast_report_chunks
    measurements = []
    for points in (8, 128):
        case = tmp_path/str(points)
        case.mkdir()
        fixture = build_completed_points_base(case, point_count=points)
        with fresh_shots(fixture) as (rebuilt, shots):
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                with ResultStorage(fixture.scratch) as storage:
                    gc.collect()
                    tracemalloc.start()
                    start = time.perf_counter()
                    try:
                        value = build_corrected_forecast_presentation(rebuilt, shots, forecast, storage)
                        report_bytes = sum(len(chunk.encode('utf-8')) for chunk in
                                           native_forecast_report_chunks({'nativeForecast': value}))
                        gc.collect()
                        retained, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    measurements.append({'completed_points': points,
                        'accepted_inputs': len(rebuilt.owner.rows('transitions')),
                        'history_rows': len(value['history']), 'current_samples': value['current']['sample_count'],
                        'python_retained_bytes': retained, 'python_peak_bytes': peak,
                        'seconds': time.perf_counter()-start, 'streamed_html_bytes': report_bytes,
                        'scope': 'INDEPENDENT_NATIVE_PREFIX_RECONSTRUCTION_OWNED_PRESENTATION_REPORT_STREAM',
                        'excludes': ['fixture_preparation', 'prior_native_owners', 'native_sqlite_cache',
                                     'native_numerical_allocations', 'independent_prediction_accuracy']})
                    assert len(value['history']) == 1+6*points
                    assert value['current']['sample_count'] == points
        unchanged(fixture)
    OBSERVATIONS.append({'actual_owned_forecast_presentation_capacity': measurements,
        'retained_growth_budget_bytes': 2*1024*1024, 'peak_growth_budget_bytes': 8*1024*1024})
    small, large = measurements
    assert large['python_retained_bytes']-small['python_retained_bytes'] < 2*1024*1024
    assert large['python_peak_bytes']-small['python_peak_bytes'] < 8*1024*1024
