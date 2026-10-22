"""Private native export sets from actual freshly corrected rule/shot owners.

The measurements and participant reviews are explicit synthetic fixtures.
These checks establish semantic export integrity and bounded ownership, never
detector, calibration, speed, long-match, or live accuracy qualification.
"""
from contextlib import contextmanager
from dataclasses import replace
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import pytest

from tests.test_corrected_rule_rebuild import build_sealed_base, unchanged
from tests.test_corrected_shot_rebuild import (
    advance_equivalent_generation, build_shot_base, no_model_imports,
)
from tests.test_corrected_forecasts import (
    build_completed_points_base, fresh_shots, republish_rule_source,
)


OBSERVATIONS = []
NATIVE_FILES = {
    'point_analytics.json', 'rallies.json', 'match_analytics.json',
    'serve_executions.json', 'shot_play_scope.json', 'scoring_events.json',
    'score_history.json', 'match_state.json', 'shot_events.json',
    'line_calls.json', 'match_events.json', 'metrics.json',
    'native_rule_replay.json', 'point_timeline_manifest.json',
    'corrected_forecasts.json', 'correction_request.json',
    'correction_audit.json', 'corrected_export_manifest.json',
    'source.mp4', 'landing_zones.json',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digests(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(directory).iterdir() if path.is_file()}


@contextmanager
def fresh_forecast(fixture, *, request=None):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            yield rebuilt, shots, forecast


def export(rebuilt, shots, forecast, scratch):
    from src.scoring.corrected_exports import write_corrected_exports
    return write_corrected_exports(rebuilt, shots, forecast, scratch)


def strict_point_load(result, fixture, *, authority=True):
    from src.api.point_timeline_contract import load_point_analysis
    from src.api.result_storage import ResultStorage
    with ResultStorage(fixture.scratch) as storage:
        proof = load_point_analysis(result.directory, 'fixture',
            frame_count=fixture.snapshot.frames, authoritative=authority, storage=storage)
        assert proof['analysisRevision'] == 1 and proof['scoringRevision'] == 0
        assert proof['sourceSha256'] == fixture.source_sha
        # Return detached small fixture evidence only, never owned lazy views.
        return {**proof, 'points': list(proof['points']),
                'rallies': list(proof['rallies']),
                'serveExecutions': list(proof['serveExecutions'])}


def assert_export_set(result, fixture):
    assert isinstance(result.directory, Path)
    assert result.directory.parent == fixture.scratch.resolve()
    assert result.directory != fixture.scratch.resolve()
    assert set(result.files) == NATIVE_FILES
    assert set(digests(result.directory)) == NATIVE_FILES
    assert all(path == result.directory/name for name, path in result.files.items())
    with pytest.raises(TypeError):
        result.files['injected.json'] = result.directory/'injected.json'
    seal = read(result.files['corrected_export_manifest.json'])
    assert seal['analysis_id'] == 'fixture' and seal['analysis_revision'] == 1
    assert seal['base_generation'] == fixture.snapshot.generation
    assert seal['source_sha256'] == fixture.source_sha
    assert seal['published'] is False and seal['independent_accuracy_validated'] is False
    assert seal['human_review_dependent'] is True
    assert seal['status'] == 'PRIVATE_PENDING_RENDER_AND_PUBLICATION'
    assert set(seal['artifacts']) == NATIVE_FILES-{'corrected_export_manifest.json'}
    for name, metadata in seal['artifacts'].items():
        path = result.files[name]
        assert metadata == {'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    source = result.files['source.mp4']
    assert source.read_bytes() == fixture.snapshot.resolve_file('source.mp4').read_bytes()
    assert source.stat().st_nlink == 1
    return seal


def test_double_fault_exports_fresh_native_semantics_and_causal_history(tmp_path):
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    fixture = build_shot_base(tmp_path)
    assert fixture.original['overview']['live_shots_recorded'] == 2
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        result = export(rebuilt, shots, forecast, fixture.scratch)
        assert_export_set(result, fixture)
        point = strict_point_load(result, fixture)
        assert point['summary']['totalPointsPlayed'] == 1
        assert point['summary']['totalRallies'] == 0 and point['rallies'] == []
        assert [row['status'] for row in point['serveExecutions']] == ['DOUBLE_FAULT']
        assert point['serveByPlayer']['player_1']['doubleFaults'] == 1
        assert point['serveByPlayer']['player_1']['secondServesIn'] == 0
        assert point['serveByPlayer']['player_1']['firstServePercentage'] is None
        members = read(result.files['shot_play_scope.json'])['shots']
        assert {row['play_scope'] for row in members} == {'DOUBLE_FAULT', 'DEAD_AFTER_PLAY'}
        assert all(row['rally_uid'] is None for row in members)
        analytics = read(result.files['match_analytics.json'])
        assert analytics['overview']['live_shots_recorded'] == 0
        assert analytics['overview']['total_shots_recorded'] == 2
        assert analytics['ball_landing_heatmap'] == []
        calls = read(result.files['line_calls.json'])['line_calls']
        assert len(calls) == 1 and calls[0]['event_id'] == 2
        assert calls[0]['decision'] == 'SERVE_FAULT'
        current = read(result.files['corrected_forecasts.json'])
        assert current['current']['sample_count'] == 1
        assert current['current']['serve_estimates'][0]['probability'] == .4
        assert len(current['history']) == 7
        assert current['history'][3]['forecast']['sample_count'] == 0
        assert current['history'][4]['forecast']['sample_count'] == 1
        assert read(result.files['correction_request.json']) == fixture.request
        replay_scratch = fixture.scratch/'roundtrip'
        replay_scratch.mkdir()
        with reproduce_rule_timeline(result.files['native_rule_replay.json'], replay_scratch,
                artifact_sha256=hashlib.sha256(result.files['native_rule_replay.json'].read_bytes()).hexdigest(),
                source_sha256=fixture.source_sha, analysis_id='fixture', analysis_revision=1) as replay:
            assert replay.engine.revision == 0 and replay.owner.revision == 1
            assert replay.engine.state.points_p2 == 2
            assert len(replay.owner.rows('points', status='COMPLETED')) == 1
            assert replay.owner.event(3)['outcome_type'] == 'DEAD_BALL_IGNORED'
            assert len(replay.shots) == 2 and replay.shots[1].is_dead_ball
    # Exported primitive inventory remains readable after all native owners close.
    assert_export_set(result, fixture)
    unchanged(fixture)
    OBSERVATIONS.append({'double_fault_export': {'original_live_shots': 2,
        'corrected_live_shots': 0, 'fresh_double_faults': 1,
        'posterior_points': 1, 'server_probability': .4,
        'native_revision_roundtrip': 1, 'independent_accuracy_validated': False}})


@pytest.mark.parametrize('scenario', ['authority_off', 'unanchored', 'reviewed_let', 'unknown'])
def test_missing_context_and_review_boundaries_stay_unavailable_in_exports(tmp_path, scenario):
    if scenario in {'reviewed_let', 'unknown'}:
        fixture = republish_rule_source(build_sealed_base(tmp_path, reviewed_let=True))
    else:
        fixture = republish_rule_source(build_shot_base(tmp_path),
            authority=scenario != 'authority_off', prepend_unknown=scenario == 'unanchored')
    request = copy.deepcopy(fixture.request)
    if scenario == 'unknown':
        request['initial_state_amendment'] = None
        original = fixture.document['inputs'][1]
        replacement = copy.deepcopy(original['source_input'])
        replacement.update(event_type='POINT_OUTCOME_UNKNOWN',
                           reviewed_evidence_ref='SYNTHETIC_NEW_CONSEQUENTIAL_REVIEW')
        request['input_amendments'] = [{'event_id': 3,
            'expected_input_fingerprint': original['input_fingerprint'],
            'replacement_source_input': replacement}]
    with fresh_forecast(fixture, request=request) as (rebuilt, shots, forecast):
        result = export(rebuilt, shots, forecast, fixture.scratch)
        assert_export_set(result, fixture)
        proof = strict_point_load(result, fixture, authority=scenario != 'authority_off')
        saved = read(result.files['corrected_forecasts.json'])
        if scenario == 'reviewed_let':
            assert saved['current']['sample_count'] == 1
            assert proof['serveByPlayer']['player_1']['serviceLets'] == 1
            assert proof['serveByPlayer']['player_1']['doubleFaults'] == 1
        else:
            assert saved['current']['status'] == 'unavailable'
            assert saved['current']['sample_count'] == 0
            assert saved['current']['probabilities'] is None
        if scenario == 'authority_off':
            assert proof['summary']['totalPointsPlayed'] is None
            assert proof['points'] == [] and proof['serveExecutions'] == []
            assert read(result.files['match_events.json'])['events'] == []
        if scenario == 'unknown':
            assert proof['summary']['totalPointsPlayed'] == 0
    unchanged(fixture)


def test_export_reduces_authoritative_owners_again_instead_of_mutable_presentation(tmp_path):
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        shots.match_analytics['overview']['total_points_played'] = 999
        shots.match_analytics['serve_analytics']['by_player']['1']['double_faults'] = 999
        detached = forecast.current
        detached['sample_count'] = 999
        detached['serve_estimates'][0]['probability'] = .999
        result = export(rebuilt, shots, forecast, fixture.scratch)
        assert read(result.files['match_analytics.json'])['overview']['total_points_played'] == 1
        assert read(result.files['match_analytics.json'])['serve_analytics']['by_player']['1']['double_faults'] == 1
        assert read(result.files['corrected_forecasts.json'])['current']['sample_count'] == 1
        assert read(result.files['corrected_forecasts.json'])['current']['serve_estimates'][0]['probability'] == .4
    unchanged(fixture)


def test_export_uses_actual_admitted_command_instead_of_mutable_audit(tmp_path):
    fixture = build_shot_base(tmp_path)
    expected = copy.deepcopy(fixture.request)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        rebuilt.audit['request'].update(reviewer_id='FORGED REVIEWER', reason='FORGED REASON')
        rebuilt.audit.update(published=True, independent_accuracy_validated=True)
        result = export(rebuilt, shots, forecast, fixture.scratch)
        assert read(result.files['correction_request.json']) == expected
        audit = read(result.files['correction_audit.json'])
        assert audit['published'] is False and audit['independent_accuracy_validated'] is False
        canonical = json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert audit['request_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    unchanged(fixture)


def test_positive_landing_is_bound_to_the_same_fresh_event_export(tmp_path, monkeypatch):
    from tests.test_corrected_landing_exports import reviewed_base
    fixture = reviewed_base(tmp_path, monkeypatch, dead_bounce=True)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        result = export(rebuilt, shots, forecast, fixture.scratch)
        assert_export_set(result, fixture)
        raw = read(result.files['landing_zones.json'])
        assert len(raw['records']) == 2 and set(raw['calibrations']) == {'frame_4'}
        events = read(result.files['match_events.json'])['events']
        assert events[1]['evidence']['analysis_revision'] == 1
        assert events[1]['evidence']['line_call'] == 'out'
        assert events[2]['evidence']['active_play'] is False
        assert 'calibration_id' not in events[2]['evidence']
        assert {row['event_id'] for row in raw['records']} == {2}
        strict_point_load(result, fixture)
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['point_manifest', 'replay', 'landing'])
def test_strict_validation_failure_cannot_leave_an_export_seal(tmp_path, monkeypatch, damage):
    from src.scoring import corrected_exports as module
    fixture = build_shot_base(tmp_path)
    function_name = {'point_manifest': 'publish_point_timeline_manifest',
                     'replay': 'write_rule_replay', 'landing': 'write_corrected_landing_zones'}[damage]
    original = getattr(module, function_name)
    def corrupt(*args, **kwargs):
        value = original(*args, **kwargs)
        if damage == 'point_manifest':
            path = Path(args[0])/'point_timeline_manifest.json'
            doc = read(path)
            doc['artifacts']['point_analytics.json']['sha256'] = '0'*64
        elif damage == 'landing':
            path = Path(args[2])/'landing_zones.json'
            doc = read(path)
            doc['analysis_revision'] = 999
        else:
            path = Path(args[0])
            doc = read(path)
            doc['final_state']['fields']['points_p2'] = 999
        path.write_text(json.dumps(doc), encoding='utf-8')
        return value
    monkeypatch.setattr(module, function_name, corrupt)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        with pytest.raises(ValueError):
            export(rebuilt, shots, forecast, fixture.scratch)
        assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
    unchanged(fixture)


@pytest.mark.parametrize('scenario', ['game_end', 'tiebreak_7', 'tiebreak_10', 'match_end'])
def test_export_preserves_preaward_server_raw_tiebreak_and_terminal_result(tmp_path, scenario):
    from src.scoring.match_state import CompletedSetScore
    from src.scoring.match_state_codec import decode_match_state, encode_match_state
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = request['initial_state_amendment']['replacement_initial_state']['fields']
    if scenario == 'game_end':
        initial['points_p2'] = 3
    elif scenario.startswith('tiebreak'):
        target = int(scenario.rsplit('_', 1)[1])
        initial.update(games_p1=6, games_p2=6, points_p1=3, points_p2=2,
            tie_break_active=True, tie_break_points_p1=target-1,
            tie_break_points_p2=target-2, tie_break_points_to_win=target,
            initial_server_id=1 if target == 7 else 2)
    else:
        state = decode_match_state(request['initial_state_amendment']['replacement_initial_state'])
        state.sets_won_p2, state.current_set = 1, 2
        state.games_p2, state.points_p2 = 5, 3
        state.completed_sets = [CompletedSetScore(1, 0, 6, winner_id=2)]
        request['initial_state_amendment']['replacement_initial_state'] = encode_match_state(state)
    with fresh_forecast(fixture, request=request) as (rebuilt, shots, forecast):
        result = export(rebuilt, shots, forecast, fixture.scratch)
        proof = strict_point_load(result, fixture)
        assert proof['summary']['totalPointsPlayed'] == 1
        assert proof['serveByPlayer']['player_1']['doubleFaults'] == 1
        saved = read(result.files['corrected_forecasts.json'])
        assert saved['current']['sample_count'] == 1
        assert saved['current']['serve_estimates'][0]['points_played'] == 1
        assert saved['current']['serve_estimates'][1]['points_played'] == 0
        replay = read(result.files['native_rule_replay.json'])
        final_fields = replay['final_state']['fields']
        if scenario == 'game_end':
            assert final_fields['server_id'] == 2 and final_fields['games_p2'] == 1
        elif scenario.startswith('tiebreak'):
            assert (final_fields['points_p1'], final_fields['points_p2']) == (3, 2)
            assert final_fields['tie_break_points_to_win'] == target
            assert final_fields['tie_break_points_p1'] == final_fields['tie_break_points_p2'] == target-1
        else:
            assert proof['summary']['matchComplete'] is True
            assert proof['summary']['winnerId'] == 2
            assert saved['current']['status'] == 'confirmed_result'
            assert saved['current']['winner_id'] == 2 and saved['current']['probabilities'] is None
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['event_alias', 'forecast_source', 'forecast_history',
                                   'forecast_current', 'forecast_audit', 'base_generation'])
def test_export_requires_exact_fresh_owner_bindings(tmp_path, damage):
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        original_events = shots.events
        argument = forecast
        if damage == 'event_alias':
            object.__setattr__(shots, 'events', list(shots.events))
        elif damage == 'forecast_source':
            argument = replace(forecast, _rebuilt=object())
        elif damage == 'forecast_history':
            argument = replace(forecast, _history=list(forecast.history))
        elif damage == 'forecast_current':
            forged = forecast.current
            forged['sample_count'] = 999
            argument = replace(forecast, _current_json=json.dumps(forged))
        elif damage == 'forecast_audit':
            forged = forecast.audit
            forged.update(analysis_id='different-analysis', source_sha256='0'*64)
            argument = replace(forecast, _audit_json=json.dumps(forged))
        else:
            argument = replace(forecast, base_generation='0'*64)
        try:
            with pytest.raises((ValueError, TypeError, AttributeError), match='owner|bound|source|generation|alias|fresh|forecast|event'):
                export(rebuilt, shots, argument, fixture.scratch)
        finally:
            object.__setattr__(shots, 'events', original_events)
        assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
    unchanged(fixture)


def test_export_rejects_closed_forecast_owner_without_a_final_seal(tmp_path):
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            pass
        with pytest.raises((ValueError, RuntimeError), match='open|owner|closed'):
            export(rebuilt, shots, forecast, fixture.scratch)
        assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
    unchanged(fixture)


def test_changed_sealed_source_is_rejected_before_an_export_set_is_sealed(tmp_path):
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        path = fixture.snapshot.resolve_file('source.mp4')
        original = path.read_bytes()
        try:
            path.write_bytes(b'changed-synthetic-source')
            with pytest.raises(ValueError, match='source|artifact|bytes|hash|different|changed'):
                export(rebuilt, shots, forecast, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
        finally:
            path.write_bytes(original)
    unchanged(fixture)


@pytest.mark.parametrize('destination', ['analysis', 'different_scratch', 'traversal'])
def test_export_stage_must_belong_to_the_guarded_forecast_scratch(tmp_path, destination):
    fixture = build_shot_base(tmp_path)
    elsewhere = tmp_path/'other_scratch'
    elsewhere.mkdir()
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        target = {'analysis': fixture.directory, 'different_scratch': elsewhere,
                  'traversal': fixture.scratch/'..'/fixture.scratch.name}[destination]
        with pytest.raises(ValueError, match='scratch|destination|directory|parent|guard|traversal'):
            export(rebuilt, shots, forecast, target)
        assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
    unchanged(fixture)


def test_successive_exports_are_independent_and_failure_preserves_the_previous_set(tmp_path, monkeypatch):
    from src.scoring import corrected_exports as module
    fixture = build_shot_base(tmp_path)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        first = export(rebuilt, shots, forecast, fixture.scratch)
        first_hashes = digests(first.directory)
        second = export(rebuilt, shots, forecast, fixture.scratch)
        assert first.directory != second.directory
        assert_export_set(second, fixture)
        original = module.write_rule_replay
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('SYNTHETIC_MID_EXPORT_FAILURE')
        monkeypatch.setattr(module, 'write_rule_replay', fail)
        with pytest.raises(RuntimeError, match='SYNTHETIC_MID_EXPORT_FAILURE'):
            export(rebuilt, shots, forecast, fixture.scratch)
        assert digests(first.directory) == first_hashes
        assert set(fixture.scratch.rglob('corrected_export_manifest.json')) == {
            first.files['corrected_export_manifest.json'], second.files['corrected_export_manifest.json']}
        assert_export_set(first, fixture)
    unchanged(fixture)


def test_generation_advance_during_export_never_seals_or_publishes_the_candidate(tmp_path, monkeypatch):
    from src.api.analysis_generation import resolve_generation
    from src.scoring import corrected_exports as module
    fixture = build_shot_base(tmp_path)
    advanced = []
    with pytest.raises(ValueError, match='current generation|base'):
        with fresh_forecast(fixture) as (rebuilt, shots, forecast):
            original = module.write_rule_replay
            def advance(*args, **kwargs):
                value = original(*args, **kwargs)
                advanced.append(advance_equivalent_generation(fixture))
                return value
            monkeypatch.setattr(module, 'write_rule_replay', advance)
            with pytest.raises(ValueError, match='current generation|base'):
                export(rebuilt, shots, forecast, fixture.scratch)
            assert not list(fixture.scratch.rglob('corrected_export_manifest.json'))
    assert len(advanced) == 1
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert digests(fixture.snapshot.files_directory) == fixture.manifest


def test_growing_completed_points_bounds_actual_export_history_and_strict_reader_memory(tmp_path):
    from src.api.result_storage import DiskSeries, ResultStorage
    measurements = []
    for count in (8, 128):
        parent = tmp_path/str(count)
        parent.mkdir()
        fixture = build_completed_points_base(parent, point_count=count)
        with fresh_forecast(fixture) as (rebuilt, shots, forecast):
            assert len(rebuilt.owner.rows('points', status='COMPLETED')) == count
            assert len(forecast.history) == 1+6*count
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            try:
                result = export(rebuilt, shots, forecast, fixture.scratch)
                gc.collect()
                retained, peak = tracemalloc.get_traced_memory()
                measurements.append({'completed_points': count, 'accepted_inputs': 3*count,
                    'forecast_history_rows': 1+6*count,
                    'retained_python_bytes_after_export': retained, 'peak_python_bytes': peak,
                    'elapsed_seconds': time.perf_counter()-started,
                    'exported_bytes': sum(path.stat().st_size for path in result.files.values())})
            finally:
                tracemalloc.stop()
            assert_export_set(result, fixture)
            with ResultStorage(fixture.scratch) as storage:
                saved = storage.read_json(result.files['corrected_forecasts.json'])
                assert saved['current']['sample_count'] == count
                assert len(saved['history']) == 1+6*count
                # The bounded parser keeps at most 64 array members in Python,
                # then spills the full sequence to its owned disk store.
                if len(saved['history']) <= 64:
                    assert isinstance(saved['history'], list)
                else:
                    assert isinstance(saved['history'], DiskSeries)
        unchanged(fixture)
    small, large = measurements
    assert large['retained_python_bytes_after_export']-small['retained_python_bytes_after_export'] < 1024*1024
    assert large['peak_python_bytes']-small['peak_python_bytes'] < 3*1024*1024
    OBSERVATIONS.append({'completed_point_export_capacity': measurements,
        'includes': 'Actual native semantic writers, fresh causal history export, source copy, strict point reader, seals',
        'excludes': 'Fixture and corrected rule/shot/forecast owner construction, SQLite native and numerical-library allocations',
        'independent_accuracy_validated': False})
