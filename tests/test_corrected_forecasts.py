"""Causal native forecasts after fresh private rule and shot reconstruction.

Sealed measurements and reviewed roles are explicit synthetic fixtures. These
tests establish causality and ownership, never calibration or video accuracy.
"""
from contextlib import ExitStack, contextmanager
import copy
import gc
import hashlib
import json
import time
import tracemalloc

import pytest

from tests.test_corrected_rule_rebuild import (
    binding_amendment, build_sealed_base, canonical_sha, unchanged,
)
from tests.test_corrected_shot_rebuild import (
    advance_equivalent_generation, assert_only_parser_diagnostics, build_shot_base,
    no_model_imports,
)


OBSERVATIONS = []


@contextmanager
def fresh_shots(fixture, *, request=None):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
            request=fixture.request if request is None else request) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch) as shots:
            yield rebuilt, shots


def republish_rule_source(fixture, *, prepend_unknown=False, authority=True):
    """Regenerate actual original rule evidence for two missing-evidence cases.

No already accepted private ledger is edited. An UNKNOWN physical prelude
opens an unanchored point; a later valid service contact cannot retrospectively
prove the missing point boundary. Authority-off retains no accepted inputs.
"""
    import numpy as np
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.match_state_codec import decode_match_state
    from src.scoring.point_timeline import PointTimeline, process_timeline_event
    from src.scoring.rule_replay import write_rule_replay
    from src.scoring.rule_replay_source import _source_event
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.scoring_journal import ScoringJournal
    source_frames = [{0: row['interval'], 1: row['identity']}
                     for row in fixture.document['source_frames']]
    records = [copy.deepcopy(row['source_input']) for row in fixture.document['inputs']]
    for record in records:
        if record['reviewed_evidence_ref'] is None:
            record['evidence'] = {'fixture': 'EXPLICIT_SYNTHETIC_FORECAST_SOURCE'}
    if prepend_unknown:
        unknown = copy.deepcopy(records[0])
        unknown.update(event_id=0, event_type='UNKNOWN_EVENT', player_id=None,
                       evidence={'fixture': 'EXPLICIT_SYNTHETIC_UNANCHORED_PRELUDE'})
        records.insert(0, unknown)
    if not authority:
        records = []
    detections = json.loads(fixture.snapshot.resolve_file('detections.json').read_text(encoding='utf-8'))
    points = [TemporalBallPoint(index, index/30., *row['ball']['position_px'], confidence=.99,
                               state=BallState.DETECTED)
              for index, row in enumerate(detections['frames'])]
    path = fixture.directory / 'forecast_fixture_rule_source.json'
    initial = decode_match_state(fixture.document['initial_state'])
    with ScoringJournal(fixture.scratch) as journal, PointAnalysisSequence(fixture.scratch,
            'fixture', revision=0, source_fingerprint=fixture.source_sha) as owner:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
        line_engine = TennisLineCallEngine()
        for record in records:
            event, context = _source_event(record, source_frames, 'fixture', fixture.source_sha)
            process_timeline_event(event, context, engine, timeline, line_engine,
                ball_points=points, homography_matrix=np.diag([.01, .01, 1.]), frames=(),
                geometry_available=True)
        timeline.finish()
        write_rule_replay(path, engine, owner, source_frames,
                          authoritative_events_enabled=authority)
    files = {item.name: item for item in fixture.snapshot.files_directory.iterdir()
             if item.name not in {'result.json', 'summary.json'}}
    files['native_rule_replay.json'] = path
    with ResultStorage(fixture.directory) as storage:
        token = publish_analysis_generation(storage, {
            'summary': {'id': 'fixture', 'frames': len(source_frames)}, 'analysisRevision': 0,
        }, fixture.directory, files=files, analysis_revision=0,
            expected_generation=fixture.snapshot.generation)
    fixture.snapshot = resolve_generation(fixture.directory)
    fixture.document = json.loads(path.read_text(encoding='utf-8'))
    fixture.request.update(base_generation=token,
        base_rule_replay_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'))
    fixture.request['initial_state_amendment']['expected_initial_state_sha256'] = canonical_sha(fixture.document['initial_state'])
    fixture.marker = (fixture.directory / '.result/current.store').read_bytes()
    fixture.manifest = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                        for item in fixture.snapshot.files_directory.iterdir()}
    return fixture


def row_for(history, phase, event_id):
    matches = [row for row in history if row['phase'] == phase and row['event_id'] == event_id]
    assert len(matches) == 1
    return matches[0]


def build_completed_points_base(tmp_path, *, point_count, last_winner=None):
    """Actual physical rallies in a deuce game, with independent sealed assets.

    Alternating terminal second bounces IN/OUT award server/receiver points.
    The game remains at deuce and never ends as cardinality grows. The last
    winner can differ for a prefix-causality check; all earlier measurements
    and accepted inputs are identical. No accepted completed row is invented.
    """
    import numpy as np
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.events.event_detector import TennisEvent, EventType
    from src.line_calling.contact_refinement import BounceContactRefiner
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.match_state import MatchState, BallPlayState, PointState
    from src.scoring.match_state_codec import encode_match_state
    from src.scoring.point_timeline import (PointTimeline, ReviewedParticipantBinding,
        candidate_event_context, process_timeline_event)
    from src.scoring.rule_replay import write_rule_replay
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.scoring_journal import ScoringJournal
    from tests.test_correction_sources import write_observation_assets

    assert type(point_count) is int and point_count > 0
    fixture = build_shot_base(tmp_path, frame_count=point_count*12+1, receiver_hit=False)
    source_frames = [{0: row['interval'], 1: row['identity']}
                     for row in fixture.document['source_frames']]
    positions = [[300., 800.] for _ in source_frames]
    events = []
    for point_index in range(point_count):
        start = point_index*12
        winner = (1 if point_index % 2 == 0 else 2)
        if last_winner is not None and point_index == point_count-1:
            winner = last_winner
        serve_x = 300. if point_index % 2 == 0 else 700.
        positions[start] = [serve_x, 760.]
        positions[start+4] = [serve_x, 800.]
        positions[start+8] = [300. if winner == 1 else 1300., 800.]
        for offset, kind in ((0, EventType.SERVE_CONTACT), (4, EventType.BOUNCE),
                             (8, EventType.BOUNCE)):
            frame = start+offset
            events.append(TennisEvent(len(events)+1, kind, frame, frame/30.,
                1 if kind is EventType.SERVE_CONTACT else None,
                tuple(positions[frame]), None, .99, 'EXPLICIT_SYNTHETIC_OBSERVATION',
                {'fixture': 'EXPLICIT_SYNTHETIC_COMPLETED_POINT_GROWTH'}))
    files = write_observation_assets(fixture.directory, source_frames,
                                     positions=positions, fps=30.)
    detections = json.loads(files['detections.json'].read_text(encoding='utf-8'))
    for row in detections['frames']:
        row['player_1'] = {'bbox': [100., 300., 200., 700.],
                           'confidence': .99, 'source_track_id': 11}
        row['player_2'] = {'bbox': [100., 100., 200., 400.],
                           'confidence': .99, 'source_track_id': 22}
    files['detections.json'].write_text(json.dumps(detections), encoding='utf-8')
    config = fixture.directory / 'completed_points_run_config.yaml'
    config.write_text(fixture.snapshot.resolve_file('run_config.yaml').read_text(
        encoding='utf-8').replace('contact_refinement_window: 3',
                                 'contact_refinement_window: 0'), encoding='utf-8')
    files['run_config.yaml'] = config
    initial = MatchState(match_id='fixture', points_p1=3, points_p2=3, serve_attempt=1,
        ball_state=BallPlayState.DEAD, point_state=PointState.WAITING_FOR_SERVE,
        score_source='EXPLICIT_SYNTHETIC_REVIEWED_INITIAL_DEUCE')
    ball_points = [TemporalBallPoint(frame, frame/30., *positions[frame], confidence=.99,
                    state=BallState.DETECTED) for frame in range(len(source_frames))]
    replay = fixture.directory / 'completed_points_rule_source.json'
    with ScoringJournal(fixture.scratch) as journal, PointAnalysisSequence(fixture.scratch,
            'fixture', revision=0, source_fingerprint=fixture.source_sha) as owner:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
        evaluator = TennisLineCallEngine(refiner=BounceContactRefiner(window_radius=0, fps=30.))
        for event in events:
            interval = source_frames[event.frame_index][0]
            binding = ReviewedParticipantBinding(event.frame_index, interval['segment_id'], 1, 1,
                'A', 'track-A', 'NEAR_COURT', 0, 'synthetic-calibration', 0,
                'EXPLICIT_SYNTHETIC_END_REVIEW', analysis_id='fixture',
                source_fingerprint=fixture.source_sha, source_frame_index=interval['source_frame_index'],
                source_pts=interval['source_pts'], source_time_base=interval['source_time_base'],
                identity_epoch=0, participant_mapping_ref='EXPLICIT_SYNTHETIC_PARTICIPANT_MAP',
                geometry_qualification_ref='EXPLICIT_SYNTHETIC_GEOMETRY_ONLY')
            context = candidate_event_context(event, source_frames, lambda *_: binding,
                analysis_id='fixture', source_fingerprint=fixture.source_sha)
            process_timeline_event(event, context, engine, timeline, evaluator,
                ball_points=ball_points, homography_matrix=np.diag([.01, .01, 1.]),
                frames=(), geometry_available=True)
        timeline.finish()
        assert len(owner.rows('points', status='COMPLETED')) == point_count
        assert engine.state.score_confirmed
        write_rule_replay(replay, engine, owner, source_frames)
    retained_files = {item.name: item for item in fixture.snapshot.files_directory.iterdir()
                      if item.name not in {'result.json', 'summary.json'}}
    retained_files.update(files)
    retained_files['native_rule_replay.json'] = replay
    with ResultStorage(fixture.directory) as storage:
        token = publish_analysis_generation(storage, {
            'summary': {'id': 'fixture', 'frames': len(source_frames)}, 'analysisRevision': 0,
        }, fixture.directory, files=retained_files, analysis_revision=0,
            expected_generation=fixture.snapshot.generation)
    fixture.snapshot = resolve_generation(fixture.directory)
    fixture.document = json.loads(replay.read_text(encoding='utf-8'))
    corrected_initial = encode_match_state(initial)
    corrected_initial['fields']['points_p1'] = corrected_initial['fields']['points_p2'] = 5
    fixture.request.update(base_generation=token,
        base_rule_replay_sha256=fixture.snapshot.file_sha256('native_rule_replay.json'))
    fixture.request['initial_state_amendment'].update(
        expected_initial_state_sha256=canonical_sha(fixture.document['initial_state']),
        replacement_initial_state=corrected_initial)
    fixture.marker = (fixture.directory / '.result/current.store').read_bytes()
    fixture.manifest = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
                        for item in fixture.snapshot.files_directory.iterdir()}
    return fixture


def test_corrected_double_fault_updates_one_causal_point_before_later_dead_contact(tmp_path):
    from src.prediction.tennis import TennisWinProbability
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.scoring.match_state_codec import encode_match_state
    fixture = build_shot_base(tmp_path)
    (fixture.directory / 'forecasts.json').write_text(
        '{"sample_count":999,"status":"confirmed_result","winner_id":1}', encoding='utf-8')
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            history = list(forecast.history)
            assert len(history) == 7
            assert [(row['phase'], row['event_id']) for row in history] == [
                ('INITIAL', None), ('BEFORE_EVENT', 1), ('AFTER_EVENT', 1),
                ('BEFORE_EVENT', 2), ('AFTER_EVENT', 2),
                ('BEFORE_EVENT', 3), ('AFTER_EVENT', 3)]
            before = row_for(history, 'BEFORE_EVENT', 2)
            after = row_for(history, 'AFTER_EVENT', 2)
            later = row_for(history, 'AFTER_EVENT', 3)
            assert before['forecast']['sample_count'] == 0
            assert before['forecast']['status'] == 'prior_only'
            assert after['forecast']['sample_count'] == 1
            assert after['forecast']['status'] == 'uncalibrated'
            assert later['forecast']['sample_count'] == 1
            assert after['uses_points_through_event_id'] == 2
            assert before['uses_points_through_event_id'] == 1
            assert later['uses_points_through_event_id'] == 3
            server, receiver = after['forecast']['serve_estimates']
            assert (server['player_id'], server['points_won'], server['points_played']) == (1, 0, 1)
            assert (server['alpha'], server['beta'], server['probability']) == (2., 3., .4)
            assert (receiver['points_won'], receiver['points_played']) == (0, 0)
            assert later['forecast']['serve_estimates'] == after['forecast']['serve_estimates']
            assert all(row['forecast']['calibration_status'] == 'not_calibrated' for row in history)
            assert all(row['analysis_id'] == 'fixture' and row['analysis_revision'] == 1
                and row['source_sha256'] == fixture.source_sha
                and row['base_generation'] == fixture.snapshot.generation for row in history)
            assert len({row['model_signature'] for row in history}) == 1
            assert all(len(row['model_signature']) == len(row['raw_state_sha256']) == 64 for row in history)
            assert before['timestamp_seconds'] == 4/30.
            assert before['source_pts'] == str(10**18 + 400)
            assert before['source_time_base'] == '1/3000'
            assert before['source_frame_index'] == 4
            assert before['frame_index'] == 4
            assert before['event_uid'] == rebuilt.owner.event(2)['uid']
            assert after['point_uid'] == rebuilt.owner.event(2)['point_uid']
            assert history[0]['raw_state_sha256'] == canonical_sha(encode_match_state(rebuilt.engine.initial_state))
            assert later['raw_state_sha256'] == canonical_sha(encode_match_state(rebuilt.engine.state))
            expected = TennisWinProbability()
            expected.observe_point(point_id=after['point_uid'], server_id=1, winner_id=2,
                                   timestamp_seconds=4/30.)
            assert forecast.current == expected.snapshot(rebuilt.engine.state, revision=1,
                                                         timestamp_seconds=8/30.)
            assert forecast.current == history[-1]['forecast']
            detached = forecast.current
            detached['serve_estimates'][0]['points_played'] = 999
            assert forecast.history[-1]['forecast']['sample_count'] == 1
            assert forecast.history[-1]['forecast']['serve_estimates'][0]['points_played'] == 1
            assert forecast.audit['published'] is False
            assert forecast.audit['independent_accuracy_validated'] is False
            borrowed = forecast.history
            OBSERVATIONS.append({'causal_double_fault': {
                'before_samples': 0, 'after_samples': 1, 'after_dead_contact_samples': 1,
                'preaward_server': 1, 'server_posterior_alpha': 2., 'server_posterior_beta': 3.,
                'independent_accuracy_validated': False}})
        with pytest.raises(RuntimeError):
            borrowed[0]
    unchanged(fixture)


def test_future_dead_contact_does_not_change_any_earlier_forecast(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    histories = []
    for name, receiver_hit in (('prefix', False), ('extended', True)):
        parent = tmp_path / name
        parent.mkdir()
        fixture = build_shot_base(parent, receiver_hit=receiver_hit)
        with fresh_shots(fixture) as (rebuilt, shots):
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                histories.append(list(forecast.history))
        unchanged(fixture)
    assert len(histories[0]) == 5 and len(histories[1]) == 7
    for earlier, extended in zip(histories[0], histories[1][:5]):
        assert earlier['forecast'] == extended['forecast']
        assert earlier['raw_state_sha256'] == extended['raw_state_sha256']
        assert {key: value for key, value in earlier.items() if key != 'base_generation'} == {
            key: value for key, value in extended.items() if key != 'base_generation'}


def test_consequential_unknown_freezes_forecasts_at_the_first_affected_transition(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = binding_amendment(fixture)
    request['input_amendments'][0]['replacement_source_input']['event_type'] = 'PLAYER_1_HIT'
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        assert not rebuilt.engine.state.score_confirmed
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            history = list(forecast.history)
            assert history[0]['forecast']['status'] == 'prior_only'
            assert row_for(history, 'BEFORE_EVENT', 1)['forecast']['status'] == 'prior_only'
            for row in history[2:]:
                assert row['forecast']['status'] == 'unavailable'
                assert row['forecast']['probabilities'] is None
                assert row['forecast']['sample_count'] == 0
            assert forecast.current['status'] == 'unavailable'
    unchanged(fixture)


def test_awarded_but_unanchored_native_point_never_updates_the_posterior(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = republish_rule_source(build_shot_base(tmp_path), prepend_unknown=True)
    with fresh_shots(fixture) as (rebuilt, shots):
        assert rebuilt.engine.state.points_p2 == 2
        assert rebuilt.owner.event(2)['outcome_type'] == 'DOUBLE_FAULT'
        point = rebuilt.owner.get('points', rebuilt.owner.event(2)['point_uid'])
        assert point['anchor_event_id'] is None
        assert point['status'] == 'PARTIAL_UNANCHORED'
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert row_for(forecast.history, 'BEFORE_EVENT', 2)['forecast']['sample_count'] == 0
            after = row_for(forecast.history, 'AFTER_EVENT', 2)['forecast']
            assert after['status'] == 'unavailable' and after['sample_count'] == 0
            assert after['probabilities'] is None
            assert forecast.current['status'] == 'unavailable'
            assert forecast.current['sample_count'] == 0
    unchanged(fixture)


def test_authority_off_has_only_an_unavailable_initial_forecast(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = republish_rule_source(build_shot_base(tmp_path), authority=False)
    with fresh_shots(fixture) as (rebuilt, shots):
        assert not rebuilt.authoritative_events_enabled and len(shots.shots) == 0
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert len(forecast.history) == 1
            assert forecast.history[0]['phase'] == 'INITIAL'
            assert forecast.current['status'] == 'unavailable'
            assert forecast.current['sample_count'] == 0
            assert forecast.current['probabilities'] is None
    unchanged(fixture)


def test_first_fault_and_dead_receiver_contact_contribute_no_completed_points(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = request['initial_state_amendment']['replacement_initial_state']['fields']
    initial['serve_attempt'] = 1
    initial['point_state']['value'] = 'WAITING_FOR_SERVE'
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        assert rebuilt.owner.event(2)['outcome_type'] == 'FIRST_SERVE_FAULT'
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert all(row['forecast']['sample_count'] == 0 for row in forecast.history)
            assert forecast.current['status'] == 'prior_only'
    unchanged(fixture)


def test_reviewed_service_let_counts_only_the_later_terminal_double_fault(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = republish_rule_source(build_sealed_base(tmp_path, reviewed_let=True))
    with fresh_shots(fixture) as (rebuilt, shots):
        assert rebuilt.owner.event(3)['outcome_type'] == 'SERVICE_LET'
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert row_for(forecast.history, 'AFTER_EVENT', 3)['forecast']['sample_count'] == 0
            assert row_for(forecast.history, 'BEFORE_EVENT', 2)['forecast']['sample_count'] == 0
            assert row_for(forecast.history, 'AFTER_EVENT', 2)['forecast']['sample_count'] == 1
            assert forecast.current['sample_count'] == 1
    unchanged(fixture)


def test_game_ending_double_fault_uses_the_server_before_the_award(tmp_path):
    from src.prediction.tennis import TennisWinProbability
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    request['initial_state_amendment']['replacement_initial_state']['fields']['points_p2'] = 3
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        assert rebuilt.engine.state.server_id == 2
        assert rebuilt.engine.state.games_p2 == 1
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            serves = forecast.current['serve_estimates']
            assert (serves[0]['player_id'], serves[0]['points_won'], serves[0]['points_played']) == (1, 0, 1)
            assert (serves[1]['player_id'], serves[1]['points_played']) == (2, 0)
            expected = TennisWinProbability()
            expected.observe_point(point_id='comparison-point', server_id=1, winner_id=2,
                                   timestamp_seconds=4/30.)
            assert forecast.current == expected.snapshot(rebuilt.engine.state, revision=1,
                                                         timestamp_seconds=8/30.)
    unchanged(fixture)


@pytest.mark.parametrize('advance_at', ['entry', 'consumer_exit'])
def test_stale_base_generation_rejects_forecasts_and_preserves_history(tmp_path, advance_at):
    from src.api.analysis_generation import resolve_generation
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    advanced = []
    with pytest.raises(ValueError, match='current generation|base'):
        with fresh_shots(fixture) as (rebuilt, shots):
            if advance_at == 'entry':
                advanced.append(advance_equivalent_generation(fixture))
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                assert advance_at == 'consumer_exit'
                assert forecast.current['sample_count'] == 1
                advanced.append(advance_equivalent_generation(fixture))
    assert len(advanced) == 1
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in fixture.snapshot.files_directory.iterdir()} == fixture.manifest
    assert_only_parser_diagnostics(fixture.scratch)


def test_consumer_failure_closes_owned_forecast_history_and_keeps_the_base(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with pytest.raises(RuntimeError, match='synthetic forecast export'):
        with fresh_shots(fixture) as (rebuilt, shots):
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                history = forecast.history
                detached = forecast.current
                assert detached['sample_count'] == 1
                raise RuntimeError('synthetic forecast export failure')
    with pytest.raises(RuntimeError):
        history[0]
    assert detached['sample_count'] == 1
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)


@pytest.mark.parametrize('damage', [
    'source', 'revision_bool', 'revision_wrong', 'scoring_revision_bool',
    'shot_owner', 'base_generation',
])
def test_forecasts_require_the_same_finished_source_and_exact_revision_owners(tmp_path, damage):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        candidate = copy.copy(shots)
        old = (rebuilt.owner.source_fingerprint, rebuilt.owner.revision,
               rebuilt.engine.revision, rebuilt.base_generation)
        try:
            if damage == 'source': rebuilt.owner.source_fingerprint = 'b'*64
            elif damage == 'revision_bool': rebuilt.owner.revision = True
            elif damage == 'revision_wrong': rebuilt.owner.revision = 2
            elif damage == 'scoring_revision_bool': rebuilt.engine.revision = False
            elif damage == 'shot_owner': candidate.owner = None
            elif damage == 'base_generation': rebuilt.base_generation = 'b'*64
            with pytest.raises((ValueError, RuntimeError)):
                with rebuild_corrected_forecasts(rebuilt, candidate, fixture.scratch):
                    pytest.fail('Mismatched forecast ownership was accepted')
        finally:
            (rebuilt.owner.source_fingerprint, rebuilt.owner.revision,
             rebuilt.engine.revision, rebuilt.base_generation) = old
    unchanged(fixture)


@pytest.mark.parametrize('flag', ['initial_state_confirmed', 'score_confirmed'])
def test_unconfirmed_initial_context_never_becomes_a_forecast_sample(tmp_path, flag):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    request['initial_state_amendment']['replacement_initial_state']['fields'][flag] = False
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert all(row['forecast']['status'] == 'unavailable' for row in forecast.history)
            assert all(row['forecast']['sample_count'] == 0 for row in forecast.history)
            assert forecast.current['probabilities'] is None
    unchanged(fixture)


@pytest.mark.parametrize('target', [7, 10])
def test_native_tiebreak_forecast_replays_separate_raw_counters_and_first_server(tmp_path, target):
    from src.prediction.tennis import TennisWinProbability
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.scoring.match_state_codec import encode_match_state
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = request['initial_state_amendment']['replacement_initial_state']['fields']
    initial.update(games_p1=6, games_p2=6, points_p1=3, points_p2=2,
        tie_break_active=True, tie_break_points_p1=target-1, tie_break_points_p2=target-2,
        tie_break_points_to_win=target, initial_server_id=1 if target == 7 else 2)
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        assert rebuilt.engine.state.tie_break_active
        assert (rebuilt.engine.state.points_p1, rebuilt.engine.state.points_p2) == (3, 2)
        assert (rebuilt.engine.state.tie_break_points_p1, rebuilt.engine.state.tie_break_points_p2) == (target-1, target-1)
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            expected = TennisWinProbability()
            expected.observe_point(point_id='comparison-point', server_id=1, winner_id=2,
                                   timestamp_seconds=4/30.)
            assert forecast.current == expected.snapshot(rebuilt.engine.state, revision=1,
                                                         timestamp_seconds=8/30.)
            assert forecast.history[-1]['raw_state_sha256'] == canonical_sha(encode_match_state(rebuilt.engine.state))
            assert forecast.current['sample_count'] == 1
    unchanged(fixture)


def test_confirmed_match_completion_has_a_result_and_no_forecast_probability(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.scoring.match_state import CompletedSetScore
    from src.scoring.match_state_codec import decode_match_state, encode_match_state
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = decode_match_state(request['initial_state_amendment']['replacement_initial_state'])
    initial.sets_won_p2, initial.current_set = 1, 2
    initial.games_p2, initial.points_p2 = 5, 3
    initial.completed_sets = [CompletedSetScore(1, 0, 6, winner_id=2)]
    request['initial_state_amendment']['replacement_initial_state'] = encode_match_state(initial)
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        assert rebuilt.engine.state.match_complete and rebuilt.engine.state.winner_id == 2
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert row_for(forecast.history, 'BEFORE_EVENT', 2)['forecast']['probabilities'] is not None
            terminal = row_for(forecast.history, 'AFTER_EVENT', 2)['forecast']
            assert terminal['status'] == 'confirmed_result'
            assert terminal['winner_id'] == 2 and terminal['probabilities'] is None
            assert forecast.current['status'] == 'confirmed_result'
            assert forecast.current['sample_count'] == 1
    unchanged(fixture)


def test_private_forecast_export_streams_the_fresh_history_and_detached_current(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts, write_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            destination = fixture.scratch / 'corrected_forecasts.json'
            assert write_corrected_forecasts(destination, forecast) == destination
            document = json.loads(destination.read_text(encoding='utf-8'))
            assert document['history'] == list(forecast.history)
            assert document['current'] == forecast.current
            assert document['audit'] == forecast.audit
            assert document['analysis_id'] == 'fixture' and document['analysis_revision'] == 1
            assert document['source_sha256'] == fixture.source_sha
            assert document['base_generation'] == fixture.snapshot.generation
            assert document['audit']['published'] is False
    assert destination.is_file()
    unchanged(fixture)


def test_failed_private_forecast_export_preserves_an_existing_destination(tmp_path, monkeypatch):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts, write_corrected_forecasts
    from src.utils.json_array_artifact import JsonArrayArtifact
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            destination = fixture.scratch / 'corrected_forecasts.json'
            destination.write_bytes(b'PREVIOUS PRIVATE FORECAST EXPORT')
            original = JsonArrayArtifact.append
            writes = []
            def fail_on_second_row(self, row):
                writes.append(row['phase'])
                if len(writes) == 2:
                    raise RuntimeError('synthetic forecast row serialization failure')
                return original(self, row)
            monkeypatch.setattr(JsonArrayArtifact, 'append', fail_on_second_row)
            with pytest.raises(RuntimeError, match='serialization failure'):
                write_corrected_forecasts(destination, forecast)
            assert len(writes) == 2
            assert destination.read_bytes() == b'PREVIOUS PRIVATE FORECAST EXPORT'
            assert not list(fixture.scratch.glob('corrected_forecasts.json.*'))
    unchanged(fixture)


def test_private_forecast_export_refuses_a_closed_owner_and_an_analysis_destination(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts, write_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            with pytest.raises(ValueError, match='scratch|destination'):
                write_corrected_forecasts(fixture.directory / 'corrected_forecasts.json', forecast)
            assert not (fixture.directory / 'corrected_forecasts.json').exists()
        with pytest.raises(RuntimeError):
            write_corrected_forecasts(fixture.scratch / 'corrected_forecasts.json', forecast)
    unchanged(fixture)


@pytest.mark.parametrize(('field', 'value'), [
    ('terminal_event_id', True), ('terminal_event_id', 2.), ('terminal_event_id', 3),
    ('anchor_event_id', True), ('anchor_event_id', 1.), ('anchor_event_id', -1),
    ('anchor_event_id', 2), ('anchor_event_id', 3), ('anchor_event_id', None),
    ('point_id', 'another-point'), ('server_id', True), ('winner_id', 2.),
])
def test_completed_point_requires_exact_terminal_and_a_proven_earlier_service_anchor(
        tmp_path, field, value):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        uid = rebuilt.owner.event(2)['point_uid']
        original = rebuilt.owner.get('points', uid)
        damaged = {**original, field: value}
        # Deliberately bypass the sealed writer only in this private scratch
        # owner, modeling an invalid producer. Committed source assets remain
        # immutable, and the original temporary row is restored on every exit.
        rebuilt.owner.execute('UPDATE records SET payload=? WHERE kind=? AND uid=?',
            (json.dumps(damaged, allow_nan=False), 'points', uid))
        try:
            with pytest.raises(ValueError, match='completed|anchor|terminal|point'):
                with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch):
                    pytest.fail('Malformed completed point entered the posterior')
        finally:
            rebuilt.owner.execute('UPDATE records SET payload=? WHERE kind=? AND uid=?',
                (json.dumps(original, allow_nan=False), 'points', uid))
    unchanged(fixture)
    assert_only_parser_diagnostics(fixture.scratch)


def test_forecast_metadata_is_detached_and_history_ownership_cannot_be_replaced(tmp_path):
    from dataclasses import FrozenInstanceError
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts, write_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    with fresh_shots(fixture) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            trusted_audit = forecast.audit
            forged = forecast.audit
            forged.update(published=True, independent_accuracy_validated=True,
                          analysis_id='forged-analysis', source_sha256='b'*64)
            forged['model_spec']['prior']['alpha'] = 999
            assert forecast.audit == trusted_audit
            for field, value in (('owner', None), ('history', []), ('base_generation', 'b'*64)):
                with pytest.raises((FrozenInstanceError, AttributeError)):
                    setattr(forecast, field, value)
            destination = fixture.scratch / 'corrected_forecasts.json'
            write_corrected_forecasts(destination, forecast)
            exported = json.loads(destination.read_text(encoding='utf-8'))
            assert exported['audit'] == trusted_audit
            assert exported['analysis_id'] == 'fixture'
            assert exported['source_sha256'] == fixture.source_sha
            assert exported['audit']['published'] is False
            assert exported['audit']['independent_accuracy_validated'] is False
    unchanged(fixture)


@pytest.mark.parametrize(('format_name', 'required_sets'), [('BEST_OF_3', 2), ('BEST_OF_5', 3)])
@pytest.mark.parametrize('damage', [
    'terminal_unflagged', 'too_many_sets', 'both_terminal', 'active_winner',
    'active_set_number', 'premature_completion', 'wrong_completed_winner',
    'wrong_completed_set_number',
])
def test_inconsistent_reviewed_match_completion_context_is_withheld_without_math(
        tmp_path, monkeypatch, format_name, required_sets, damage):
    from src.prediction import tennis
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = request['initial_state_amendment']['replacement_initial_state']['fields']
    initial['format']['value'] = format_name
    if damage == 'terminal_unflagged':
        initial.update(sets_won_p1=required_sets, current_set=required_sets+1)
    elif damage == 'too_many_sets':
        initial.update(sets_won_p1=required_sets+1, current_set=required_sets+2)
    elif damage == 'both_terminal':
        initial.update(sets_won_p1=required_sets, sets_won_p2=required_sets,
                       current_set=2*required_sets+1)
    elif damage == 'active_winner':
        initial.update(winner_id=1)
    elif damage == 'active_set_number':
        initial.update(sets_won_p1=1, current_set=1)
    elif damage == 'premature_completion':
        initial.update(match_complete=True, winner_id=1)
    elif damage == 'wrong_completed_winner':
        initial.update(match_complete=True, sets_won_p1=required_sets,
                       winner_id=2, current_set=required_sets)
    elif damage == 'wrong_completed_set_number':
        initial.update(match_complete=True, sets_won_p1=required_sets,
                       winner_id=1, current_set=required_sets+1)
    attempts = []
    def forbidden_math(*args, **kwargs):
        attempts.append(True)
        pytest.fail('An invalid reviewed match context reached probability mathematics')
    monkeypatch.setattr(tennis, '_RuleProbabilities', forbidden_math)
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert forecast.audit['withheld_reason_code'] == 'INVALID_TERMINAL_SCORE'
            assert all(row['forecast']['status'] == 'unavailable' for row in forecast.history)
            assert all(row['forecast']['sample_count'] == 0 for row in forecast.history)
            assert all(row['forecast']['probabilities'] is None for row in forecast.history)
            assert forecast.current['winner_id'] is None
    assert not attempts
    unchanged(fixture)


@pytest.mark.parametrize('format_name', ['BEST_OF_3', 'BEST_OF_5'])
def test_reviewed_active_set_counts_need_no_invented_past_set_rows(tmp_path, format_name):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    fixture = build_shot_base(tmp_path)
    request = copy.deepcopy(fixture.request)
    initial = request['initial_state_amendment']['replacement_initial_state']['fields']
    initial['format']['value'] = format_name
    initial.update(sets_won_p1=1, current_set=2)
    assert initial['completed_sets'] == []
    with fresh_shots(fixture, request=request) as (rebuilt, shots):
        with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
            assert forecast.audit['withheld_reason_code'] is None
            assert forecast.history[0]['forecast']['status'] == 'prior_only'
            assert forecast.current['status'] == 'uncalibrated'
            assert forecast.current['sample_count'] == 1
            assert forecast.current['probabilities'] is not None
    unchanged(fixture)


def test_changing_a_later_completed_winner_preserves_its_before_and_every_earlier_forecast(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    histories = []
    for name, winner in (('receiver-final', 2), ('server-final', 1)):
        parent = tmp_path / name
        parent.mkdir()
        fixture = build_completed_points_base(parent, point_count=2, last_winner=winner)
        with fresh_shots(fixture) as (rebuilt, shots):
            assert len(rebuilt.owner.rows('points', status='COMPLETED')) == 2
            with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                histories.append(list(forecast.history))
                assert forecast.current['sample_count'] == 2
        unchanged(fixture)
    receiver_history, server_history = histories
    assert len(receiver_history) == len(server_history) == 13
    # Only event 6's outcome/measurement differs. Its BEFORE_EVENT row has the
    # same score and only the first completed point in its causal posterior.
    for receiver, server in zip(receiver_history[:-1], server_history[:-1]):
        assert receiver['forecast'] == server['forecast']
        assert receiver['raw_state_sha256'] == server['raw_state_sha256']
    assert row_for(receiver_history, 'BEFORE_EVENT', 6)['forecast']['sample_count'] == 1
    receiver_final, server_final = receiver_history[-1]['forecast'], server_history[-1]['forecast']
    assert receiver_final['serve_estimates'][0]['points_won'] == 1
    assert server_final['serve_estimates'][0]['points_won'] == 2
    assert receiver_final['probabilities'] != server_final['probabilities']
    OBSERVATIONS.append({'later_completed_outcome_causality': {
        'equal_initial_and_preterminal_rows': 12, 'completed_points_each': 2,
        'changed_terminal_event_id': 6, 'before_terminal_sample_count': 1,
        'independent_accuracy_validated': False}})


def test_growing_completed_points_bounds_forecast_history_and_deduplication_in_python(tmp_path):
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    measurements = []
    for count in (8, 128):
        parent = tmp_path / str(count)
        parent.mkdir()
        fixture = build_completed_points_base(parent, point_count=count)
        with fresh_shots(fixture) as (rebuilt, shots):
            assert len(rebuilt.owner.rows('points', status='COMPLETED')) == count
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            try:
                with rebuild_corrected_forecasts(rebuilt, shots, fixture.scratch) as forecast:
                    assert forecast.current['sample_count'] == count
                    assert forecast.audit['posterior_point_count'] == count
                    assert len(forecast.owner.point_index) == count
                    assert len(forecast.history) == 1+6*count
                    assert not isinstance(forecast.history, list)
                    assert not isinstance(forecast.owner.point_index, dict)
                    for player in forecast.current['serve_estimates']:
                        assert player['points_played'] == (count if player['player_id'] == 1 else 0)
                    gc.collect()
                    retained, peak = tracemalloc.get_traced_memory()
                    measurements.append({'completed_points': count,
                        'accepted_inputs': 3*count, 'history_rows': 1+6*count,
                        'deduplicated_point_ids': len(forecast.owner.point_index),
                        'retained_python_bytes_inside_context': retained,
                        'peak_python_bytes': peak,
                        'forecast_owner_disk_bytes': forecast.owner.path.stat().st_size,
                        'elapsed_seconds_inside_and_entry': time.perf_counter()-started})
                    borrowed = forecast.history
            finally:
                tracemalloc.stop()
            with pytest.raises(RuntimeError):
                borrowed[0]
        unchanged(fixture)
    small, large = measurements
    assert large['retained_python_bytes_inside_context']-small['retained_python_bytes_inside_context'] < 768*1024
    assert large['peak_python_bytes']-small['peak_python_bytes'] < 2*1024*1024
    OBSERVATIONS.append({'completed_point_forecast_capacity': measurements,
        'includes': 'Actual accepted-rule replay, statistical snapshots, disk history and point dedup index',
        'excludes': 'Fixture construction, corrected rules/shots, SQLite native memory and numerical-library allocations',
        'independent_accuracy_validated': False})
