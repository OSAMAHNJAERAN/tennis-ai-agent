"""Private corrected replay with real rules, geometry and sealed synthetic observations.

The measured ball path and court mapping are explicit fixtures. These checks
establish correction routing and source integrity, never vision accuracy.
"""
import builtins
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
from types import SimpleNamespace

import pytest


FORBIDDEN = {'torch', 'ultralytics'}
OBSERVATIONS = []


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def browser_number_encoding(value):
    """Portable fixture for JSON.stringify's integral-number formatting.

    Actual Node parse/clone/stringify is exercised in the retained wire probe.
    """
    def number(text):
        parsed = float(text)
        return int(parsed) if parsed.is_integer() else parsed
    return json.loads(json.dumps(value, allow_nan=False), parse_float=number)


@pytest.fixture(autouse=True)
def no_inference_imports(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            raise AssertionError('Private correction must not import inference: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def build_sealed_base(tmp_path, *, frame_count=12, reviewed_let=False):
    import numpy as np
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.match_state import MatchState, BallPlayState, PointState
    from src.scoring.match_state_codec import encode_match_state
    from src.scoring.scoring_engine import TennisScoringEngine
    from src.scoring.point_timeline import (PointTimeline, ReviewedParticipantBinding, ReviewedRuleSignal,
        candidate_event_context, process_timeline_event)
    from src.scoring.rule_replay import write_rule_replay
    from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
    from src.utils.point_analysis_sequence import PointAnalysisSequence
    from src.utils.scoring_journal import ScoringJournal
    from tests.test_correction_sources import write_observation_assets

    directory = tmp_path / 'analysis'
    directory.mkdir()
    scratch = tmp_path / 'private-rebuild'
    scratch.mkdir()
    source = directory / 'source.mp4'
    source.write_bytes(b'EXPLICIT SYNTHETIC SOURCE; NO VIDEO DECODE OR MATCH ACCURACY CLAIM')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_frames = [{0: {
        'analysis_id': 'fixture', 'frame_index': index, 'timestamp_seconds': index / 30.,
        'source_pts': str(10**18 + index * 100), 'source_time_base': '1/3000',
        'source_frame_index': index, 'segment_id': 'synthetic-camera-0',
        'analysis_allowed': True, 'measurement_transition_allowed': index > 0,
        'anchor_frame_index': 0, 'review_evidence_id': 'EXPLICIT_SYNTHETIC_NORMAL_SPEED',
    }, 1: {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0, 'observations': [
        {'track_key': 'track-A', 'player_id': 'A', 'state': 'CONFIRMED'},
        {'track_key': 'track-B', 'player_id': 'B', 'state': 'CONFIRMED'},
    ]}} for index in range(frame_count)]
    positions = [[300., 800. - 10. * abs(index - 4)] if index <= 8 else [300., 760.]
                 for index in range(frame_count)]
    homography = [[.01, 0., 0.], [0., .01, 0.], [0., 0., 1.]]
    files = write_observation_assets(directory, source_frames,
                                    homography=homography, positions=positions, fps=30.)
    files['source.mp4'] = source
    config = directory / 'run_config.yaml'
    config.write_text('line_calling:\n  contact_refinement_window: 3\n', encoding='utf-8')
    files[config.name] = config
    replay_path = directory / 'native_rule_replay.json'
    initial = MatchState(match_id='fixture', serve_attempt=2,
        ball_state=BallPlayState.DEAD, point_state=PointState.SECOND_SERVE_WAITING,
        score_source='EXPLICIT_SYNTHETIC_REVIEWED_INITIAL_SCORE')
    points = [TemporalBallPoint(index, index/30., *positions[index], confidence=.99,
                               state=BallState.DETECTED) for index in range(frame_count)]
    events = [SimpleNamespace(event_id=1, frame_index=0, timestamp_s=0.,
        event_type=SimpleNamespace(value='SERVE_CONTACT'), player_id=1, confidence=.99,
        ball_position_px=positions[0], court_position_m=None,
        trajectory_state='EXPLICIT_SYNTHETIC_OBSERVATION', evidence='SYNTHETIC_CONTACT'),
        SimpleNamespace(event_id=2, frame_index=4, timestamp_s=4/30.,
        event_type=SimpleNamespace(value='BOUNCE'), player_id=None, confidence=.99,
        ball_position_px=positions[4], court_position_m=None,
        trajectory_state='EXPLICIT_SYNTHETIC_OBSERVATION', evidence='SYNTHETIC_BOUNCE')]
    if reviewed_let:
        restart = copy.deepcopy(events[0])
        restart.event_id, restart.frame_index, restart.timestamp_s = 4, 3, 3/30.
        restart.ball_position_px = positions[3]
        signal = SimpleNamespace(event_id=3, frame_index=2, timestamp_s=2/30.)
        events[1:1] = [signal, restart]
    with ScoringJournal(directory) as journal:
        engine = TennisScoringEngine(initial_state=initial, journal=journal.revision())
        with PointAnalysisSequence(directory, 'fixture', revision=0,
                                   source_fingerprint=source_sha) as owner:
            timeline = PointTimeline(owner, initial.to_dict(), source_frames=source_frames)
            evaluator = TennisLineCallEngine()
            for event in events:
                interval = source_frames[event.frame_index][0]
                binding = ReviewedParticipantBinding(event.frame_index, interval['segment_id'], 1, 1,
                    'A', 'track-A', 'NEAR_COURT', 0, 'synthetic-calibration', 0,
                    'EXPLICIT_SYNTHETIC_END_REVIEW', analysis_id='fixture',
                    source_fingerprint=source_sha, source_frame_index=interval['source_frame_index'],
                    source_pts=interval['source_pts'], source_time_base=interval['source_time_base'],
                    identity_epoch=0, participant_mapping_ref='EXPLICIT_SYNTHETIC_PARTICIPANT_MAP',
                    geometry_qualification_ref='EXPLICIT_SYNTHETIC_GEOMETRY_ONLY')
                context = candidate_event_context(event, source_frames, lambda *_: binding,
                    analysis_id='fixture', source_fingerprint=source_sha)
                assert context.binding_current
                if reviewed_let and event.event_id == 3:
                    event = ReviewedRuleSignal(3, 2, 2/30., 'SERVICE_LET',
                                              'EXPLICIT_SYNTHETIC_REVIEWED_LET', context)
                process_timeline_event(event, context, engine, timeline, evaluator,
                    ball_points=points, homography_matrix=np.asarray(homography), frames=(),
                    geometry_available=True)
            timeline.finish()
            assert engine.state.points_p2 == 0 and engine.state.score_confirmed
            write_rule_replay(replay_path, engine, owner, source_frames)
    files[replay_path.name] = replay_path
    with ResultStorage(directory) as storage:
        token = publish_analysis_generation(storage, {
            'summary': {'id': 'fixture', 'frames': len(source_frames)}, 'analysisRevision': 0,
        }, directory, files=files, analysis_revision=0)
    snapshot = resolve_generation(directory)
    assert snapshot.generation == token
    document = json.loads(replay_path.read_text(encoding='utf-8'))
    corrected_initial = encode_match_state(initial)
    corrected_initial['fields']['points_p2'] = 1
    corrected_initial['fields']['service_side']['value'] = 'AD'
    request = {
        'schema_version': 1, 'analysis_id': 'fixture', 'source_sha256': source_sha,
        'base_generation': token, 'base_analysis_revision': 0,
        'base_rule_replay_sha256': snapshot.file_sha256(replay_path.name),
        'correction_id': 'SYNTHETIC-REVIEW-1', 'reviewer_id': 'SYNTHETIC-REVIEWER',
        'evidence_ref': 'SYNTHETIC-SCOREBOARD-REFERENCE',
        'reason': 'Explicit software fixture: initial receiver score was 15, not zero.',
        'initial_state_amendment': {
            'expected_initial_state_sha256': canonical_sha(document['initial_state']),
            'replacement_initial_state': corrected_initial,
        }, 'input_amendments': [],
    }
    return SimpleNamespace(directory=directory, scratch=scratch, source_sha=source_sha,
        request=request, snapshot=snapshot, document=document,
        marker=(directory / '.result/current.store').read_bytes(),
        manifest={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in snapshot.files_directory.iterdir()}, events=events)


@pytest.fixture
def sealed_base(tmp_path):
    return build_sealed_base(tmp_path)


def unchanged(fixture):
    assert (fixture.directory / '.result/current.store').read_bytes() == fixture.marker
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.snapshot.files_directory.iterdir()} == fixture.manifest


def test_initial_score_correction_recomputes_actual_later_target_and_double_fault(sealed_base, monkeypatch):
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    requests = []
    original = TennisLineCallEngine.evaluate_bounce
    def observed(self, *args, **kwargs):
        evidence = original(self, *args, **kwargs)
        requests.append((kwargs['event_id'], kwargs['context'].value,
                         kwargs['target_service_box'].value, evidence.decision.value,
                         evidence.refinement_method))
        return evidence
    monkeypatch.setattr(TennisLineCallEngine, 'evaluate_bounce', observed)
    fixture = sealed_base
    # Root aliases can change after a commit. Only the sealed generation binds
    # this rebuild's measurements and rule source.
    (fixture.directory / 'detections.json').write_text('{}', encoding='utf-8')
    (fixture.directory / 'court_geometry.json').write_text('{}', encoding='utf-8')
    (fixture.directory / 'native_rule_replay.json').write_text('{}', encoding='utf-8')
    (fixture.directory / 'run_config.yaml').write_text('INVALID ROOT CONFIG', encoding='utf-8')
    (fixture.directory / 'source.mp4').write_bytes(b'MUTABLE ROOT SOURCE ALIAS')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=fixture.request) as rebuilt:
        assert rebuilt.owner.analysis_id == 'fixture' and rebuilt.owner.revision == 1
        assert rebuilt.owner.source_fingerprint == fixture.source_sha
        assert rebuilt.engine.revision == 0
        assert rebuilt.owner.sealed is False
        assert rebuilt.engine.initial_state.points_p2 == 1
        assert rebuilt.engine.state.points_p1 == 0 and rebuilt.engine.state.points_p2 == 2
        assert rebuilt.engine.state.score_confirmed is True
        transition = rebuilt.owner.event(2)
        assert transition['outcome_type'] == 'DOUBLE_FAULT'
        assert transition['outcome']['winner_id'] == 2
        assert len(rebuilt.owner.rows('transitions')) == 2
        assert len(rebuilt.owner.rows('points', status='COMPLETED')) == 1
        assert len(rebuilt.owner.member_rows()) == 0
        assert rebuilt.base_generation == fixture.snapshot.generation
        assert len(requests) == 1
        assert requests[0][:4] == (2, 'SERVE', 'FAR_AD', 'SERVE_FAULT')
        assert requests[0][4] == 'PIECEWISE_IMPACT_INTERSECTION'
        original_call = fixture.document['inputs'][1]['scoring_input']['line_call']
        assert original_call['decision'] == 'SERVE_IN'
        assert 'FAR_DEUCE' in original_call['reason']
        assert original_call['bounce_position_m'] == pytest.approx([3., 8.])
        OBSERVATIONS.append({'original_target': 'FAR_DEUCE',
            'original_decision': original_call['decision'], 'corrected_evaluator_request': requests[0],
            'corrected_receiver_points': rebuilt.engine.state.points_p2,
            'source_scope': 'EXPLICIT_SYNTHETIC_MEASUREMENTS; NO ACCURACY_QUALIFICATION'})
        owner, sources = rebuilt.owner, rebuilt.sources
    unchanged(fixture)
    with pytest.raises(RuntimeError):
        len(owner.rows('transitions'))
    with pytest.raises(RuntimeError):
        len(sources.ball_points)


@pytest.mark.parametrize('damage', [
    'extra_field', 'schema_bool', 'revision_bool', 'wrong_analysis', 'wrong_source',
    'wrong_generation', 'wrong_revision', 'wrong_replay_sha', 'blank_reviewer',
    'blank_reason', 'blank_evidence', 'no_amendment', 'wrong_initial_sha',
    'foreign_initial_match', 'initial_bool_counter', 'cached_line_call',
])
def test_invalid_or_stale_correction_never_changes_the_committed_generation(sealed_base, damage):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    request = copy.deepcopy(fixture.request)
    if damage == 'extra_field': request['unreviewed'] = True
    elif damage == 'schema_bool': request['schema_version'] = True
    elif damage == 'revision_bool': request['base_analysis_revision'] = False
    elif damage == 'wrong_analysis': request['analysis_id'] = 'foreign-analysis'
    elif damage == 'wrong_source': request['source_sha256'] = 'b'*64
    elif damage == 'wrong_generation': request['base_generation'] = 'b'*64
    elif damage == 'wrong_revision': request['base_analysis_revision'] = 1
    elif damage == 'wrong_replay_sha': request['base_rule_replay_sha256'] = 'b'*64
    elif damage == 'blank_reviewer': request['reviewer_id'] = ' '
    elif damage == 'blank_reason': request['reason'] = ' '
    elif damage == 'blank_evidence': request['evidence_ref'] = ' '
    elif damage == 'no_amendment': request['initial_state_amendment'] = None
    elif damage == 'wrong_initial_sha': request['initial_state_amendment']['expected_initial_state_sha256'] = 'b'*64
    elif damage == 'foreign_initial_match': request['initial_state_amendment']['replacement_initial_state']['fields']['match_id'] = 'foreign'
    elif damage == 'initial_bool_counter': request['initial_state_amendment']['replacement_initial_state']['fields']['points_p2'] = True
    elif damage == 'cached_line_call': request['initial_state_amendment']['line_call'] = {'decision': 'IN'}
    with pytest.raises((ValueError, FileNotFoundError)):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request):
            pytest.fail('Invalid correction yielded a timeline')
    unchanged(fixture)


def binding_amendment(fixture):
    request = copy.deepcopy(fixture.request)
    request['initial_state_amendment'] = None
    original = fixture.document['inputs'][0]
    replacement = copy.deepcopy(original['source_input'])
    replacement['context']['binding']['evidence_ref'] = 'SYNTHETIC-REVIEWED-BINDING-REFERENCE-2'
    request['input_amendments'] = [{
        'event_id': original['source_input']['event_id'],
        'expected_input_fingerprint': original['input_fingerprint'],
        'replacement_source_input': replacement,
    }]
    return request


@pytest.mark.parametrize('browser_numbers', [False, True])
def test_original_event_amendment_is_revalidated_and_does_not_reuse_accepted_calls(sealed_base, monkeypatch, browser_numbers):
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    calls = []
    original = TennisLineCallEngine.evaluate_bounce
    def observe(self, *args, **kwargs):
        calls.append(kwargs['event_id'])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(TennisLineCallEngine, 'evaluate_bounce', observe)
    request = binding_amendment(fixture)
    if browser_numbers:
        request = browser_number_encoding(request)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
        assert rebuilt.engine.state.points_p2 == 0
        transition = rebuilt.owner.event(1)
        assert transition['context']['binding']['evidence_ref'] == 'SYNTHETIC-REVIEWED-BINDING-REFERENCE-2'
        assert transition['input_fingerprint'] != fixture.document['inputs'][0]['input_fingerprint']
        assert calls == [2]
    unchanged(fixture)


@pytest.mark.parametrize('browser_numbers', [False, True])
def test_reviewed_let_can_be_amended_to_consequential_unknown_with_rechecked_source_context(tmp_path, browser_numbers):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_sealed_base(tmp_path, reviewed_let=True)
    request = copy.deepcopy(fixture.request)
    request['initial_state_amendment'] = None
    original = fixture.document['inputs'][1]
    assert original['source_input']['event_type'] == 'SERVICE_LET'
    replacement = copy.deepcopy(original['source_input'])
    replacement['event_type'] = 'POINT_OUTCOME_UNKNOWN'
    replacement['reviewed_evidence_ref'] = 'SYNTHETIC_NEW_CONSEQUENTIAL_REVIEW'
    request['input_amendments'] = [{
        'event_id': 3, 'expected_input_fingerprint': original['input_fingerprint'],
        'replacement_source_input': replacement,
    }]
    if browser_numbers:
        request = browser_number_encoding(request)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
        assert rebuilt.owner.event(3)['rule_event_type'] == 'POINT_OUTCOME_UNKNOWN'
        assert rebuilt.owner.event(3)['reviewed_rule_evidence_ref'] == 'SYNTHETIC_NEW_CONSEQUENTIAL_REVIEW'
        assert rebuilt.owner.event(3)['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert rebuilt.owner.event(4)['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert rebuilt.owner.event(2)['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert rebuilt.engine.state.score_confirmed is False
        assert rebuilt.engine.state.points_p1 == rebuilt.engine.state.points_p2 == 0
        assert rebuilt.audit['applied_event_ids'] == [3]
        assert rebuilt.audit['human_review_dependent'] is True
        assert rebuilt.audit['independent_accuracy_validated'] is False
    unchanged(fixture)


@pytest.mark.parametrize('browser_numbers', [False, True])
def test_semantic_contact_review_recomputes_rally_call_but_keeps_consequential_score_frozen(sealed_base, monkeypatch, browser_numbers):
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    request = binding_amendment(fixture)
    request['input_amendments'][0]['replacement_source_input']['event_type'] = 'PLAYER_1_HIT'
    if browser_numbers:
        request = browser_number_encoding(request)
    calls = []
    original = TennisLineCallEngine.evaluate_bounce
    def observe(self, *args, **kwargs):
        evidence = original(self, *args, **kwargs)
        calls.append((kwargs['event_id'], kwargs['context'].value,
                      kwargs['target_service_box'], evidence.decision.value))
        return evidence
    monkeypatch.setattr(TennisLineCallEngine, 'evaluate_bounce', observe)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
        assert rebuilt.owner.event(1)['rule_event_type'] == 'PLAYER_1_HIT'
        assert rebuilt.owner.event(1)['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert rebuilt.owner.event(2)['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert rebuilt.engine.state.score_confirmed is False
        assert rebuilt.engine.state.points_p1 == rebuilt.engine.state.points_p2 == 0
        assert calls == [(2, 'RALLY', None, 'IN')]
        assert rebuilt.audit['recorded_line_calls_reused'] is False
        assert rebuilt.audit['published'] is False
    unchanged(fixture)


def test_current_generation_advance_during_actual_evaluation_rejects_pending_revision(sealed_base, monkeypatch):
    from src.api.analysis_generation import publish_analysis_generation, resolve_generation
    from src.api.result_storage import ResultStorage
    from src.line_calling.line_call_engine import TennisLineCallEngine
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    original = TennisLineCallEngine.evaluate_bounce
    advanced = []
    def advance(self, *args, **kwargs):
        evidence = original(self, *args, **kwargs)
        files = {path.name: path for path in fixture.snapshot.files_directory.iterdir()
                 if path.name not in {'result.json', 'summary.json'}}
        with ResultStorage(fixture.directory) as storage:
            advanced.append(publish_analysis_generation(storage, {
                'summary': {'id': 'fixture', 'frames': 12}, 'analysisRevision': 1,
            }, fixture.directory, files=files, analysis_revision=1,
                expected_generation=fixture.snapshot.generation))
        return evidence
    monkeypatch.setattr(TennisLineCallEngine, 'evaluate_bounce', advance)
    with pytest.raises(ValueError, match='current generation changed'):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                            request=fixture.request):
            pytest.fail('An obsolete private revision was yielded')
    assert len(advanced) == 1
    assert resolve_generation(fixture.directory).generation == advanced[0]
    assert resolve_generation(fixture.directory, fixture.snapshot.generation) == fixture.snapshot
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.snapshot.files_directory.iterdir()} == fixture.manifest


def test_consumer_failure_closes_private_owners_without_publication(sealed_base):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    with pytest.raises(RuntimeError, match='Synthetic downstream failure'):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                            request=fixture.request) as rebuilt:
            owner, sources = rebuilt.owner, rebuilt.sources
            source_parser = rebuilt.source_frames.rows
            raise RuntimeError('Synthetic downstream failure')
    unchanged(fixture)
    for read in (lambda: len(owner.rows('transitions')), lambda: len(sources.ball_points)):
        with pytest.raises(RuntimeError):
            read()
    # Small arrays are detached plain records and remain harmless values.
    # Their parser owns no borrowed SQLite reader after the context exits.
    if hasattr(source_parser, '_storage'):
        assert source_parser._storage._connection is None


def test_combined_corrected_builder_keeps_buffers_bounded_as_source_duration_grows(tmp_path):
    from src.api.result_storage import DiskSeries
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    samples = []
    for count in (257, 4097):
        parent = tmp_path / ('frames-' + str(count))
        parent.mkdir()
        fixture = build_sealed_base(parent, frame_count=count)
        gc.collect()
        started = time.perf_counter()
        tracemalloc.start()
        try:
            with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                                request=fixture.request) as rebuilt:
                assert rebuilt.engine.state.points_p2 == 2
                assert len(rebuilt.owner.rows('transitions')) == 2
                assert len(rebuilt.source_frames) == count
                assert isinstance(rebuilt.source_frames.rows, DiskSeries)
                assert isinstance(rebuilt.sources._detections['frames'], DiskSeries)
                assert isinstance(rebuilt.sources._camera['frames'], DiskSeries)
                retained, peak = tracemalloc.get_traced_memory()
            closed_retained, closed_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        sample = {'frame_count': count, 'accepted_input_count': 2,
            'retained_bytes_inside': retained, 'peak_bytes_inside': peak,
            'retained_bytes_after_cleanup': closed_retained,
            'peak_bytes_including_cleanup': closed_peak,
            'elapsed_seconds': time.perf_counter() - started,
            'scope': 'FULL_CORRECTED_REBUILD_ENTRY_PARSERS_REAL_LINE_CALL_OWNED_VIEWS_AND_CLEANUP; FIXTURE_BUILD_EXCLUDED'}
        samples.append(sample)
        unchanged(fixture)
    assert max(sample['peak_bytes_including_cleanup'] for sample in samples) < 16 * 1024 * 1024
    assert samples[1]['peak_bytes_including_cleanup'] <= samples[0]['peak_bytes_including_cleanup'] + 2 * 1024 * 1024
    assert max(sample['retained_bytes_inside'] for sample in samples) < 2 * 1024 * 1024
    OBSERVATIONS.append({'combined_builder_capacity': samples,
                         'native_sqlite_and_numpy_allocations_outside_tracemalloc': True})


@pytest.mark.parametrize('damage', [
    'unknown_id', 'wrong_fingerprint', 'duplicate_id', 'too_many', 'change_event_id',
    'change_frame', 'change_timestamp', 'change_source_pts', 'change_source_sha',
    'change_pixels', 'change_confidence', 'bool_role', 'float_role', 'inject_call',
    'bool_identity_epoch', 'bool_binding_source_index', 'float_binding_source_index',
    'bool_mapping_ref',
])
def test_event_amendment_preserves_source_identity_measurements_and_current_binding(sealed_base, damage):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    request = binding_amendment(fixture)
    amendment = request['input_amendments'][0]
    replacement = amendment['replacement_source_input']
    if damage == 'unknown_id': amendment['event_id'] = 999
    elif damage == 'wrong_fingerprint': amendment['expected_input_fingerprint'] = 'b'*64
    elif damage == 'duplicate_id': request['input_amendments'].append(copy.deepcopy(amendment))
    elif damage == 'too_many': request['input_amendments'] *= 129
    elif damage == 'change_event_id': replacement['event_id'] = 999
    elif damage == 'change_frame': replacement['frame_index'] = 1
    elif damage == 'change_timestamp': replacement['timestamp_s'] = .001
    elif damage == 'change_source_pts': replacement['context']['source_pts'] = '999'
    elif damage == 'change_source_sha': replacement['context']['binding']['source_fingerprint'] = 'b'*64
    elif damage == 'change_pixels': replacement['ball_position_px'] = [4., 5.]
    elif damage == 'change_confidence': replacement['confidence'] = .7
    elif damage == 'bool_role': replacement['player_id'] = True
    elif damage == 'float_role': replacement['player_id'] = 1.0
    elif damage == 'bool_identity_epoch': replacement['context']['binding']['identity_epoch'] = False
    elif damage == 'bool_binding_source_index': replacement['context']['binding']['source_frame_index'] = False
    elif damage == 'float_binding_source_index': replacement['context']['binding']['source_frame_index'] = 0.0
    elif damage == 'bool_mapping_ref': replacement['context']['binding']['participant_mapping_ref'] = True
    elif damage == 'inject_call': replacement['line_call'] = {'decision': 'IN'}
    with pytest.raises(ValueError):
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request):
            pytest.fail('Invalid event amendment yielded a timeline')
    unchanged(fixture)


def test_admitted_review_command_survives_caller_and_audit_mutation(sealed_base):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = sealed_base
    request = copy.deepcopy(fixture.request)
    expected = copy.deepcopy(request)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
        request['reason'] = 'CHANGED CALLER VALUE'
        rebuilt.audit['request']['reviewer_id'] = 'CHANGED AUDIT REVIEWER'
        detached = rebuilt.request
        detached['reason'] = 'CHANGED DETACHED VALUE'
        assert rebuilt.request == expected
        for name in ('request', '_request_json'):
            with pytest.raises(AttributeError, match='immutable'):
                setattr(rebuilt, name, {})
            with pytest.raises(AttributeError, match='immutable'):
                delattr(rebuilt, name)
    unchanged(fixture)


def wire_measurement_example():
    return {'event_id': 1, 'frame_index': 0, 'timestamp_s': -0.0,
        'event_type': 'SERVE_CONTACT', 'reviewed_evidence_ref': None,
        'ball_position_px': [300.0, 760.0], 'court_position_m': None,
        'trajectory_state': 'SYNTHETIC', 'confidence': 1.0,
        'evidence': {'nested': [{'value': 1.0, 'flag': False}], 'counter': 2**53 + 1},
        'context': {'frame_index': 0, 'source_pts': '1000000000000000000',
            'source_time_base': '1/3000', 'source_frame_index': 0, 'segment_id': 'toy',
            'analysis_allowed': True, 'review_evidence_ref': None, 'anchor_frame_index': 0}}


def test_wire_number_normalization_retains_exact_original_measurements_without_mutating_command():
    from src.scoring.corrected_rule_rebuild import _canonical, _replacement
    original = wire_measurement_example()
    replacement = browser_number_encoding(original)
    replacement['event_type'] = 'UNKNOWN_EVENT'
    command = {'replacement_source_input': replacement}
    before_original, before_command = _canonical(original), _canonical(command)
    normalized = _replacement(original, command)
    assert type(replacement['timestamp_s']) is int
    assert _canonical(normalized['timestamp_s']) == '-0.0'
    for key in ('timestamp_s', 'confidence', 'ball_position_px', 'court_position_m', 'evidence'):
        assert _canonical(normalized[key]) == _canonical(original[key])
    normalized['evidence']['nested'][0]['value'] = 5
    assert _canonical(original) == before_original and _canonical(command) == before_command


@pytest.mark.parametrize('damage', [
    'timestamp_bool', 'timestamp_string', 'timestamp_changed', 'timestamp_nan', 'timestamp_inf',
    'pixel_bool', 'pixel_shape', 'missing_evidence', 'evidence_bool', 'evidence_flag',
    'evidence_rounded_integer', 'evidence_changed', 'evidence_extra_key',
    'frame_float', 'context_bool', 'pts_number', 'formatting_only',
])
def test_wire_number_normalization_rejects_changed_measurements_and_primitive_types(damage):
    from src.scoring.corrected_rule_rebuild import _replacement
    original = wire_measurement_example()
    replacement = browser_number_encoding(original)
    replacement['event_type'] = 'UNKNOWN_EVENT'
    if damage == 'timestamp_bool': replacement['timestamp_s'] = False
    elif damage == 'timestamp_string': replacement['timestamp_s'] = '0'
    elif damage == 'timestamp_changed': replacement['timestamp_s'] = 5e-324
    elif damage == 'timestamp_nan': replacement['timestamp_s'] = float('nan')
    elif damage == 'timestamp_inf': replacement['timestamp_s'] = float('inf')
    elif damage == 'pixel_bool': replacement['ball_position_px'][0] = True
    elif damage == 'pixel_shape': replacement['ball_position_px'].append(0)
    elif damage == 'missing_evidence': del replacement['evidence']
    elif damage == 'evidence_bool': replacement['evidence']['nested'][0]['value'] = True
    elif damage == 'evidence_flag': replacement['evidence']['nested'][0]['flag'] = 0
    elif damage == 'evidence_rounded_integer': replacement['evidence']['counter'] = float(2**53 + 1)
    elif damage == 'evidence_changed': replacement['evidence']['nested'][0]['value'] = 1.0000000000000002
    elif damage == 'evidence_extra_key': replacement['evidence']['new'] = None
    elif damage == 'frame_float': replacement['frame_index'] = 0.0
    elif damage == 'context_bool': replacement['context']['analysis_allowed'] = 1
    elif damage == 'pts_number': replacement['context']['source_pts'] = 1000000000000000000
    elif damage == 'formatting_only': replacement['event_type'] = original['event_type']
    with pytest.raises(ValueError):
        _replacement(original, {'replacement_source_input': replacement})
