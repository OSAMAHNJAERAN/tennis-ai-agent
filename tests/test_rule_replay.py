"""Lossless replay evidence and bounded atomic export; synthetic rules only."""
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tracemalloc
from types import ModuleType

import pytest

from tests.test_scoring_service_phase import actual_classes
from tests.test_point_timeline import frames, event, shot, SuppliedCalls
from src.shot_analysis.shot_types import ShotDirection
from src.scoring.match_state_codec import decode_match_state
from src.scoring.point_timeline import (
    PointTimeline, ReviewedParticipantBinding, ReviewedRuleSignal,
    candidate_event_context, context_record, event_input_record,
    event_input_fingerprint, process_timeline_event,
)
from src.scoring import rule_replay as replay_module
from src.scoring.rule_replay import write_rule_replay
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.scoring_journal import ScoringJournal
from src.utils import json_array_artifact as json_module


SOURCE_SHA = 'a' * 64
CAPACITY_OBSERVATIONS = []


@pytest.fixture
def api(monkeypatch):
    actual = actual_classes()
    for name in ('src.scoring.match_state', 'src.scoring.point_outcome_resolver',
                 'src.line_calling.line_call_engine'):
        module = ModuleType(name)
        module.__dict__.update(vars(actual))
        monkeypatch.setitem(sys.modules, name, module)
    return actual


def context(ev, source, engine):
    role = getattr(ev, 'player_id', None) or engine.state.server_id
    identity = 'A' if role == 1 else 'B'
    interval = source[ev.frame_index][0]
    binding = ReviewedParticipantBinding(
        ev.frame_index, interval['segment_id'], role, role, identity, 'track-' + identity,
        'NEAR_COURT' if role == 1 else 'FAR_COURT', 0, 'synthetic-calibration',
        0, 'synthetic-end-proof', analysis_id='fixture', source_fingerprint=SOURCE_SHA,
        source_frame_index=interval['source_frame_index'], source_pts=interval['source_pts'],
        source_time_base=interval['source_time_base'], identity_epoch=0,
        participant_mapping_ref='synthetic-participant-map',
        geometry_qualification_ref='synthetic-geometry-proof',
    )
    return candidate_event_context(ev, source, lambda *_: binding,
        analysis_id='fixture', source_fingerprint=SOURCE_SHA)


def original_fingerprint(ev, ctx):
    # Preserve the exact canonical form from before event_input_record existed.
    reviewed = isinstance(ev, ReviewedRuleSignal)
    payload = {name: getattr(ev, name, None) for name in (
        'event_id', 'frame_index', 'timestamp_s', 'player_id', 'confidence',
        'ball_position_px', 'court_position_m', 'trajectory_state', 'evidence')}
    payload['event_type'] = ev.rule_type if reviewed else ev.event_type.value
    payload['reviewed_evidence_ref'] = ev.evidence_ref if reviewed else None
    payload['context'] = context_record(ctx)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@contextmanager
def accepted_analysis(api, directory, *, journal=False, source=None, empty=False):
    source = frames(12) if source is None else source
    initial = api.MatchState(match_id='fixture', points_p1=7, points_p2=6,
        tie_break_points_p1=17, tie_break_points_p2=16, score_source='synthetic-reviewed-initial')
    with ScoringJournal(directory) as scoring_owner:
        engine = api.TennisScoringEngine(initial_state=initial,
            journal=scoring_owner.revision() if journal else None)
        with PointAnalysisSequence(directory, 'fixture', source_fingerprint=SOURCE_SHA) as owner:
            timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
            events = [] if empty else [event(ident, kind, player) for ident, kind, player in (
                (1, 'SERVE_CONTACT', 1), (2, 'BOUNCE', None),
                (3, 'SERVE_CONTACT', 1), (4, 'BOUNCE', None),
                (5, 'PLAYER_2_HIT', 2), (6, 'BOUNCE', None))]
            calls = SuppliedCalls(api, {2: 'SERVE_FAULT', 4: 'SERVE_IN', 6: 'OUT'})
            for ev in events:
                ev.ball_position_px = (3.123456789012345, 6.987654321098765)
                ev.confidence = .987654321012345
                ev.evidence = {'source': 'synthetic-only', 'nested': {'raw': [.123456789012345]}}
                ev._replay_context = context(ev, source, engine)
                process_timeline_event(ev, ev._replay_context, engine, timeline, calls,
                    ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
            timeline.finish()
            yield engine, owner, source, events, calls


@pytest.mark.parametrize('reviewed', [False, True])
def test_event_input_record_matches_prior_canonical_fingerprint_and_detaches(api, reviewed):
    source = frames()
    engine = api.TennisScoringEngine()
    ev = event(1, 'SERVE_CONTACT', 1)
    ev.ball_position_px = (11.123456789012345, 22.987654321098765)
    ev.court_position_m = (1.234567890123456, 2.345678901234567)
    ev.confidence = .981234567890123
    ev.evidence = {'raw': [.123456789012345], 'unicode': '\u062f\u0644\u064a\u0644'}
    ctx = context(ev, source, engine)
    if reviewed:
        ev = ReviewedRuleSignal(1, 0, .1, 'SERVICE_LET', 'explicit-synthetic-review', ctx)
    before = original_fingerprint(ev, ctx)
    recorded = event_input_record(ev, ctx)
    assert event_input_fingerprint(ev, ctx) == before
    assert hashlib.sha256(json.dumps(recorded, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest() == before
    assert json.loads(json.dumps(recorded, allow_nan=False)) == recorded
    assert recorded['context']['source_pts'] == str(10**18)
    if not reviewed:
        assert recorded['ball_position_px'] == [11.123456789012345, 22.987654321098765]
        assert recorded['confidence'] == .981234567890123
        ev.evidence['raw'][0] = 99
        ev.ball_position_px = (0, 0)
        assert recorded['evidence']['raw'] == [.123456789012345]
        assert recorded['ball_position_px'][0] == 11.123456789012345


@pytest.mark.parametrize('journal', [False, True])
def test_accepted_rule_inputs_detach_and_preserve_typed_line_call(api, tmp_path, journal):
    with accepted_analysis(api, tmp_path, journal=journal) as (engine, owner, *_):
        encoded = engine.accepted_rule_input(6)
        assert encoded == list(engine.iter_rule_inputs())[-1]
        assert encoded['line_call']['decision'] == 'OUT'
        typed = engine._decode_input(encoded)
        assert type(typed['line_call']) is api.LineCallEvidence
        assert typed['line_call'].decision is api.LineCallDecision.OUT
        encoded['line_call']['reason'] = 'changed response'
        typed['line_call'].reason = 'changed detached instance'
        assert engine.accepted_rule_input(6)['line_call']['reason'] != 'changed response'
        iterator = engine.iter_rule_inputs()
        first = next(iterator)
        first['confidence'] = 0
        iterator.close()
        assert next(engine.iter_rule_inputs())['confidence'] == .987654321012345
        with pytest.raises(ValueError):
            engine.accepted_rule_input(1)
        assert len(owner.rows('replay_inputs')) == len(engine._inputs) == 6


@pytest.mark.parametrize('journal', [False, True])
def test_writer_keeps_raw_payload_initial_final_state_and_every_frame(api, tmp_path, journal):
    source = frames(19)
    # An interval with no rule event still belongs in the saved continuity proof.
    source[17][0]['analysis_allowed'] = False
    source[17][0]['measurement_transition_allowed'] = False
    with accepted_analysis(api, tmp_path, journal=journal, source=source) as (engine, owner, _, events, _):
        original = copy.deepcopy(owner.rows('replay_inputs')[0]['source_input'])
        events[0].evidence['nested']['raw'][0] = 42
        events[0].evidence['landing_zone'] = 'downstream-derived-only'
        events[0].confidence = .1
        destination = tmp_path / 'replay.json'
        write_rule_replay(destination, engine, owner, source)
        saved = json.loads(destination.read_text(encoding='utf-8'))
        assert saved['inputs'][0]['source_input'] == original
        assert saved['inputs'][0]['rule_environment'] == {'geometry_available': True}
        assert saved['inputs'][0]['source_input']['confidence'] == .987654321012345
        assert 'landing_zone' not in saved['inputs'][0]['source_input']['evidence']
        assert saved['input_count'] == len(saved['inputs']) == 6
        assert saved['source_frame_count'] == len(saved['source_frames']) == 19
        for index, frame in enumerate(saved['source_frames']):
            assert frame['frame_index'] == index
            assert frame['interval'] == source[index][0]
            assert frame['identity'] == source[index][1]
        assert saved['source_frames'][17]['interval']['analysis_allowed'] is False
        initial = decode_match_state(saved['initial_state'])
        final = decode_match_state(saved['final_state'])
        assert asdict(initial) == asdict(engine.initial_state)
        assert asdict(final) == asdict(engine.get_current_state())
        assert (initial.points_p1, initial.points_p2,
                initial.tie_break_points_p1, initial.tie_break_points_p2) == (7, 6, 17, 16)
        assert saved['source_sha256'] == SOURCE_SHA
        assert saved['replay_scope'] == 'UNCHANGED_RULE_REPRODUCTION_ONLY'
        assert saved['independent_accuracy_validated'] is False
        assert not list(tmp_path.glob('.replay.json.*.tmp'))


def test_duplicate_retry_adds_no_second_replay_record(api, tmp_path):
    source = frames()
    with accepted_analysis(api, tmp_path, source=source) as (engine, owner, _, events, calls):
        # A duplicate retry before sealing is an unchanged accepted transition.
        owner.sealed = False
        ev = events[-1]
        process_timeline_event(ev, ev._replay_context, engine,
            PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source), calls,
            ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        owner.seal()
        assert len(owner.rows('replay_inputs')) == len(owner.rows('transitions')) == len(engine._inputs) == 6


@pytest.mark.parametrize('field', ['confidence', 'evidence'])
def test_nonfinite_original_evidence_fails_before_rule_or_timeline_mutation(api, tmp_path, field):
    source = frames()
    engine = api.TennisScoringEngine(initial_state=api.MatchState(match_id='fixture'))
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint=SOURCE_SHA) as owner:
        timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
        ev = event(1, 'SERVE_CONTACT', 1)
        setattr(ev, field, float('nan') if field == 'confidence' else {'hidden': float('inf')})
        before = engine.state.to_dict()
        with pytest.raises(ValueError):
            process_timeline_event(ev, context(ev, source, engine), engine, timeline,
                SuppliedCalls(api, {}), ball_points=(), homography_matrix=None,
                frames=(), geometry_available=True)
        assert engine.state.to_dict() == before and len(engine._inputs) == 0
        assert len(owner.rows('transitions')) == len(owner.rows('replay_inputs')) == 0


def test_reviewed_rule_signal_is_retained_separately_from_physical_payload(api, tmp_path):
    source = frames()
    initial = api.MatchState(match_id='fixture')
    engine = api.TennisScoringEngine(initial_state=initial)
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint=SOURCE_SHA) as owner:
        timeline = PointTimeline(owner, initial.to_dict(), source_frames=source)
        serve = event(1, 'SERVE_CONTACT', 1)
        stub = event(2, 'UNKNOWN_EVENT')
        signal = ReviewedRuleSignal(2, 1, .2, 'SERVICE_LET',
                                   'explicit-synthetic-service-let', context(stub, source, engine))
        for ev in (serve, signal):
            ctx = ev.context if isinstance(ev, ReviewedRuleSignal) else context(ev, source, engine)
            process_timeline_event(ev, ctx, engine, timeline, SuppliedCalls(api, {}),
                ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        timeline.finish()
        write_rule_replay(tmp_path / 'replay.json', engine, owner, source)
        saved = json.loads((tmp_path / 'replay.json').read_text(encoding='utf-8'))
        row = saved['inputs'][1]
        assert row['source_input']['event_type'] == 'SERVICE_LET'
        assert row['source_input']['reviewed_evidence_ref'] == 'explicit-synthetic-service-let'
        assert row['source_input']['ball_position_px'] is None
        assert row['accepted_transition']['outcome_type'] == 'SERVICE_LET'
        assert row['scoring_input']['event_type'] == 'SERVICE_LET'


@pytest.mark.parametrize('failure', ['unsealed', 'corrected_engine', 'missing_frames',
                                   'source_sha', 'initial_owner', 'authority_type', 'authority_off'])
def test_writer_rejects_unadmitted_replay_without_changing_previous_output(api, tmp_path, failure):
    with accepted_analysis(api, tmp_path) as (engine, owner, source, *_):
        kwargs = {}
        if failure == 'unsealed': owner.sealed = False
        if failure == 'corrected_engine': engine.revision = 1
        if failure == 'missing_frames': source = None
        if failure == 'source_sha': owner.source_fingerprint = 'unbound-source'
        if failure == 'initial_owner': engine.initial_state.match_id = 'another-analysis'
        if failure == 'authority_type': kwargs['authoritative_events_enabled'] = 1
        if failure == 'authority_off': kwargs['authoritative_events_enabled'] = False
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'prior complete result')
        with pytest.raises(ValueError):
            write_rule_replay(destination, engine, owner, source, **kwargs)
        assert destination.read_bytes() == b'prior complete result'
        assert not owner.readers and not list(tmp_path.glob('.replay.json.*.tmp'))


def test_authority_off_replay_preserves_empty_inputs_and_actual_false_flag(api, tmp_path):
    with accepted_analysis(api, tmp_path, empty=True) as (engine, owner, source, *_):
        write_rule_replay(tmp_path / 'replay.json', engine, owner, source,
                          authoritative_events_enabled=False)
        saved = json.loads((tmp_path / 'replay.json').read_text(encoding='utf-8'))
        assert saved['authoritative_events_enabled'] is False
        assert saved['inputs'] == [] and saved['input_count'] == 0
        assert saved['initial_state'] == saved['final_state']


def test_geometry_unavailable_environment_and_automatic_abstention_are_captured(api, tmp_path):
    source = frames()
    initial = api.MatchState(match_id='fixture')
    engine = api.TennisScoringEngine(initial_state=initial)
    caller = SuppliedCalls(api, {})
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint=SOURCE_SHA) as owner:
        timeline = PointTimeline(owner, initial.to_dict(), source_frames=source)
        for ev in (event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE')):
            process_timeline_event(ev, context(ev, source, engine), engine, timeline, caller,
                ball_points=(), homography_matrix=None, frames=(), geometry_available=False)
        timeline.finish()
        write_rule_replay(tmp_path / 'replay.json', engine, owner, source)
        saved = json.loads((tmp_path / 'replay.json').read_text(encoding='utf-8'))
        assert [row['rule_environment'] for row in saved['inputs']] == [
            {'geometry_available': False}, {'geometry_available': False}]
        assert saved['inputs'][1]['scoring_input']['line_call']['reason_code'] == 'SERVICE_CONTEXT_UNAVAILABLE'
        assert saved['inputs'][1]['scoring_input']['line_call']['contact_patch_model'] == 'NOT_EVALUATED'
        assert caller.requests == []


@pytest.mark.parametrize('missing_shot', [False, True])
def test_raw_shots_precede_destructive_membership_attachment_and_are_sealed_together(api, tmp_path, missing_shot):
    with accepted_analysis(api, tmp_path) as (engine, owner, source, *_):
        owner.sealed = False
        timeline = PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source)
        shots = [shot(1, 'SERVE_CONTACT', 1), shot(3, 'SERVE_CONTACT', 1),
                 shot(5, 'PLAYER_2_HIT', 2)]
        shots[0].bounce_event_id, shots[0].bounce_frame = 6, 5
        shots[0].landing_court_position_m = (1.23456789012345, 4.56789012345678)
        shots[0].direction = ShotDirection.CROSS_COURT
        shots[0].direction_confidence = .987654321012345
        original = json.loads(json.dumps(shots[0].to_dict(), allow_nan=False))
        timeline.attach_shots(shots)
        timeline.finish()
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old complete shot evidence')
        if missing_shot:
            owner.execute("DELETE FROM records WHERE kind='replay_shots' AND position=1")
            with pytest.raises(ValueError, match='shots and accepted memberships differ'):
                write_rule_replay(destination, engine, owner, source)
            assert destination.read_bytes() == b'old complete shot evidence'
            return
        write_rule_replay(destination, engine, owner, source)
        saved = json.loads(destination.read_text(encoding='utf-8'))
        assert saved['shot_count'] == len(saved['shots']) == len(saved['expected_memberships']) == 3
        assert saved['shots'][0]['raw'] == original
        assert saved['shots'][0]['raw']['bounce_event_id'] == 6
        assert saved['shots'][0]['raw']['direction'] == 'CROSS_COURT'
        assert saved['shots'][0]['linked']['bounce_event_id'] is None
        assert saved['shots'][0]['linked']['direction'] == 'UNKNOWN'
        assert saved['shots'][0]['linked']['outcome'] == 'FAULT'
        assert set(saved['expected_timeline']) == {'points', 'rallies', 'attempts'}
        assert len(saved['expected_timeline']['points']) == 1
        assert len(saved['expected_timeline']['attempts']) == 2
        assert all('shot_ids' not in row for rows in saved['expected_timeline'].values() for row in rows)


@pytest.mark.parametrize('failure', ['fingerprint', 'transition', 'rule_input', 'history', 'count', 'source_owner', 'source_order'])
def test_failed_writer_preserves_prior_destination_and_closes_readers(api, tmp_path, failure):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        if failure in ('fingerprint', 'transition', 'rule_input'):
            row = owner.rows('replay_inputs')[2]
            if failure == 'fingerprint': row['source_input']['confidence'] = .2
            if failure == 'transition': row['accepted_transition']['outcome_type'] = 'UNKNOWN_EVENT'
            if failure == 'rule_input': row['scoring_input']['confidence'] = .2
            owner.sealed = False
            owner.put('replay_inputs', row['accepted_transition']['uid'], row,
                event_id=row['source_input']['event_id'], frame=row['source_input']['frame_index'])
            owner.seal()
        elif failure == 'history':
            # Alter one accepted history through its actual journal storage.
            journal_owner = engine._journal.owner
            row = journal_owner._execute('SELECT history FROM transitions WHERE revision=0 AND position=2', one=True)
            history = json.loads(row[0]); history['point_outcome'] = 'UNKNOWN_EVENT'
            journal_owner._execute('UPDATE transitions SET history=? WHERE revision=0 AND position=2',
                                   (json.dumps(history),))
        elif failure == 'count':
            owner.execute("DELETE FROM records WHERE kind='replay_inputs' AND position=2")
        elif failure == 'source_owner':
            source[10][0]['analysis_id'] = 'another-analysis'
        else:
            source[10][0]['frame_index'] = 999
        destination = tmp_path / 'replay.json'
        previous = b'previous complete synthetic revision'
        destination.write_bytes(previous)
        with pytest.raises(ValueError):
            write_rule_replay(destination, engine, owner, source)
        assert destination.read_bytes() == previous
        assert not owner.readers and not engine._journal.owner.readers
        assert not list(tmp_path.glob('.replay.json.*.tmp'))


@pytest.mark.parametrize('failure', ['replace', 'fsync', 'serialize'])
def test_storage_failures_preserve_previous_file(api, tmp_path, monkeypatch, failure):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old complete revision')
        def fail(*_args, **_kwargs):
            raise OSError('synthetic writer storage failure')
        if failure == 'serialize':
            original_dump = json_module.json.dump
            def dump(value, stream, **kwargs):
                if isinstance(value, dict) and 'source_input' in value:
                    fail()
                return original_dump(value, stream, **kwargs)
            monkeypatch.setattr(json_module.json, 'dump', dump)
        else:
            monkeypatch.setattr(json_module.os, failure, fail)
        with pytest.raises(OSError, match='synthetic writer storage failure'):
            write_rule_replay(destination, engine, owner, source)
        assert destination.read_bytes() == b'old complete revision'
        assert not owner.readers and not engine._journal.owner.readers
        assert not list(tmp_path.glob('.replay.json.*.tmp'))


@pytest.mark.parametrize('field,value', [
    ('sequence_number', 999), ('timestamp_seconds', 999.),
    ('event_type', 'DIFFERENT_RULE_INPUT'), ('confidence', .123),
    ('reason', 'different accepted outcome reason'),
])
def test_every_accepted_history_field_is_bound_before_publication(api, tmp_path, field, value):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        journal_owner = engine._journal.owner
        stored = journal_owner._execute(
            'SELECT history FROM transitions WHERE revision=0 AND position=2', one=True)
        history = json.loads(stored[0])
        history[field] = value
        journal_owner._execute('UPDATE transitions SET history=? WHERE revision=0 AND position=2',
                               (json.dumps(history),))
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old accepted revision')
        with pytest.raises(ValueError, match='Replay evidence differs'):
            write_rule_replay(destination, engine, owner, source)
        assert destination.read_bytes() == b'old accepted revision'
        assert not owner.readers and not journal_owner.readers


@pytest.mark.parametrize('field,value', [('event_type', 'UNKNOWN_EVENT'), ('player_id', 2)])
def test_translated_rule_input_must_match_timeline_even_if_journal_and_capture_agree(api, tmp_path, field, value):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        journal_owner = engine._journal.owner
        stored = journal_owner._execute(
            'SELECT input,history FROM transitions WHERE revision=0 AND position=2', one=True)
        rule_input, history = json.loads(stored[0]), json.loads(stored[1])
        rule_input[field] = value
        if field == 'event_type': history[field] = value
        journal_owner._execute('UPDATE transitions SET input=?,history=? WHERE revision=0 AND position=2',
                               (json.dumps(rule_input), json.dumps(history)))
        row = owner.rows('replay_inputs')[2]
        row['scoring_input'] = copy.deepcopy(rule_input)
        owner.sealed = False
        owner.put('replay_inputs', row['accepted_transition']['uid'], row,
                  event_id=row['source_input']['event_id'], frame=row['source_input']['frame_index'])
        owner.seal()
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old accepted revision')
        with pytest.raises(ValueError, match='Replay evidence differs'):
            write_rule_replay(destination, engine, owner, source)
        assert destination.read_bytes() == b'old accepted revision'


def test_raw_final_state_must_match_last_accepted_transition(api, tmp_path):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        engine.state.state_version += 1
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old accepted revision')
        with pytest.raises(ValueError, match='Replay final state differs'):
            write_rule_replay(destination, engine, owner, source)
        assert destination.read_bytes() == b'old accepted revision'


@pytest.mark.parametrize('primary_failure', [False, True])
def test_iterator_close_failure_is_before_publication_and_preserves_primary(api, tmp_path, monkeypatch, primary_failure):
    with accepted_analysis(api, tmp_path, journal=True) as (engine, owner, source, *_):
        original = engine.iter_rule_inputs
        closed = []
        class CloseFailure:
            def __init__(self): self.iterator = original()
            def __iter__(self): return self
            def __next__(self): return next(self.iterator)
            def close(self):
                self.iterator.close()
                closed.append(True)
                raise OSError('synthetic reader cleanup failure')
        monkeypatch.setattr(engine, 'iter_rule_inputs', CloseFailure)
        if primary_failure:
            row = owner.rows('replay_inputs')[1]
            row['input_fingerprint'] = 'b' * 64
            owner.sealed = False
            owner.put('replay_inputs', row['accepted_transition']['uid'], row,
                event_id=row['source_input']['event_id'], frame=row['source_input']['frame_index'])
            owner.seal()
        destination = tmp_path / 'replay.json'
        destination.write_bytes(b'old revision must survive cleanup failure')
        error_type = ValueError if primary_failure else OSError
        with pytest.raises(error_type) as error:
            write_rule_replay(destination, engine, owner, source)
        assert closed == [True]
        if primary_failure:
            assert any('synthetic reader cleanup failure' in note for note in error.value.__notes__)
        assert destination.read_bytes() == b'old revision must survive cleanup failure'
        assert not owner.readers and not engine._journal.owner.readers
        assert not list(tmp_path.glob('.replay.json.*.tmp'))


@pytest.mark.parametrize('count', [257, 4097])
def test_source_frame_writer_retains_one_frame_without_materializing_input(api, tmp_path, count):
    template = frames(1)[0]
    class LazyFrames:
        def __init__(self): self.accesses = 0
        def __len__(self): return count
        def __iter__(self): raise AssertionError('Whole source sequence materialization forbidden')
        def __getitem__(self, index):
            assert 0 <= index < count
            self.accesses += 1
            interval = {**template[0], 'source_pts': str(10**18 + index * 200),
                        'source_frame_index': index}
            return {0: interval, 1: template[1]}
    source = LazyFrames()
    with accepted_analysis(api, tmp_path, journal=True, source=source, empty=True) as (engine, owner, *_):
        tracemalloc.start()
        try:
            write_rule_replay(tmp_path / 'replay.json', engine, owner, source)
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        CAPACITY_OBSERVATIONS.append({'source_frames': count, 'retained_bytes': retained,
                                      'peak_bytes': peak, 'source_accesses': source.accesses})
        assert source.accesses <= 2 * count
        assert retained < 128 * 1024
        assert peak < 512 * 1024
        # Full parse is intentionally outside the writer's measured scope.
        saved = json.loads((tmp_path / 'replay.json').read_text(encoding='utf-8'))
        assert len(saved['source_frames']) == count and saved['inputs'] == []


@pytest.mark.parametrize('count', [129, 513])
def test_accepted_input_writer_is_bounded_with_actual_rule_and_owner_records(api, tmp_path, count):
    source = frames(count)
    with accepted_analysis(api, tmp_path, journal=True, source=source, empty=True) as (engine, owner, *_):
        owner.sealed = False
        timeline = PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source)
        for ident in range(1, count + 1):
            ev = event(ident, 'UNKNOWN_EVENT')
            ev.evidence = {'synthetic': True, 'unrounded': .123456789012345}
            process_timeline_event(ev, context(ev, source, engine), engine, timeline,
                SuppliedCalls(api, {}), ball_points=(), homography_matrix=None,
                frames=(), geometry_available=True)
        timeline.finish()
        tracemalloc.start()
        try:
            write_rule_replay(tmp_path / 'replay.json', engine, owner, source)
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        CAPACITY_OBSERVATIONS.append({'accepted_inputs': count, 'source_frames': count,
                                      'retained_bytes': retained, 'peak_bytes': peak})
        assert retained < 256 * 1024
        assert peak < 768 * 1024
        saved = json.loads((tmp_path / 'replay.json').read_text(encoding='utf-8'))
        assert len(saved['inputs']) == saved['input_count'] == count
        assert not owner.readers and not engine._journal.owner.readers
