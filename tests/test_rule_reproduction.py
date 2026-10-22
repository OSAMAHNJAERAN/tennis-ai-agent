"""Unchanged rule reproduction with real rules and synthetic source evidence.

The artifact checksum represents an external sealed-generation binding. These
tests never infer events, calibrate geometry or qualify physical accuracy.
"""
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
import builtins
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_point_timeline import (
    api, binding_context, event, frames, primary_events, shot, SuppliedCalls,
)
from src.scoring.match_state_codec import encode_match_state
from src.scoring.point_timeline import (
    PointTimeline, ReviewedRuleSignal, candidate_event_context, process_timeline_event,
)
from src.scoring.rule_replay import write_rule_replay
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.scoring_journal import ScoringJournal
from src.api.result_storage import ResultStorage, DiskSeries


SOURCE_SHA = 'a' * 64
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'PIL', 'pandas'}
CAPACITY_OBSERVATIONS = []


@pytest.fixture(autouse=True)
def no_model_or_image_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            raise AssertionError('Rule reproduction must stay model/image free: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded)


@pytest.fixture(autouse=True)
def coherent_actual_scoring_definitions(api, monkeypatch):
    # Each api fixture executes the unchanged real definitions afresh. Keep the
    # reproducer's ordinary imports on that same set of enum/dataclass types.
    for name in ('src.scoring.scoring_engine', 'src.scoring.scoring_rules',
                 'src.scoring.event_history'):
        module = ModuleType(name)
        module.__dict__.update(vars(api))
        monkeypatch.setitem(sys.modules, name, module)


def plain(value):
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [plain(item) for item in value]
    return value


def context(ev, source, engine):
    original = binding_context(ev, source, engine)
    binding = replace(original.binding, source_fingerprint=SOURCE_SHA)
    return candidate_event_context(ev, source, lambda *_: binding,
        analysis_id='fixture', source_fingerprint=SOURCE_SHA)


def snapshot(owner):
    return {**{kind: plain(owner.rows(kind)) for kind in
               ('points', 'attempts', 'rallies', 'transitions', 'replay_inputs', 'replay_shots')},
            'memberships': plain(owner.member_rows())}


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_replay(api, directory, *, scenario='first_fault', frame_count=129, authority=True,
                 geometry_available=True):
    directory.mkdir(parents=True, exist_ok=True)
    source = frames(frame_count)
    initial = api.MatchState(match_id='fixture', points_p1=3,
        tie_break_points_p1=17, tie_break_points_p2=16,
        score_source='SYNTHETIC_RULE_REPRODUCTION_ONLY')
    # Retain distinct raw tie-break counters even when standard points display.
    if scenario == 'first_fault':
        events = primary_events()
        decisions = {2: 'SERVE_FAULT', 4: 'IN', 6: 'SERVE_IN', 8: 'OUT',
                     10: 'SERVE_FAULT', 11: 'IN', 13: 'SERVE_FAULT'}
    elif scenario == 'double_fault':
        events = [event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE'),
                  event(3, 'SERVE_CONTACT', 1), event(4, 'BOUNCE')]
        decisions = {2: 'SERVE_FAULT', 4: 'SERVE_FAULT'}
    elif scenario in ('first_let', 'second_let'):
        second = scenario == 'second_let'
        events = [event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE')] if second else []
        decisions = {2: 'SERVE_FAULT'} if second else {}
        start = 3 if second else 1
        events += [event(start, 'SERVE_CONTACT', 1),
                   event(start + 1, 'UNKNOWN'), event(start + 2, 'SERVE_CONTACT', 1),
                   event(start + 3, 'BOUNCE'), event(start + 4, 'PLAYER_2_HIT', 2),
                   event(start + 5, 'BOUNCE')]
        decisions.update({start + 3: 'SERVE_IN', start + 5: 'OUT'})
    elif scenario in ('camera_break', 'sparse_rally'):
        events = [event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE'),
                  event(3, 'PLAYER_2_HIT', 2), event(4, 'BOUNCE')]
        for ev, frame in zip(events, (0, 1, 8, 10), strict=True):
            ev.frame_index = frame
        decisions = {2: 'SERVE_IN', 4: 'OUT'}
        if scenario == 'camera_break':
            source[5][0]['measurement_transition_allowed'] = False
    else:
        raise AssertionError('Unsupported explicit fixture scenario: ' + scenario)
    if not authority:
        events = []
    path = directory / 'native_rule_replay.json'
    with ScoringJournal(directory) as journal:
        engine = api.TennisScoringEngine(initial_state=initial, journal=journal.revision())
        with PointAnalysisSequence(directory, 'fixture', revision=3,
                                   source_fingerprint=SOURCE_SHA) as owner:
            timeline = PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source)
            caller = SuppliedCalls(api, decisions)
            accepted = []
            for ev in events:
                ctx = context(ev, source, engine)
                if ev.event_type.value == 'UNKNOWN':
                    ev = ReviewedRuleSignal(ev.event_id, ev.frame_index, ev.timestamp_s,
                        'SERVICE_LET', 'SYNTHETIC_EXPLICIT_REVIEWED_LET', ctx)
                else:
                    ev.confidence = .987654321012345
                    ev.ball_position_px = (1.23456789012345, 9.87654321012345)
                    ev.evidence = {'scope': 'SYNTHETIC_RULE_REPRODUCTION_ONLY',
                                   'raw': [.123456789012345, '\u062f\u0644\u064a\u0644']}
                process_timeline_event(ev, ctx, engine, timeline, caller,
                    ball_points=(), homography_matrix=None, frames=(),
                    geometry_available=geometry_available)
                accepted.append(ev)
            shots = [shot(ev.event_id, ev.event_type.value, ev.player_id)
                     for ev in accepted if not isinstance(ev, ReviewedRuleSignal) and ev.player_id]
            for value in shots:
                value.shot_confidence = .912345678901234
                value.source_court_position_m = (3.123456789012345, 6.987654321098765)
            if scenario == 'first_fault' and shots:
                # Cross-attempt landing must be cleared by actual attachment.
                shots[0].bounce_event_id, shots[0].bounce_frame = 6, 5
                shots[0].landing_court_position_m = (4.123456789012345, 7.987654321098765)
            timeline.attach_shots(shots)
            timeline.finish()
            expected = snapshot(owner)
            final_raw = encode_match_state(engine.get_current_state())
            initial_raw = encode_match_state(engine.initial_state)
            linked = [plain(value.to_dict()) for value in shots]
            write_rule_replay(path, engine, owner, source, authoritative_events_enabled=authority)
    assert not list(directory.glob('point_analysis_*.sqlite3*'))
    return SimpleNamespace(path=path, digest=file_sha(path), source=source, expected=expected,
        final_raw=final_raw, initial_raw=initial_raw, linked_shots=linked,
        authority=authority, source_sha=SOURCE_SHA, analysis_id='fixture', revision=3,
        line_call_requests=list(caller.requests))


@contextmanager
def reproduce(fixture, directory, **overrides):
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    kwargs = {'source_sha256': fixture.source_sha, 'analysis_id': fixture.analysis_id,
              'analysis_revision': fixture.revision, 'artifact_sha256': fixture.digest}
    kwargs.update(overrides)
    directory.mkdir(parents=True, exist_ok=True)
    with reproduce_rule_timeline(fixture.path, directory, **kwargs) as value:
        yield value


@pytest.mark.parametrize('scenario', [
    'first_fault', 'double_fault', 'first_let', 'second_let', 'camera_break', 'sparse_rally',
])
def test_unchanged_reproduction_matches_all_rules_raw_state_timeline_and_shots(api, tmp_path, scenario):
    fixture = build_replay(api, tmp_path / 'original', scenario=scenario)
    destination = tmp_path / 'reproduction'
    saved_owner = saved_shots = None
    with reproduce(fixture, destination) as replay:
        assert replay.authoritative_events_enabled is True
        assert replay.owner.analysis_id == 'fixture' and replay.owner.revision == 3
        assert replay.owner.source_fingerprint == SOURCE_SHA and replay.owner.sealed
        assert replay.engine.revision == 0
        assert encode_match_state(replay.engine.initial_state) == fixture.initial_raw
        assert encode_match_state(replay.engine.get_current_state()) == fixture.final_raw
        assert snapshot(replay.owner) == fixture.expected
        assert [plain(value.to_dict()) for value in replay.shots] == fixture.linked_shots
        assert len(replay.source_frames) == len(fixture.source)
        assert all(plain(replay.source_frames[index][0]) == source[0]
                   and plain(replay.source_frames[index][1]) == source[1]
                   for index, source in enumerate(fixture.source))
        saved_owner, saved_shots = replay.owner, replay.shots
        if scenario == 'first_fault':
            assert [row['status'] for row in replay.owner.rows('attempts')] == [
                'FIRST_SERVE_FAULT', 'SERVE_IN_PLAY', 'FIRST_SERVE_FAULT', 'DOUBLE_FAULT']
            assert replay.shots[0].bounce_event_id is None
            assert fixture.expected['replay_shots'][0]['raw']['bounce_event_id'] == 6
        elif scenario.endswith('_let'):
            attempts = list(replay.owner.rows('attempts'))
            let_index = 1 if scenario == 'second_let' else 0
            assert attempts[let_index]['status'] == 'SERVICE_LET'
            assert attempts[let_index + 1]['replay_index'] == 1
        elif scenario == 'double_fault':
            assert len(replay.owner.rows('rallies')) == 0
        elif scenario == 'camera_break':
            assert list(replay.owner.rows('rallies'))[0]['duration_s'] is None
        elif scenario == 'sparse_rally':
            assert list(replay.owner.rows('rallies'))[0]['duration_s'] is not None
    assert saved_owner.connection is None
    assert not list(destination.glob('point_analysis_*.sqlite3*'))
    assert not list(destination.glob('scoring_journal_*.sqlite3*'))
    with pytest.raises(RuntimeError):
        len(saved_owner.rows('transitions'))
    # The caller receives owned views; no stale second in-memory shot list.
    assert saved_shots is not None


def test_authority_off_reproduces_initial_state_without_inventing_events(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original', authority=False)
    with reproduce(fixture, tmp_path / 'reproduction') as replay:
        assert replay.authoritative_events_enabled is False
        assert encode_match_state(replay.engine.get_current_state()) == fixture.initial_raw
        assert snapshot(replay.owner) == fixture.expected
        assert len(replay.shots) == 0
        assert all(len(replay.owner.rows(kind)) == 0 for kind in ('transitions', 'points', 'attempts', 'rallies'))


def test_reproducer_uses_actual_spill_parser_for_all_source_frames(api, tmp_path, monkeypatch):
    fixture = build_replay(api, tmp_path / 'original', frame_count=257)
    observed = []
    read_json = ResultStorage.read_json

    def tracked(storage, path, *args, **kwargs):
        value = read_json(storage, path, *args, **kwargs)
        if Path(path) == fixture.path:
            observed.append(isinstance(value['source_frames'], DiskSeries))
        return value

    monkeypatch.setattr(ResultStorage, 'read_json', tracked)
    with reproduce(fixture, tmp_path / 'reproduction') as replay:
        assert len(replay.source_frames) == 257
        assert snapshot(replay.owner) == fixture.expected
    assert observed == [True]


def test_unavailable_geometry_reproduces_original_automatic_abstentions(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original', scenario='sparse_rally',
                           geometry_available=False)
    assert fixture.line_call_requests == []
    source_inputs = fixture.expected['replay_inputs']
    assert all(row['rule_environment'] == {'geometry_available': False} for row in source_inputs)
    calls = [row['scoring_input']['line_call'] for row in source_inputs
             if row['scoring_input']['line_call'] is not None]
    assert calls and all(call['decision'] == 'REVIEW_REQUIRED' for call in calls)
    assert all(call['bounce_position_px'] is None and call['bounce_position_m'] is None
               and call['confidence'] == 0.0 for call in calls)
    with reproduce(fixture, tmp_path / 'reproduction') as replay:
        assert encode_match_state(replay.engine.get_current_state()) == fixture.final_raw
        assert snapshot(replay.owner) == fixture.expected
        assert [plain(value.to_dict()) for value in replay.shots] == fixture.linked_shots
        assert all(row['rule_environment'] == {'geometry_available': False}
                   for row in replay.owner.rows('replay_inputs'))


def test_source_frame_reproduction_keeps_traced_python_memory_bounded(api, tmp_path):
    # Same real accepted rule/timeline/shot workload; only the saved source audit
    # grows by ~16x. Fixture objects, exporter and independent hashes are outside
    # the measurement. Native SQLite pages are not measured by tracemalloc.
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    del reproduce_rule_timeline  # Load code consistently before either sample.
    observations = []
    for frame_count in (257, 4097):
        fixture = build_replay(api, tmp_path / f'original_{frame_count}',
                               scenario='sparse_rally', frame_count=frame_count)
        destination = tmp_path / f'reproduction_{frame_count}'
        destination.mkdir()
        source_digest_before = file_sha(fixture.path)
        gc.collect()
        assert not tracemalloc.is_tracing()
        tracemalloc.start()
        started = time.perf_counter()
        try:
            # Entry includes actual ResultStorage open/parse, the two integrity
            # checks, every source-frame check, and all replay/verification work.
            with reproduce(fixture, destination) as replay:
                assert len(replay.source_frames) == frame_count
                assert isinstance(replay.source_frames.rows, DiskSeries)
                assert len(replay.owner.rows('transitions')) == 4
                assert len(replay.owner.rows('points')) == 1
                assert len(replay.owner.rows('rallies')) == 1
                assert len(replay.shots) == 2
                assert snapshot(replay.owner) == fixture.expected
                assert encode_match_state(replay.engine.get_current_state()) == fixture.final_raw
                for index in (0, frame_count // 2, frame_count - 1):
                    assert plain(replay.source_frames[index][0]) == fixture.source[index][0]
                assert replay.owner.execute('PRAGMA cache_size', one=True)[0] == -2048
                assert replay.owner.execute('PRAGMA mmap_size', one=True)[0] == 0
                gc.collect()
                retained_open, peak_open = tracemalloc.get_traced_memory()
            # Closing all three private owners is included in peak measurement.
            gc.collect()
            retained_closed, peak_complete = tracemalloc.get_traced_memory()
            elapsed = time.perf_counter() - started
        finally:
            tracemalloc.stop()
        assert file_sha(fixture.path) == source_digest_before == fixture.digest
        assert not list(destination.glob('point_analysis_*.sqlite3*'))
        assert not list(destination.glob('scoring_journal_*.sqlite3*'))
        row = {'source_frames': frame_count, 'accepted_inputs': 4,
               'points': 1, 'rallies': 1, 'shots': 2,
               'artifact_bytes': fixture.path.stat().st_size,
               'retained_open_bytes': retained_open, 'retained_closed_bytes': retained_closed,
               'peak_open_bytes': peak_open, 'peak_complete_bytes': peak_complete,
               'elapsed_seconds': elapsed, 'artifact_sha256': fixture.digest,
               'scope': 'TRACED_PYTHON_OPEN_PARSE_REPLAY_VERIFY_CLOSE_NO_MEDIA_OR_INFERENCE'}
        observations.append(row)
        CAPACITY_OBSERVATIONS.append(row)
    small, large = observations
    assert large['artifact_bytes'] > small['artifact_bytes'] * 12
    # A 4097-row Python source tree would exceed this allowance. Permit fixed
    # parser buffers/hash chunks while rejecting duration-proportional retention.
    assert large['retained_open_bytes'] < 1024 * 1024
    assert large['retained_open_bytes'] <= small['retained_open_bytes'] + 256 * 1024
    assert large['peak_complete_bytes'] < 4 * 1024 * 1024
    assert large['peak_complete_bytes'] <= small['peak_complete_bytes'] + 2 * 1024 * 1024


@pytest.mark.parametrize('override', [
    {'source_sha256': 'b' * 64}, {'analysis_id': 'different-analysis'},
    {'analysis_revision': 4}, {'artifact_sha256': 'c' * 64},
])
def test_external_identity_binding_mismatch_refuses_reproduction(api, tmp_path, override):
    fixture = build_replay(api, tmp_path / 'original')
    with pytest.raises(ValueError):
        with reproduce(fixture, tmp_path / 'reproduction', **override):
            pytest.fail('Mismatched externally bound replay was admitted')


def rewrite(fixture, mutation):
    document = json.loads(fixture.path.read_text(encoding='utf-8'))
    mutation(document)
    fixture.path.write_text(json.dumps(document, allow_nan=False), encoding='utf-8')


@pytest.mark.parametrize('mutation', [
    lambda d: d['source_frames'][5]['interval'].__setitem__('measurement_transition_allowed', False),
    lambda d: d['inputs'][0]['source_input'].__setitem__('confidence', .1),
    lambda d: d['inputs'][1]['scoring_input']['line_call'].__setitem__('decision', 'SERVE_FAULT'),
    lambda d: d['inputs'][0]['accepted_transition'].__setitem__('outcome_type', 'UNKNOWN_EVENT'),
    lambda d: d.__setitem__('scoring_revision', 1),
    lambda d: d.__setitem__('replay_scope', 'CORRECTED_OUTPUT_PUBLICATION'),
])
def test_original_external_artifact_digest_rejects_any_changed_saved_bytes(api, tmp_path, mutation):
    fixture = build_replay(api, tmp_path / 'original', scenario='sparse_rally')
    rewrite(fixture, mutation)
    with pytest.raises(ValueError):
        with reproduce(fixture, tmp_path / 'reproduction'):
            pytest.fail('Changed externally sealed bytes were admitted')


def mutate_semantic(document, kind):
    if kind == 'source_context':
        document['source_frames'][0]['interval']['source_pts'] = str(10 ** 18 + 1)
    elif kind == 'intervening_camera_break':
        document['source_frames'][5]['interval']['measurement_transition_allowed'] = False
    elif kind == 'source_order':
        document['source_frames'][4], document['source_frames'][5] = document['source_frames'][5], document['source_frames'][4]
    elif kind == 'source_omitted':
        document['source_frames'].pop(5)
        document['source_frame_count'] -= 1
    elif kind == 'source_input':
        document['inputs'][0]['source_input']['confidence'] = .1
    elif kind == 'binding':
        document['inputs'][0]['source_input']['context']['binding']['source_fingerprint'] = 'b' * 64
    elif kind == 'rule_input':
        document['inputs'][1]['scoring_input']['line_call']['decision'] = 'SERVE_FAULT'
    elif kind == 'transition':
        document['inputs'][0]['accepted_transition']['outcome_type'] = 'UNKNOWN_EVENT'
    elif kind == 'input_order':
        document['inputs'][0], document['inputs'][1] = document['inputs'][1], document['inputs'][0]
    elif kind == 'input_omitted':
        document['inputs'].pop()
        document['input_count'] -= 1
    elif kind == 'final_raw_state':
        document['final_state']['fields']['tie_break_points_p1'] += 1
    elif kind == 'initial_match_id':
        document['initial_state']['fields']['match_id'] = 'different-analysis'
    elif kind == 'corrected_revision':
        document['scoring_revision'] = 1
    elif kind == 'wrong_scope':
        document['replay_scope'] = 'CORRECTED_OUTPUT_PUBLICATION'
    elif kind == 'authority_off_with_inputs':
        document['authoritative_events_enabled'] = False
    elif kind == 'linked_shot':
        document['shots'][0]['linked']['shot_confidence'] = .1
    elif kind == 'raw_shot':
        document['shots'][0]['raw']['shot_confidence'] = .1
    elif kind == 'expected_point':
        document['expected_timeline']['points'][0]['winner_id'] = 2
    elif kind == 'expected_membership':
        document['expected_memberships'][0]['play_scope'] = 'UNKNOWN'
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize('kind', [
    'source_context', 'intervening_camera_break', 'source_order', 'source_omitted',
    'source_input', 'binding', 'rule_input', 'transition', 'input_order', 'input_omitted',
    'final_raw_state', 'initial_match_id', 'corrected_revision', 'wrong_scope',
    'authority_off_with_inputs', 'linked_shot', 'raw_shot', 'expected_point', 'expected_membership',
])
def test_internally_inconsistent_evidence_refused_even_with_updated_external_digest(api, tmp_path, kind):
    fixture = build_replay(api, tmp_path / 'original', scenario='sparse_rally')
    rewrite(fixture, lambda document: mutate_semantic(document, kind))
    fixture.digest = file_sha(fixture.path)
    destination = tmp_path / 'reproduction'
    with pytest.raises(ValueError):
        with reproduce(fixture, destination):
            pytest.fail('Internally inconsistent evidence was admitted: ' + kind)
    assert not list(destination.glob('point_analysis_*.sqlite3*'))
    assert not list(destination.glob('scoring_journal_*.sqlite3*'))
