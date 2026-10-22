"""Actual rule/replay parity and bounded journal integration; no media accuracy."""
import copy
import gc
import json
import tracemalloc

import pytest

from src.utils.scoring_journal import ScoringJournal
from tests.test_scoring_service_phase import actual_classes, call


@pytest.fixture
def api():
    return actual_classes()


def exercise(api, engines):
    events = [
        dict(event_id=1, event_type='SERVE_CONTACT', timestamp_seconds=0., player_id=1),
        dict(event_id=2, event_type='BOUNCE', timestamp_seconds=.1, line_call=call(api, 'SERVE_FAULT', 2)),
        dict(event_id=3, event_type='PLAYER_2_HIT', timestamp_seconds=.2, player_id=2),
        dict(event_id=4, event_type='SERVE_CONTACT', timestamp_seconds=.3, player_id=1),
        dict(event_id=5, event_type='SERVICE_LET', timestamp_seconds=.4),
        dict(event_id=6, event_type='SERVE_CONTACT', timestamp_seconds=.5, player_id=1),
        dict(event_id=7, event_type='BOUNCE', timestamp_seconds=.6, line_call=call(api, 'SERVE_IN', 7)),
        dict(event_id=8, event_type='PLAYER_2_HIT', timestamp_seconds=.7, player_id=2),
        dict(event_id=9, event_type='BOUNCE', timestamp_seconds=.8, line_call=call(api, 'OUT', 9, 'RALLY'), point_id='point-one'),
        dict(event_id=10, event_type='SERVE_CONTACT', timestamp_seconds=.9, player_id=1, point_id='point-one'),
        dict(event_id=11, event_type='CONSEQUENTIAL_GAP', timestamp_seconds=1.),
        dict(event_id=12, event_type='SERVE_CONTACT', timestamp_seconds=1.1, player_id=1),
    ]
    for event in events:
        outcomes = [engine.process_match_event(**event)[1].to_dict() for engine in engines]
        assert outcomes[0] == outcomes[1]
        assert engines[0].state.to_dict() == engines[1].state.to_dict()
    return events


def test_actual_rules_idempotency_history_and_completed_ids_match_legacy(api, tmp_path):
    legacy = api.TennisScoringEngine()
    with ScoringJournal(tmp_path) as owner:
        disk = api.TennisScoringEngine(journal=owner.revision())
        events = exercise(api, (legacy, disk))
        assert disk.history.to_list() == legacy.history.to_list()
        assert set(disk.history.processed_event_ids) == legacy.history.processed_event_ids
        assert set(disk.completed_point_ids) == legacy.completed_point_ids == {'point-one'}
        assert len(disk._inputs) == len(legacy._inputs)
        old = copy.deepcopy(disk.state.to_dict())
        for engine in (legacy, disk):
            _, outcome = engine.process_match_event(**events[0])
            assert outcome.outcome_type == api.PointOutcomeType.UNKNOWN_EVENT
        assert disk.state.to_dict() == old
    assert not list(tmp_path.glob('*.sqlite3*'))


def review_stream(api, engine):
    engine.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
    engine.process_match_event(2, 'BOUNCE', .1, line_call=call(api, 'REVIEW_REQUIRED', 2))
    engine.process_match_event(3, 'PLAYER_2_HIT', .2, player_id=2)
    engine.process_match_event(4, 'BOUNCE', .3, line_call=call(api, 'OUT', 4, 'RALLY'))


def test_corrected_replay_and_nested_atomic_export_match_actual_legacy(api, tmp_path):
    legacy = api.TennisScoringEngine()
    with ScoringJournal(tmp_path) as owner:
        disk = api.TennisScoringEngine(journal=owner.revision())
        for engine in (legacy, disk):
            review_stream(api, engine)
            engine.correct_event(2, reason='Explicit synthetic serve review', expected_revision=0,
                line_call=call(api, 'SERVE_IN', 2))
        assert disk.state.to_dict() == legacy.state.to_dict()
        assert disk.state.points_p1 == 1 and disk.revision == legacy.revision == 1
        assert disk.corrections[0]['previous_history'][3]['new_state']['score_confirmed'] is False
        assert disk.history.to_list() == legacy.history.to_list()
        for engine, name in ((disk, 'disk.json'), (legacy, 'legacy.json')):
            engine.save_history_json(str(tmp_path / name))
        assert json.loads((tmp_path / 'disk.json').read_text()) == json.loads((tmp_path / 'legacy.json').read_text())
        for engine in (legacy, disk):
            engine.correct_event(4, reason='Explicit synthetic line correction', expected_revision=1,
                line_call=call(api, 'IN', 4, 'RALLY'))
        assert disk.state.to_dict() == legacy.state.to_dict()
        disk.save_history_json(str(tmp_path / 'second-disk.json'))
        legacy.save_history_json(str(tmp_path / 'second-legacy.json'))
        assert json.loads((tmp_path / 'second-disk.json').read_text()) == json.loads((tmp_path / 'second-legacy.json').read_text())


@pytest.mark.parametrize('replacement', [dict(timestamp_seconds=-1.), dict(timestamp_seconds=100.), dict(confidence=float('nan'))])
def test_failed_correction_keeps_revision_state_resolver_and_audit(api, tmp_path, replacement):
    with ScoringJournal(tmp_path) as owner:
        disk = api.TennisScoringEngine(journal=owner.revision())
        review_stream(api, disk)
        prior = (copy.deepcopy(disk.state.to_dict()), copy.deepcopy(vars(disk.resolver)), disk.history.to_list())
        with pytest.raises(ValueError):
            disk.correct_event(2, reason='Synthetic invalid replacement', expected_revision=0, **replacement)
        assert disk.revision == 0 and len(disk.corrections) == 0
        assert (disk.state.to_dict(), vars(disk.resolver), disk.history.to_list()) == prior
        disk.correct_event(2, reason='Retry valid review', expected_revision=0, line_call=call(api, 'SERVE_IN', 2))
        assert disk.revision == 1 and disk.state.points_p1 == 1


def test_failed_append_restores_both_rule_and_resolver_then_retry_succeeds(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        journal = owner.revision()
        class RejectWrites:
            def __getattr__(self, name):
                return getattr(journal, name)
            def append_transition(self, *args, **kwargs):
                raise OSError('Explicit synthetic disk write failure before persistence')
        disk = api.TennisScoringEngine(journal=RejectWrites())
        prior = (copy.deepcopy(disk.state.to_dict()), copy.deepcopy(vars(disk.resolver)))
        with pytest.raises(OSError, match='disk write failure'):
            disk.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
        assert (disk.state.to_dict(), vars(disk.resolver)) == prior
        assert len(disk.history.entries) == len(disk._inputs) == 0
        disk._journal = journal
        disk.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
        assert disk.state.ball_state == api.BallPlayState.SERVE_STARTED
        assert len(disk._inputs) == len(disk.history.entries) == 1


def test_nullable_line_call_survives_typed_replay_and_nonfinite_input_is_rejected_first(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        disk = api.TennisScoringEngine(journal=owner.revision())
        disk.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
        evidence = api.LineCallEvidence.unavailable(2, 2, api.LineCallContext.SERVE, 'Synthetic missing proof')
        disk.process_match_event(2, 'BOUNCE', .1, line_call=evidence)
        disk.correct_event(1, reason='Synthetic confidence review', expected_revision=0, confidence=.8)
        stored = disk._decode_input(disk._inputs[1])['line_call']
        assert stored.to_dict() == evidence.to_dict()
        assert stored.decision is api.LineCallDecision.REVIEW_REQUIRED and stored.bounce_position_m is None
        invalid = copy.deepcopy(evidence)
        invalid.confidence = float('nan')
        prior = (disk.state.to_dict(), len(disk._inputs))
        with pytest.raises(ValueError):
            disk.process_match_event(3, 'BOUNCE', .2, line_call=invalid)
        assert (disk.state.to_dict(), len(disk._inputs)) == prior


def test_nonempty_journal_cannot_be_rebound_to_a_fresh_initial_state(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        journal = owner.revision()
        engine = api.TennisScoringEngine(journal=journal)
        engine.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
        with pytest.raises(ValueError, match='empty journal'):
            api.TennisScoringEngine(journal=journal)


def test_scoped_file_journals_reset_state_and_have_separate_idempotency(api, tmp_path):
    configured = api.TennisScoringEngine(initial_server_id=2)
    configured.process_match_event(99, 'CONSEQUENTIAL_GAP', 0.)
    configured_state = copy.deepcopy(configured.state.to_dict())
    for _ in range(2):
        with configured.journal_context(tmp_path) as engine:
            assert engine is not configured and engine.state.score_confirmed
            assert engine.state.server_id == 2 and len(engine.history.entries) == 0
            engine.process_match_event(1, 'SERVE_CONTACT', 0., player_id=2)
            assert engine.state.ball_state == api.BallPlayState.SERVE_STARTED
        assert configured.state.to_dict() == configured_state
        assert not list(tmp_path.glob('scoring_journal_*.sqlite3*'))


@pytest.mark.parametrize('commit_then_raise', [False, True])
def test_uncertain_correction_commit_disables_old_engine_and_all_exports(api, tmp_path, commit_then_raise):
    from tests.test_scoring_journal import ConnectionProxy
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        review_stream(api, engine)
        owner.flush()
        class AmbiguousCommit(ConnectionProxy):
            def commit(self):
                if commit_then_raise:
                    self.connection.commit()
                raise OSError('Synthetic uncertain correction COMMIT')
        owner.connection = AmbiguousCommit(owner.connection)
        with pytest.raises(OSError, match='uncertain correction'):
            engine.correct_event(2, reason='Explicit synthetic serve review', expected_revision=0,
                line_call=call(api, 'SERVE_IN', 2))
        target = tmp_path / 'state.json'
        target.write_text('prior output', encoding='utf-8')
        for action in (engine.get_current_state, lambda: engine.save_state_json(str(target)),
                lambda: engine.save_history_json(str(target)), lambda: len(engine._inputs)):
            with pytest.raises(RuntimeError, match='invalid after a transaction failure'):
                action()
        assert target.read_text() == 'prior output'


def test_closed_disk_engine_rejects_state_read_and_export(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
    target = tmp_path / 'state.json'
    target.write_text('prior output')
    for action in (engine.get_current_state, lambda: engine.save_state_json(str(target))):
        with pytest.raises(RuntimeError, match='not open'):
            action()
    assert target.read_text() == 'prior output'


def test_journal_rule_retention_and_correction_capacity(api, tmp_path):
    """Synthetic engine replay capacity, distinct from inference/match capacity."""
    rows = []
    for count in (257, 1025, 4097):
        directory = tmp_path / str(count)
        directory.mkdir()
        gc.collect()
        tracemalloc.start()
        with ScoringJournal(directory) as owner:
            disk = api.TennisScoringEngine(journal=owner.revision())
            for index in range(count):
                disk.process_match_event(index + 1, 'UNRECOGNIZED_CONTACT', index / 50., confidence=.9)
            retained, peak = tracemalloc.get_traced_memory()
            assert not isinstance(disk.history.entries, list) and not isinstance(disk._inputs, list)
            assert not isinstance(disk.history.processed_event_ids, set)
            disk.correct_event(1, reason='Explicit synthetic confidence correction', expected_revision=0, confidence=.8)
            corrected_retained, corrected_peak = tracemalloc.get_traced_memory()
            disk.save_history_json(str(directory / 'score_history.json'))
            export_retained, export_peak = tracemalloc.get_traced_memory()
            assert len(disk.history.entries) == len(disk._inputs) == count
            assert len(disk.corrections[0]['previous_history']) == count
            assert disk.revision == 1
            rows.append(dict(events=count, retained_traced_bytes=retained, peak_traced_bytes=peak,
                correction_retained_traced_bytes=corrected_retained, correction_peak_traced_bytes=corrected_peak,
                export_retained_traced_bytes=export_retained, export_peak_traced_bytes=export_peak,
                models_used=False, media_used=False, qualification_evidence=False))
        tracemalloc.stop()
        assert not list(directory.glob('*.sqlite3*'))
    (tmp_path / 'capacity.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    assert rows[-1]['export_peak_traced_bytes'] < rows[0]['export_peak_traced_bytes'] + 2 * 1024 * 1024
