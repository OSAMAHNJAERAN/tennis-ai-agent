"""Native source binding and typed journal views; synthetic rule inputs only."""
from contextlib import ExitStack
import copy
from dataclasses import replace
import json
import sys
from types import ModuleType

import pytest

from src.scoring.point_timeline import PointTimeline, process_timeline_event
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.scoring_journal import ScoringJournal
from tests.test_point_timeline import binding_context, event, frames
from tests.test_scoring_service_phase import actual_classes, call


@pytest.fixture
def api(monkeypatch):
    actual = actual_classes()
    for name in ('src.scoring.match_state', 'src.scoring.point_outcome_resolver',
                 'src.line_calling.line_call_engine'):
        module = ModuleType(name)
        module.__dict__.update(vars(actual))
        monkeypatch.setitem(sys.modules, name, module)
    return actual


class SuppliedEvidence:
    def __init__(self, evidence):
        self.evidence = evidence
        self.requests = []

    def evaluate_bounce(self, **kwargs):
        self.requests.append(kwargs)
        return self.evidence


def run_event(ev, source, engine, timeline, caller, *, geometry_available=True):
    return process_timeline_event(ev, binding_context(ev, source, engine), engine,
        timeline, caller, ball_points=(), homography_matrix=None, frames=(),
        geometry_available=geometry_available)


def snapshot(engine, timeline):
    """Include accepted SQL records and active context, not just visible score."""
    cursor = timeline.owner.connection.execute(
        'SELECT kind,position,uid,parent,status,event_id,frame,payload '
        'FROM records ORDER BY kind,position')
    try:
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return copy.deepcopy((engine.state.to_dict(), vars(engine.resolver),
        engine.history.to_list(), list(engine._inputs), set(engine.completed_point_ids),
        rows, timeline.point, timeline.attempt, timeline.rally,
        timeline.previous_context, timeline.replay_index))


@pytest.mark.parametrize('field,value', [
    ('event_id', 2), ('event_id', True), ('event_id', 1.0),
    ('event_id', '1'), ('event_id', None),
    ('bounce_frame', 2), ('bounce_frame', True), ('bounce_frame', 1.0),
    ('bounce_frame', '1'), ('bounce_frame', None),
])
def test_malformed_native_provider_is_rejected_before_any_acceptance(
        api, tmp_path, field, value):
    source = frames()
    with ExitStack() as stack:
        journal = stack.enter_context(ScoringJournal(tmp_path))
        owner = stack.enter_context(PointAnalysisSequence(tmp_path, 'fixture',
            source_fingerprint='synthetic-source-v1'))
        engine = api.TennisScoringEngine(journal=journal.revision())
        timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
        # IDs need not equal frame indices. Here True == 1 for both fields,
        # exposing coercion/equality-only validation rather than just mismatch.
        contact = event(9, 'SERVE_CONTACT', 1)
        contact.frame_index, contact.timestamp_s = 0, .1
        bounce = event(1, 'BOUNCE')
        bounce.frame_index, bounce.timestamp_s = 1, .2
        valid = call(api, 'SERVE_IN', 1)
        run_event(contact, source, engine, timeline, SuppliedEvidence(valid))
        before = snapshot(engine, timeline)
        caller = SuppliedEvidence(replace(valid, **{field: value}))

        with pytest.raises(ValueError, match='(?i)(line.call|evidence|bounce|binding|source)'):
            run_event(bounce, source, engine, timeline, caller)

        assert len(caller.requests) == 1
        assert snapshot(engine, timeline) == before
        assert owner.event(bounce.event_id) is None
        assert bounce.event_id not in engine.history.processed_event_ids

        # A valid retry of the identical physical input is still accepted once.
        state, outcome, accepted = run_event(bounce, source, engine, timeline,
            SuppliedEvidence(valid))
        assert accepted.to_dict() == valid.to_dict()
        assert state.ball_state is api.BallPlayState.LIVE
        assert outcome.outcome_type is api.PointOutcomeType.SERVE_IN_PLAY
        assert len(engine.history.entries) == len(engine._inputs) == 2
        assert len(owner.rows('transitions')) == 2


@pytest.mark.parametrize('geometry_available', [False, True])
def test_native_unavailable_evidence_keeps_nulls_and_is_an_accepted_abstention(
        api, tmp_path, geometry_available):
    source = frames()
    with ExitStack() as stack:
        journal = stack.enter_context(ScoringJournal(tmp_path))
        owner = stack.enter_context(PointAnalysisSequence(tmp_path, 'fixture',
            source_fingerprint='synthetic-source-v1'))
        engine = api.TennisScoringEngine(journal=journal.revision())
        timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
        contact, bounce = event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE')
        evidence = api.LineCallEvidence.unavailable(2, 1, api.LineCallContext.SERVE,
            'Explicit synthetic unavailable measurement', 'SYNTHETIC_MISSING_GEOMETRY')
        caller = SuppliedEvidence(evidence)
        run_event(contact, source, engine, timeline, caller)
        state, outcome, accepted = run_event(bounce, source, engine, timeline,
            caller, geometry_available=geometry_available)
        assert len(caller.requests) == int(geometry_available)
        assert accepted.event_id == 2 and accepted.bounce_frame == 1
        assert accepted.decision is api.LineCallDecision.REVIEW_REQUIRED
        assert accepted.bounce_position_px is accepted.bounce_position_m is None
        assert accepted.ball_edge_margin_cm is accepted.position_uncertainty_cm is None
        assert state.points_p1 == state.points_p2 == 0
        assert state.ball_state is api.BallPlayState.SERVE_STARTED
        assert outcome.outcome_type is api.PointOutcomeType.POINT_REVIEW_REQUIRED
        assert owner.event(2) is not None and len(engine.history.entries) == 2
        recovered = list(engine.native_line_calls())
        assert len(recovered) == 1 and recovered[0].to_dict() == accepted.to_dict()


def test_native_line_call_view_reconstructs_finite_typed_evidence_and_nulls(api, tmp_path):
    with ScoringJournal(tmp_path) as journal:
        engine = api.TennisScoringEngine(journal=journal.revision())
        measured = call(api, 'SERVE_IN', 2)
        measured.bounce_frame = 7
        unavailable = api.LineCallEvidence.unavailable(3, 9, api.LineCallContext.RALLY,
            'Explicit synthetic abstention', 'SYNTHETIC_MISSING_PROOF')
        expected = [measured.to_dict(), unavailable.to_dict()]
        engine.process_match_event(1, 'SERVE_CONTACT', 0., player_id=1)
        engine.process_match_event(2, 'BOUNCE', .1, line_call=measured)
        engine.process_match_event(3, 'BOUNCE', .2, line_call=unavailable)
        measured.bounce_position_m = (123., 456.)
        view = engine.native_line_calls()
        recovered = list(view)
        assert len(view) == 2 and [item.to_dict() for item in recovered] == expected
        assert all(isinstance(item, api.LineCallEvidence) for item in recovered)
        assert recovered[0].decision is api.LineCallDecision.SERVE_IN
        assert recovered[0].decision_context is api.LineCallContext.SERVE
        assert recovered[0].nearest_line is api.CourtLineType.NEAR_SERVICE_LINE
        assert recovered[0].bounce_position_m == (3., 6.)
        assert type(recovered[0].bounce_position_m) is tuple
        assert recovered[1].decision is api.LineCallDecision.REVIEW_REQUIRED
        assert recovered[1].nearest_line is api.CourtLineType.NONE
        assert recovered[1].reason_code == 'SYNTHETIC_MISSING_PROOF'
        assert recovered[1].confidence == 0. and recovered[1].bounce_position_m is None
        serialized = json.loads(json.dumps(expected, allow_nan=False))
        assert serialized[0]['bounce_position_m'] == [3., 6.]
        assert serialized[1]['bounce_position_m'] is None
        assert view[0].to_dict() == recovered[0].to_dict()


def test_legacy_engine_cannot_claim_a_native_line_call_journal(api):
    with pytest.raises(RuntimeError, match='(?i)journal'):
        api.TennisScoringEngine().native_line_calls()


def test_native_line_call_view_and_new_view_reject_closed_owner(api, tmp_path):
    with ScoringJournal(tmp_path) as journal:
        engine = api.TennisScoringEngine(journal=journal.revision())
        engine.process_match_event(1, 'BOUNCE', 0., line_call=call(api, 'IN', 1))
        view = engine.native_line_calls()
        assert len(view) == 1
        suspended = iter(view)
        assert next(suspended).event_id == 1
        suspended_frames = iter(view.by_frame(1, 1))
        assert next(suspended_frames)[0] == 1
    for operation in (engine.native_line_calls, lambda: len(view), lambda: list(view),
                      lambda: view[0], lambda: next(suspended),
                      lambda: next(suspended_frames)):
        with pytest.raises(RuntimeError, match='(?i)(journal|owner)'):
            operation()
    assert not list(tmp_path.glob('scoring_journal_*.sqlite3*'))


def test_native_line_call_view_rejects_actual_uncertain_commit_owner(api, tmp_path):
    class RejectCommit:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            raise OSError('Explicit synthetic uncertain native line-call COMMIT')

    with ScoringJournal(tmp_path) as journal:
        engine = api.TennisScoringEngine(journal=journal.revision())
        engine.process_match_event(1, 'BOUNCE', 0., line_call=call(api, 'IN', 1))
        view = engine.native_line_calls()
        assert len(view) == 1
        journal.connection = RejectCommit(journal.connection)
        with pytest.raises(OSError, match='uncertain native line-call COMMIT'):
            journal.flush()
        for operation in (engine.native_line_calls, lambda: len(view), lambda: list(view),
                          lambda: view[0]):
            with pytest.raises(RuntimeError, match='(?i)(invalid|failure)'):
                operation()
