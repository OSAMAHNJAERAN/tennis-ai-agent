"""Independent regressions for native rule/context revision boundaries.

These reuse the actual model-free scoring definitions and small synthetic
fixtures. They neither run inference nor qualify real identity/clock accuracy.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_point_timeline import api, frames, event, binding_context, drive, SuppliedCalls
from src.scoring.point_timeline import (PointTimeline, ReviewedRuleSignal,
    candidate_event_context, process_timeline_event, merge_rule_inputs)
from src.utils.point_analysis_sequence import PointAnalysisSequence


@pytest.mark.parametrize('changed_field', ['identity_revision', 'calibration_revision',
                                          'participant_mapping_ref', 'geometry_qualification_ref'])
def test_active_rally_requires_explicit_revision_when_current_proof_changes(api, tmp_path, changed_field):
    source = frames()
    if changed_field == 'identity_revision':
        for row in source[2:]:
            row[1]['revision'] = 1

    def context_at(ev, engine):
        binding = binding_context(ev, source, engine).binding
        if ev.frame_index >= 2:
            binding = replace(binding, **{changed_field: 1 if changed_field == 'identity_revision'
                                         else 'explicitly-different-reviewed-proof'})
        return candidate_event_context(ev, source, lambda *args: binding,
            analysis_id='fixture', source_fingerprint='synthetic-source-v1')

    inputs = [event(1, 'SERVE_CONTACT', 1), event(2, 'BOUNCE'),
              event(3, 'PLAYER_2_HIT', 2), event(4, 'BOUNCE')]
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint='synthetic-source-v1') as owner:
        engine, timeline, _ = drive(api, owner, inputs, source,
            decisions={2: 'SERVE_IN', 4: 'OUT'}, contexts=context_at)
        returned = owner.event(3)
        assert returned['rule_event_type'] == 'CONSEQUENTIAL_GAP'
        assert returned['outcome_type'] == 'POINT_REVIEW_REQUIRED'
        assert engine.state.score_confirmed is False
        assert len(owner.rows('points', status='COMPLETED')) == 0
        assert len(owner.rows('rallies', status='COMPLETED')) == 0
        assert next(iter(owner.rows('points')))['winner_id'] is None


@pytest.mark.parametrize('owner_analysis,owner_source', [
    ('different-analysis', 'synthetic-source-v1'),
    ('fixture', 'different-source'),
    ('different-analysis', 'different-source'),
])
def test_native_handoff_rejects_context_not_bound_to_its_owner(api, tmp_path, owner_analysis, owner_source):
    source = frames()
    engine = api.TennisScoringEngine()
    ev = event(1, 'SERVE_CONTACT', 1)
    context = binding_context(ev, source, engine)
    assert context.binding_current  # Valid within its own frozen analysis/source.
    with PointAnalysisSequence(tmp_path, owner_analysis, source_fingerprint=owner_source) as owner:
        timeline = PointTimeline(owner, engine.state.to_dict(), source_frames=source)
        caller = SuppliedCalls(api, {})
        version = engine.state.state_version
        with pytest.raises(ValueError):
            process_timeline_event(ev, context, engine, timeline, caller,
                ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert engine.state.state_version == version
        assert engine.history.entries == []
        assert len(owner.rows('transitions')) == len(owner.rows('attempts')) == 0
        assert caller.requests == []


@pytest.mark.parametrize('mutation', ['player_id', 'timestamp_s', 'confidence', 'binding_evidence_ref',
                                     'participant_mapping_ref', 'geometry_qualification_ref'])
def test_duplicate_id_is_retry_only_when_raw_payload_and_full_context_match(api, tmp_path, mutation):
    source = frames()
    original = event(1, 'SERVE_CONTACT', 1)
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint='synthetic-source-v1') as owner:
        engine, timeline, caller = drive(api, owner, [original], source)
        version = engine.state.state_version
        retry = SimpleNamespace(**vars(original))
        if mutation in ('player_id', 'timestamp_s', 'confidence'):
            setattr(retry, mutation, {'player_id': 2, 'timestamp_s': .9, 'confidence': .1}[mutation])
        context = binding_context(retry, source, engine)
        if mutation in ('binding_evidence_ref', 'participant_mapping_ref', 'geometry_qualification_ref'):
            field = 'evidence_ref' if mutation == 'binding_evidence_ref' else mutation
            changed = replace(context.binding, **{field: 'different-explicit-reviewed-evidence'})
            context = candidate_event_context(retry, source, lambda *args: changed,
                analysis_id='fixture', source_fingerprint='synthetic-source-v1')
            assert context.binding_current
        with pytest.raises(ValueError):
            process_timeline_event(retry, context, engine, timeline, caller,
                ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert engine.state.state_version == version
        assert len(owner.rows('transitions')) == len(owner.rows('attempts')) == 1
        # A rejected changed payload must not poison the original exact retry.
        _, outcome, _ = process_timeline_event(original, binding_context(original, source, engine),
            engine, timeline, caller, ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert outcome.outcome_type.value == 'SERVE_IN_PLAY'
        assert engine.state.state_version == version


def test_duplicate_reviewed_rule_signal_requires_same_evidence_reference(api, tmp_path):
    source = frames()
    with PointAnalysisSequence(tmp_path, 'fixture', source_fingerprint='synthetic-source-v1') as owner:
        engine, timeline, caller = drive(api, owner, [event(1, 'SERVE_CONTACT', 1)], source)
        stub = event(2, 'UNKNOWN')
        context = binding_context(stub, source, engine)
        signal = ReviewedRuleSignal(2, 1, .2, 'SERVICE_LET', 'reviewed-let-source-one', context)
        _, first, _ = process_timeline_event(signal, context, engine, timeline, caller,
            ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert first.outcome_type.value == 'SERVICE_LET'
        version = engine.state.state_version
        with pytest.raises(ValueError):
            process_timeline_event(replace(signal, evidence_ref='different-reviewed-let-source'), context,
                engine, timeline, caller, ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert engine.state.state_version == version
        _, retried, _ = process_timeline_event(signal, context, engine, timeline, caller,
            ball_points=(), homography_matrix=None, frames=(), geometry_available=True)
        assert retried.outcome_type.value == 'SERVICE_LET'
        assert engine.state.state_version == version


@pytest.mark.parametrize('signal_time,bounce_time', [(.2, .3), (.2, .2)])
def test_same_frame_reviewed_signal_merges_by_explicit_time_and_event_id(api, signal_time, bounce_time):
    source = frames()
    engine = api.TennisScoringEngine()
    serve = event(1, 'SERVE_CONTACT', 1)
    bounce = event(3, 'BOUNCE')
    bounce.frame_index, bounce.timestamp_s = 1, bounce_time
    context = binding_context(bounce, source, engine)
    signal = ReviewedRuleSignal(2, 1, signal_time, 'SERVICE_LET', 'explicit-reviewed-earlier-let', context)
    merged = list(merge_rule_inputs([serve, bounce], [signal]))
    assert [value.event_id for value in merged] == [1, 2, 3]
    assert [value.timestamp_s for value in merged] == sorted(value.timestamp_s for value in merged)


def test_same_frame_same_time_same_id_different_rule_types_are_ambiguous(api):
    source = frames()
    engine = api.TennisScoringEngine()
    bounce = event(3, 'BOUNCE')
    bounce.frame_index, bounce.timestamp_s = 1, .2
    signal = ReviewedRuleSignal(3, 1, .2, 'SERVICE_LET', 'reviewed-order-insufficient',
        binding_context(bounce, source, engine))
    with pytest.raises(ValueError):
        list(merge_rule_inputs([event(1, 'SERVE_CONTACT', 1), bounce], [signal]))


def test_later_frame_with_reversed_declared_time_is_rejected_before_that_event(api):
    source = frames()
    engine = api.TennisScoringEngine()
    first = event(1, 'SERVE_CONTACT', 1)
    first.timestamp_s = .4
    signal = ReviewedRuleSignal(2, 1, .2, 'SERVICE_LET', 'reviewed-clock-conflict',
        binding_context(event(2, 'UNKNOWN'), source, engine))
    merged = iter(merge_rule_inputs([first], [signal]))
    assert next(merged).event_id == 1
    with pytest.raises(ValueError):
        next(merged)
