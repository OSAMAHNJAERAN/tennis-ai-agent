"""Native annotation query/export contracts; finite synthetic evidence only."""
import ast
from contextlib import ExitStack
from dataclasses import replace
import json
import os
import time
from types import SimpleNamespace

import pytest

from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.scoring_journal import ScoringJournal
from tests.test_native_line_call_retention import BOUNCE, LazyBounces, source_nodes
from tests.test_scoring_service_phase import actual_classes, call


@pytest.fixture
def api():
    return actual_classes()


def append(api, engine, ident, frame, *, unavailable=False):
    evidence = (api.LineCallEvidence.unavailable(ident, frame, api.LineCallContext.RALLY,
                'Explicit synthetic missing geometry') if unavailable else
                replace(call(api, 'IN', ident, 'RALLY'), bounce_frame=frame))
    engine.process_match_event(ident, 'BOUNCE', ident / 10., line_call=evidence)
    return evidence


def test_indexed_window_keeps_last_per_frame_in_first_insertion_order(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        originals = [append(api, engine, ident, frame) for ident, frame in
                     ((1, 97), (2, 2), (3, 97), (4, 20), (5, 2), (6, 21))]
        calls = engine.native_line_calls()
        assert [item.to_dict() for item in calls] == [item.to_dict() for item in originals]
        assert [(frame, item.event_id) for frame, item in calls.by_frame(2, 97)] == [(97, 3), (2, 5), (20, 4), (21, 6)]
        assert [(frame, item.event_id) for frame, item in calls.by_frame(2, 20)] == [(2, 5), (20, 4)]
        assert list(calls.by_frame(21, 20)) == []
        assert list(calls.by_frame(-18, 0)) == []
        assert [(frame, item.event_id) for frame, item in calls.by_frame(2, 2)] == [(2, 5)]


@pytest.mark.parametrize('selection', [0, -1, slice(None, None, -1), slice(1, 3), slice(None, None, 2)])
def test_typed_view_indexing_and_slices_preserve_nullable_values(api, tmp_path, selection):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        originals = [append(api, engine, ident, ident, unavailable=ident % 2 == 0) for ident in range(1, 5)]
        view = engine.native_line_calls()
        selected, expected = view[selection], originals[selection]
        if isinstance(selection, slice):
            assert [item.to_dict() for item in selected] == [item.to_dict() for item in expected]
        else:
            assert selected.to_dict() == expected.to_dict()
            assert selected is not view[selection]
        changed = view[0]
        changed.reason = 'mutation of a detached snapshot'
        assert view[0].reason == originals[0].reason


def test_view_ignores_non_call_inputs_duplicate_retry_and_empty_queries(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        view = engine.native_line_calls()
        assert len(view) == 0 and list(view) == list(view.by_frame(0, 100)) == []
        with pytest.raises(IndexError):
            _ = view[0]
        engine.process_match_event(1, 'UNRECOGNIZED_CONTACT', .1)
        original = append(api, engine, 2, 10, unavailable=True)
        engine.process_match_event(2, 'BOUNCE', .2, line_call=original)
        assert len(view) == 1 and view[0].to_dict() == original.to_dict()
        assert len(engine._inputs) == 2
        with pytest.raises(IndexError):
            _ = view[-2]


def test_window_query_uses_the_frame_range_index(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        for ident in range(1, 34):
            append(api, engine, ident, ident)
        statements = []
        owner.connection.set_trace_callback(statements.append)
        assert len(list(engine.native_line_calls().by_frame(10, 28))) == 19
        owner.connection.set_trace_callback(None)
        statement = next(item for item in statements if item.startswith('SELECT selected.frame'))
        cursor = owner.connection.execute('EXPLAIN QUERY PLAN '+statement)
        try:
            plans = cursor.fetchall()
        finally:
            cursor.close()
        assert any('scoring_line_call_frame' in str(row) and 'SEARCH' in str(row) for row in plans)


def test_line_call_views_remain_pinned_across_real_score_correction(api, tmp_path):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        original = append(api, engine, 1, 5)
        old = engine.native_line_calls()
        corrected = replace(original, reason='Explicit synthetic corrected evidence')
        engine.correct_event(1, reason='Synthetic native-view revision test', expected_revision=0,
                             line_call=corrected)
        current = engine.native_line_calls()
        assert old[0].to_dict() == original.to_dict()
        assert current[0].to_dict() == corrected.to_dict()
        assert list(old.by_frame(5, 5))[0][1].reason == original.reason
        assert list(current.by_frame(5, 5))[0][1].reason == corrected.reason


@pytest.mark.parametrize('failure', ['sanitize', 'fsync', 'replace'])
def test_actual_native_export_failure_preserves_previous_file_and_closes_reader(api, tmp_path, monkeypatch, failure):
    with ScoringJournal(tmp_path) as owner:
        engine = api.TennisScoringEngine(journal=owner.revision())
        for ident in (1, 2, 3):
            append(api, engine, ident, ident)
        target = tmp_path / 'line_calls.json'
        previous = b'previous complete artifact\n'
        target.write_bytes(previous)
        def sanitize(value):
            if failure == 'sanitize' and value['event_id'] == 2:
                raise RuntimeError('Explicit synthetic serializer failure')
            return value
        def reject(*args):
            raise OSError('Explicit synthetic publication failure')
        if failure in ('fsync', 'replace'):
            monkeypatch.setattr(os, failure, reject)
        path, _, export, _ = source_nodes()
        namespace = dict(disk_players=True, output_dir=str(tmp_path), os=os,
            write_json_array_artifact=write_json_array_artifact, _sanitize=sanitize,
            line_call_evidences=engine.native_line_calls())
        with pytest.raises((RuntimeError, OSError), match='Explicit synthetic'):
            exec(compile(ast.Module(body=[export], type_ignores=[]), str(path), 'exec'), namespace)
        assert target.read_bytes() == previous
        assert not list(tmp_path.glob('.line_calls.json.*.tmp'))
        assert not owner.readers
        assert len(engine.native_line_calls()) == 3


def test_dead_event_set_tracks_only_current_transition_outcomes_without_cache(tmp_path):
    with PointAnalysisSequence(tmp_path, 'synthetic') as owner:
        view = owner.dead_event_ids()
        assert not isinstance(view, (set, frozenset))
        assert len(view) == 0 and 1 not in view
        for ident, outcome in ((3, 'DEAD_BALL_IGNORED'), (1, 'RALLY_CONTINUES'), (8, 'DEAD_BALL_IGNORED')):
            owner.put('transitions', str(ident), {'event_id': ident, 'outcome_type': outcome},
                      event_id=ident, frame=ident)
        owner.put('points', 'ignored', {'outcome_type': 'DEAD_BALL_IGNORED'}, event_id=99, frame=99)
        assert len(view) == 2 and list(view) == [3, 8]
        assert 3 in view and 8 in view and 1 not in view and 99 not in view
        assert 3.0 in view and '3' not in view and b'3' not in view
        owner.put('transitions', '3', {'event_id': 3, 'outcome_type': 'RALLY_CONTINUES'}, event_id=3, frame=3)
        assert len(view) == 1 and 3 not in view and list(view) == [8]
        with pytest.raises(TypeError):
            _ = [] in view
        owner.seal()
        assert 8 in view and list(view) == [8]


def test_actual_legacy_phase6_still_uses_its_list_map_set_and_existing_export(api, tmp_path):
    path, statements, export, _ = source_nodes()
    initial = api.MatchState(match_status='SUSPENDED')
    engine = api.TennisScoringEngine(initial_state=initial)
    class SuppliedCalls:
        def evaluate_bounce(self, **kwargs):
            return replace(call(api, 'IN', kwargs['event_id'], 'RALLY'), bounce_frame=kwargs['bounce_frame'])
    events = LazyBounces(7)
    namespace = dict(vars(api), self=SimpleNamespace(scoring_engine=engine, line_call_engine=SuppliedCalls()),
        disk_players=False, authoritative_enabled=True, detected_events=events,
        EventType=SimpleNamespace(BOUNCE=BOUNCE), is_h_valid=True, ball_points=(),
        homography_matrix=None, frames=(), output_dir=str(tmp_path), os=os,
        time=time, t0=time.time(), json=json, _sanitize=lambda value: value)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), namespace)
    assert isinstance(namespace['line_call_evidences'], list)
    assert isinstance(namespace['line_call_map'], dict)
    assert isinstance(namespace['dead_event_ids'], set)
    assert len(namespace['line_call_evidences']) == len(namespace['dead_event_ids']) == 7
    assert [(frame, value.event_id) for frame, value in namespace['line_call_map'].items()] == [(0, 2), (1, 4), (2, 6), (3, 7)]
    exec(compile(ast.Module(body=[export], type_ignores=[]), str(path), 'exec'), namespace)
    exported = json.loads((tmp_path / 'line_calls.json').read_text())
    expected = json.loads(json.dumps({'schema_version': '1.0', 'line_calls':
        [value.to_dict() for value in namespace['line_call_evidences']]}))
    assert exported == expected and engine.state.to_dict() == initial.to_dict()


@pytest.mark.parametrize('primary', [None, RuntimeError('Explicit synthetic consumer failure')])
def test_dead_set_suspended_reader_cleanup_preserves_neighbor_and_primary(tmp_path, primary):
    neighbor = tmp_path / 'neighbor.sqlite3'
    neighbor.write_bytes(b'unrelated fixture')
    owner = PointAnalysisSequence(tmp_path, 'synthetic')
    try:
        with owner:
            owner.put('transitions', '1', {'outcome_type': 'DEAD_BALL_IGNORED'}, event_id=1, frame=1)
            view = owner.dead_event_ids()
            iterator = iter(view)
            assert next(iterator) == 1 and owner.readers
            if primary is not None:
                raise primary
    except RuntimeError as error:
        assert error is primary
    assert neighbor.read_bytes() == b'unrelated fixture'
    assert not owner.readers and not owner.path.exists()
    iterator.close()
    with pytest.raises(RuntimeError, match='not open'):
        len(view)
    with pytest.raises(RuntimeError, match='not open'):
        _ = 1 in view
    with pytest.raises(RuntimeError, match='not open'):
        list(view)
