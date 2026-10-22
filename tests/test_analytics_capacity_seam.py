"""Real model-free analytics source blocks; synthetic rows are not match GT."""
import ast
import asyncio
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import gc
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import weakref

import numpy as np
import pytest

from src.scoring.match_state import MatchState
from src.shot_analysis.shot_types import (ShotEventEvidence, ShotType, ShotDirection,
    ShotClassificationSource, ShotOutcome, CourtZone3x3)
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision
from src.utils.shot_evidence_sequence import ShotEvidenceSequence, require_shot_result_owner
from src.utils import shot_evidence_sequence as storage
from src.utils.analytics_json_artifact import (write_membership_analytics, write_match_analytics,
                                              native_match_analytics_summary)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/analytics_capacity_implementation_v1/baseline/src'
PINS = {
    'rally_analyzer.py': '4dce1459717852d9679223a5f2ca565fb2d3759e142d6f7d0c7eca5a8a7560b7',
    'shot_statistics.py': 'f0c16cb44dbde33cea571d120b97f924793a3c65c3bbe429bf1d3f5c7c8a349e',
    'serve_analyzer.py': 'faa33d91de481c16639c8ca5fd1bc67ffa6d24a457f65ad52bb8f4d2b20b9d85',
    'match_analytics.py': 'dbf3c6d619c0e730cc264ce11f318c2882308573fcc4c9ecbd46e4a9d01525b0',
    'phase6_pipeline.py': '975d1cdb478a30ed3b4ea6cc4491fda95f08e9d3517427e71be3e02814bce9e3',
}


def consumers(prechange=False):
    namespace = dict(globals())
    for filename in ('rally_analyzer.py', 'shot_statistics.py', 'serve_analyzer.py', 'match_analytics.py'):
        path = (BASELINE if prechange else ROOT / 'src') / 'analytics' / filename
        if prechange:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[filename]
        nodes = [node for node in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def shot(index=0, **changes):
    value = dict(shot_id=index % 7, match_event_id=index+1, frame_index=index*8, timestamp_s=index*.321,
        player_id=[1, 2, None][index % 3], shot_type=list(ShotType)[index % len(ShotType)],
        shot_confidence=.8, classification_source=ShotClassificationSource.GEOMETRY_BASELINE,
        direction=list(ShotDirection)[index % len(ShotDirection)], direction_confidence=.7,
        direction_source='SYNTHETIC_ONLY', source_court_position_m=(3., None),
        landing_court_position_m=None if index % 4 == 0 else (0. if index % 2 else 8.12345, 5.67895),
        landing_zone=None if index % 4 == 0 else list(CourtZone3x3)[index % len(CourtZone3x3)],
        bounce_frame=None if index % 4 == 0 else index*8+4,
        outcome=list(ShotOutcome)[index % len(ShotOutcome)], is_dead_ball=index % 17 == 16)
    value.update(changes)
    return ShotEventEvidence(**value)


def normalize(value):
    if hasattr(value, '__dataclass_fields__'):
        return {key: normalize(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, Sequence)) and not isinstance(value, (str, bytes)):
        return [normalize(item) for item in value]
    return value


def all_results(classes, rows, owner=None, *, rallies=True, fps=25.):
    state = MatchState(match_id='synthetic-capacity-only')
    calls = [SimpleNamespace(event_id=1, bounce_frame=12, decision=LineCallDecision.SERVE_FAULT),
             SimpleNamespace(event_id=2, bounce_frame=12, decision=LineCallDecision.SERVE_IN)]
    kwargs = {'result_owner': owner} if owner is not None else {}
    rally = classes['RallyAnalyzer'].analyze_rallies(rows, fps=fps, **kwargs)
    points = classes['MatchAnalyticsAggregator'].build_point_analytics(state, rows, rally if rallies else [], calls,
        p1_distance_m=float('nan'), p2_distance_m=-0., **kwargs)
    match = classes['MatchAnalyticsAggregator'].build_match_analytics(state, rows, rally, calls,
        [(0., 0.), None, (10.97, 23.77)], [], None, -0., float('inf'), 3.33333,
        ball_speed_summary={'available': False}, **kwargs)
    return {'rallies': rally, 'points': points, 'match': match}


@pytest.mark.parametrize('count', [0, 1, 7, 55])
@pytest.mark.parametrize('dead', [False, True])
@pytest.mark.parametrize('rallies', [False, True])
@pytest.mark.parametrize('fps', [None, 25.])
def test_native_and_legacy_outputs_match_exact_prechange(tmp_path, count, dead, rallies, fps):
    values = [shot(index, **({'is_dead_ball': True} if dead else {})) for index in range(count)]
    if len(values) > 2:
        # Preserve source order, duplicated IDs and irregular/tied timestamps.
        values[0].timestamp_s, values[1].timestamp_s = .5, .5
        values[-1].timestamp_s = .499
    expected = all_results(consumers(True), values, rallies=rallies, fps=fps)
    classes = consumers()
    assert normalize(all_results(classes, values, rallies=rallies, fps=fps)) == normalize(expected)
    with ShotEvidenceSequence(tmp_path) as rows:
        for value in values:
            rows.append(value)
        rows.seal()
        result = all_results(classes, rows, rows, rallies=rallies, fps=fps)
        assert normalize(result) == normalize(expected)
        assert not isinstance(result['points'][0]['shot_ids'], list)
        assert not isinstance(result['match']['ball_landing_heatmap'], list)


def phase6_parts(prechange=False):
    path = (BASELINE if prechange else ROOT / 'src') / 'pipeline/phase6_pipeline.py'
    if prechange:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[path.name]
    tree = ast.parse(path.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    start = next(i for i, node in enumerate(run.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'final_st' for target in node.targets))
    end = next(i for i, node in enumerate(run.body[start:], start) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'print'
               and 'Generated' in ast.unparse(node))
    exports = []
    for node in run.body:
        if isinstance(node, (ast.Expr, ast.With)) and any(name in ast.unparse(node)
            for name in ('"rallies.json"', "'rallies.json'", '"point_analytics.json"', "'point_analytics.json'",
                         '"match_analytics.json"', "'match_analytics.json'")):
            exports.append(node)
    assert len(exports) == 3
    returned = next(node for node in reversed(run.body) if isinstance(node, ast.Return))
    sanitize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sanitize_json_value')
    return path, run.body[start:end], exports, returned, sanitize


def phase6_run(rows, directory, *, authority=True, prechange=False, disk=True):
    classes = consumers(prechange)
    path, statements, exports, returned, sanitize = phase6_parts(prechange)
    state = MatchState(match_id='synthetic-capacity-only')
    namespace = dict(classes, self=SimpleNamespace(scoring_engine=SimpleNamespace(get_current_state=lambda: state)),
        shot_evidences=rows, shot_output=rows if disk else None, disk_players=disk,
        authoritative_enabled=authority, fps=25., line_call_evidences=[], p1_dist=None, p2_dist=None,
        p1_speed=None, p2_speed=None, p1_positions=[], p2_positions=[], speed_summary={'available': False},
        p1_motion={'distance_estimator': 'SYNTHETIC_ONLY'}, output_dir=str(directory),
        fps_proc=0., t_total=0., player_crop_provider=None, inference_resume_record=None)
    exec(compile(ast.Module(body=[sanitize]+statements, type_ignores=[]), str(path), 'exec'), namespace)
    namespace['_sanitize'] = namespace['sanitize_json_value']
    exec(compile(ast.Module(body=exports, type_ignores=[]), str(path), 'exec'), namespace)
    # Execute the exact return expression after the actual exporters.
    namespace['returned'] = eval(compile(ast.Expression(returned.value), str(path), 'eval'), namespace)
    return namespace


@pytest.mark.parametrize('authority', [False, True])
@pytest.mark.parametrize('disk', [False])
def test_actual_legacy_phase6_step10_exports_and_return_remain_compatible(tmp_path, authority, disk):
    # Native v2 intentionally fixes the old whole-file point/attempt semantics.
    # Its exact source blocks and exports are exercised in test_point_timeline.
    old_dir, new_dir = tmp_path / 'old', tmp_path / 'new'
    old_dir.mkdir(); new_dir.mkdir()
    values = [shot(index) for index in range(39)]
    expected = phase6_run(values, old_dir, authority=authority, prechange=True, disk=disk)
    with ShotEvidenceSequence(new_dir) as owner:
        for value in values:
            owner.append(value)
        actual = phase6_run(owner if disk else values, new_dir, authority=authority, disk=disk)
        for name in ('rallies.json', 'point_analytics.json', 'match_analytics.json'):
            assert json.loads((new_dir / name).read_text()) == json.loads((old_dir / name).read_text())
        returned = actual['returned']['match_analytics']
        if disk:
            assert 'ball_landing_heatmap' not in returned
            assert returned['ball_landing_heatmap_artifact'] == {
                'path': 'match_analytics.json', 'array_key': 'ball_landing_heatmap',
                'row_count': sum(value.landing_court_position_m is not None for value in values)}
            # This would reject a lazy Sequence/SQLite connection escaping.
            json.dumps(returned, allow_nan=False)
        else:
            assert returned == expected['returned']['match_analytics']
        assert owner._connection is not None
        assert owner._readers == set()
    assert not list(new_dir.glob('shot_evidence_*'))
    json.dumps(returned, allow_nan=False)  # Still usable after the source closes.


def test_views_require_exact_open_sealed_owner_and_freeze_mutations(tmp_path):
    with ShotEvidenceSequence(tmp_path) as rows, ShotEvidenceSequence(tmp_path) as other:
        rows.append(shot())
        with pytest.raises(ValueError, match='sealed'):
            rows.select()
        with pytest.raises(ValueError, match='sealed'):
            require_shot_result_owner(rows, rows)
        rows.seal()
        for source, owner in ((list(rows), rows), (rows, other)):
            with pytest.raises(ValueError, match='exact'):
                require_shot_result_owner(source, owner)
        with pytest.raises(ValueError, match='sealed'):
            rows.append(shot())
        with pytest.raises(ValueError, match='sealed'):
            rows[0] = shot()
        view = rows.select().field('shot_id')
        assert list(view) == [0]
        assert list(view[::-1]) == [0]
    with pytest.raises(RuntimeError, match='not open'):
        list(view)


@pytest.mark.parametrize('fps', [0., -1.])
@pytest.mark.parametrize('all_dead', [False, True])
def test_invalid_fps_retains_prechange_empty_group_behavior(tmp_path, fps, all_dead):
    values = [shot(index, is_dead_ball=all_dead) for index in range(4)]
    old, new = consumers(True)['RallyAnalyzer'], consumers()['RallyAnalyzer']
    with ShotEvidenceSequence(tmp_path) as rows:
        for value in values:
            rows.append(value)
        rows.seal()
        if all_dead:
            assert old.analyze_rallies(values, fps=fps) == new.analyze_rallies(rows, fps=fps, result_owner=rows) == []
        else:
            for analyzer, source, options in ((old, values, {}), (new, rows, {'result_owner': rows})):
                with pytest.raises(ValueError, match='fps must be positive'):
                    analyzer.analyze_rallies(source, fps=fps, **options)


def test_partial_landing_does_not_invent_a_missing_coordinate(tmp_path):
    value = shot(1, landing_court_position_m=(3., None), shot_type=ShotType.FOREHAND)
    with ShotEvidenceSequence(tmp_path) as rows:
        rows.append(value)
        rows.seal()
        with pytest.raises(TypeError):
            consumers(True)['ShotStatisticsAnalyzer'].compute_bounce_landing_heatmap([value])
        view = consumers()['ShotStatisticsAnalyzer'].compute_bounce_landing_heatmap(rows, result_owner=rows)
        with pytest.raises(TypeError):
            list(view)
        assert rows._readers == set()


@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize('export', ['rallies', 'points', 'match'])
def test_interrupted_nested_reader_preserves_destination_and_closes_owned_resources(tmp_path, failure, export):
    destination = tmp_path / 'completed.json'
    destination.write_bytes(b'preserve completed result')
    unrelated = tmp_path / 'unrelated.sqlite3'
    unrelated.write_bytes(b'keep')
    with ShotEvidenceSequence(tmp_path) as rows:
        for index in range(9):
            rows.append(shot(index))
        rows.seal()
        results = all_results(consumers(), rows, rows)
        calls = 0
        def interrupt(value):
            nonlocal calls
            calls += 1
            if (export == 'match' and isinstance(value, dict) and 'shot_id' in value) or (
                    export != 'match' and isinstance(value, int) and calls > 19):
                assert rows._readers  # The failure occurs while a SQL reader is suspended.
                raise failure('Synthetic serializer interruption')
            return value
        with pytest.raises(failure, match='Synthetic serializer'):
            if export == 'match':
                write_match_analytics(destination, results['match'], interrupt)
            else:
                write_membership_analytics(destination, export, results[export], interrupt)
        assert rows._readers == set()
        assert rows._connection is not None
        assert destination.read_bytes() == b'preserve completed result'
        assert not list(tmp_path.glob('.completed.json.*.tmp'))
        suspended = iter(rows.select().field('shot_id'))
        next(suspended)
        assert rows._readers
    assert rows._readers == set()
    suspended.close()
    assert unrelated.read_bytes() == b'keep'
    assert not list(tmp_path.glob('shot_evidence_*'))


@pytest.mark.parametrize('export', ['membership', 'match'])
def test_publication_failure_preserves_completed_destination_and_removes_only_owned_temp(tmp_path, monkeypatch, export):
    from src.utils import analytics_json_artifact as module
    destination = tmp_path / 'completed.json'
    destination.write_bytes(b'previous completed output')
    unrelated = tmp_path / '.completed.json.unrelated.tmp'
    unrelated.write_bytes(b'unrelated')
    with ShotEvidenceSequence(tmp_path) as rows:
        rows.append(shot(1))
        rows.seal()
        result = all_results(consumers(), rows, rows)
        def reject_replace(source, target):
            assert Path(target) == destination
            assert Path(source).parent == tmp_path
            raise PermissionError('Synthetic publication failure')
        monkeypatch.setattr(module.os, 'replace', reject_replace)
        with pytest.raises(PermissionError, match='Synthetic publication'):
            if export == 'membership':
                write_membership_analytics(destination, 'rallies', result['rallies'], lambda value: value)
            else:
                write_match_analytics(destination, result['match'], lambda value: value)
        assert rows._readers == set()
        assert destination.read_bytes() == b'previous completed output'
        assert list(tmp_path.glob('.completed.json.*.tmp')) == [unrelated]
        assert unrelated.read_bytes() == b'unrelated'


@pytest.mark.parametrize('count', [257, 1025, 4097])
def test_legacy_analytics_owner_views_and_exports_remain_bounded(tmp_path, monkeypatch, count):
    live, peak = weakref.WeakSet(), 0
    original = storage.ShotEventEvidence
    class ObservedShot(original):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            nonlocal peak
            super().__init__(*args, **kwargs)
            live.add(self)
            peak = max(peak, len(live))
    monkeypatch.setattr(storage, 'ShotEventEvidence', ObservedShot)
    with ShotEvidenceSequence(tmp_path) as rows:
        for index in range(count):
            rows.append(ObservedShot(**shot(index).__dict__))
        peak = 0
        rows.seal()
        results = all_results(consumers(), rows, rows)
        write_membership_analytics(tmp_path/'rallies.json', 'rallies', results['rallies'], lambda value:value)
        write_membership_analytics(tmp_path/'point_analytics.json', 'points', results['points'], lambda value:value)
        write_match_analytics(tmp_path/'match_analytics.json', results['match'], lambda value:value)
        actual = {'rallies':results['rallies'], 'point_analytics':results['points'], 'match_analytics':results['match']}
        assert peak <= 8 and len(live) == 0
        assert not isinstance(actual['rallies'][0].shot_ids, list)
        assert not isinstance(actual['point_analytics'][0]['shot_ids'], list)
        assert not isinstance(actual['match_analytics']['ball_landing_heatmap'], list)
        assert rows._readers == set()
        assert rows._execute('PRAGMA cache_size', fetch=True) == (-2048,)
        assert rows._execute('PRAGMA mmap_size', fetch=True) == (0,)
        receipt = {'input_shots': count, 'peak_live_shot_objects': peak, 'retained_live_shot_objects': len(live),
            'retained_rally_shot_id_list': False, 'retained_point_shot_id_list': False,
            'retained_match_landing_list': False, 'nested_export_rows': count,
            'scope': 'LEGACY_ANALYTICS_HELPERS; NATIVE_V2_POINT_TIMELINE_HAS_SEPARATE_TESTS',
            'full_match_capacity_proven': False, 'qualification_evidence': False}
        (tmp_path / 'capacity_counts.json').write_text(json.dumps(receipt, indent=2)+'\n')
