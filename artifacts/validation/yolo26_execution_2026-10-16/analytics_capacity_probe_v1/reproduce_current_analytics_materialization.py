"""Read-only RED probe of actual analytics over lazy typed shot evidence.

No production source is edited. Exact consumer class/function AST nodes and
the actual Phase6 step-10 statements are executed without the model runtime.
Synthetic positions/events do not establish match correctness or capacity.
"""
import ast
from collections.abc import Sequence
from contextlib import redirect_stdout
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import weakref

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np
from src.shot_analysis.shot_types import (ShotEventEvidence, ShotType, ShotDirection,
    ShotClassificationSource, ShotOutcome, CourtZone3x3)
from src.scoring.match_state import MatchState
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision

BASE = Path(__file__).resolve().parent
PATHS = [ROOT / value for value in (
    'src/analytics/rally_analyzer.py', 'src/analytics/shot_statistics.py',
    'src/analytics/serve_analyzer.py', 'src/analytics/match_analytics.py',
    'src/pipeline/phase6_pipeline.py', 'src/utils/shot_evidence_sequence.py',
    'src/shot_analysis/shot_linker.py', 'src/shot_analysis/shot_types.py')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                               'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    status = Counters()
    status.cb = ctypes.sizeof(status)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(status), status.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'owned_rss_bytes': status.rss, 'peak_owned_rss_bytes': status.peak_rss}


def actual_consumers():
    namespace = dict(globals())
    names = {
        'rally_analyzer.py': {'RallySegment', 'RallyAnalyzer'},
        'shot_statistics.py': {'ShotStatisticsAnalyzer'},
        'serve_analyzer.py': {'ServeAnalyzer'},
        'match_analytics.py': {'_round_known', 'MatchAnalyticsAggregator'},
    }
    for file, wanted in names.items():
        path = ROOT / 'src/analytics' / file
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
        assert {node.name for node in nodes} == wanted
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    path = ROOT / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text())
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    def assignment(name):
        return next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    # Exactly the current rally, point and match calls, including source order.
    blocks = run.body[assignment('final_st'):assignment('match_analytics')+1]
    calls = [node for node in ast.walk(ast.Module(body=blocks, type_ignores=[])) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr in
             {'analyze_rallies', 'build_point_analytics', 'build_match_analytics'}]
    assert len(calls) == 3
    namespace['_phase6_step10'] = compile(ast.Module(body=blocks, type_ignores=[]), str(path), 'exec')
    return namespace


class LazyShots(Sequence):
    """Fresh typed snapshots only; the source never caches records or slices."""
    def __init__(self, count):
        self.count, self.reads, self.peak = count, 0, 0
        self.live = weakref.WeakSet()
        source = self
        class ObservedShot(ShotEventEvidence):
            __hash__ = object.__hash__
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                source.live.add(self)
                source.peak = max(source.peak, len(source.live))
        self.record_type = ObservedShot

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError('The input is lazy; no source list or slice is supplied')
        if not 0 <= index < len(self):
            raise IndexError(index)
        self.reads += 1
        dead = index % 17 == 16
        kind = ShotType.SERVE if index % 5 == 0 else ShotType.BACKHAND if index % 3 == 0 else ShotType.FOREHAND
        return self.record_type(shot_id=index+1, match_event_id=index*2+1, frame_index=index*8,
            timestamp_s=index*.321, player_id=1 if index % 2 == 0 else 2,
            shot_type=kind, shot_confidence=.8, classification_source=ShotClassificationSource.GEOMETRY_BASELINE,
            direction=ShotDirection.CROSS_COURT, direction_confidence=.8, direction_source='GEOMETRY_DERIVED',
            source_court_position_m=(3., 19.), landing_court_position_m=(8., 5.), landing_zone=CourtZone3x3.DEEP_RIGHT,
            bounce_event_id=index*2+2, bounce_frame=index*8+4, speed_kmh=None,
            outcome=ShotOutcome.UNKNOWN if dead else ShotOutcome.IN_PLAY,
            is_dead_ball=dead, confidence_components={'synthetic_only': 1.}, reason='SYNTHETIC_CAPACITY_FIXTURE')


def phase6_step10(namespace, shots):
    values = dict(namespace, self=SimpleNamespace(scoring_engine=SimpleNamespace(
        get_current_state=lambda: MatchState(match_id='synthetic-capacity-only'))), shot_evidences=shots,
        line_call_evidences=[], fps=25., p1_dist=None, p2_dist=None, p1_speed=None, p2_speed=None,
        p1_positions=[], p2_positions=[], speed_summary={'available': False})
    exec(namespace['_phase6_step10'], values)
    return {'rallies': values['rallies'], 'points': values['point_analytics'], 'match': values['match_analytics']}


def output_summary(name, result):
    if name == 'rally':
        return {'rallies': len(result), 'retained_shot_ids': sum(len(value.shot_ids) for value in result),
                'live_strokes': sum(value.total_strokes_including_serve for value in result)}
    if name == 'player_statistics':
        return {'retained_fixed_player_counter_keys': len(result['shot_type_counts'])+len(result['direction_counts'])+
                len(result['landing_zone_counts']), 'counted_live_player1_shots': result['shot_type_counts']['total']}
    if name == 'serve_statistics':
        return {'first_serves_attempted': result['first_serves_attempted'],
                'retained_fixed_top_level_keys': len(result), 'placement_total': sum(result['placement_distribution'].values())}
    if name == 'bounce_heatmap':
        return {'retained_bounce_rows': len(result)}
    if name == 'point_without_rally':
        return {'points': len(result), 'retained_shot_ids': sum(len(value['shot_ids']) for value in result),
                'live_shots': result[0]['total_shots_including_serve']}
    return {'rallies': len(result['rallies']), 'points': len(result['points']),
            'retained_rally_shot_ids': sum(len(value.shot_ids) for value in result['rallies']),
            'retained_point_shot_ids': sum(len(value['shot_ids']) for value in result['points']),
            'retained_match_bounce_rows': len(result['match']['ball_landing_heatmap']),
            'reported_total_points_played': result['match']['overview']['total_points_played']}


def main():
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    before = {str(path.relative_to(ROOT)): sha(path) for path in PATHS}
    classes = actual_consumers()
    Rally, Statistics, Serves, Aggregator = [classes[value] for value in
        ('RallyAnalyzer', 'ShotStatisticsAnalyzer', 'ServeAnalyzer', 'MatchAnalyticsAggregator')]
    consumers = {
        'rally': lambda shots: Rally.analyze_rallies(shots, fps=25.),
        'player_statistics': lambda shots: Statistics.compute_player_shot_stats(shots, player_id=1),
        'serve_statistics': lambda shots: Serves.compute_serve_statistics(shots, [], server_id=1),
        'bounce_heatmap': Statistics.compute_bounce_landing_heatmap,
        'point_without_rally': lambda shots: Aggregator.build_point_analytics(
            MatchState(match_id='synthetic-capacity-only'), shots, [], []),
        'actual_phase6_step10': lambda shots: phase6_step10(classes, shots),
    }
    cases = []
    resources_before = memory()
    for count in (257, 1025, 4097):
        for name, consume in consumers.items():
            source = LazyShots(count)
            gc.collect()
            stage_start = time.perf_counter()
            tracemalloc.start()
            result = consume(source)
            current, maximum = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            summary = output_summary(name, result)
            cases.append(dict(consumer=name, input_shots=count, source_reads=source.reads,
                peak_live_shot_objects=source.peak, retained_live_shot_objects=len(source.live),
                traced_peak_bytes=maximum, traced_retained_bytes=current,
                elapsed_seconds=time.perf_counter()-stage_start, output=summary))
            result = None
            gc.collect()
            assert len(source.live) == 0
    # Demonstrate current API/call-site semantics separately from allocation.
    # A large gap and another serve are not independent point-boundary evidence.
    # The API receives no point-boundary input and still emits one supplied group.
    boundary = [LazyShots(2)[0], LazyShots(2)[1]]
    boundary[0].frame_index, boundary[0].timestamp_s = 0, 0.
    boundary[1].frame_index, boundary[1].timestamp_s = 15000, 600.
    boundary[1].shot_type = ShotType.SERVE
    boundary_rallies = Rally.analyze_rallies(boundary, fps=25.)
    semantics = dict(two_synthetic_serves_separated_seconds=600., point_boundary_input_available=False,
        observed_rally_count=len(boundary_rallies), reported_point_ids=[value.point_id for value in boundary_rallies],
        statement='Current Phase6 supplies whole-file shots to a one-group API; no per-point segmentation is established',
        is_capacity_fix=False, accuracy_claimed=False)
    # Exact current timing behavior, recorded without redefining it.
    boundary[1].frame_index, boundary[1].timestamp_s = 10, .71
    timing = Rally.analyze_rallies(boundary, fps=25.)[0]
    semantics['timing_example'] = {'first_timestamp_s': 0., 'last_hit_timestamp_s': .71,
        'last_frame': 10, 'linked_bounce_frame': boundary[1].bounce_frame,
        'fps': 25., 'reported_duration_s': timing.duration_s,
        'method': 'CURRENT_FRAME_INDEX_DIVIDED_BY_FPS_OVERRIDES_HIT_TIMESTAMPS',
        'physical_timing_qualified': False}
    after = {str(path.relative_to(ROOT)): sha(path) for path in PATHS}
    offenders = [case for case in cases if case['peak_live_shot_objects'] > 8
        or case['output'].get('retained_bounce_rows', 0) > 8
        or case['output'].get('retained_shot_ids', 0) > 8
        or case['output'].get('retained_match_bounce_rows', 0) > 8]
    receipt = dict(status='REPRODUCED_ANALYTICS_MATERIALIZATION_RED', started_utc=started,
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
        cases=cases, separate_behavior_findings=semantics, source_bindings_before=before,
        source_bindings_after=after, sources_changed=[key for key in before if before[key] != after[key]],
        resources_before=resources_before, resources_after=memory(),
        input_is_lazy_typed_snapshots=True, input_fixture='MIXED_TYPES_ALTERNATING_PLAYERS_EVERY17TH_DEAD_ALL_WITH_LANDINGS',
        traced_bytes_include_weakref_observer_overhead=True, line_call_input_count=0,
        player_ground_position_inputs_count=0, source_clock_or_geometry_accuracy_exercised=False,
        torch_imported='torch' in sys.modules, ultralytics_imported='ultralytics' in sys.modules,
        model_used=False, media_used=False, training_performed=False, qualification_evidence=False,
        full_match_capacity_proven=False, expected_bounded_assertion_passed=not offenders,
        failing_consumer_count=len(offenders), python_executable=sys.executable,
        python_version=sys.version, numpy_version=np.__version__, script_sha256=sha(__file__))
    with (BASE / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'receipt': str(BASE / 'receipt.json'), 'status': receipt['status'],
        'cases': cases, 'resources': receipt['resources_after'], 'torch_imported': receipt['torch_imported'],
        'ultralytics_imported': receipt['ultralytics_imported'], 'sources_changed': receipt['sources_changed']}))
    assert not offenders, 'Lazy shots are re-materialized by rally, statistics and Phase6 analytics consumers'


if __name__ == '__main__':
    main()
