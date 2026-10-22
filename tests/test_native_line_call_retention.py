"""Actual native Phase6 retention baseline; synthetic rules, never video accuracy.

The original RED source and receipts remain in native_line_call_retention_red_v1.
Source extraction runs the real Step 7 statements and real rule/point owners. Only
camera/identity inputs and supplied line-call decisions are synthetic. No model,
image, geometry inference, or duration-long fixture input arrays are involved.
"""
from __future__ import annotations

import ast
from collections.abc import Sequence
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_scoring_service_phase import actual_classes
from src.scoring.point_timeline import (
    PointTimeline, ReviewedParticipantBinding, ReviewedRuleSignal,
    candidate_event_context, merge_rule_inputs, process_timeline_event,
)
from src.utils.point_analysis_sequence import PointAnalysisSequence
from src.utils.json_array_artifact import write_json_array_artifact
from src.api.result_storage import ResultStorage


ROOT = Path(__file__).resolve().parents[1]
COUNTS = (257, 1025, 4097)
ANALYSIS_ID = 'synthetic-native-line-call-retention'
BOUNCE = SimpleNamespace(value='BOUNCE')
OBSERVATIONS = []


class LazyFrames(Sequence):
    """Fresh constant-size source/identity snapshots, with exact source PTS."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        return {
            0: dict(analysis_id=ANALYSIS_ID, source_pts=str(index * 40),
                    source_time_base='1/1000', source_frame_index=index,
                    segment_id='synthetic-camera', analysis_allowed=True,
                    review_evidence_id='SYNTHETIC_NORMAL_SPEED_INPUT',
                    measurement_transition_allowed=index > 0, anchor_frame_index=0),
            1: dict(analysis_id=ANALYSIS_ID, epoch=0, revision=0,
                    observations=[dict(track_key='synthetic-A', player_id='A', state='CONFIRMED')]),
        }


class LazyBounces(Sequence):
    """Two distinct source events per frame exercise same-frame replacement."""
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        frame = index // 2
        return SimpleNamespace(event_id=index + 1, frame_index=frame,
            timestamp_s=frame / 25., event_type=BOUNCE, player_id=None,
            confidence=.99, ball_position_px=None, court_position_m=None,
            trajectory_state='SYNTHETIC_ONLY', evidence={'synthetic_input_position': index})


def expected_evidence(api, ident, frame):
    return api.LineCallEvidence(ident, frame, api.LineCallDecision.IN,
        api.LineCallContext.RALLY, api.CourtLineType.NONE,
        (500., 400.), (3., 6.), 'SYNTHETIC_ONLY', 20., 'SYNTHETIC_ONLY',
        1., 19., 0., 'SYNTHETIC_ONLY', .99,
        'SUPPLIED_RULE_FIXTURE_NOT_GEOMETRIC_GROUND_TRUTH', 'SYNTHETIC_ONLY',
        line_strip_info={'fixture': 'finite nested evidence', 'vertices': [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]})


def source_nodes():
    path = ROOT / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    start = next(index for index, node in enumerate(run.body) if isinstance(node, ast.AnnAssign)
                 and getattr(node.target, 'id', None) == 'line_call_evidences')
    end = next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
               and any(getattr(target, 'id', None) == 'authoritative_events' for target in node.targets))
    exports = [node for node in run.body if isinstance(node, (ast.With, ast.If)) and 'line_calls.json' in ast.unparse(node)]
    assert len(exports) == 1, 'Capture the actual line-call export, not a mirrored serializer'
    render = next(node for node in run.body if isinstance(node, ast.With)
                  and any(getattr(item.optional_vars, 'id', None) == 'render_stack' for item in node.items))
    loop = next(node for node in render.body if isinstance(node, ast.For) and getattr(node.target, 'id', None) == 'i')
    active = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                  and any(getattr(target, 'id', None) == 'active_line_calls' for target in node.targets))
    return path, run.body[start:end + 1], exports[0], loop.body[active:active + 2]


def _digest(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def _probe(api, directory, count, monkeypatch):
    directory.mkdir()
    source = directory / 'synthetic_source_identity.bin'
    source.write_bytes(b'Synthetic source identity only; not media or an accuracy reference.')
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    frames = LazyFrames((count + 1) // 2)
    events = LazyBounces(count)
    initial = api.MatchState(match_id=ANALYSIS_ID, match_status='SUSPENDED')
    initial_engine = api.TennisScoringEngine(initial_state=initial)
    initial_state = initial_engine.state.to_dict()
    path, statements, export, banner = source_nodes()

    def binding(event, interval, identity):
        return ReviewedParticipantBinding(event.frame_index, 'synthetic-camera', 1, 1,
            'A', 'synthetic-A', 'NEAR_COURT', 0, 'SYNTHETIC_CALIBRATION', 0,
            'SYNTHETIC_END_BINDING', analysis_id=ANALYSIS_ID, source_fingerprint=fingerprint,
            source_frame_index=event.frame_index, source_pts=interval['source_pts'],
            source_time_base=interval['source_time_base'], identity_epoch=0,
            participant_mapping_ref='SYNTHETIC_PARTICIPANT_BINDING',
            geometry_qualification_ref='SYNTHETIC_SUPPLIED_GEOMETRY_PERMISSION')

    class SuppliedCalls:
        count = 0
        def evaluate_bounce(self, **kwargs):
            self.count += 1
            assert kwargs['context'] is api.LineCallContext.RALLY
            assert kwargs['target_service_box'] is None
            return expected_evidence(api, kwargs['event_id'], kwargs['bounce_frame'])

    calls = SuppliedCalls()
    caller = SimpleNamespace(scoring_engine=initial_engine, line_call_engine=calls,
        reviewed_participant_binding_provider=binding,
        config={'match_configuration': {'match_id': ANALYSIS_ID}})
    namespace = dict(vars(api), self=caller, PointAnalysisSequence=PointAnalysisSequence,
        PointTimeline=PointTimeline, ReviewedRuleSignal=ReviewedRuleSignal,
        candidate_event_context=candidate_event_context, merge_rule_inputs=merge_rule_inputs,
        process_timeline_event=process_timeline_event, disk_players=True,
        authoritative_enabled=True, input_video_path=str(source), output_dir=str(directory),
        candidate_frames=frames, frames=(), ball_points=(), homography_matrix=None,
        is_h_valid=True, detected_events=events, event_output=SimpleNamespace(events_by_frame=lambda: iter(events)),
        hashlib=hashlib, time=time, os=os, t0=time.time(), _sanitize=lambda value: value,
        write_json_array_artifact=write_json_array_artifact)

    for name in ('src.scoring.match_state', 'src.scoring.point_outcome_resolver', 'src.line_calling.line_call_engine'):
        module = ModuleType(name)
        module.__dict__.update(vars(api))
        monkeypatch.setitem(sys.modules, name, module)

    gc.collect()
    tracemalloc.start()
    try:
        with ExitStack() as stack:
            namespace['export_stack'] = stack
            exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), namespace)
            gc.collect()
            retained, peak = tracemalloc.get_traced_memory()
            evidence = namespace['line_call_evidences']
            frame_map = namespace['line_call_map']
            dead_ids = namespace['dead_event_ids']
            owner = namespace['point_owner']
            disk_engine = caller.scoring_engine
            assert len(evidence) == len(dead_ids) == calls.count == count
            assert frame_map is None
            assert not isinstance(evidence, (list, tuple)) and not isinstance(dead_ids, (set, frozenset))
            assert namespace['authoritative_events'] is events
            assert not isinstance(events, (list, tuple)) and not isinstance(frames, (list, tuple))
            assert len(disk_engine.history.entries) == len(disk_engine._inputs) == count
            assert len(owner.rows('transitions')) == count
            assert not isinstance(disk_engine.history.entries, list)
            assert not isinstance(disk_engine._inputs, list)
            assert disk_engine.state.to_dict() == initial_state

            # Exact accepted evidence, source frames and PTS are checked one row at a time.
            for index, (item, stored, transition, history) in enumerate(zip(
                    evidence, disk_engine._inputs, owner.rows('transitions'),
                    disk_engine.history.entries, strict=True)):
                expected = expected_evidence(api, index + 1, index // 2)
                assert item.to_dict() == expected.to_dict()
                assert disk_engine._decode_input(stored)['line_call'].to_dict() == expected.to_dict()
                assert transition['event_id'] == history.event_id == index + 1
                assert transition['frame'] == index // 2
                assert transition['timestamp_s'] == (index // 2) / 25.
                assert transition['context']['source_pts'] == str((index // 2) * 40)
                assert transition['outcome_type'] == history.point_outcome == 'DEAD_BALL_IGNORED'
                assert index + 1 in dead_ids
            for frame, (key, item) in enumerate(evidence.by_frame(0, len(frames) - 1)):
                assert key == frame and item.event_id == min(2 * frame + 2, count)
                assert item.to_dict() == expected_evidence(api, item.event_id, frame).to_dict()

            class Canvas:
                FONT_HERSHEY_SIMPLEX = LINE_AA = 0
                seen = 0
                @staticmethod
                def rectangle(*args):
                    return None
                @classmethod
                def putText(cls, *args):
                    cls.seen += 1
            namespace.update(cv2=Canvas, ann_frame=object())
            for frame in (0, 18, 19, len(frames) - 1):
                namespace['i'] = frame
                Canvas.seen = 0
                exec(compile(ast.Module(body=banner, type_ignores=[]), str(path), 'exec'), namespace)
                assert Canvas.seen == min(frame + 1, 19)

            row = dict(events=count, retained_line_calls=len(evidence), retained_frame_map=0,
                retained_dead_event_ids=len(dead_ids), retained_traced_bytes=retained,
                peak_traced_bytes=peak, line_call_view_shallow_bytes=sys.getsizeof(evidence),
                dead_id_view_shallow_bytes=sys.getsizeof(dead_ids),
                evidence_values_sha256=_digest(item.to_dict() for item in evidence),
                frame_map_order_sha256=_digest((frame, item.event_id) for frame, item in evidence.by_frame(0, len(frames) - 1)),
                source_mapping_sha256=_digest((event.event_id, event.frame_index, event.timestamp_s,
                    str(event.frame_index * 40), '1/1000') for event in events),
                actual_scoring_history_records=len(disk_engine.history.entries), score_unchanged=True,
                same_frame_last_wins=True, all_transitions_preserve_source_pts=True,
                lazy_constant_size_input_fixtures=True, native_history_is_disk_backed=True,
                native_annotations_are_disk_views=True, actual_banner_inclusive_18_frame_window=True)
            exec(compile(ast.Module(body=[export], type_ignores=[]), str(path), 'exec'), namespace)
            _, export_peak = tracemalloc.get_traced_memory()
            row['export_peak_traced_bytes'] = export_peak
            row['line_calls_json_bytes'] = (directory / 'line_calls.json').stat().st_size
            # Release only the three duration-growing annotations, retaining all owners.
            del evidence, frame_map, dead_ids
            for name in ('line_call_evidences', 'line_call_map', 'dead_event_ids'):
                namespace.pop(name)
            gc.collect()
            after_release, _ = tracemalloc.get_traced_memory()
            row['retained_after_annotation_release_bytes'] = after_release
            row['annotation_release_delta_bytes'] = retained - after_release
            assert len(disk_engine.history.entries) == count
        assert caller.scoring_engine is initial_engine
        assert not list(directory.glob('*.sqlite3*'))
        tracemalloc.stop()
        # Parse the actual export into the existing bounded JSON owner, not a full list.
        with ResultStorage(directory) as storage:
            exported = storage.read_json(directory / 'line_calls.json')
            assert exported['schema_version'] == '1.0' and len(exported['line_calls']) == count
            row['export_values_sha256'] = _digest(exported['line_calls'])
            assert row['export_values_sha256'] == row['evidence_values_sha256']
        return row
    finally:
        tracemalloc.stop()


@pytest.fixture(scope='module')
def observations(tmp_path_factory):
    api = actual_classes()
    directory = tmp_path_factory.mktemp('native_line_call_retention')
    with pytest.MonkeyPatch.context() as monkeypatch:
        for count in COUNTS:
            OBSERVATIONS.append(_probe(api, directory / str(count), count, monkeypatch))
    return OBSERVATIONS


@pytest.mark.parametrize('position', range(len(COUNTS)))
def test_actual_native_phase6_annotations_preserve_values_order_frames_and_source_time(observations, position):
    row = observations[position]
    assert row['events'] == row['retained_line_calls'] == row['retained_dead_event_ids'] == COUNTS[position]
    assert row['same_frame_last_wins'] and row['all_transitions_preserve_source_pts']


def test_native_annotation_retention_does_not_grow_with_match_duration(observations):
    """The exact capacity contract that failed in the retained original RED."""
    growth = observations[-1]['retained_traced_bytes'] - observations[0]['retained_traced_bytes']
    assert growth <= 256 * 1024, (
        'Native Phase6 retains duration-growing line-call list, frame map and dead-event set: '
        f'{growth:,} extra traced bytes between {COUNTS[0]} and {COUNTS[-1]} lazy synthetic events')
