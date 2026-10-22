"""Bounded native shot linking; synthetic evidence does not qualify accuracy."""
import ast
import asyncio
from contextlib import closing
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple
import weakref

import numpy as np
import pytest

from src.events import event_detector as events_module
from src.shot_analysis.shot_types import (ShotType, ShotDirection, ShotOutcome, ShotClassificationSource,
    ShotEventEvidence, PlayerHandedness, CourtZone3x3)
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.court_zones import CourtZoneEngine
from src.tracking.temporal_ball_tracker import TemporalBallPoint
from src.utils.bbox_utils import BBox
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.shot_evidence_sequence import ShotEvidenceSequence
from src.utils import shot_evidence_sequence as storage_module


def actual_classes(*, prechange=False):
    """Execute exact model-free classes, excluding only their heavy import graph."""
    root = Path(__file__).resolve().parents[1]
    namespace = dict(globals(), PoseFeatureExtractor=object, Sequence=__import__('collections.abc').abc.Sequence,
                     TennisEvent=events_module.TennisEvent, EventType=events_module.EventType)
    for file, name in [('shot_classifier.py', 'TennisShotClassifier'), ('shot_linker.py', 'TennisShotLinker')]:
        path = root / 'src/shot_analysis' / file
        if prechange and file == 'shot_linker.py':
            path = root / 'artifacts/validation/yolo26_execution_2026-10-16/shot_result_capacity_probe_v1/prechange_shot_linker.py'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == '25e3c123de3d145c93495ebfe08eba14ee926dae4b31e2f59f901279825c6e25'
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
        assert len(selected) == 1
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['TennisShotLinker'], namespace['TennisShotClassifier']


def event(ident, frame, kind=events_module.EventType.PLAYER_1_HIT, player=1, timestamp=None):
    return events_module.TennisEvent(ident, kind, frame, frame / 25. if timestamp is None else timestamp,
        player, (50., 100.), (3., 19.) if player else (8., 5.), .9, 'DETECTED')


def shot(ident=1, frame=2, **changes):
    record = dict(shot_id=ident, match_event_id=ident, frame_index=frame, timestamp_s=frame/25., player_id=1,
        shot_type=ShotType.FOREHAND, shot_confidence=.8, classification_source=ShotClassificationSource.GEOMETRY_BASELINE,
        direction=ShotDirection.CROSS_COURT, direction_confidence=.9, direction_source='GEOMETRY_DERIVED')
    record.update(changes)
    return ShotEventEvidence(**record)


def serialize(rows):
    return json.dumps([value.to_dict() for value in rows], sort_keys=True)


@pytest.mark.parametrize('events', [
    [event(1, 20), event(2, 10, events_module.EventType.SERVE_CONTACT, 2),
     event(3, 25, events_module.EventType.BOUNCE, None), event(4, 15, events_module.EventType.BOUNCE, None)],
    [event(1, 10), event(2, 10, events_module.EventType.PLAYER_2_HIT, 2),
     event(3, 10, events_module.EventType.BOUNCE, None), event(4, 15, events_module.EventType.BOUNCE, None),
     event(5, 15, events_module.EventType.BOUNCE, None), event(6, 20)],
    [event(1, 5), event(2, 10), event(3, 16, events_module.EventType.BOUNCE, None)],
    [event(1, 7, player=None, timestamp=.301), event(2, 8, events_module.EventType.UNKNOWN_EVENT, None),
     event(3, 12, events_module.EventType.BOUNCE, None, timestamp=.519)],
    [],
])
@pytest.mark.parametrize('dead', [set(), {1, 3}])
def test_native_nonempty_linking_is_exactly_legacy_with_real_classifier_direction_and_zones(tmp_path, events, dead):
    Linker, Classifier = actual_classes()
    classifier = Classifier(handedness_map={1: PlayerHandedness.RIGHT_HANDED, 2: PlayerHandedness.LEFT_HANDED},
                            court_orientation_map={1: 1., 2: -1.})
    linker = Linker(classifier)
    points = [TemporalBallPoint(index, index/25., 100., 150., court_x_m=3., court_y_m=19., speed_kmh=None)
              for index in range(40)]
    boxes = [BBox(10., 10., 50., 200.)] * len(points)
    PreviousLinker, _ = actual_classes(prechange=True)
    expected = PreviousLinker(classifier).link_shots(events, points, boxes, boxes, dead_event_ids=dead)
    assert serialize(linker.link_shots(events, points, boxes, boxes, dead_event_ids=dead)) == serialize(expected)
    assert isinstance(expected, list)
    if any(value.event_type in (events_module.EventType.SERVE_CONTACT, events_module.EventType.PLAYER_1_HIT,
                                events_module.EventType.PLAYER_2_HIT) for value in events):
        assert expected
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        for value in events:
            source.events.append(value)
        actual = linker.link_shots(source.events, points, boxes, boxes, dead_event_ids=dead,
                                   event_output=source, result_output=output)
        assert actual is output
        assert serialize(actual) == serialize(expected)
        assert [value.shot_id for value in actual] == list(range(1, len(expected)+1))
        assert list(actual[::-1]) == list(reversed(list(actual)))
        assert not isinstance(actual, list)
    assert not list(tmp_path.glob('shot_evidence_*'))


def test_first_bounce_endpoint_ties_and_reuse_are_preserved(tmp_path):
    Linker, Classifier = actual_classes()
    values = [event(1, 10), event(2, 10, events_module.EventType.PLAYER_2_HIT, 2), event(3, 20),
        event(4, 10, events_module.EventType.BOUNCE, None), event(5, 15, events_module.EventType.BOUNCE, None),
        event(6, 15, events_module.EventType.BOUNCE, None), event(7, 25, events_module.EventType.BOUNCE, None)]
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        for value in values:
            source.events.append(value)
        Linker(Classifier()).link_shots(source.events, [], [], [], event_output=source, result_output=output)
        assert [(value.match_event_id, value.bounce_event_id) for value in output] == [(1, 5), (2, 5), (3, 7)]


def test_typed_snapshots_preserve_enums_ieee_tuples_and_explicit_writeback(tmp_path):
    nan = struct.unpack('!d', bytes.fromhex('7ff8000000000041'))[0]
    value = shot(confidence_components={'signed': -0., 'nan': nan, 'array': np.array([1., -0.])},
                 landing_zone=CourtZone3x3.DEEP_RIGHT, source_court_position_m=(3., 19.))
    with ShotEvidenceSequence(tmp_path) as rows:
        rows.append(value)
        restored = rows[0]
        assert restored.shot_type is ShotType.FOREHAND and restored.landing_zone is CourtZone3x3.DEEP_RIGHT
        assert type(restored.source_court_position_m) is tuple
        assert struct.pack('!d', restored.confidence_components['nan']) == struct.pack('!d', nan)
        assert struct.pack('!d', restored.confidence_components['signed']) == struct.pack('!d', -0.)
        assert restored.confidence_components['array'].tobytes() == value.confidence_components['array'].tobytes()
        restored.reason = 'reviewed correction'
        assert rows[0].reason == ''
        rows[0] = restored
        assert rows[0].reason == 'reviewed correction'


def test_badge_query_is_last_per_frame_and_bounded_to_inclusive_window(tmp_path):
    with ShotEvidenceSequence(tmp_path) as rows:
        for ident, frame in enumerate((1, 7, 7, 3, 20), 1):
            rows.append(shot(ident, frame))
        assert [(value.frame_index, value.shot_id) for value in rows.badges_by_frame(1, 7)] == [(1, 1), (7, 3), (3, 4)]
        assert [(value.frame_index, value.shot_id) for value in rows.badges_by_frame(3, 7)] == [(7, 3), (3, 4)]


@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
def test_cancel_or_failure_removes_only_owned_scratch_even_with_suspended_query(tmp_path, failure):
    unrelated = tmp_path / 'preserve.sqlite3'
    unrelated.write_bytes(b'keep')
    rows = ShotEvidenceSequence(tmp_path)
    iterator = None
    with pytest.raises(failure):
        with rows:
            rows.append(shot())
            iterator = rows.badges_by_frame(0, 20)
            assert next(iterator).shot_id == 1
            raise failure('Synthetic consumer interruption')
    iterator.close()
    assert unrelated.read_bytes() == b'keep' and not rows.path.exists()
    with pytest.raises(RuntimeError, match='not open'):
        rows[0]


def test_output_owner_requires_open_empty_store_and_exact_authoritative_event_alias(tmp_path):
    Linker, Classifier = actual_classes()
    linker = Linker(Classifier())
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        source.events.append(event(1, 2))
        with pytest.raises(ValueError, match='authoritative event'):
            linker.link_shots(list(source.events), [], [], [], event_output=source, result_output=output)
        with pytest.raises(ValueError, match='authoritative event'):
            linker.link_shots(source.events, [], [], [], result_output=output)
        # Withheld authority does not read the diagnostic event store.
        assert linker.link_shots([], [], [], [], result_output=output) is output
        output.append(shot())
        with pytest.raises(ValueError, match='empty'):
            linker.link_shots(source.events, [], [], [], event_output=source, result_output=output)


@pytest.mark.parametrize('count', [257, 1025])
def test_native_linker_keeps_live_event_and_shot_objects_constant_across_commit_batches(tmp_path, monkeypatch, count):
    Linker, Classifier = actual_classes()
    live_events, live_shots = weakref.WeakSet(), weakref.WeakSet()
    peaks = dict(events=0, shots=0)
    original_event, original_shot = events_module.TennisEvent, storage_module.ShotEventEvidence
    class ObservedEvent(original_event):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_events.add(self)
            peaks['events'] = max(peaks['events'], len(live_events))
    class ObservedShot(original_shot):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_shots.add(self)
            peaks['shots'] = max(peaks['shots'], len(live_shots))
    monkeypatch.setattr(events_module, 'TennisEvent', ObservedEvent)
    monkeypatch.setattr(storage_module, 'ShotEventEvidence', ObservedShot)
    # The exact linker's class globals are its AST isolation namespace.
    Linker.link_shots.__globals__['ShotEventEvidence'] = ObservedShot
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        for index in range(count):
            source.events.append(event(index*2+1, index*2))
            source.events.append(event(index*2+2, index*2+1, events_module.EventType.BOUNCE, None))
        result = Linker(Classifier()).link_shots(source.events, [], [], [], event_output=source, result_output=output)
        assert len(result) == count and result[-1].shot_id == count
        assert peaks['events'] <= 8 and peaks['shots'] <= 2
        assert len(live_events) == len(live_shots) == 0
        assert output._execute('PRAGMA cache_size', fetch=True) == (-2048,)
        assert output._execute('PRAGMA mmap_size', fetch=True) == (0,)
        assert output._execute('PRAGMA temp_store', fetch=True) == (1,)
        assert output._pending < output.transaction_size <= 256
        (tmp_path / 'live_object_counts.json').write_text(json.dumps({
            'input_events': count*2, 'linked_shots': len(output),
            'peak_live_event_objects': peaks['events'], 'peak_live_shot_objects': peaks['shots'],
            'retained_live_event_objects': len(live_events), 'retained_live_shot_objects': len(live_shots),
            'qualification_evidence': False, 'full_match_capacity_proven': False,
        }, indent=2) + '\n')


@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
def test_classifier_failure_closes_native_event_readers_before_owner_exit(tmp_path, failure):
    Linker, _ = actual_classes()
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        for value in (event(1, 2), event(2, 4), event(3, 6, events_module.EventType.BOUNCE, None)):
            source.events.append(value)
        active = []
        original = source.events_by_frame
        def observed(*args, **kwargs):
            generator = original(*args, **kwargs)
            active.append(True)
            try:
                yield from generator
            finally:
                generator.close()
                active.pop()
        source.events_by_frame = observed
        class Classifier:
            def classify_shot(self, **kwargs):
                assert len(active) == 2
                raise failure('Synthetic classifier interruption')
        with pytest.raises(failure, match='Synthetic classifier'):
            Linker(Classifier()).link_shots(source.events, [], [], [], event_output=source, result_output=output)
        assert active == []
        assert len(output) == 0
        assert source._connection is not None and output._connection is not None


def test_unknown_partial_ground_pair_preserves_legacy_value_instead_of_repairing_it(tmp_path):
    Linker, Classifier = actual_classes()
    point = TemporalBallPoint(0, .031, 50., 100., court_x_m=3., court_y_m=None)
    values = [event(1, 0, timestamp=.031)]
    # A missing landing makes the current direction classifier abstain before
    # arithmetic, so the partial source pair can be retained without inventing y.
    expected = Linker(Classifier()).link_shots(values, [point], [None], [None])
    assert expected[0].source_court_position_m == (3., None)
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        source.events.append(values[0])
        Linker(Classifier()).link_shots(source.events, [point], [None], [None], event_output=source, result_output=output)
        assert serialize(output) == serialize(expected)


@pytest.mark.parametrize('value', [0, -1, 257, True])
def test_storage_commit_batch_is_explicitly_bounded(value, tmp_path):
    with pytest.raises(ValueError):
        ShotEvidenceSequence(tmp_path, transaction_size=value)


def test_malformed_payloads_and_nonfinite_active_timestamps_are_rejected_before_append(tmp_path):
    with ShotEvidenceSequence(tmp_path) as output:
        with pytest.raises(TypeError, match='typed'):
            output.append({'shot_id': 1})
        with pytest.raises(TypeError, match='enum'):
            output.append(shot(shot_type='FOREHAND'))
        for value in (float('nan'), float('inf'), True, '0.8', 2**53+1):
            with pytest.raises(ValueError, match='timestamp'):
                output.append(shot(timestamp_s=value))
        assert len(output) == 0
        output.append(shot())
        output._execute('UPDATE shots SET payload=? WHERE position=0', ('{"schema_version":true,"record":[]}',))
        with pytest.raises(ValueError, match='envelope'):
            output[0]


@pytest.mark.parametrize('authority', [False, True])
def test_exact_phase6_shot_owner_render_query_and_json_consume_nonempty_native_results(tmp_path, authority):
    """Execute exact changed source blocks, not a duplicate orchestration."""
    from contextlib import ExitStack
    import os
    import time
    from src.utils.json_array_artifact import write_json_array_artifact
    root = Path(__file__).resolve().parents[1]
    path = root / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text())
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    start = next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'shot_output' for target in node.targets))
    assignments = run.body[start:start+2]
    assert ast.unparse(assignments[1]).startswith('shot_evidences = self.shot_linker.link_shots(')
    render = next(node for node in run.body if isinstance(node, ast.With)
                  and any(isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == 'render_stack'
                          for item in node.items))
    frame_map = next(node for node in render.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'shot_frame_map' for target in node.targets))
    loop = next(node for node in render.body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'i')
    active = next(index for index, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'active_shots' for target in node.targets))
    badges = loop.body[active:active+2]
    destination = next(node for node in run.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == 'write_json_array_artifact'
        and 'shot_events.json' in ast.unparse(node.value.args[0]))
    sanitize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'sanitize_json_value')
    Linker, Classifier = actual_classes()
    values = [event(1, 2, events_module.EventType.SERVE_CONTACT, 1), event(2, 2),
              event(3, 6, events_module.EventType.BOUNCE, None), event(4, 9, events_module.EventType.PLAYER_2_HIT, 2)]
    expected = Linker(Classifier()).link_shots(values if authority else [], [], [], [], dead_event_ids={4})
    badges_seen = []
    class Annotator:
        @staticmethod
        def rectangle(*args):
            return None
        @staticmethod
        def putText(_frame, text, *args):
            badges_seen.append(text)
        FONT_HERSHEY_SIMPLEX = LINE_AA = 0
    with ExitStack() as stack:
        source = stack.enter_context(EventAnalysisSequence(tmp_path))
        for value in values:
            source.events.append(value)
        namespace = dict(np=np, Any=object, os=os, time=time, ShotEvidenceSequence=ShotEvidenceSequence,
            write_json_array_artifact=write_json_array_artifact,
            self=SimpleNamespace(shot_linker=Linker(Classifier())), export_stack=stack,
            output_dir=str(tmp_path), disk_players=True, event_output=source, authoritative_enabled=authority,
            detected_events=source.events if authority else [], ball_points=[], p1_boxes=[], p2_boxes=[], frames=[],
            dead_event_ids={4}, cv2=Annotator, i=10, ann_frame=object())
        exec(compile(ast.Module(body=[sanitize]+assignments+[frame_map]+badges, type_ignores=[]), str(path), 'exec'), namespace)
        output = namespace['shot_output']
        assert namespace['shot_evidences'] is output
        assert namespace['shot_frame_map'] is None
        namespace['_sanitize'] = namespace['sanitize_json_value']
        exec(compile(ast.Module(body=[destination], type_ignores=[]), str(path), 'exec'), namespace)
        loaded = json.loads((tmp_path / 'shot_events.json').read_text())
        assert loaded == json.loads(json.dumps({'schema_version': '1.0', 'shot_events':
                                                [value.to_dict() for value in expected]}))
        assert len(output) == (3 if authority else 0)
        assert len(badges_seen) == (1 if authority else 0)  # Last same-frame shot wins; dead shot hidden.
        assert output.path.exists() and output._connection is not None
    assert not output.path.exists() and not list(tmp_path.glob('shot_evidence_*'))


@pytest.mark.parametrize('seed', [7, 709])
def test_shuffled_ties_and_unknowns_match_exact_prechange_linker(tmp_path, seed):
    import random
    rng = random.Random(seed)
    Linker, Classifier = actual_classes()
    PreviousLinker, _ = actual_classes(prechange=True)
    kinds = [events_module.EventType.SERVE_CONTACT, events_module.EventType.PLAYER_1_HIT,
             events_module.EventType.PLAYER_2_HIT, events_module.EventType.BOUNCE,
             events_module.EventType.UNKNOWN_EVENT]
    values = [event(index+1, rng.randrange(70), rng.choice(kinds), rng.choice([None, 1, 2])) for index in range(180)]
    rng.shuffle(values)
    dead = {value.event_id for value in values if value.event_id % 17 == 0}
    expected = PreviousLinker(Classifier()).link_shots(values, [], [], [], dead_event_ids=dead)
    assert expected and any(value.is_dead_ball for value in expected)
    assert serialize(Linker(Classifier()).link_shots(values, [], [], [], dead_event_ids=dead)) == serialize(expected)
    with EventAnalysisSequence(tmp_path) as source, ShotEvidenceSequence(tmp_path) as output:
        for value in values:
            source.events.append(value)
        Linker(Classifier()).link_shots(source.events, [], [], [], dead_event_ids=dead,
                                       event_output=source, result_output=output)
        assert serialize(output) == serialize(expected)
