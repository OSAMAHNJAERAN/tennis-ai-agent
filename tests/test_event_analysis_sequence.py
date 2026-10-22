"""Typed disk event results; these checks do not establish vision accuracy."""
import ast
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import weakref
from collections.abc import Sequence

import numpy as np
import pytest

from src.events import event_detector as module
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils import event_analysis_sequence as storage_module
from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


def candidate(frame=0, timestamp=0., **changes):
    values = dict(frame_index=frame, timestamp_s=timestamp, score=.8,
                  ball_position_px=(50., 100.), trajectory_state='DETECTED',
                  discovery_frame_index=frame, discovery_timestamp_s=timestamp)
    values.update(changes)
    return module.EventCandidate(**values)


def event(frame=0, timestamp=0., confidence=.8, **changes):
    values = dict(event_id=0, event_type=module.EventType.BOUNCE, frame_index=frame,
                  timestamp_s=timestamp, player_id=None, ball_position_px=(50., 100.),
                  court_position_m=None, confidence=confidence, trajectory_state='DETECTED')
    values.update(changes)
    return module.TennisEvent(**values)


def trace(candidate_id):
    return {'candidate_id': candidate_id, 'stage_pass': {'temporal_suppression': False,
            'final_authoritative_event': False}, 'verification_pass': True, 'final_event_emitted': False,
            'rejection_stage': None, 'rejection_reason': None}


def snapshot(analysis):
    return json.dumps({'candidates': [asdict(value) for value in analysis.candidates],
        'events': [asdict(value) for value in analysis.events], 'traces': list(analysis.verification_traces),
        'settings': analysis.settings, 'coordinate_system': analysis.coordinate_system},
        sort_keys=True, allow_nan=True, default=lambda value: value.value)


def points():
    return [TemporalBallPoint(index, index / 25. + (index % 3) * .0001,
                             300. + 3. * index if index != 9 else None,
                             180. + 8. * abs(index - 24) if index != 9 else None,
                             state=BallState.DETECTED if index != 9 else BallState.MISSING,
                             confidence=.85 if index != 9 else None) for index in range(50)]


@pytest.mark.parametrize('compensate', [False, True])
def test_actual_detector_disk_results_equal_legacy_and_outlive_feature_context(tmp_path, compensate):
    trajectory = points()
    boxes = [None] * len(trajectory)
    offsets = [(index * .1, index * .05) if index != 17 else None for index in range(len(trajectory))]
    legacy = module.TennisEventDetector(enable_camera_motion_compensation=compensate)
    expected = legacy.analyze(trajectory, boxes, boxes, fps=25., frame_size=(1280, 720), camera_offsets_px=offsets)
    with EventAnalysisSequence(tmp_path) as output, TemporalBallPointSequence(tmp_path) as stored:
        stored.extend(trajectory)
        detector = module.TennisEventDetector(enable_camera_motion_compensation=compensate)
        actual = detector.analyze(stored, boxes, boxes, fps=25., frame_size=(1280, 720),
                                  camera_offsets_px=offsets, scratch_directory=tmp_path, result_output=output)
        assert actual.candidates is output.candidates and actual.events is output.events
        assert actual.verification_traces is output.verification_traces
        assert snapshot(actual) == snapshot(expected)
        assert isinstance(expected.candidates, list) and isinstance(expected.events, list)
        assert not list(tmp_path.glob('event_features_*'))
        assert detector.last_analysis is actual
        assert list(output.events_by_frame()) == sorted(actual.events, key=lambda value: value.frame_index)
    assert not list(tmp_path.glob('event_analysis_*'))


def test_codec_preserves_ieee_payloads_tuples_arrays_and_explicit_snapshot_writeback(tmp_path):
    nan = struct.unpack('!d', bytes.fromhex('7ff8000000000041'))[0]
    evidence = {'signed': -0., 'nan': nan, 'infinity': math.inf, 'nested': (['a'], {1: np.float32(.3)}),
                'array': np.asarray([1., -0., nan], dtype=np.float64)}
    with EventAnalysisSequence(tmp_path) as output:
        output.candidates.append(candidate(evidence=evidence))
        restored = output.candidates[0]
        assert struct.pack('!d', restored.evidence['signed']) == struct.pack('!d', -0.)
        assert struct.pack('!d', restored.evidence['nan']) == struct.pack('!d', nan)
        assert restored.evidence['infinity'] == math.inf
        assert isinstance(restored.evidence['nested'], tuple)
        assert type(restored.evidence['nested'][1][1]) is np.float32
        assert restored.evidence['array'].tobytes() == evidence['array'].tobytes()
        restored.evidence['signed'] = 10.
        assert struct.pack('!d', output.candidates[0].evidence['signed']) == struct.pack('!d', -0.)
        output.candidates[0] = restored
        assert output.candidates[0].evidence['signed'] == 10.
        output.events.append(event(event_id=1, evidence={'before': True}))
        landing = output.events[0]
        landing.evidence['calibration_id'] = 'frame_0'
        output.events[0] = landing
        assert output.events[0].event_type is module.EventType.BOUNCE
        assert output.events[0].evidence['calibration_id'] == 'frame_0'


@pytest.mark.parametrize(('timestamps', 'interval', 'expected_frames'), [
    ([.3, .4, 1.], .1, [0, 1, 2]),
    ([.5, .625, 1.], .125, [0, 2]),
    ([0., math.nextafter(.1, 0.), .2], .1, [0, 2]),
    ([0., math.nextafter(.1, math.inf), .3], .1, [0, 1, 2]),
    ([2., 2., 3.], -.01, [0, 1, 2]),
    ([2., 2., 3.], 0., [0, 2]),
    ([1.e308, math.nextafter(1.e308, math.inf), -1.e308], 1.e292, [2, 0, 1]),
])
@pytest.mark.parametrize('spatial', [False, True])
def test_final_disk_rank_preserves_stable_ties_and_exact_time_endpoints(tmp_path, timestamps, interval,
                                                                      expected_frames, spatial):
    with EventAnalysisSequence(tmp_path) as output:
        for index, timestamp in enumerate(timestamps):
            output.candidates.append(candidate(index, index * .25))
            output.verification_traces.append(trace(index + 1))
            output.add_provisional(index + 1, event(index, timestamp, .8))
        output.finalize(module.EventDetectorSettings(final_event_min_interval_seconds=interval,
                                                      enable_spatial_duplicate_clustering=spatial))
        assert [value.frame_index for value in output.events] == expected_frames
        assert [value.event_id for value in output.events] == list(range(1, len(expected_frames) + 1))
        for index, record in enumerate(output.verification_traces):
            if index in expected_frames:
                assert record['final_event_emitted'] is True
                assert record['stage_pass']['temporal_suppression'] is True
                assert record['final_event_id'] == expected_frames.index(index) + 1
            else:
                assert record['verification_pass'] is False
                assert record['rejection_stage'] == 'TEMPORAL_SUPPRESSION'
        with pytest.raises(ValueError, match='only once'):
            output.finalize(module.EventDetectorSettings())


def test_final_rank_prefers_confidence_then_time_then_candidate_id(tmp_path):
    with EventAnalysisSequence(tmp_path) as output:
        for index, (timestamp, confidence) in enumerate(((.0, .7), (.05, .9), (.05, .9), (.4, .7))):
            output.verification_traces.append(trace(index + 1))
            output.add_provisional(index + 1, event(index, timestamp, confidence))
        output.finalize(module.EventDetectorSettings(final_event_min_interval_seconds=.1))
        assert [value.frame_index for value in output.events] == [1, 3]
        assert output.verification_traces[2]['rejection_reason'] == 'DUPLICATE_VERIFIED_EVENT'
        plan = output._open().execute('EXPLAIN QUERY PLAN SELECT 1 FROM accepted WHERE timestamp>=? AND timestamp<=? '
            'AND abs(timestamp-?)<=? LIMIT 1', (0., .2, .1, .1)).fetchall()
        assert any('SEARCH accepted USING COVERING INDEX accepted_event_time' in row[-1] for row in plan)


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan, True, '0.8', 2 ** 53 + 1])
def test_invalid_active_rank_keys_are_not_coerced(tmp_path, value):
    with EventAnalysisSequence(tmp_path) as output:
        with pytest.raises(ValueError, match='finite binary64'):
            output.add_provisional(1, event(confidence=value))
        assert output._execute('SELECT COUNT(*) FROM provisional', fetch=True) == (0,)


@pytest.mark.parametrize('count', [257, 4097])
def test_native_selector_retains_constant_live_candidate_snapshots_across_commits(tmp_path, monkeypatch, count):
    original = module.EventCandidate
    live = weakref.WeakSet()
    seen = {'peak': 0}

    class ObservedCandidate(original):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live.add(self)
            seen['peak'] = max(seen['peak'], len(live))

    monkeypatch.setattr(module, 'EventCandidate', ObservedCandidate)
    with EventAnalysisSequence(tmp_path) as output, module._FeatureSequence(tmp_path) as features:
        for index in range(count):
            features.append(module._Feature(index, index * .25, (50., 100.), (50., 100.),
                'DETECTED', .9, score=.8, normalized_speed=.1, strengths={'kinematic': .8}))
        detector = module.TennisEventDetector()
        result = detector._candidates_from_features(features, 25., 1280, 720, 'DISABLED', output=output.candidates)
        assert result is output.candidates and len(result) == count
        assert len(live) == 0 and seen['peak'] <= 2
        for index in range(count):
            value = result[index]
            assert value.frame_index == index
        value = None
        assert seen['peak'] <= 2
        assert output._execute('PRAGMA cache_size', fetch=True) == (-2048,)
        assert output._execute('PRAGMA mmap_size', fetch=True) == (0,)
        assert output._execute('PRAGMA temp_store', fetch=True) == (1,)
        assert output._pending < output.transaction_size <= 256


@pytest.mark.parametrize('failure', [RuntimeError, asyncio.CancelledError])
def test_failure_and_cancel_close_remove_only_owned_scratch_and_suspended_queries(tmp_path, failure):
    unrelated = tmp_path / 'unrelated.sqlite3'
    unrelated.write_bytes(b'keep')
    output = EventAnalysisSequence(tmp_path)
    iterator = None
    with pytest.raises(failure):
        with output:
            output.events.append(event(event_id=1))
            iterator = output.events_by_frame()
            assert next(iterator).event_id == 1
            raise failure('synthetic pipeline interruption')
    iterator.close()
    assert not output.path.exists()
    assert unrelated.read_bytes() == b'keep'
    with pytest.raises(RuntimeError, match='not open'):
        output.events[0]


def test_frame_queries_preserve_stable_order_and_inclusive_bounds_without_event_list(tmp_path):
    with EventAnalysisSequence(tmp_path) as output:
        for ident, frame in enumerate((9, 2, 9, 4), 1):
            output.events.append(event(frame, ident * .25, event_id=ident))
        assert [value.event_id for value in output.events_by_frame()] == [2, 4, 1, 3]
        assert [value.event_id for value in output.events_by_frame(4, 9)] == [4, 1, 3]
        assert [value.event_id for value in output.events_by_frame(upper=3)] == [2]
        assert list(output.events_by_frame(event_type=module.EventType.PLAYER_1_HIT)) == []
        assert output.events[::-1][0].event_id == 4


def test_result_option_requires_native_scratch_and_an_open_empty_caller_owned_context(tmp_path):
    detector = module.TennisEventDetector()
    trajectory = points()
    boxes = [None] * len(trajectory)
    unopened = EventAnalysisSequence(tmp_path)
    with pytest.raises(ValueError, match='native feature scratch'):
        detector.analyze(trajectory, boxes, boxes, result_output=unopened)
    with pytest.raises(RuntimeError, match='not open'):
        detector.analyze(trajectory, boxes, boxes, scratch_directory=tmp_path, result_output=unopened)
    with EventAnalysisSequence(tmp_path) as occupied:
        occupied.candidates.append(candidate())
        with pytest.raises(ValueError, match='cannot be reused'):
            detector.analyze(trajectory, boxes, boxes, scratch_directory=tmp_path, result_output=occupied)


def test_three_streamed_wire_arrays_keep_loaded_legacy_values_without_importing_phase6_runtime(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'sanitize_json_value', 'event_export_records', 'event_candidate_export_records'}
    namespace = {'np': np, 'Any': object}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    with EventAnalysisSequence(tmp_path) as output:
        output.candidates.append(candidate(evidence={'finite': np.float32(.5), 'unknown': math.inf}))
        output.events.append(event(event_id=1, confidence=.87654, court_position_m=(1., 2.)))
        output.verification_traces.append({'candidate_id': 1, 'unknown': math.nan})
        write_json_array_artifact(tmp_path / 'events.json', {'events_count': 1}, 'events', namespace['event_export_records'](output.events))
        write_json_array_artifact(tmp_path / 'candidates.json', {'scope': 'diagnostic'}, 'candidates', namespace['event_candidate_export_records'](output.candidates))
        write_json_array_artifact(tmp_path / 'traces.json', {'scope': 'audit'}, 'traces',
                                  (namespace['sanitize_json_value'](item) for item in output.verification_traces))
    actual = json.loads((tmp_path / 'events.json').read_text())
    assert actual == {'events_count': 1, 'events': [{'event_id': 1, 'event_type': 'BOUNCE', 'frame': 0,
        'timestamp_s': 0., 'player_id': None, 'ball_position_px': [50., 100.], 'court_position_m': [1., 2.],
        'confidence': .877, 'trajectory_state': 'DETECTED', 'evidence': {}}]}
    assert json.loads((tmp_path / 'candidates.json').read_text())['candidates'][0]['evidence'] == {'finite': .5, 'unknown': None}
    assert json.loads((tmp_path / 'traces.json').read_text()) == {'scope': 'audit', 'traces': [{'candidate_id': 1, 'unknown': None}]}


def forced_native_analysis(detector, output, count, tmp_path, monkeypatch):
    """Actual analysis control flow with lazy input and synthetic verification."""
    class Timeline(Sequence):
        def __len__(self):
            return count
        def __getitem__(self, index):
            if not 0 <= index < count:
                raise IndexError(index)
            return TemporalBallPoint(index, index * .25, 50., 100., confidence=.9, state=BallState.DETECTED)

    timeline = Timeline()
    monkeypatch.setattr(detector, '_positions', lambda *args: (None, 'DISABLED'))

    def features(*args, output=None):
        assert output is not None, 'Native analysis must keep its single disk feature pass'
        for index in range(count):
            output.append(module._Feature(index, index * .25, (50., 100.), (50., 100.),
                'DETECTED', .9, score=.8, normalized_speed=.1, strengths={'kinematic': .8}))
        return output

    monkeypatch.setattr(detector, '_features', features)
    monkeypatch.setattr(detector, '_refined_frame', lambda item, *args: item.frame_index)
    monkeypatch.setattr(detector, '_legacy_feature_fallback', lambda item, *args: item)
    if '_verify' not in detector.__dict__:
        monkeypatch.setattr(detector, '_verify', lambda ident, item, feature, *args, **kwargs:
            (event(feature.frame, feature.timestamp, evidence={'synthetic_verification_only': True}), trace(ident)))
    return detector.analyze(timeline, timeline, timeline, fps=4., frame_size=(1280, 720),
                             scratch_directory=tmp_path, result_output=output)


@pytest.mark.parametrize('count', [257, 1025])
def test_actual_native_analyze_keeps_candidate_event_and_trace_objects_bounded(tmp_path, monkeypatch, count):
    live = {kind: weakref.WeakSet() for kind in ('candidate', 'event', 'trace')}
    peaks = dict.fromkeys(live, 0)

    def observe(kind, value):
        live[kind].add(value)
        peaks[kind] = max(peaks[kind], len(live[kind]))

    original_candidate, original_event = module.EventCandidate, module.TennisEvent

    class ObservedCandidate(original_candidate):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observe('candidate', self)

    class ObservedEvent(original_event):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observe('event', self)

    class ObservedTrace(dict):
        __hash__ = object.__hash__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observe('trace', self)

    monkeypatch.setattr(module, 'EventCandidate', ObservedCandidate)
    monkeypatch.setattr(module, 'TennisEvent', ObservedEvent)
    original_decode = storage_module._decode_record
    monkeypatch.setattr(storage_module, '_decode_record', lambda kind, payload:
                        ObservedTrace(original_decode(kind, payload)) if kind == 'traces' else original_decode(kind, payload))
    detector = module.TennisEventDetector()
    monkeypatch.setattr(detector, '_verify', lambda ident, item, feature, *args, **kwargs:
        (event(feature.frame, feature.timestamp, evidence={'synthetic_verification_only': True}), ObservedTrace(trace(ident))))
    with EventAnalysisSequence(tmp_path) as output:
        analysis = forced_native_analysis(detector, output, count, tmp_path, monkeypatch)
        assert len(analysis.candidates) == len(analysis.events) == len(analysis.verification_traces) == count
        assert all(not isinstance(rows, list) for rows in
                   (analysis.candidates, analysis.events, analysis.verification_traces))
        assert max(peaks.values()) <= 4
        assert all(len(values) == 0 for values in live.values())
        assert output._execute('SELECT COUNT(*) FROM provisional', fetch=True) == (count,)
        assert output._execute('SELECT COUNT(*) FROM accepted', fetch=True) == (count,)
        assert analysis.events[-1].event_id == count
        assert analysis.verification_traces[-1]['final_event_id'] == count
    assert not list(tmp_path.glob('event_features_*')) and not list(tmp_path.glob('event_analysis_*'))


@pytest.mark.parametrize('gate', ['activity', 'provenance'])
def test_native_rejection_gate_mutations_are_written_back_before_continue(tmp_path, monkeypatch, gate):
    if gate == 'activity':
        detector = module.TennisEventDetector(enable_activity_state_gating=True)
        monkeypatch.setattr(module, 'VisualActivityStateClassifier', lambda **kwargs: SimpleNamespace(
            classify_trajectory=lambda trajectory, **kwargs: [module.VisualActivityState.DEAD_BALL_VISUAL] * len(trajectory)))
        expected_stage = 'VISION_ACTIVITY_FILTER'
    else:
        detector = module.TennisEventDetector(enable_provenance_gating=True)
        monkeypatch.setattr(module, 'ProvenanceAuditor', lambda **kwargs: SimpleNamespace(
            audit=lambda *args: SimpleNamespace(provenance_trust_score=0., low_trust_reason='SYNTHETIC_LOW_TRUST',
                                                to_dict=lambda: {'synthetic_only': True})))
        expected_stage = 'PROVENANCE_FILTER'
    with EventAnalysisSequence(tmp_path) as output:
        analysis = forced_native_analysis(detector, output, 7, tmp_path, monkeypatch)
        assert len(analysis.candidates) == len(analysis.verification_traces) == 7
        assert len(analysis.events) == 0
        for record in analysis.verification_traces:
            assert record['verification_pass'] is False and record['suppressed'] is True
            assert record['rejection_stage'] == expected_stage
