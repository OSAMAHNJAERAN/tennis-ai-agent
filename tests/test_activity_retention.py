"""Actual optional classifier retention; synthetic values are not activity GT."""
import gc
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Sequence
import sys
import tracemalloc
from types import ModuleType

import pytest

from src.events.activity_state import ActivityStateSettings, VisualActivityStateClassifier
from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


OBSERVATIONS = []
ROOT = Path(__file__).resolve().parents[1]
RED = ROOT / 'artifacts/validation/yolo26_execution_2026-10-17/activity_retention_checks_v1/retention_red_v1'
RED_SHA256 = 'e5288a39b8b14ee31b5fa0698e5ec807d45c70e64bad6aafe8ea8f3683d76c50'


def red_rows():
    data = (RED / 'receipt.json').read_bytes()
    assert hashlib.sha256(data).hexdigest() == RED_SHA256
    return json.loads(data)['rows']


def baseline_classifier(scene):
    module = ModuleType('retained_activity_baseline')
    source = RED / 'source_snapshot/src/events/activity_state.py'
    receipt = json.loads((RED / 'receipt.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == receipt['source_before']['src/events/activity_state.py']
    sys.modules[module.__name__] = module
    try:
        exec(compile(source.read_text(), str(source), 'exec'), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    settings = module.ActivityStateSettings(max_pred_gap_for_dead=100000) if scene.startswith('preplay') else None
    return module.VisualActivityStateClassifier(fps=5., frame_size=(1280, 720), settings=settings)


class Trajectory(Sequence):
    """Fresh typed values per indexed read, with no resident trajectory list."""
    def __init__(self, count, scene='mixed', indices=None):
        self.count, self.scene = count, scene
        self.indices = range(count) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.count, self.scene, self.indices[index])
        i = self.indices[index]
        x, y, state, confidence = 100. + (i % 97) * 17., 220. + (i % 23) * 9., BallState.DETECTED, .7
        if self.scene == 'missing':
            x = y = None
            state, confidence = BallState.MISSING, None
        elif self.scene == 'static':
            x, y, confidence = 136., 74., .03
        elif self.scene in ('preplay', 'preplay_then_active') and (self.scene == 'preplay' or i < 53):
            x, y, state, confidence = 300. + (i % 3) * .001, 180., BallState.PREDICTED, 0.
        elif self.scene == 'mixed' and 20 <= i % 97 < 47:
            state, confidence = BallState.PREDICTED, 0.
        elif self.scene == 'mixed' and i % 97 >= 73:
            x = y = None
            state, confidence = BallState.MISSING, None
        return TemporalBallPoint(i, i / 5. + (i % 3) * .00007, x, y,
                                 state=state, confidence=confidence)


def classifier(scene):
    settings = ActivityStateSettings(max_pred_gap_for_dead=100000) if scene.startswith('preplay') else None
    return VisualActivityStateClassifier(fps=5., frame_size=(1280, 720), settings=settings)


def digest(instance):
    value = hashlib.sha256()
    for i in range(len(instance._frame_states)):
        value.update(json.dumps([instance.state_at_frame(i).value, instance.evidence_at_frame(i)],
                                sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    return value.hexdigest()


@pytest.mark.parametrize('scene', ['moving', 'mixed', 'missing', 'static', 'preplay', 'preplay_then_active'])
def test_actual_classifier_retains_ordered_state_and_evidence(tmp_path, scene):
    from src.utils.activity_state_sequence import ActivityStateSequence
    instance = classifier(scene)
    with ActivityStateSequence(tmp_path) as output:
        states = instance.classify_trajectory(Trajectory(129, scene), output=output)
        assert len(states) == 129 and states is output.states
        expected = next(row for row in red_rows() if row['scene'] == scene)
        assert digest(instance) == expected['values_sha256']
        ordinary = classifier(scene)
        assert [state.value for state in ordinary.classify_trajectory(Trajectory(129, scene))] == [state.value for state in states]
        assert digest(ordinary) == expected['values_sha256']
        OBSERVATIONS.append(dict(scene=scene, frames=129, values_sha256=digest(instance),
                                 retained_red_values_equal=True, qualification_evidence=False))


@pytest.mark.parametrize('count', [257, 1025, 4097])
def test_actual_classifier_retention_at_increasing_frame_counts(tmp_path, count):
    from src.utils.activity_state_sequence import ActivityStateSequence
    instance = classifier('mixed')
    trajectory = Trajectory(count)
    with ActivityStateSequence(tmp_path) as output:
        gc.collect()
        tracemalloc.start()
        try:
            instance.classify_trajectory(trajectory, output=output)
            retained, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        expected = next(row for row in red_rows() if row['scene'] == 'capacity' and row['frames'] == count)
        assert digest(instance) == expected['values_sha256']
        OBSERVATIONS.append(dict(scene='capacity', frames=count,
            retained_traced_bytes=retained, peak_traced_bytes=peak, values_sha256=digest(instance),
            actual_classifier=True, lazy_typed_source=True, retained_red_values_equal=True,
            qualification_evidence=False))


def test_classifier_has_no_duration_sized_resident_state():
    rows = [row for row in OBSERVATIONS if row['scene'] == 'capacity']
    assert [row['frames'] for row in rows] == [257, 1025, 4097]
    assert max(row['retained_traced_bytes'] for row in rows) < 512 * 1024


@pytest.mark.parametrize('setting', ['active_min_detection_confidence', 'static_lock_max_spread_px',
                                    'active_min_normalised_speed', 'pre_serve_max_spread_normalised'])
@pytest.mark.parametrize('direction', [-math.inf, math.inf])
def test_threshold_decisions_and_evidence_match_retained_math_exactly(tmp_path, setting, direction):
    from src.events.activity_state import _position_spread_px, _median_grounded_speed
    from src.utils.activity_state_sequence import ActivityStateSequence
    scene = 'preplay' if setting.startswith('pre_serve') else 'moving'
    points = Trajectory(129, scene)
    baseline, actual = baseline_classifier(scene), classifier(scene)
    if setting == 'active_min_detection_confidence':
        threshold = .7
    elif setting == 'static_lock_max_spread_px':
        threshold = _position_spread_px(list(points[27:38]))
    elif setting == 'active_min_normalised_speed':
        threshold = _median_grounded_speed(list(points[30:35]), actual.diagonal)
        baseline.settings.active_min_detection_confidence = actual.settings.active_min_detection_confidence = 1.
    else:
        threshold = _position_spread_px(list(points)) / actual.diagonal
    value = math.nextafter(threshold, direction)
    setattr(baseline.settings, setting, value)
    setattr(actual.settings, setting, value)
    baseline.classify_trajectory(points)
    with ActivityStateSequence(tmp_path) as output:
        actual.classify_trajectory(points, output=output)
        assert digest(actual) == digest(baseline)
        assert actual.dump_timeline() == baseline.dump_timeline()


@pytest.mark.parametrize('count', [0, 1, 15, 16])
def test_empty_and_prefix_boundary_match_retained_classifier(tmp_path, count):
    from src.utils.activity_state_sequence import ActivityStateSequence
    points = Trajectory(count, 'preplay')
    baseline, actual = baseline_classifier('preplay'), classifier('preplay')
    baseline.classify_trajectory(points)
    with ActivityStateSequence(tmp_path) as output:
        actual.classify_trajectory(points, output=output)
        assert digest(actual) == digest(baseline)


def test_empty_trajectory_retains_legacy_early_return_before_window_settings(tmp_path):
    from src.utils.activity_state_sequence import ActivityStateSequence
    baseline, actual = baseline_classifier('moving'), classifier('moving')
    baseline.fps = actual.fps = math.nan
    assert baseline.classify_trajectory([]) == []
    assert actual.classify_trajectory([]) == []
    with ActivityStateSequence(tmp_path) as output:
        assert list(actual.classify_trajectory([], output=output)) == []


@pytest.mark.parametrize('interruption', [OSError, KeyboardInterrupt])
def test_classification_failure_publishes_no_partial_classifier_and_cleans_scratch(tmp_path, interruption):
    from src.utils.activity_state_sequence import ActivityStateSequence
    actual = classifier('moving')
    previous = actual.classify_trajectory(Trajectory(3, 'moving'))
    class Interrupted(Trajectory):
        def __getitem__(self, index):
            if not isinstance(index, slice) and index < len(self) and self.indices[index] == 22:
                raise interruption('Supplied typed-source interruption')
            return super().__getitem__(index)
    with pytest.raises(interruption, match='Supplied typed-source'):
        with ActivityStateSequence(tmp_path) as output:
            path = output.path
            actual.classify_trajectory(Interrupted(129, 'moving'), output=output)
    assert actual._frame_states is previous
    assert not path.exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_native_detector_activity_ownership_and_legacy_results(tmp_path, monkeypatch, enabled):
    from dataclasses import asdict
    from src.events import event_detector as module
    from src.utils import activity_state_sequence as storage
    from src.utils.event_analysis_sequence import EventAnalysisSequence
    from src.utils.temporal_ball_point_sequence import TemporalBallPointSequence
    original = module.VisualActivityStateClassifier
    observed = []
    class ObservedClassifier(original):
        def classify_trajectory(self, values, **kwargs):
            if kwargs.get('output') is not None:
                observed.append(kwargs['output'])
            return super().classify_trajectory(values, **kwargs)
    monkeypatch.setattr(module, 'VisualActivityStateClassifier', ObservedClassifier)
    if not enabled:
        class UnexpectedStore:
            def __init__(self, *args, **kwargs):
                raise AssertionError('Disabled gating must not create activity scratch')
        monkeypatch.setattr(storage, 'ActivityStateSequence', UnexpectedStore)
    points = [TemporalBallPoint(i, i / 25., 300. + 3. * i, 610. - 8. * abs(i - 30),
                               state=BallState.DETECTED, confidence=.85) for i in range(60)]
    absent = [None] * len(points)
    def snapshot(result):
        return json.dumps({'events': [asdict(row) for row in result.events],
            'candidates': [asdict(row) for row in result.candidates],
            'traces': list(result.verification_traces)}, sort_keys=True, default=lambda row: row.value)
    detector = lambda: module.TennisEventDetector(enable_activity_state_gating=enabled)
    expected = detector().analyze(points, absent, absent, fps=25., frame_size=(1280, 720))
    assert len(expected.candidates) > 0 and len(expected.verification_traces) > 0
    with TemporalBallPointSequence(tmp_path) as source, EventAnalysisSequence(tmp_path) as result:
        source.extend(points)
        actual = detector().analyze(source, absent, absent, fps=25., frame_size=(1280, 720),
                                   scratch_directory=tmp_path, result_output=result)
        assert snapshot(actual) == snapshot(expected)
        assert len(observed) == int(enabled)
        if enabled:
            assert not observed[0].path.exists()
            with pytest.raises(RuntimeError): observed[0].states[0]
    assert not list(tmp_path.glob('activity_state_*'))
