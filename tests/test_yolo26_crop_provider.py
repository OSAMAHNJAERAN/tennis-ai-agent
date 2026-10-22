"""Post-player crop transitions use injected models, with no inference runtime."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.detection import yolo26_adapters as adapters


class Hook:
    def __init__(self, error=None):
        self.removed, self.error = 0, error

    def remove(self):
        self.removed += 1
        if self.error is not None:
            raise self.error


class Tracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class Model:
    def __init__(self, device='cuda:2', *, cached=True, error=None):
        self.device = device
        self.overrides = {'imgsz': [640, 672], 'single_cls': False}
        self.hook, self.tracker = Hook(), Tracker()
        self.predictor = (SimpleNamespace(device=device, trackers=[self.tracker],
            _hook=self.hook, _feats=object(),
            args=SimpleNamespace(conf=.4, imgsz=960, classes=[0])) if cached else None)
        self.callbacks = {'on_predict_start': ['tracking'], 'on_predict_postprocess_end': ['tracking']}
        self.callback_resets, self.calls, self.track_calls = 0, [], 0
        self.error, self.return_device = error, None

    def reset_callbacks(self):
        self.callback_resets += 1
        self.callbacks = {'on_predict_start': ['default'], 'on_predict_postprocess_end': ['default']}

    def predict(self, image, **kwargs):
        self.calls.append((image, deepcopy(kwargs)))
        assert all('tracking' not in callbacks for callbacks in self.callbacks.values())
        if self.error is not None:
            raise self.error
        if self.predictor is None:
            self.predictor = SimpleNamespace(device=self.return_device or kwargs['device'])
        return ['native pose output']

    def track(self, image, **kwargs):
        self.track_calls += 1
        return []


def detector(model, *, device='auto'):
    return adapters.YOLO26PlayerPoseDetector('selected-player.pt', tracker='pinned.yaml',
                                            device=device, _model=model)


def test_transition_disposes_pollution_without_loading_or_replacing_selected_model(monkeypatch):
    monkeypatch.setattr(adapters, '_load', lambda *args, **kwargs: pytest.fail('second model load'))
    model = Model()
    player = detector(model)
    original_predictor = model.predictor
    player.reset()
    assert model.tracker.resets == 1
    assert model.predictor is original_predictor and model.hook.removed == 0
    provider = player.finish_tracking_for_crops(admitted_device='cuda:2')
    assert player.model is provider.model is model
    assert model.predictor is None and model.callback_resets == 1
    assert model.hook.removed == 1 and not hasattr(original_predictor, '_hook')
    assert original_predictor._feats is None
    # Finishing disposes the cached predictor; it does not reset IDs mid-pass.
    assert model.tracker.resets == 1
    assert provider.predict('crop', device='cpu', conf=.99, imgsz=32, nms=True) == ['native pose output']
    arguments = model.calls[0][1]
    assert arguments == {'imgsz': [640, 672], 'conf': .25, 'verbose': False,
        'device': 'cuda:2', 'end2end': True, 'nms': False, 'batch': 1,
        'rect': True, 'half': False, 'max_det': 300, 'iou': .7, 'augment': False,
        'agnostic_nms': False, 'classes': None, 'single_cls': False,
        'mode': 'predict', 'save': False}
    provider.assert_success()
    record = provider.record()
    assert record['prediction_calls'] == record['completed_prediction_calls'] == 1
    assert record['failed'] is False and record['configured_device'] == 'auto'
    assert json.loads(json.dumps(record)) == record


def test_checkpoint_options_are_snapshotted_before_tracking_and_record_is_detached():
    model = Model()
    player = detector(model)
    model.overrides['imgsz'][0] = 960
    model.overrides['single_cls'] = True
    provider = player.finish_tracking_for_crops(admitted_device=2)
    record = provider.record()
    record['input_spec']['imgsz'][0] = 32
    record['input_spec']['device'] = 'cpu'
    provider.predict('crop')
    assert model.calls[0][1]['imgsz'] == [640, 672]
    assert model.calls[0][1]['single_cls'] is False
    assert provider.record()['resolved_device'] == 'cuda:2'
    assert provider.record()['selected_model_path'] == 'selected-player.pt'


def test_finish_is_one_way_and_observing_saved_pose_remains_available():
    model = Model()
    player = detector(model)
    player.detect_and_track('frame')
    assert model.track_calls == 1
    player.finish_tracking_for_crops(admitted_device='cuda:2')
    for operation in (lambda: player.detect('frame'), lambda: player.detect_and_track('frame'),
                      lambda: player.finish_tracking_for_crops(admitted_device='cuda:2')):
        with pytest.raises(RuntimeError, match='closed|already finished'):
            operation()
    assert model.track_calls == 1 and not model.calls
    assert player.observe(None, {1: None}) == {1: None}
    player.reset()  # The existing camera-cut reset remains a no-op without a predictor.


@pytest.mark.parametrize('failure', [ValueError('native crop failure'), KeyboardInterrupt('interrupted')])
def test_native_failures_stay_latched_and_count_only_underlying_invocations(failure):
    model = Model(error=failure)
    provider = detector(model).finish_tracking_for_crops(admitted_device='cuda:2')
    try:
        provider.predict('crop')
    except BaseException as error:
        assert error is failure
    else:
        pytest.fail('native failure was swallowed')
    # Simulate the legacy extractor continuing after catching a predict error.
    with pytest.raises(RuntimeError, match='crop inference failed') as caught:
        provider.assert_success()
    assert caught.value.__cause__ is failure
    with pytest.raises(RuntimeError, match='crop inference failed'):
        provider.predict('another crop')
    assert provider.failure is failure
    assert len(model.calls) == provider.prediction_calls == 1
    assert provider.record()['completed_prediction_calls'] == 0
    assert provider.record()['failed'] is True


@pytest.mark.parametrize('admitted,expected', [('cpu', 'cpu'), (0, 'cuda:0'), ('2', 'cuda:2')])
def test_fresh_automatic_device_resume_requires_and_uses_the_sealed_device(admitted, expected):
    model = Model(cached=False)
    provider = detector(model).finish_tracking_for_crops(admitted_device=admitted, expected_device=expected)
    provider.predict('crop', device='different extractor default')
    assert model.calls[0][1]['device'] == expected
    assert provider.record()['resolved_device'] == expected


@pytest.mark.parametrize('configured,admitted,expected,cached,observed,match', [
    ('auto', 'cpu', None, False, 'cpu', 'saved crop device'),
    ('auto', 'cuda:2', 'cpu', False, 'cpu', 'Saved crop device'),
    ('cpu', 'cuda:2', None, False, 'cpu', 'Configured crop device'),
    ('cuda:2', 'cpu', None, False, 'cpu', 'Configured crop device'),
    ('auto', 'cuda:2', 'cuda:2', True, 'cpu', 'Player predictor'),
])
def test_device_mismatches_fail_before_disposal(configured, admitted, expected, cached, observed, match):
    model = Model(observed, cached=cached)
    player = detector(model, device=configured)
    original = model.predictor
    with pytest.raises(ValueError, match=match):
        player.finish_tracking_for_crops(admitted_device=admitted, expected_device=expected)
    assert model.predictor is original and model.callback_resets == model.hook.removed == 0
    player.detect_and_track('frame')
    assert model.track_calls == 1


@pytest.mark.parametrize('device', [None, True, np.bool_(True), -1, 'auto', 'cuda', 'cuda:-1', 'mps', '', ' cpu'])
def test_unresolved_or_invalid_admission_never_selects_a_fallback(device):
    model = Model()
    with pytest.raises(ValueError, match='explicit CPU or indexed CUDA'):
        detector(model).finish_tracking_for_crops(admitted_device=device)
    assert model.callback_resets == 0 and not model.calls


@pytest.mark.parametrize('imgsz', [True, 31, 4097, [], [640, 640, 640], [640., 640], None])
def test_bad_checkpoint_crop_shape_is_rejected_before_tracking_is_disposed(imgsz):
    model = Model()
    model.overrides['imgsz'] = imgsz
    player = detector(model)
    with pytest.raises(ValueError, match='crop image size'):
        player.finish_tracking_for_crops(admitted_device='cuda:2')
    assert model.hook.removed == model.callback_resets == 0


@pytest.mark.parametrize('imgsz', [640, [640], (640, 672)])
def test_checkpoint_native_scalar_and_rectangular_sizes_are_preserved(imgsz):
    model = Model()
    model.overrides = {'imgsz': imgsz, 'single_cls': True}
    provider = detector(model).finish_tracking_for_crops(admitted_device='cuda:2')
    expected = imgsz if type(imgsz) is int else list(imgsz)
    assert provider.record()['input_spec']['imgsz'] == expected
    assert provider.record()['input_spec']['single_cls'] is True


@pytest.mark.parametrize('overrides', [None, {'single_cls': 1}])
def test_missing_or_invalid_checkpoint_policy_does_not_mutate_predictor(overrides):
    model = Model()
    model.overrides = overrides
    with pytest.raises(ValueError, match='overrides|class configuration'):
        detector(model).finish_tracking_for_crops(admitted_device='cuda:2')
    assert model.callback_resets == 0


def test_failed_hook_cleanup_still_clears_callbacks_predictor_and_closes_tracking():
    model = Model()
    failure = RuntimeError('hook removal failed')
    model.hook.error = failure
    player = detector(model)
    with pytest.raises(RuntimeError) as caught:
        player.finish_tracking_for_crops(admitted_device='cuda:2')
    assert caught.value is failure
    assert model.predictor is None and model.callback_resets == 1
    with pytest.raises(RuntimeError, match='closed'):
        player.detect_and_track('frame')


@pytest.mark.parametrize('pollution', ['tracking', 'device', 'model'])
def test_later_state_pollution_fails_without_another_prediction(pollution):
    model = Model()
    player = detector(model)
    provider = player.finish_tracking_for_crops(admitted_device='cuda:2')
    if pollution == 'tracking':
        model.predictor = SimpleNamespace(device='cuda:2', trackers=[])
    elif pollution == 'device':
        model.predictor = SimpleNamespace(device='cpu')
    else:
        player.model = object()
    with pytest.raises(RuntimeError):
        provider.predict('crop')
    assert provider.record()['failed'] is True
    assert provider.record()['prediction_calls'] == 0 and not model.calls


def test_runtime_device_fallback_is_latched_after_the_actual_call():
    model = Model()
    model.return_device = 'cpu'
    provider = detector(model).finish_tracking_for_crops(admitted_device='cuda:2')
    with pytest.raises(RuntimeError, match='admitted device'):
        provider.predict('crop')
    assert provider.prediction_calls == 1 and provider.completed_prediction_calls == 0
    with pytest.raises(RuntimeError, match='crop inference failed'):
        provider.assert_success()
