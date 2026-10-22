"""Actual orchestration with deterministic observations and no model/media jobs."""
from collections.abc import Sequence
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest
import yaml

from src.pipeline import phase6_pipeline as current
from src.tracking.player_motion_tracking import JOINT_NAMES
from src.utils.bbox_utils import BBox


BASELINE = Path('artifacts/validation/product_execution/pose_racket_storage_2026-10-06/before/src/pipeline/phase6_pipeline.py')
BASELINE_SHA256 = '93b1bcf44a56ed70d76264ae7ab8cd2e8423616821632da09904d71ddc3c75ec'
FPS = 60000 / 1001
SIZE = (640, 360)


@pytest.fixture(scope='module')
def frozen_pipeline():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
    spec = importlib.util.spec_from_file_location('frozen_pose_pipeline', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BeforeShotLinkStopped(RuntimeError):
    pass


class ObservationFrame(dict):
    """A weak-referenceable normal mapping, without retaining a reference log."""


class GeneratedFrames(Sequence):
    def __init__(self, count):
        self.count, self.closed = count, False
        self.metadata = SimpleNamespace(fps=FPS)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def missing_racket():
    return {'state': 'MISSING', 'position_px': None, 'bbox_xyxy': None,
            'confidence': None, 'velocity_px_per_s': None,
            'motion_direction_deg': None, 'contact_event': None}


def observation_values(index, identity, kind, numpy_values=False, nonfinite_array=False):
    if kind == 'pose':
        if index % 4 == 1 and identity == 2:
            return None
        joints = {}
        for offset, name in enumerate(JOINT_NAMES):
            x = np.nextafter(float(80 + identity * 140 + offset), np.inf)
            position = [-0.0 if offset == 0 else float(x), float(150 + offset) + index / FPS]
            if index % 3 == 0 and name == 'left_wrist':
                position = None
            confidence = float('nan') if name == 'nose' and index == 2 else .9000000000000001
            joints[name] = {'position_px': np.asarray(position, dtype=np.float64) if numpy_values and position is not None else position,
                            'confidence': np.float32(confidence) if numpy_values else confidence}
        return joints
    if index % 3 == 1 and identity == 2:
        return missing_racket()
    box = [float(180 + identity * 50), 160.0, float(205 + identity * 50), 185.0]
    velocity = [-0.0, index / FPS]
    if nonfinite_array and index == 2:
        velocity = np.asarray([np.nan, np.inf], dtype=np.float64)
    elif numpy_values:
        velocity = np.asarray(velocity, dtype=np.float32)
    return {'state': 'DETECTED', 'position_px': [box[0] + 12.5, 172.5],
            'bbox_xyxy': np.asarray(box, dtype=np.float64) if numpy_values else box,
            'confidence': np.float32(.875) if numpy_values else .875,
            'velocity_px_per_s': velocity, 'motion_direction_deg': None,
            'contact_event': None, 'native_timestamp_probe': index / FPS,
            'zero_probe': 0, 'boolean_probe': np.bool_(True) if numpy_values else True}


def build_harness(module, directory, monkeypatch, count=5, enabled=(True, True),
                  stop_before_shots=False, failure=None, release_failure=False,
                  numpy_values=False, nonfinite_array=False):
    """Stub only inference/media boundaries, retaining actual orchestration/export."""
    directory.mkdir(parents=True)
    frames = GeneratedFrames(count)
    state = SimpleNamespace(pose_live=0, racket_live=0, pose_produced=0,
                            racket_produced=0, before_shots=None, captures=[], writers=[])

    class CourtDetector:
        def __init__(self, **_):
            pass

        def predict(self, frame):
            return np.zeros((14, 2))

    class PlayerDetector:
        model = object()

        def __init__(self, **_):
            pass

        def detect_and_track(self, frame, persist=True):
            assert persist is True
            return [BBox(80, 120, 140, 230, confidence=.9, track_id=11),
                    BBox(310, 40, 350, 120, confidence=.9, track_id=22)]

    class BallDetector:
        def __init__(self, **_):
            pass

        def predict_stream(self, source):
            for _ in source:
                yield []

    class Tracker:
        kind = 'pose'

        def __init__(self, **_):
            pass

        def observe(self, frame, boxes, timestamp=None):
            index = getattr(state, f'{self.kind}_produced')
            assert list(boxes) == [1, 2]
            if self.kind == 'racket':
                assert timestamp == index / FPS
                if enabled[0]:
                    assert state.pose_produced == index + 1
            elif enabled[1]:
                assert state.racket_produced == index
            values = ObservationFrame((identity, observation_values(index, identity, self.kind, numpy_values, nonfinite_array))
                                      for identity in ([2, 1] if index % 2 else [1, 2]))
            setattr(state, f'{self.kind}_produced', index + 1)
            setattr(state, f'{self.kind}_live', getattr(state, f'{self.kind}_live') + 1)

            def released(kind=self.kind):
                setattr(state, f'{kind}_live', getattr(state, f'{kind}_live') - 1)

            weakref.finalize(values, released)
            return values

    class RacketTracker(Tracker):
        kind = 'racket'

    class Capture:
        def __init__(self, path):
            self.index, self.released = 0, False
            state.captures.append(self)

        def read(self):
            if failure == 'read':
                raise OSError('Injected read failure')
            if self.index == count:
                return False, None
            self.index += 1
            return True, frames[self.index - 1]

        def release(self):
            self.released = True
            if release_failure:
                raise RuntimeError('Injected capture release failure')

    class Writer:
        def __init__(self, path, fourcc, fps, size):
            assert fps == FPS and size == SIZE
            self.released, self.frame_digests = False, []
            state.writers.append(self)

        def isOpened(self):
            return failure != 'writer_open'

        def write(self, frame):
            if failure == 'write':
                raise OSError('Injected write failure')
            self.frame_digests.append(hashlib.sha256(frame.tobytes()).hexdigest())

        def release(self):
            self.released = True

    monkeypatch.setattr(module, 'CourtKeypointDetector', CourtDetector)
    monkeypatch.setattr(module, 'PlayerDetector', PlayerDetector)
    monkeypatch.setattr(module, 'WASBBallDetector', BallDetector)
    monkeypatch.setattr(module, 'PlayerMotionTracking', Tracker)
    monkeypatch.setattr(module, 'RacketTracking', RacketTracker)
    monkeypatch.setattr(module, 'VideoFrameSequence', lambda path: frames)
    monkeypatch.setattr(module.cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(module.cv2, 'VideoWriter', Writer)
    monkeypatch.setattr(module.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(module, 'tqdm', lambda values, **_: values)
    if failure == 'draw':
        def fail_draw(*args, **kwargs):
            raise OSError('Injected draw failure')
        monkeypatch.setattr(module.VideoAnnotator, 'draw_player_boxes', fail_draw)

    config = yaml.safe_load(Path('configs/phase6_analytics/pipeline.yaml').read_text())
    config['court_detection']['model_path'] = 'never-opened'
    config['player_detection']['model'] = 'never-opened'
    config['ball_detection'].update({'backend': 'wasb', 'model': 'never-opened',
                                     'source_directory': 'never-opened', 'temporal_step': 1})
    config['temporal_tracking']['strategy'] = 'model_top1'
    config['event_detection']['authoritative_enabled'] = False
    config['pose_estimation']['enabled'] = False
    config['player_motion'] = {'enabled': enabled[0]}
    config['racket_tracking'] = {'enabled': enabled[1]}
    config_path = directory / 'pipeline.yaml'
    config_path.write_text(yaml.safe_dump(config))
    pipeline = module.Phase6Pipeline(str(config_path))
    if stop_before_shots:
        def stop(**kwargs):
            assert kwargs['raw_frames'] is frames
            state.before_shots = {'pose_live': state.pose_live, 'racket_live': state.racket_live,
                                  'pose_produced': state.pose_produced, 'racket_produced': state.racket_produced}
            raise BeforeShotLinkStopped('Stopped before shot linking; no model or media inference')
        pipeline.shot_linker.link_shots = stop
    return pipeline, frames, state, directory / 'output'


@pytest.mark.parametrize('count', [64, 512])
def test_pose_and_racket_original_records_are_bounded_before_shot_linking(tmp_path, monkeypatch, count):
    pipeline, frames, state, output = build_harness(current, tmp_path / 'run', monkeypatch, count,
                                                  stop_before_shots=True)
    with pytest.raises(BeforeShotLinkStopped):
        pipeline.run('generated-frames-only.mp4', str(output))
    assert state.before_shots['pose_produced'] == count
    assert state.before_shots['racket_produced'] == count
    assert state.before_shots['pose_live'] <= 2, state.before_shots
    assert state.before_shots['racket_live'] <= 2, state.before_shots
    assert frames.closed


@pytest.mark.parametrize('failure', ['writer_open', 'read', 'draw', 'write'])
def test_render_handles_release_on_failure_before_traceback_or_gc_cleanup(tmp_path, monkeypatch, failure):
    pipeline, frames, state, output = build_harness(current, tmp_path / 'run', monkeypatch, failure=failure)
    error = RuntimeError if failure == 'writer_open' else OSError
    with pytest.raises(error, match='Unable to open output video writer' if failure == 'writer_open' else f'Injected {failure} failure') as caught:
        pipeline.run('generated-frames-only.mp4', str(output))
    # Retain the exception and its frames. Release must be explicit, before GC.
    assert caught.value.__traceback__ is not None
    assert len(state.writers) == 1 and state.writers[0].released
    assert len(state.captures) == (0 if failure == 'writer_open' else 1)
    assert all(capture.released for capture in state.captures)
    assert frames.closed


def test_render_cleanup_failure_preserves_primary_and_releases_other_handle(tmp_path, monkeypatch):
    pipeline, frames, state, output = build_harness(current, tmp_path / 'run', monkeypatch,
                                                  failure='write', release_failure=True)
    with pytest.raises(OSError, match='Injected write failure') as caught:
        pipeline.run('generated-frames-only.mp4', str(output))
    assert state.captures[0].released and state.writers[0].released
    assert any('Injected capture release failure' in note for note in caught.value.__notes__)
    assert frames.closed


@pytest.mark.parametrize('enabled', [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize('numpy_values', [False, True])
def test_complete_pose_racket_raw_bytes_and_render_frames_match_frozen_pipeline(tmp_path, monkeypatch, frozen_pipeline, enabled, numpy_values):
    outputs, renders = [], []
    for name, module in [('frozen', frozen_pipeline), ('current', current)]:
        with monkeypatch.context() as patch:
            pipeline, frames, state, output = build_harness(module, tmp_path / name, patch,
                                                          enabled=enabled, numpy_values=numpy_values)
            result = pipeline.run('generated-frames-only.mp4', str(output))
            assert result['status'] == 'success' and frames.closed
            assert state.writers[0].released and state.captures[0].released
            renders.append(state.writers[0].frame_digests)
            outputs.append({filename: (output / filename).read_bytes()
                            for filename in ('player_motion.json', 'racket_tracking.json')})
    assert outputs[0] == outputs[1]
    assert renders[0] == renders[1] and len(renders[1]) == 5
    assert all(not payload.endswith((b'\n', b'\r')) for payload in outputs[1].values())
    motion = json.loads(outputs[1]['player_motion.json'])
    rackets = json.loads(outputs[1]['racket_tracking.json'])
    assert len(motion['pose_frames']) == (5 if enabled[0] else 0)
    assert len(rackets['frames']) == (5 if enabled[1] else 0)
    if enabled[0]:
        assert list(motion['pose_frames'][1]) == ['2', '1']
        assert motion['pose_frames'][1]['2'] is None
        assert motion['pose_frames'][2]['1']['nose']['confidence'] is None
    if enabled[1]:
        assert rackets['frames'][1]['2'] == missing_racket()
        assert rackets['frames'][2]['1']['native_timestamp_probe'] == 2 / FPS
        assert rackets['frames'][0]['1']['boolean_probe'] is True
        assert rackets['frames'][0]['1']['velocity_px_per_s'][0] == 0
        assert b'-0.0' in outputs[1]['racket_tracking.json']


def test_integer_pose_identities_keep_distinct_render_colors_and_detected_rackets(tmp_path, monkeypatch):
    original_line, original_rectangle = current.cv2.line, current.cv2.rectangle
    colors, racket_boxes = set(), []

    def line(frame, first, second, color, *args):
        if color in ((220, 210, 40), (40, 180, 255)):
            colors.add(color)
        return original_line(frame, first, second, color, *args)

    def rectangle(frame, first, second, color, *args):
        if color == (80, 240, 160):
            racket_boxes.append((first, second))
        return original_rectangle(frame, first, second, color, *args)

    monkeypatch.setattr(current.cv2, 'line', line)
    monkeypatch.setattr(current.cv2, 'rectangle', rectangle)
    pipeline, _, _, output = build_harness(current, tmp_path / 'run', monkeypatch)
    assert pipeline.run('generated-frames-only.mp4', str(output))['status'] == 'success'
    assert colors == {(220, 210, 40), (40, 180, 255)}
    assert ((230, 160), (255, 185)) in racket_boxes
    assert ((280, 160), (305, 185)) in racket_boxes


def test_nonfinite_numpy_array_export_failure_matches_frozen_pipeline(tmp_path, monkeypatch, frozen_pipeline):
    for name, module in [('frozen', frozen_pipeline), ('current', current)]:
        with monkeypatch.context() as patch:
            pipeline, frames, state, output = build_harness(module, tmp_path / name, patch,
                                                          nonfinite_array=True)
            with pytest.raises(ValueError, match='Out of range float values'):
                pipeline.run('generated-frames-only.mp4', str(output))
            assert frames.closed and state.writers[0].released and state.captures[0].released
