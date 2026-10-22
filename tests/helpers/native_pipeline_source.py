"""Complete native orchestration with an explicit injected vision boundary.

The pipeline's definitions, methods, numerical consumers, disk owners and render
code execute unchanged. Only top-level neural imports are omitted. Its actual
constructor receives one three-adapter bundle over deterministic NumPy results;
checkpoint validation/loading and the fixture's reviewed interval are synthetic.
The inert ``torch`` name below is a local namespace object, never a module.
This helper supplies no actual neural inference, accuracy or capacity evidence.
"""
from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.court.court_geometry import TennisCourtGeometry
from src.detection.yolo26_adapters import (
    YOLO26BallDetector, YOLO26CourtPoseDetector, YOLO26PlayerPoseDetector,
)


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = ROOT / 'src/pipeline/phase6_pipeline.py'
WIDTH, HEIGHT, FPS = 320, 240, 25
NEURAL_IMPORTS = {
    'src.court.court_keypoint_detector',
    'src.detection.player_detector',
    'src.detection.yolo11_ball_detector',
    'src.detection.wasb_ball_detector',
    'src.shot_analysis.pose_feature_extractor',
    'src.tracking.racket_tracking',
}
FORBIDDEN_ROOTS = {'torch', 'torchvision', 'ultralytics'}


def forbidden_constructor(*args, **kwargs):
    raise AssertionError('Injected native fixture reached an additional neural constructor')


@contextmanager
def no_inference_imports():
    """Reject before cache lookup and retain violations caught by consumers.

    The yielded list exposes every blocked attempt to the caller. A normal
    context exit still fails if application code caught a guard exception. An
    existing body failure retains its identity, with violations attached as a
    note. No modules are created or replaced in ``sys.modules``.
    """
    native_import = builtins.__import__
    attempts = []

    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN_ROOTS:
            attempts.append({'mechanism': 'builtins.__import__', 'name': name,
                             'root': name.split('.', 1)[0]})
            raise AssertionError('Model-free native fixture attempted a neural import: ' + name)
        return native_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        yield attempts
    except BaseException as error:
        if attempts:
            error.add_note('Blocked neural import attempts retained by fixture guard: '
                           + ', '.join(value['name'] for value in attempts))
        raise
    else:
        if attempts:
            raise AssertionError('Model-free native context retained forbidden import attempts: '
                                 + ', '.join(value['name'] for value in attempts))
    finally:
        builtins.__import__ = native_import


def load_native_pipeline_source():
    """Load actual definitions, omitting only the named neural import nodes."""
    source = PIPELINE_PATH.read_bytes()
    tree = ast.parse(source, filename=str(PIPELINE_PATH))
    kept, omitted = [], []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name in FORBIDDEN_ROOTS for alias in node.names):
            if len(node.names) != 1 or node.names[0].name != 'torch':
                raise AssertionError('Review the changed pipeline neural import boundary')
            omitted.append('torch')
        elif isinstance(node, ast.ImportFrom) and node.module in NEURAL_IMPORTS:
            omitted.append(node.module)
        else:
            kept.append(node)
    if set(omitted) != {'torch', *NEURAL_IMPORTS}:
        raise AssertionError('Review the changed pipeline import inventory')
    inert_cuda = SimpleNamespace(is_available=lambda: False, empty_cache=forbidden_constructor)
    namespace = dict(__name__=__name__, torch=SimpleNamespace(cuda=inert_cuda))
    for name in ('CourtKeypointDetector', 'PlayerDetector', 'YOLO11BallDetector',
                 'WASBBallDetector', 'WASBEnsembleDetector', 'PoseFeatureExtractor', 'RacketTracking'):
        namespace[name] = forbidden_constructor
    with no_inference_imports() as source_import_attempts:
        exec(compile(ast.Module(body=kept, type_ignores=[]), str(PIPELINE_PATH), 'exec'), namespace)
    # The motion module's constants/functions are safe imports; its historical
    # model constructor must still fail if this candidate unexpectedly uses it.
    namespace['PlayerMotionTracking'] = forbidden_constructor
    return SimpleNamespace(**namespace, fixture_source_sha256=hashlib.sha256(source).hexdigest(),
                           fixture_omitted_imports=tuple(sorted(omitted)),
                           fixture_forbidden_import_roots=tuple(sorted(FORBIDDEN_ROOTS)),
                           fixture_source_import_attempts=source_import_attempts)


class Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class SyntheticTracker:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class InjectedModel:
    """Only the selected adapter's NumPy result boundary is replaced."""
    def __init__(self, role, trace, *, fail_ball_at=None, fail_ball_error=None):
        self.role, self.trace, self.fail_ball_at = role, trace, fail_ball_at
        self.fail_ball_error = (fail_ball_error if fail_ball_error is not None
                                else RuntimeError('Injected ball result boundary failure'))
        self.device = 'cpu'
        self.overrides = {'imgsz': 640, 'single_cls': False}
        self.predictor = None
        self.callbacks = {'on_predict_start': ['default'], 'on_predict_postprocess_end': ['default']}
        self.calls, self.tracking_calls, self.callback_resets = [], [], 0
        self.frame_indices, self.device_moves = [], []
        self.tracker = SyntheticTracker()
        self.names = {0: {'players': 'person', 'court': 'court', 'ball': 'tennis-ball'}[role]}

    def reset_callbacks(self):
        self.callback_resets += 1
        self.callbacks = {'on_predict_start': ['default'], 'on_predict_postprocess_end': ['default']}

    def to(self, device):
        self.device = str(device)
        self.device_moves.append(self.device)
        return self

    @staticmethod
    def frame_index(image):
        # A 32x32 constant patch survives this fixture's lossy MP4V codec. Use
        # its interior, away from the block edge, to bind results to content
        # rather than to a model instance's fresh local call counter.
        return int(round(float(np.asarray(image)[8:24, 8:24].mean()) / 20.))

    def _result(self, index):
        if self.role == 'players':
            boxes = np.array([[88., 142., 112., 202.], [188., 22., 212., 72.]])
            points = np.zeros((2, 17, 3), dtype=float)
            for person, (x1, y1, x2, y2) in enumerate(boxes):
                points[person, :, :2] = [(x1 + x2) / 2, (y1 + y2) / 2]
                points[person, :, 2] = .95
                points[person, 15, :2] = [(x1 + x2) / 2 - 2, y2 - 2]
                points[person, 16, :2] = [(x1 + x2) / 2 + 2, y2 - 2]
            return SimpleNamespace(boxes=Boxes(xyxy=boxes, conf=np.array([.95, .94]),
                                               id=np.array([31, 84])),
                                   keypoints=SimpleNamespace(data=points))
        if self.role == 'court':
            xy = TennisCourtGeometry.get_canonical_keypoints() * [20., 8.] + [40., 20.]
            points = np.column_stack((xy, np.full(14, .99)))[None]
            return SimpleNamespace(boxes=Boxes(xyxy=np.array([[40., 20., 260., 212.]]),
                                               conf=np.array([.99]), id=None),
                                   keypoints=SimpleNamespace(data=points))
        x, y = 120. + index * 4, 102. + index * 2
        return SimpleNamespace(boxes=Boxes(xyxy=np.array([[x-2, y-2, x+2, y+2]]),
                                           conf=np.array([.9]), id=None), keypoints=None)

    def track(self, image, **options):
        if self.role != 'players':
            raise AssertionError('Only the selected participant adapter may track')
        index = len(self.tracking_calls)
        self.tracking_calls.append(deepcopy(options))
        self.trace.append(('players', 'track', index))
        self.callbacks = {'on_predict_start': ['tracking'], 'on_predict_postprocess_end': ['tracking']}
        if self.predictor is None:
            self.predictor = SimpleNamespace(device='cpu', trackers=[self.tracker], _feats=None)
        return [self._result(index)]

    def predict(self, image, **options):
        invocation = len(self.calls)
        index = self.frame_index(image)
        self.calls.append(deepcopy(options))
        self.frame_indices.append(index)
        self.trace.append((self.role, 'predict', index))
        if self.role == 'players':
            raise AssertionError('Disabled crop analysis attempted a participant prediction')
        if self.role == 'ball' and invocation == self.fail_ball_at:
            raise self.fail_ball_error
        return [self._result(index)]


@dataclass(frozen=True)
class NativePipelineFixture:
    pipeline: object
    module: object
    directory: Path
    source: Path
    config: dict
    models: dict
    trace: list
    frame_count: int
    import_attempts: list


def _write_source(directory, frame_count):
    source = directory / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))
    try:
        if not writer.isOpened():
            raise RuntimeError('Synthetic native integration fixture cannot open MP4V encoder')
        yy, xx = np.indices((HEIGHT, WIDTH))
        board = (((xx // 12 + yy // 12) % 2) * 85 + 55).astype(np.uint8)
        background = np.repeat(board[:, :, None], 3, axis=2)
        background[142:202, 88:112] = [30, 40, 220]
        background[22:72, 188:212] = [220, 40, 30]
        for index in range(frame_count):
            image = background.copy()
            image[:32, :32] = index * 20
            writer.write(image)
    finally:
        writer.release()
    with (directory / 'source_frame_mapping.jsonl').open('x', encoding='utf-8') as stream:
        for index in range(frame_count):
            stream.write(json.dumps({'frame_index': index, 'timestamp_seconds': index / FPS,
                'source_pts': str(index), 'source_time_base': '1/25', 'source_frame_index': index,
                'validMeasurementTransitionFromPrevious': index > 0}) + '\n')
    return source


@contextmanager
def injected_native_pipeline(directory, *, frame_count=8, fail_ball_at=None,
                             fail_ball_error=None, existing=False, inference_identity=None):
    """Prepare real synthetic normalized input and run the real constructor.

    Caller invokes ``fixture.pipeline.run(str(fixture.source), str(directory))``.
    No pipeline run or test executes merely by importing this helper.
    """
    directory = Path(directory)
    config_path = directory / 'inference_config.yaml'
    if existing:
        if not directory.is_dir():
            raise FileNotFoundError('Existing fixture must retain its prepared source directory')
        source = directory / 'source.mp4'
        for path in (source, config_path, directory / 'source_frame_mapping.jsonl'):
            if not path.is_file():
                raise FileNotFoundError('Existing fixture input is missing: ' + path.name)
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    else:
        directory.mkdir(parents=True, exist_ok=False)
        source = _write_source(directory, frame_count)
        config = yaml.safe_load((ROOT / 'configs/production/yolo26.yaml').read_text(encoding='utf-8'))
        config['match_configuration']['match_id'] = 'synthetic_native_pipeline_only'
        config['model_system']['device'] = 'cpu'
        config['pose_estimation']['enabled'] = False
        config['player_motion']['enabled'] = False
        config['ball_detection']['tiles']['enabled'] = False
        config['camera_intervals'] = {'reviews': [{'evidence_id': 'SYNTHETIC_REVIEW_GATE_NOT_GROUND_TRUTH',
            'start_source_seconds': 0., 'end_source_seconds': 100., 'label': 'match_play',
            'viewpoint': 'elevated_endcourt_singles', 'normal_speed_verified': True,
            'provenance': 'manual_review'}]}
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    trace = []
    models = {role: InjectedModel(role, trace, fail_ball_at=fail_ball_at, fail_ball_error=fail_ball_error)
              for role in ('players', 'ball', 'court')}
    bundle = SimpleNamespace(
        player_detector=YOLO26PlayerPoseDetector('SYNTHETIC-selected-player.pt', tracker='SYNTHETIC-pinned.yaml',
            confidence=config['player_detection']['confidence'], imgsz=config['player_detection']['imgsz'],
            device='cpu', _model=models['players']),
        ball_detector=YOLO26BallDetector('SYNTHETIC-selected-ball.pt', imgsz=config['ball_detection']['imgsz'],
            high_conf=config['ball_detection']['high_conf'], low_conf=config['ball_detection']['low_conf'],
            device='cpu', tiles=config['ball_detection']['tiles'], _model=models['ball']),
        court_detector=YOLO26CourtPoseDetector('SYNTHETIC-selected-court.pt',
            imgsz=config['court_detection']['imgsz'], device='cpu', _model=models['court']),
        artifact_manifest={'status': 'SYNTHETIC_INJECTED_MODEL_BOUNDARY_NOT_QUALIFIED'})
    with no_inference_imports() as import_attempts:
        module = load_native_pipeline_source()
        from src.detection import yolo26_bundle
        with patch.object(yolo26_bundle, 'build_yolo26_bundle', return_value=bundle) as build:
            pipeline = module.Phase6Pipeline(str(config_path), inference_identity=inference_identity)
            if build.call_count != 1:
                raise AssertionError('Native constructor did not consume exactly one injected three-role bundle')
        yield NativePipelineFixture(pipeline, module, directory, source, config, models, trace,
                                    frame_count, import_attempts)
