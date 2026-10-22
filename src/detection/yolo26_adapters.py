"""The selected three-checkpoint YOLO26 runtime, with no legacy fallback.

Pinned to the locally verified Ultralytics 8.4.36 API. In this version
``end2end=True`` selects the one-to-one head; ``nms=False`` is also explicit.
Ultralytics Results are already in the input image's pixel coordinates.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.tracking.temporal_ball_tracker import BallObservation
from src.tracking.player_motion_tracking import decode_pose
from src.utils.bbox_utils import BBox


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _torch_device(device):
    """Translate an Ultralytics integer GPU ID for PyTorch module placement."""
    if isinstance(device, (bool, np.bool_)):
        raise ValueError('A GPU device ID must be a nonnegative integer')
    if isinstance(device, (int, np.integer)):
        if device < 0:
            raise ValueError('A GPU device ID must be a nonnegative integer')
        return f'cuda:{int(device)}'
    if isinstance(device, str) and device.isdecimal():
        return f'cuda:{int(device)}'
    return device


def _load(model_path, task, scale, keypoint_count=None, device="auto"):
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Selected YOLO26 {task} checkpoint is missing: {path}. No fallback is permitted.")
    model = YOLO(str(path), task=task)
    config = model.model.yaml
    architecture = Path(str(config.get("yaml_file", ""))).name.lower()
    if "yolo26" not in architecture or config.get("scale") != scale or not config.get("end2end"):
        raise ValueError(f"Checkpoint must be YOLO26{scale} {task}, got {architecture!r}")
    if model.task != task:
        raise ValueError(f"Expected {task} checkpoint, got {model.task}")
    if keypoint_count is not None and list(config.get("kpt_shape", [])) != [keypoint_count, 3]:
        raise ValueError(f"Expected kpt_shape [{keypoint_count}, 3]")
    if device != "auto":
        model.to(_torch_device(device))
    return model


def _inference_args(imgsz, confidence, device):
    options = dict(imgsz=imgsz, conf=confidence, verbose=False, end2end=True, nms=False)
    if device != "auto":
        options["device"] = device
    return options


def _resolved_crop_device(value):
    """Accept an admitted device, never automatic accelerator selection."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Crop inference needs an explicit CPU or indexed CUDA device')
    if isinstance(value, (int, np.integer)) and value >= 0:
        return f'cuda:{int(value)}'
    if type(value) is str:
        if value == 'cpu':
            return value
        index = value[5:] if value.startswith('cuda:') else value
        if index.isascii() and index.isdecimal():
            return f'cuda:{int(index)}'
    raise ValueError('Crop inference needs an explicit CPU or indexed CUDA device')


def _native_crop_options(overrides, device):
    if type(overrides) is not dict:
        raise ValueError('Selected checkpoint crop overrides are unavailable')
    imgsz = overrides.get('imgsz', 640)
    dimensions = [imgsz] if type(imgsz) is int else imgsz
    if (type(dimensions) not in (list, tuple) or not 1 <= len(dimensions) <= 2
            or any(type(value) is not int or not 32 <= value <= 4096 for value in dimensions)):
        raise ValueError('Selected checkpoint crop image size is invalid')
    single_cls = overrides.get('single_cls', False)
    if type(single_cls) is not bool:
        raise ValueError('Selected checkpoint crop class configuration is invalid')
    return {'imgsz': imgsz if type(imgsz) is int else list(imgsz), 'conf': .25,
            'verbose': False, 'device': device, 'end2end': True, 'nms': False,
            'batch': 1, 'rect': True, 'half': False, 'max_det': 300, 'iou': .7,
            'augment': False, 'agnostic_nms': False, 'classes': None,
            'single_cls': single_cls, 'mode': 'predict', 'save': False}


class PredictionOnlyCropProvider:
    """Reuse one selected model after tracking, retaining swallowed failures."""

    def __init__(self, detector, options):
        self._detector, self.model = detector, detector.model
        self._options = deepcopy(options)
        self.failure = None
        self.prediction_calls = self.completed_prediction_calls = 0

    def assert_success(self):
        if self.failure is not None:
            raise RuntimeError('Selected-player crop inference failed') from self.failure

    def _check(self):
        self.assert_success()
        if self._detector.model is not self.model:
            raise RuntimeError('Selected player model changed after tracking')
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None:
            if hasattr(predictor, 'trackers') or getattr(predictor, '_hook', None) is not None:
                raise RuntimeError('Tracking state reappeared in prediction-only crops')
            device = getattr(predictor, 'device', None)
            if device is None or _resolved_crop_device(str(device)) != self._options['device']:
                raise RuntimeError('Crop predictor differs from the admitted device')

    def predict(self, image, **_arguments):
        # The legacy extractor supplies its own device/defaults. Only the
        # admitted immutable options define this selected-checkpoint operation.
        try:
            self._check()
            self.prediction_calls += 1
            result = self.model.predict(image, **deepcopy(self._options))
            self._check()
            self.completed_prediction_calls += 1
            return result
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def record(self):
        return {'schema_version': 1, 'role': 'players',
                'selected_model_path': self._detector.model_path,
                'tracking_policy': 'DISABLED_BOUND_SOURCE_CROPS',
                'configured_device': str(self._detector.device),
                'resolved_device': self._options['device'],
                'input_spec': deepcopy(self._options),
                'prediction_calls': self.prediction_calls,
                'completed_prediction_calls': self.completed_prediction_calls,
                'failed': self.failure is not None}


@dataclass
class PoseBBox(BBox):
    """Local track ID and anatomical evidence; neither establishes match identity."""

    keypoints_px: tuple = field(default_factory=tuple)
    ground_point_px: tuple | None = None
    ground_point_source: str = "BBOX_BOTTOM_CENTER_ESTIMATE"
    ground_point_quality: str = "GROUND_CONTACT_UNVERIFIED"
    ground_point_confidence: float | None = None


class YOLO26PlayerPoseDetector:
    def __init__(self, model_path, *, confidence=.4, imgsz=960, tracker,
                 min_pose_confidence=.35, device="auto", _model=None):
        if not 0 <= confidence <= 1 or not 0 <= min_pose_confidence <= 1:
            raise ValueError("Player/keypoint confidence must be in [0, 1]")
        self.model = _model or _load(model_path, "pose", "s", 17, device)
        self.model_path, self.tracker = str(model_path), str(tracker)
        self.confidence, self.imgsz = confidence, imgsz
        self.min_pose_confidence, self.device = min_pose_confidence, device
        overrides = getattr(self.model, 'overrides', None)
        self._checkpoint_crop_overrides = ({key: deepcopy(overrides[key])
                                           for key in ('imgsz', 'single_cls') if key in overrides}
                                          if type(overrides) is dict else None)
        self._tracking_finished = False

    def _decode(self, results):
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = _array(result.boxes.xyxy)
            scores = _array(result.boxes.conf)
            ids = _array(result.boxes.id) if result.boxes.id is not None else None
            keypoints = _array(result.keypoints.data) if result.keypoints is not None else None
            for index, (coords, score) in enumerate(zip(boxes, scores)):
                if not np.isfinite(coords).all() or not np.isfinite(score) or coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                raw_id = ids[index] if ids is not None else None
                track_id = (int(raw_id) if raw_id is not None and np.isfinite(raw_id)
                            and raw_id >= 0 and int(raw_id) == raw_id else None)
                box = PoseBBox(*map(float, coords), confidence=float(score), class_id=0, track_id=track_id)
                if keypoints is not None and keypoints[index].shape == (17, 3):
                    points = keypoints[index]
                    box.keypoints_px = tuple(tuple(map(float, point)) for point in points)
                    ankles = points[[15, 16]]
                    valid = np.isfinite(ankles).all() and (ankles[:, 2] >= self.min_pose_confidence).all()
                    valid = valid and (ankles[:, :2] >= 0).all() and (ankles[:, :2] != 0).any()
                    if valid:
                        box.ground_point_px = tuple(map(float, ankles[:, :2].mean(axis=0)))
                        box.ground_point_source = "POSE_ANKLE_MIDPOINT_ESTIMATE"
                        box.ground_point_quality = "ANKLES_VISIBLE_GROUND_CONTACT_UNVERIFIED"
                        box.ground_point_confidence = float(ankles[:, 2].min())
                detections.append(box)
        return detections

    def detect(self, frame):
        if self._tracking_finished:
            raise RuntimeError('Full-frame player inference is closed after tracking finishes')
        return self._decode(self.model.predict(frame, classes=[0],
                            **_inference_args(self.imgsz, self.confidence, self.device)))

    def detect_and_track(self, frame, persist=True):
        if self._tracking_finished:
            raise RuntimeError('Full-frame player inference is closed after tracking finishes')
        return self._decode(self.model.track(frame, classes=[0], persist=persist, tracker=self.tracker,
                            **_inference_args(self.imgsz, self.confidence, self.device)))

    def observe(self, frame, boxes):
        """Reuse the selected boxes' pose evidence; do not run another network."""
        return {identity: decode_pose(np.asarray(box.keypoints_px), (0, 0), self.min_pose_confidence)
                if box is not None and len(getattr(box, "keypoints_px", ())) == 17 else None
                for identity, box in boxes.items()}

    def reset(self):
        """One detector per job; call at a discontinuity before reconnecting identity."""
        predictor = getattr(self.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", ()):
            tracker.reset()

    def finish_tracking_for_crops(self, *, admitted_device, expected_device=None):
        """Close the player pass and return prediction-only crops on this model.

        Call only after complete player/role evidence is consumed. A fresh
        auto-configured resume must supply the previously sealed device as
        ``expected_device`` as well as the currently admitted device. This
        method admits no hardware, loads no model and never guesses a device.
        Failed disposal still closes full-frame inference permanently.
        """
        if self._tracking_finished:
            raise RuntimeError('Player tracking has already finished')
        device = _resolved_crop_device(admitted_device)
        if expected_device is not None and _resolved_crop_device(expected_device) != device:
            raise ValueError('Saved crop device differs from the admitted device')
        if self.device != 'auto' and _resolved_crop_device(str(_torch_device(self.device))) != device:
            raise ValueError('Configured crop device differs from the admitted device')
        predictor = getattr(self.model, 'predictor', None)
        if predictor is not None:
            observed = getattr(predictor, 'device', None)
            if observed is None or _resolved_crop_device(str(observed)) != device:
                raise ValueError('Player predictor differs from the admitted device')
        elif self.device == 'auto' and expected_device is None:
            raise ValueError('A fresh automatic-device resume needs its saved crop device')
        options = _native_crop_options(self._checkpoint_crop_overrides, device)
        reset_callbacks = getattr(self.model, 'reset_callbacks', None)
        if not callable(reset_callbacks):
            raise ValueError('Selected model callback reset is unavailable')

        self._tracking_finished = True
        failures = []
        if predictor is not None:
            hook = getattr(predictor, '_hook', None)
            if hook is not None:
                try:
                    hook.remove()
                    del predictor._hook
                except BaseException as error:
                    failures.append(error)
            if hasattr(predictor, '_feats'):
                predictor._feats = None
        try:
            reset_callbacks()
        except BaseException as error:
            failures.append(error)
        finally:
            self.model.predictor = None
        if failures:
            error = failures[0]
            for failure in failures[1:]:
                error.add_note('Player predictor disposal also failed: ' + str(failure))
            raise error
        return PredictionOnlyCropProvider(self, options)


def tile_regions(width, height, tile_size, overlap):
    """Half-open native-pixel rectangles, including edge tiles without resizing."""
    if tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be positive and overlap in [0, 1)")
    stride = max(1, int(round(tile_size * (1 - overlap))))
    def offsets(length):
        last = max(0, length - tile_size)
        return sorted(set([*range(0, last + 1, stride), last]))
    return [(x, y, min(width, x + tile_size), min(height, y + tile_size))
            for y in offsets(height) for x in offsets(width)]


def _iou(first, second):
    intersection = max(0., min(first.x2, second.x2) - max(first.x1, second.x1)) * max(0., min(first.y2, second.y2) - max(first.y1, second.y1))
    area = (first.x2 - first.x1) * (first.y2 - first.y1) + (second.x2 - second.x1) * (second.y2 - second.y1) - intersection
    return intersection / area if area > 0 else 0.


class YOLO26BallDetector:
    def __init__(self, model_path, *, imgsz=1280, high_conf=.25, low_conf=.05,
                 class_id=0, device="auto", tiles=None, _model=None):
        if not 0 <= low_conf <= high_conf <= 1:
            raise ValueError("Ball confidence must satisfy 0 <= low <= high <= 1")
        self.model = _model or _load(model_path, "detect", "m", device=device)
        self.model_path, self.imgsz, self.device = str(model_path), imgsz, device
        self.high_conf, self.low_conf, self.class_id = high_conf, low_conf, class_id
        self.tiles = dict(tiles or {})

    def extract_candidates(self, frame, imgsz=None):
        height, width = frame.shape[:2]
        regions = [(0, 0, width, height)]
        if self.tiles.get("enabled", False):
            regions += tile_regions(width, height, self.tiles["size_px"], self.tiles["overlap"])
        regions = list(dict.fromkeys(regions))
        proposals = []
        for left, top, right, bottom in regions:
            result = self.model.predict(frame[top:bottom, left:right], classes=[self.class_id],
                                       **_inference_args(imgsz or self.imgsz, self.low_conf, self.device))[0]
            if result.boxes is None:
                continue
            for xyxy, score in zip(_array(result.boxes.xyxy), _array(result.boxes.conf)):
                if not np.isfinite(xyxy).all() or not np.isfinite(score):
                    continue
                # Results have already undone letterbox within the tile. Add its
                # origin exactly once, then clip to the source image's bounds.
                x1, y1, x2, y2 = xyxy + np.array([left, top, left, top])
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                box = BBox(float(x1), float(y1), float(x2), float(y2), confidence=float(score), class_id=self.class_id)
                proposals.append(BallObservation((x1 + x2) / 2, (y1 + y2) / 2, float(score), box))
        accepted = []
        for proposal in sorted(proposals, key=lambda item: item.confidence, reverse=True):
            if any(_iou(proposal.bbox, other.bbox) >= self.tiles.get("dedup_iou", .5)
                   or np.hypot(proposal.x_px - other.x_px, proposal.y_px - other.y_px) <= self.tiles.get("dedup_center_px", 2.)
                   for other in accepted):
                continue
            accepted.append(proposal)
        return accepted

    def detect_best(self, frame, imgsz=None, conf=None):
        threshold = self.high_conf if conf is None else conf
        candidates = [candidate for candidate in self.extract_candidates(frame, imgsz) if candidate.confidence >= threshold]
        return max(candidates, key=lambda item: item.confidence).bbox if candidates else None


class YOLO26CourtPoseDetector:
    def __init__(self, model_path, *, imgsz=1280, confidence=.4, keypoint_confidence=.5,
                 device="auto", _model=None):
        self.model = _model or _load(model_path, "pose", "s", 14, device)
        self.model_path, self.imgsz, self.device = str(model_path), imgsz, device
        self.confidence, self.keypoint_confidence = confidence, keypoint_confidence
        self.last_observation: dict[str, Any] = {}

    def predict(self, frame):
        result = self.model.predict(frame, classes=[0],
                                   **_inference_args(self.imgsz, self.confidence, self.device))[0]
        points = np.full((14, 2), np.nan, dtype=float)
        scores = np.zeros(14, dtype=float)
        if result.boxes is not None and result.keypoints is not None and len(result.boxes):
            index = int(np.argmax(_array(result.boxes.conf)))
            data = _array(result.keypoints.data)[index]
            if data.shape != (14, 3):
                raise ValueError("Court checkpoint returned an incompatible landmark schema")
            scores = data[:, 2]
            valid = np.isfinite(data).all(axis=1) & (scores >= self.keypoint_confidence)
            points[valid] = data[valid, :2]
        self.last_observation = {"keypoint_confidences": scores.tolist(),
                                 "visible_keypoints": int(np.isfinite(points).all(axis=1).sum()),
                                 "schema": "tennis_court_14_v1", "source": self.model_path,
                                 "independent_accuracy_validated": False}
        return points

    def predict_batch(self, frames):
        return [self.predict(frame) for frame in frames]
