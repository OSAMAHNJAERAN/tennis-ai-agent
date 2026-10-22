"""Bounded observations for a private, source-bound rule correction rebuild.

The generation and rule-source decoder establish provenance. This adapter
recovers only fields actually retained by the native exports: image ball
centres, confidence, state, native playback time, and current frame geometry.
It does not recover omitted tracker internals, qualify physical accuracy, or
reuse the initial court homography after a new camera anchor.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

from src.api.analysis_generation import GenerationSnapshot, resolve_generation
from src.api.result_storage import DiskSeries, ResultStorage
from src.scoring.point_timeline import candidate_event_context, exact_source_time

if TYPE_CHECKING:
    from src.scoring.point_timeline import EventContext
    from src.tracking.temporal_ball_tracker import TemporalBallPoint


_SHA = re.compile('[0-9a-f]{64}')
_STATES = {'DETECTED', 'TRACKED', 'PREDICTED', 'INTERPOLATED', 'OCCLUDED', 'MISSING'}
_OPTICAL_STATES = {'DETECTED', 'TRACKED'}


def _require(condition, message):
    if not condition:
        raise ValueError('Correction sources: ' + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _plain(value):
    # A current frame record is materialized, never a duration-long root array.
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, DiskSeries)):
        return [_plain(item) for item in value]
    return value


def _same(left, right):
    # Canonical JSON distinguishes bool/int and exact original clock tokens.
    return json.dumps(_plain(left), sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        _plain(right), sort_keys=True, separators=(',', ':'), allow_nan=False)


def _array(value, name):
    _require(isinstance(value, (list, DiskSeries)), name + ' must be an array')
    return value


def _matrix(value, name):
    """Validate a single small homography without importing numerical runtimes."""
    rows = _plain(value)
    _require(isinstance(rows, list) and len(rows) == 3 and all(
        isinstance(row, list) and len(row) == 3 and all(_number(item) for item in row) for row in rows),
        name + ' must be a finite 3 by 3 matrix')
    a, b, c = rows
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1])
                   - a[1] * (b[0] * c[2] - b[2] * c[0])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]))
    _require(math.isfinite(determinant) and abs(determinant) > 1e-12,
             name + ' is degenerate')
    return tuple(tuple(row) for row in rows)


def _same_mapping(candidate, reference):
    """Tight projective equivalence; an arbitrary matrix scale is irrelevant.

    Normalize by maximum absolute entry, with the reference's dominant pivot
    choosing a shared sign. This accepts positive/negative scalar multiples
    without converting a real camera motion into a refinement observation.
    The 1e-12 comparison is a representation tolerance, not measured accuracy.
    """
    if candidate is None or reference is None:
        return False
    left = tuple(float(item) for row in _matrix(candidate, 'candidate refinement mapping') for item in row)
    right = tuple(float(item) for row in _matrix(reference, 'event refinement mapping') for item in row)
    pivot = max(range(9), key=lambda index: abs(right[index]))
    left_scale = math.copysign(max(abs(item) for item in left), left[pivot])
    right_scale = right[pivot]
    return all(abs(a / left_scale - b / right_scale) <= 1e-12 for a, b in zip(left, right, strict=True))


@dataclass(frozen=True)
class EventGeometry:
    """Current-frame observed mapping, separate from independent qualification."""

    homography_matrix: object | None
    usable: bool
    reason: str


class _FrozenSourceFrames(Sequence):
    """Copy the validated view's current rows into this adapter's owned store."""

    def __init__(self, rows, original):
        self._rows = rows
        self.analysis_id = original.analysis_id
        self.analysis_revision = original.analysis_revision
        self.source_sha256 = original.source_sha256
        self.artifact_sha256 = original.artifact_sha256

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Frozen source frames require an indexed read')
        row = self._rows[index]
        return {0: row[0], 1: row[1]}


class _BallPoints(Sequence['TemporalBallPoint']):
    def __init__(self, owner, indices=None, event=None, context_usable=True, radius=3):
        self.owner = owner
        self.indices = range(owner.frame_count) if indices is None else indices
        self.event, self.context_usable, self.radius = event, context_usable, radius

    def __len__(self):
        self.owner._require_open()
        return len(self.indices)

    def __getitem__(self, index):
        self.owner._require_open()
        if isinstance(index, slice):
            return _BallPoints(self.owner, self.indices[index], self.event, self.context_usable, self.radius)
        if type(index) is not int:
            raise TypeError('Ball frame index must be an integer')
        frame = self.indices[index]
        point = self.owner._ball_point(frame)
        if self.event is not None and (point.state.value not in _OPTICAL_STATES
                                      or point.confidence is None or point.confidence <= 0
                                      or not self.context_usable or abs(frame - self.event) > self.radius
                                      or not self.owner._continuous(self.event, frame)):
            # Keep the actual time and global frame index. A contact fitter may
            # index/slice this view but cannot fit predicted/interpolated points
            # or see observations across a cut. Raw .ball_points keeps them.
            from src.tracking.temporal_ball_tracker import BallState
            return replace(point, x_px=None, y_px=None, confidence=None, state=BallState.MISSING,
                           source='correction_continuity_unavailable')
        return point


class CorrectionSources:
    """Read-only views whose lifetime is owned by ``correction_sources``."""

    def __init__(self, snapshot, source_frames, detections, camera, court):
        self.snapshot, self.source_frames = snapshot, source_frames
        self.frame_count = snapshot.frames
        self.fps = detections['metadata']['fps']
        self._detections, self._camera, self._court = detections, camera, court
        self._open = True
        self.ball_points = _BallPoints(self)

    def _require_open(self):
        if not self._open:
            raise RuntimeError('Correction sources are closed')

    def _index(self, frame):
        self._require_open()
        if type(frame) is not int or not 0 <= frame < self.frame_count:
            raise ValueError('Correction sources require an existing global frame index')

    def _interval(self, frame):
        self._require_open()
        return self.source_frames[frame][0]

    def _ball_point(self, frame):
        # Import the actual native type only when a caller requests a point.
        # Its current defining module imports NumPy/pandas; model packages are
        # never needed. JSON decoding preserves the native unrounded floats.
        from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint
        interval = self._interval(frame)
        ball = self._detections['frames'][frame]['ball']
        position = ball['position_px']
        return TemporalBallPoint(frame_index=frame, timestamp_seconds=interval['timestamp_seconds'],
            x_px=None if position is None else position[0], y_px=None if position is None else position[1],
            confidence=ball['confidence'], state=BallState(ball['state']), source='sealed_detections')

    def _context_usable(self, frame, context):
        self._index(frame)
        from src.scoring.point_timeline import EventContext
        if not isinstance(context, EventContext) or context.frame_index != frame:
            return False
        current = candidate_event_context(SimpleNamespace(frame_index=frame), self.source_frames,
            lambda *_: context.binding, analysis_id=self.snapshot.analysis_id,
            source_fingerprint=self.source_frames.source_sha256)
        if current != context or not current.binding_current:
            return False
        binding = current.binding
        interval = self._interval(frame)
        return bool(binding.analysis_id == self.snapshot.analysis_id
                    and binding.source_fingerprint == self.source_frames.source_sha256
                    and binding.anchor_frame_index == interval.get('anchor_frame_index')
                    and binding.calibration_revision and binding.geometry_qualification_ref
                    and interval.get('reanchor_required') is not True)

    def _continuous(self, event, frame):
        current, target = self._interval(event), self._interval(frame)
        event_matrix = self._camera['frames'][event].get('image_to_court') if self._camera['enabled'] else None
        if (self._camera['enabled'] is not True
                or self._camera['frames'][event].get('is_valid') is not True
                or self._camera['frames'][frame].get('is_valid') is not True
                or current.get('analysis_allowed') is not True or target.get('analysis_allowed') is not True
                or current.get('segment_id') != target.get('segment_id')
                or current.get('anchor_frame_index') != target.get('anchor_frame_index')
                or not _same_mapping(self._camera['frames'][frame].get('image_to_court'), event_matrix)):
            return False
        for index in range(min(event, frame) + 1, max(event, frame) + 1):
            interval = self._interval(index)
            if (interval.get('measurement_transition_allowed') is not True
                    or interval.get('analysis_allowed') is not True
                    or interval.get('local_reset') is True or interval.get('discontinuity_reasons')
                    or self._camera['frames'][index].get('is_valid') is not True
                    or not _same_mapping(self._camera['frames'][index].get('image_to_court'), event_matrix)
                    or interval.get('segment_id') != current.get('segment_id')
                    or interval.get('anchor_frame_index') != current.get('anchor_frame_index')):
                return False
        return True

    def event_ball_points(self, frame_index, context, *, window_radius=3):
        """Lazy optical trajectory masked at continuity and window edges.

        Supply the actual contact refiner radius when it differs from three.
        Slices remain range-backed; even ``[:]`` does not allocate all frames.
        The existing refiner fits raw image coordinates. Until an independently
        justified stabilization exists, camera mappings must remain projectively
        equivalent throughout a contributing interval; a moving mapping keeps
        only the event's observed point for its raw discrete fallback. This
        never registers airborne points by a court-plane warp.
        """
        self._index(frame_index)
        if type(window_radius) is not int or not 0 <= window_radius <= 128:
            raise ValueError('Contact refinement radius must be an integer from 0 to 128')
        return _BallPoints(self, event=frame_index,
                           context_usable=self._context_usable(frame_index, context), radius=window_radius)

    def event_geometry(self, frame_index, context):
        """Resolve the observed homography at this event's own camera anchor.

        Reviewed geometry qualification is supplied by the typed current event
        binding. Native camera files do not themselves establish independent
        accuracy or carry that external review's calibration revision string.
        """
        self._index(frame_index)
        if not self._context_usable(frame_index, context):
            return EventGeometry(None, False, 'EVENT_CONTEXT_NOT_CURRENT')
        if self._camera['enabled'] is not True:
            return EventGeometry(None, False, 'CURRENT_FRAME_REGISTRATION_UNAVAILABLE')
        row = self._camera['frames'][frame_index]
        if row.get('is_valid') is not True or row.get('image_to_court') is None:
            return EventGeometry(None, False, 'CURRENT_FRAME_REGISTRATION_UNAVAILABLE')
        if ('calibration_revision' in row
                and row['calibration_revision'] != context.binding.calibration_revision):
            return EventGeometry(None, False, 'CALIBRATION_REVISION_DIFFERS')
        ball = self._detections['frames'][frame_index]['ball']
        if (ball['position_px'] is None or ball['state'] not in _OPTICAL_STATES
                or ball['confidence'] is None or ball['confidence'] <= 0):
            return EventGeometry(None, False, 'BOUNCE_OBSERVATION_UNAVAILABLE')
        import numpy as np
        matrix = np.asarray(_matrix(row['image_to_court'], 'current frame homography'), dtype=np.float64)
        matrix.setflags(write=False)
        return EventGeometry(matrix, True, 'CURRENT_REVIEW_BOUND_FRAME_REGISTRATION')


def _source_interval(source_frames, frame, analysis_id):
    row = source_frames[frame]
    _require(isinstance(row, Mapping) and set(row) == {0, 1} and isinstance(row[0], Mapping),
             'source interval/identity view is malformed')
    interval = row[0]
    _require(interval.get('analysis_id') == analysis_id
             and type(interval.get('frame_index')) is int and interval['frame_index'] == frame
             and _number(interval.get('timestamp_seconds')) and interval['timestamp_seconds'] >= 0
             and type(interval.get('analysis_allowed')) is bool
             and type(interval.get('measurement_transition_allowed')) is bool,
             'source interval identity/frame/clock differs')
    pts, base, source_index = (interval.get(name) for name in
                              ('source_pts', 'source_time_base', 'source_frame_index'))
    time = None
    if pts is not None or base is not None or source_index is not None:
        try:
            _require(isinstance(pts, str) and isinstance(base, str)
                     and type(source_index) is int and source_index >= 0, 'partial source clock')
            time = exact_source_time(pts, base)
            _require(math.isfinite(float(time)), 'source clock is not finite')
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            raise ValueError('Correction sources: invalid exact source clock') from error
    if interval['analysis_allowed']:
        _require(time is not None and isinstance(interval.get('segment_id'), str)
                 and bool(interval['segment_id']) and type(interval.get('anchor_frame_index')) is int
                 and 0 <= interval['anchor_frame_index'] <= frame,
                 'allowed interval lacks exact source/camera anchor')
    if 'source_timestamp_seconds' in interval:
        actual = interval['source_timestamp_seconds']
        _require(actual is None if time is None else _number(actual) and actual == float(time),
                 'derived source clock differs from its exact PTS tokens')
    return interval, time, row[1]


def _validate(snapshot, source_frames, detections, camera, court, frozen):
    _require(isinstance(detections, dict) and detections.get('schema_version') == '1.0'
             and isinstance(detections.get('metadata'), dict), 'unsupported detections envelope')
    metadata = detections['metadata']
    # Current native exports bind these through the sealed file inventory and
    # per-frame intervals. Future explicit envelope fields cannot contradict
    # that identity while otherwise retaining the established native shape.
    for document in (detections, metadata, camera, court):
        _require(isinstance(document, dict), 'native artifact envelope must be an object')
        if 'analysis_id' in document:
            _require(document['analysis_id'] == snapshot.analysis_id, 'native analysis identity differs')
        if 'analysis_revision' in document:
            _require(type(document['analysis_revision']) is int
                     and document['analysis_revision'] == snapshot.analysis_revision,
                     'native analysis revision differs')
        if 'source_sha256' in document:
            _require(document['source_sha256'] == source_frames.source_sha256, 'native source digest differs')
    _require(type(metadata.get('frames')) is int and metadata['frames'] == snapshot.frames
             and _number(metadata.get('fps')) and metadata['fps'] > 0
             and all(type(metadata.get(name)) is int and metadata[name] > 0 for name in ('width', 'height')),
             'detection metadata dimensions/frame count differ')
    rows = _array(detections.get('frames'), 'detection frames')
    _require(len(rows) == snapshot.frames, 'detection frame count differs')
    _require(isinstance(camera, dict) and camera.get('schema_version') in ('1.0', '1.1')
             and type(camera.get('enabled')) is bool, 'unsupported camera envelope')
    cameras = _array(camera.get('frames'), 'camera frames')
    _require(len(cameras) == (snapshot.frames if camera['enabled'] else 0), 'camera frame count differs')
    _require(isinstance(court, dict) and court.get('schema_version') == '2.0'
             and type(court.get('is_valid')) is bool, 'unsupported court envelope')
    if court['is_valid']:
        _require(court.get('rejection_reason') is None, 'valid initial court carries a rejection')
        _matrix(court.get('homography_matrix'), 'initial court homography')
    else:
        _require(court.get('homography_matrix') is None, 'invalid initial court exposes a homography')
    previous, previous_time = None, None
    for frame in range(snapshot.frames):
        interval, exact, identity = _source_interval(source_frames, frame, snapshot.analysis_id)
        if previous is not None:
            _require(interval['timestamp_seconds'] > previous['timestamp_seconds'],
                     'native playback clock is not increasing')
        if interval['measurement_transition_allowed']:
            _require(previous is not None and previous.get('analysis_allowed') is True
                     and interval.get('analysis_allowed') is True and exact is not None
                     and previous_time is not None and exact > previous_time
                     and interval['segment_id'] == previous.get('segment_id')
                     and interval['anchor_frame_index'] == previous.get('anchor_frame_index')
                     and interval.get('local_reset') is not True and not interval.get('discontinuity_reasons'),
                     'allowed source transition crosses an unsupported clock/camera break')
        row = rows[frame]
        _require(isinstance(row, dict) and type(row.get('frame_index')) is int and row['frame_index'] == frame
                 and type(row.get('court_registration_valid')) is bool
                 and _same(row.get('camera_interval'), interval), 'detection/source frame join differs')
        ball = row.get('ball')
        _require(isinstance(ball, dict) and set(ball) == {'position_px', 'confidence', 'state'}
                 and isinstance(ball.get('state'), str) and ball['state'] in _STATES,
                 'ball observation fields/state differ')
        position, confidence = ball['position_px'], ball['confidence']
        _require(confidence is None or _number(confidence) and 0 <= confidence <= 1,
                 'ball confidence is invalid')
        if position is not None:
            position = _plain(position)
            _require(isinstance(position, list) and len(position) == 2 and all(_number(item) for item in position),
                     'ball position must contain two finite unrounded coordinates')
            _require(ball['state'] != 'MISSING', 'missing ball cannot contain a measured position')
        if camera['enabled']:
            registered = cameras[frame]
            _require(isinstance(registered, dict) and type(registered.get('frame_index')) is int
                     and registered['frame_index'] == frame and type(registered.get('is_valid')) is bool
                     and _same(registered.get('camera_interval'), interval)
                     and registered['is_valid'] == row['court_registration_valid'],
                     'camera/source/detection frame join differs')
            if registered['is_valid']:
                _require(interval['analysis_allowed'] and registered.get('rejection_reason') is None,
                         'valid camera registration belongs to an unavailable interval')
                _matrix(registered.get('image_to_court'), 'current frame homography')
                if registered.get('anchor_to_frame_px') is not None:
                    _matrix(registered['anchor_to_frame_px'], 'current anchor transform')
            else:
                _require(registered.get('image_to_court') is None, 'invalid frame exposes a homography')
        # Native camera-disabled runs can retain a valid static initial fit in
        # this diagnostic detection field. Accept that historical wire format;
        # event_geometry still withholds without observed frame registration.
        frozen.append([_plain(interval), _plain(identity)])
        previous, previous_time = interval, exact


@contextmanager
def correction_sources(snapshot: GenerationSnapshot, directory, *, source_sha256, source_frames):
    """Open sealed native observations alongside a validated rule-source view.

    ``source_frames`` must be the live view returned by the trusted
    ``open_rule_replay_source`` decoder. Its analysis, revision, source digest
    and replay-artifact digest must all match this committed generation. The
    caller supplies an existing private scratch directory. Inputs are never
    modified; successful exit removes only this parser's owned SQLite files.
    """
    _require(isinstance(snapshot, GenerationSnapshot), 'a committed generation snapshot is required')
    committed = resolve_generation(snapshot._root.parent, snapshot.generation)
    _require(committed == snapshot, 'snapshot is not the committed manifest')
    _require(isinstance(source_sha256, str) and _SHA.fullmatch(source_sha256)
             and isinstance(source_frames, Sequence) and len(source_frames) == snapshot.frames
             and getattr(source_frames, 'analysis_id', None) == snapshot.analysis_id
             and type(getattr(source_frames, 'analysis_revision', None)) is int
             and source_frames.analysis_revision == snapshot.analysis_revision
             and getattr(source_frames, 'source_sha256', None) == source_sha256
             and getattr(source_frames, 'artifact_sha256', None) == snapshot.file_sha256('native_rule_replay.json'),
             'validated replay view belongs to another generation/revision/source')
    paths = {}
    for name in ('detections.json', 'camera_registration.json', 'court_geometry.json', 'native_rule_replay.json'):
        paths[name] = snapshot.resolve_file(name)
        _require(paths[name] is not None, 'mandatory sealed artifact is absent: ' + name)
    directory = Path(directory)
    _require(directory.is_dir(), 'private scratch directory must already exist')
    with ResultStorage(directory) as storage:
        def finite(token):
            raise ValueError('Correction sources require finite JSON: ' + token)
        parsed = {name: storage.read_json(paths[name], parse_constant=finite)
                  for name in ('detections.json', 'camera_registration.json', 'court_geometry.json')}
        for name, path in paths.items():
            _require(snapshot.resolve_file(name) == path, 'sealed source changed during parsing')
        frozen = storage.series()
        _validate(snapshot, source_frames, parsed['detections.json'], parsed['camera_registration.json'],
                  parsed['court_geometry.json'], frozen)
        frozen_frames = _FrozenSourceFrames(frozen, source_frames)
        sources = CorrectionSources(snapshot, frozen_frames, parsed['detections.json'],
                                    parsed['camera_registration.json'], parsed['court_geometry.json'])
        try:
            yield sources
        finally:
            sources._open = False
