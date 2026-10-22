"""Fresh source-aligned private video from current corrected native evidence.

This CPU render is not inference or publication. Ground measurements and
calibrated predictions remain unavailable until their independent gates pass.
"""
from contextlib import closing, contextmanager, ExitStack
from dataclasses import dataclass
from fractions import Fraction
from itertools import zip_longest
import json
import math
from pathlib import Path
import tempfile
from types import MappingProxyType

import cv2
import numpy as np

from src.api.worker import binary, execute
from src.scoring.corrected_exports import (
    CorrectedExportResult, EXPORT_FILES, MANIFEST_NAME as EXPORT_MANIFEST,
    _admission, _object, _same, _physical_events,
)
from src.scoring.corrected_rule_rebuild import _canonical, _digest, _scratch
from src.scoring.corrected_shot_sources import _box, _ephemeral_storage as _parser
from src.utils.landing_zone_sequence import LandingZoneSequence
from src.utils.point_timeline_artifact import artifact_sha256
from src.visualization import landing_zones
from src.visualization.mini_court import MiniCourt
from src.visualization.video_annotator import VideoAnnotator


MANIFEST_NAME = 'corrected_render_manifest.json'
VIDEO_FILES = frozenset(('annotated.mp4', 'annotated_web.mp4'))


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected rendering: ' + message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _guard_exports(rebuilt, shots, forecast, exports, scratch):
    directory, command, binding = _admission(rebuilt, shots, forecast, scratch)
    _require(isinstance(exports, CorrectedExportResult), 'native sealed export result is required')
    stage = _scratch(exports.directory, rebuilt.snapshot.files_directory.parent.parent.resolve())
    names = EXPORT_FILES | {EXPORT_MANIFEST}
    _require(stage.parent == directory and set(exports.files) == names
             and {path.name for path in stage.iterdir()} == names
             and all(exports.files[name] == stage/name for name in names),
             'export inventory or file path differs from its guarded stage')
    for name in names:
        path = stage/name
        _require(path.is_file() and not path.is_symlink()
                 and not getattr(path, 'is_junction', lambda: False)(), 'export artifact path is invalid')
    manifest_path = stage/EXPORT_MANIFEST
    _require(manifest_path.stat().st_size <= 64*1024
             and type(exports.manifest_sha256) is str
             and artifact_sha256(manifest_path) == exports.manifest_sha256,
             'export manifest bytes changed after its native seal')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected = {**binding, 'schema_version': '1.0', 'scoring_revision': 0,
        'status': 'PRIVATE_PENDING_RENDER_AND_PUBLICATION', 'request_sha256': _digest(command),
        'human_review_dependent': True, 'independent_accuracy_validated': False, 'published': False}
    _require(type(manifest) is dict and set(manifest) == set(expected) | {'artifacts'},
             'export manifest fields differ')
    _same({key: manifest[key] for key in expected}, expected, 'export manifest source binding differs')
    artifacts = manifest['artifacts']
    _require(type(artifacts) is dict and set(artifacts) == EXPORT_FILES, 'export artifact inventory differs')
    for name, record in artifacts.items():
        _require(type(record) is dict and set(record) == {'bytes', 'sha256'}
                 and type(record['bytes']) is int and record['bytes'] >= 0
                 and record['bytes'] == (stage/name).stat().st_size
                 and record['sha256'] == artifact_sha256(stage/name), 'export artifact bytes/hash changed: '+name)
    _require(artifacts['source.mp4']['sha256'] == binding['source_sha256'], 'export source bytes differ')
    # The native physical event owner is intentionally writable for earlier
    # relinking. It cannot diverge from the already sealed presentation now.
    generation = shots.event_owner.events_generation
    with _parser(directory) as storage:
        document = storage.read_json(stage/'match_events.json')
        _same({key: document[key] for key in ('analysis_id', 'analysis_revision', 'events_count')},
              {'analysis_id': binding['analysis_id'], 'analysis_revision': binding['analysis_revision'],
               'events_count': len(shots.events)}, 'live physical event binding differs from export')
        sentinel = object()
        with _reader(document['events']) as saved, closing(_physical_events(shots.events)) as actual:
            for left, right in zip_longest(actual, saved, fillvalue=sentinel):
                _require(left is not sentinel and right is not sentinel,
                         'live physical event count differs from sealed export')
                _same(left, right, 'live physical event differs from sealed export')
        _require(shots.event_owner.events_generation == generation,
                 'physical events changed during sealed export comparison')
    return directory, binding, stage, generation


@contextmanager
def _reader(values):
    """Small parsed arrays are lists; owned large arrays have closeable readers."""
    iterator = iter(values)
    primary = None
    try:
        yield iterator
    except BaseException as error:
        primary = error
        raise
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            try:
                close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note('Render reader cleanup also failed: '+str(cleanup))


@contextmanager
def _video_handle(handle):
    primary = None
    try:
        yield handle
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            handle.release()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note('Video handle release also failed: '+str(cleanup))


def full_video_check(path, *, frames, fps, width, height):
    """Full decoder count rather than a container's advertised duration."""
    document = json.loads(execute([binary('ffprobe'), '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=nb_read_frames,width,height,avg_frame_rate', '-of', 'json', str(path)]))
    streams = document.get('streams')
    _require(type(streams) is list and len(streams) == 1, 'decoded video stream is unavailable')
    stream = streams[0]
    _require(int(stream['nb_read_frames']) == frames and stream['width'] == width
             and stream['height'] == height
             and math.isclose(float(Fraction(stream['avg_frame_rate'])), fps, rel_tol=1e-6, abs_tol=1e-6),
             'decoded video frame count, dimensions or frame rate differ')


def _camera(sources, frame):
    if not sources._camera['enabled']:
        return None
    row = sources._camera['frames'][frame]
    return np.asarray(row['image_to_court'], dtype=float) if row['is_valid'] else None


class _DisplayGeometryError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _project(points, matrix):
    try:
        projected = landing_zones.project_polygon(points, matrix)
    except ValueError as error:
        if str(error) in ('Zone projection intersects the projective horizon',
                          'Zone crosses the projective horizon'):
            raise _DisplayGeometryError('PROJECTIVE_HORIZON') from error
        if str(error) == 'A finite invertible calibrated image-to-court homography is required':
            raise _DisplayGeometryError('UNUSABLE_DISPLAY_MATRIX') from error
        raise
    # This is a drawing-library representability bound, not a court accuracy
    # or image-visibility gate. Legitimate offscreen points remain permitted.
    if not np.isfinite(projected).all() or np.max(np.abs(projected), initial=0.) > 2**30:
        raise _DisplayGeometryError('PROJECTED_COORDINATE_RANGE')
    return projected


def _display_geometry(sources, frame):
    matrix = _camera(sources, frame)
    if matrix is None:
        return None, 'CURRENT_FRAME_REGISTRATION_UNAVAILABLE'
    canonical = getattr(sources, '_court', {}).get('canonical_keypoints_m')
    if canonical is not None:
        canonical = np.asarray(canonical, dtype=float)
        _require(canonical.shape == (14, 2) and np.isfinite(canonical).all(),
                 'canonical projected court geometry differs')
    try:
        _project(((0., 0.), (10.97, 0.), (10.97, 23.77), (0., 23.77)), matrix)
        if canonical is not None:
            _project(canonical, matrix)
    except _DisplayGeometryError as error:
        return None, error.reason
    return matrix, None


def _identity_key(value):
    # Per-frame audit IDs, timestamps, support and hypotheses are observations,
    # not changes of the reviewed match binding.
    return _canonical([value.get('analysis_id'), value.get('epoch'), value.get('revision'),
        [[item.get(key) for key in ('track_key', 'player_id', 'state', 'anchor_id')]
         for item in value.get('observations', ())]])


def _continuous(sources, previous, frame):
    interval, identity = sources.source_frames[frame][0], sources.source_frames[frame][1]
    before, identity_before = previous
    return (interval.get('analysis_allowed') is True and before.get('analysis_allowed') is True
        and interval.get('measurement_transition_allowed') is True
        and interval.get('local_reset') is not True and not interval.get('discontinuity_reasons')
        and identity.get('local_reset') is not True
        and interval.get('segment_id') == before.get('segment_id')
        and interval.get('anchor_frame_index') == before.get('anchor_frame_index')
        and _identity_key(identity) == _identity_key(identity_before)
        and _display_geometry(sources, frame)[0] is not None
        and _display_geometry(sources, frame-1)[0] is not None)


def _text(image, value, position, *, scale=.65, color=(240, 240, 240)):
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _pose_frames(rebuilt, storage):
    """Retained image-pose visuals only; never ground or contact qualification."""
    path = rebuilt.snapshot.resolve_file('player_motion.json')
    if path is None:
        return None
    document = storage.read_json(path)
    _require(type(document) is dict and document.get('schema_version') == '2.0',
             'sealed pose presentation envelope differs')
    values = document.get('pose_frames', [])
    _require(len(values) in (0, rebuilt.snapshot.frames), 'sealed pose frame count differs')
    return values or None


def _draw_pose(image, row):
    from src.tracking.player_motion_tracking import JOINT_NAMES, SKELETON_EDGES
    _require(type(row) is dict, 'pose frame must be a role object')
    for role in ('1', '2'):
        joints = row.get(role)
        if joints is None:
            continue
        _require(type(joints) is dict and set(joints) == set(JOINT_NAMES), 'pose joint names differ')
        points = {}
        for name, joint in joints.items():
            _require(type(joint) is dict, 'pose joint must be an object')
            point = joint.get('position_px')
            _require(point is None or type(point) is list and len(point) == 2
                     and all(_finite(value) for value in point), 'pose image point is invalid')
            points[name] = tuple(map(int, point)) if point is not None else None
        color = (220, 210, 40) if role == '1' else (40, 180, 255)
        for a, b in SKELETON_EDGES:
            first, second = points[JOINT_NAMES[a]], points[JOINT_NAMES[b]]
            if first is not None and second is not None:
                cv2.line(image, first, second, color, 2, cv2.LINE_AA)
        for point in points.values():
            if point is not None:
                cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)


def _draw_frame(image, frame, rebuilt, shots, landings, partition, mini, trail, barrier,
                prediction, pose, initial_state, geometry_stats=None):
    sources, owner = rebuilt.sources, rebuilt.owner
    interval = sources.source_frames[frame][0]
    timestamp = interval['timestamp_seconds']
    allowed = interval['analysis_allowed'] is True
    matrix, geometry_reason = (_display_geometry(sources, frame) if allowed else
                              (None, 'OUTSIDE_SUPPORTED_INTERVAL'))
    if matrix is None and geometry_stats is not None:
        geometry_stats['geometry_withheld_frame_count'] += 1
        reasons = geometry_stats['geometry_withheld_reason_counts']
        reasons[geometry_reason] = reasons.get(geometry_reason, 0)+1
    if allowed:
        detections = sources._detections['frames'][frame]
        boxes = {f'ROLE {role}': _box(detections['player_'+str(role)]) for role in (1, 2)}
        image = VideoAnnotator.draw_player_boxes(image, {key: value for key, value in boxes.items() if value},
            {'ROLE 1': (255, 0, 0), 'ROLE 2': (0, 0, 255)})
        if pose is not None:
            _draw_pose(image, pose[frame])
        # Raw per-frame landmark detections were not retained. Any visible
        # points below are explicitly current calibration projections.
        canonical = sources._court.get('canonical_keypoints_m')
        if matrix is not None and canonical is not None:
            canonical = np.asarray(canonical, dtype=float)
            _require(canonical.shape == (14, 2) and np.isfinite(canonical).all(),
                     'canonical projected court geometry differs')
            projected = _project(canonical, matrix)
            image = VideoAnnotator.draw_court_keypoints(image, np.asarray(projected))
            _text(image, 'Projected calibration geometry', (30, image.shape[0]-85), scale=.48)
        ball = sources.ball_points[frame]
        if ball.x_px is not None and ball.y_px is not None:
            trail.append((ball.x_px, ball.y_px))
            del trail[:-25]
        else:
            trail.clear()
        image = VideoAnnotator.draw_ball(image,
            (ball.x_px, ball.y_px) if ball.x_px is not None and ball.y_px is not None else None, ball.state)
        image = VideoAnnotator.draw_ball_trajectory(image, trail, max_trail=25)
        # Every display contact is already qualified by the fresh export.
        # Same-segment continuity and the display-frame mapping are additional
        # rendering conditions; the event-frame polygon is never frozen in view.
        mini_image = mini.draw_court() if matrix is not None else None
        if matrix is not None:
            with closing(iter(landings.active_records(partition, timestamp))) as rows:
                for record in rows:
                    contact_frame = landings.calibrations[record['calibration_id']]['frame_index']
                    if contact_frame < barrier:
                        continue
                    try:
                        projected = {**record, 'contact_position_px': _project(
                            [record['contact_position_m']], matrix)[0], 'zones': [{**zone,
                            'polygon_px': _project(zone['polygon_m'], matrix)} for zone in record['zones']]}
                    except _DisplayGeometryError:
                        if geometry_stats is not None:
                            geometry_stats['withheld_contact_draws'] += 1
                        _text(image, 'Contact overlay unavailable', (30, image.shape[0]-105), scale=.48)
                        continue
                    image = landing_zones.draw_landing(image, projected, timestamp_seconds=timestamp)
                    mini_image = landing_zones.draw_landing(mini_image, record,
                        timestamp_seconds=timestamp, transform=mini.court_to_mini_court)
        with closing(iter(shots.event_owner.events_by_frame(max(barrier, frame-6), frame))) as events:
            for event in events:
                image = VideoAnnotator.draw_event_badge(image, event.event_type.value, (0, 255, 255), frame)
        with closing(iter(shots.shots.badges_by_frame(max(barrier, frame-20), frame))) as badges:
            for shot in badges:
                if shot.is_dead_ball:
                    continue
                cv2.rectangle(image, (40, 165), (750, 205), (180, 80, 0), -1)
                _text(image, f'SHOT: [{shot.shot_type.value}] ({shot.direction.value}, Conf: {shot.shot_confidence:.2f})',
                      (50, 193), scale=.65)
        with closing(iter(rebuilt.engine.native_line_calls().by_frame(max(barrier, frame-18), frame))) as calls:
            for _, call in calls:
                text = call.overlay_text()
                decision = call.decision.value
                color = (0, 180, 0) if 'IN' in decision else ((0, 0, 200)
                    if 'OUT' in decision or 'FAULT' in decision else (0, 165, 255))
                cv2.rectangle(image, (40, 110), (1200, 155), color, -1)
                _text(image, text, (55, 142), scale=.70)
        dead = owner.execute("SELECT 1 FROM records WHERE kind='transitions' AND frame BETWEEN ? AND ? "
            "AND json_extract(payload,'$.outcome_type')='DEAD_BALL_IGNORED' LIMIT 1",
            (max(barrier, frame-max(1, round(.5*sources.fps))), frame), one=True)
        if dead is not None:
            cv2.rectangle(image, (40, 215), (550, 255), (50, 50, 50), -1)
            _text(image, 'IGNORED - BALL NOT IN PLAY (DEAD BALL)', (50, 242), scale=.55, color=(0, 255, 255))
    else:
        trail.clear()
        mini_image = None
    if matrix is None:
        _text(image, 'Court geometry unavailable: '+geometry_reason,
              (30, image.shape[0]-85), scale=.48, color=(0, 180, 255))
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 90), (20, 20, 20), -1)
    cv2.line(image, (0, 90), (width, 90), (0, 255, 255), 2)
    hud = owner.frame_state(frame, initial_state)
    if rebuilt.authoritative_events_enabled and allowed:
        for player, y in ((1, 38), (2, 75)):
            info = hud['player_'+str(player)]
            text = (f"PLAYER {player}: Games {info['games']} | Points {info['points_display']}"
                    if hud['score_confirmed'] else f'PLAYER {player}: Score requires review')
            _text(image, text, (40, y), scale=.85,
                  color=(0, 255, 255) if hud['server_id'] == player else (255, 255, 255))
        state_label = str(hud['point_state']).replace('_', ' ').capitalize()
        label = f"Serve {hud['serve_attempt']} ({hud['service_side'].lower()}) | {state_label}"
        left, right = max(40, width-720), width-mini.width-15
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)[0][0]
        scale = min(.55, .55*max(1, right-left)/max(1, text_width))
        _text(image, label, (left, 55), scale=scale, color=(200, 200, 200))
    else:
        _text(image, 'Tennis analysis - score unavailable', (30, 38), scale=.8)
        _text(image, 'Outside supported interval' if not allowed else 'Event authority unavailable', (30, 72))
    payload = prediction['forecast']
    prediction_label = (f"Prediction uncalibrated | completed samples {payload['sample_count']}"
        if allowed and payload['status'] in ('prior_only', 'uncalibrated') else
        f"Confirmed result: PLAYER {payload['winner_id']}" if allowed and payload['status'] == 'confirmed_result'
        else 'Prediction unavailable (uncalibrated)')
    _text(image, prediction_label, (30, image.shape[0]-55), scale=.55)
    _text(image, 'Image roles / bbox foot proxy; ground positions and physical ball speed unavailable',
          (30, image.shape[0]-30), scale=.46)
    _text(image, f'Analysis revision {owner.revision} | source frame {frame}',
          (30, image.shape[0]-9), scale=.42)
    if mini_image is not None:
        image = VideoAnnotator.compose_frame(image, mini_image)
    return image


@dataclass(frozen=True)
class CorrectedRenderResult:
    directory: Path
    files: object
    manifest_sha256: str


def render_corrected_video(rebuilt, shots, forecast, exports, scratch):
    """Render and fully decode two fresh videos, then seal a private candidate."""
    directory, binding, export_stage, event_generation = _guard_exports(rebuilt, shots, forecast, exports, scratch)
    metadata = rebuilt.sources._detections['metadata']
    frames, fps = rebuilt.snapshot.frames, rebuilt.sources.fps
    width, height = metadata['width'], metadata['height']
    stage = Path(tempfile.mkdtemp(prefix='corrected-render-', dir=directory))
    seal_path = stage/MANIFEST_NAME
    source = export_stage/'source.mp4'
    geometry_stats = {'geometry_withheld_frame_count': 0, 'geometry_withheld_reason_counts': {},
                      'withheld_contact_draws': 0}
    try:
        with ExitStack() as stack:
            storage = stack.enter_context(_parser(directory))
            raw = storage.read_json(export_stage/'landing_zones.json', indexed_root_objects=('calibrations',))
            landings = stack.enter_context(LandingZoneSequence(directory, binding['analysis_id'], binding['analysis_revision']))
            with _reader(raw['records']) as rows:
                for record in rows:
                    landings.append(record, raw['calibrations'][record['calibration_id']])
            landings.seal()
            from src.scoring.corrected_landing_exports import _settings
            _, partition, _ = _settings(rebuilt.snapshot)
            pose = _pose_frames(rebuilt, storage)
            mini = MiniCourt(width=min(250, max(60, width//4)), height=min(500, max(120, height-95)))
            capture = stack.enter_context(_video_handle(cv2.VideoCapture(str(source))))
            _require(capture.isOpened(), 'source video cannot be decoded')
            _require(capture.get(cv2.CAP_PROP_FRAME_COUNT) == frames, 'source frame count differs')
            _require(capture.get(cv2.CAP_PROP_FRAME_WIDTH) == width
                     and capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == height, 'source video dimensions differ from metadata')
            _require(math.isclose(capture.get(cv2.CAP_PROP_FPS), fps, rel_tol=1e-6, abs_tol=1e-6),
                     'source frame rate differs from normalized metadata')
            writer = stack.enter_context(_video_handle(cv2.VideoWriter(str(stage/'annotated.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))))
            _require(writer.isOpened(), 'annotated video writer cannot be opened')
            history = stack.enter_context(closing(iter(forecast.history)))
            next_prediction, prediction = next(history), None
            initial_state = rebuilt.engine.initial_state.to_dict()
            previous, barrier, trail = None, 0, []
            for frame in range(frames):
                _require(shots.event_owner.events_generation == event_generation,
                         'physical events changed during rendering')
                ok, image = capture.read()
                _require(ok and image is not None, f'source video ended at frame {frame}; expected {frames}')
                _require(image.shape == (height, width, 3), 'decoded frame dimensions differ')
                current = rebuilt.sources.source_frames[frame]
                interval, identity = current[0], current[1]
                _require(math.isclose(interval['timestamp_seconds'], frame/fps, rel_tol=1e-6, abs_tol=1e-6),
                         'source normalized frame clock differs from video playback')
                if previous is None or not _continuous(rebuilt.sources, previous, frame):
                    barrier = frame
                    trail.clear()
                previous = interval, identity
                while next_prediction is not None and next_prediction['frame_index'] <= frame:
                    _require(next_prediction['timestamp_seconds'] <= interval['timestamp_seconds'],
                             'forecast includes a future source time')
                    prediction = next_prediction
                    next_prediction = next(history, None)
                _require(prediction is not None, 'initial causal prediction is missing')
                image = _draw_frame(image, frame, rebuilt, shots, landings, partition, mini,
                                    trail, barrier, prediction, pose, initial_state, geometry_stats)
                _require(shots.event_owner.events_generation == event_generation,
                         'physical events changed during rendering')
                writer.write(image)
                if frame % max(1, round(fps)) == 0:
                    _admission(rebuilt, shots, forecast, scratch)
            ok, extra = capture.read()
            _require(not ok and extra is None, 'source contains extra frames after analysis')
        # Independent codecs and complete decode count retain the worker's
        # browser recipe. Bound x264 threads; never reuse an old annotation.
        execute([binary('ffmpeg'), '-nostdin', '-n', '-v', 'error', '-xerror', '-i', str(stage/'annotated.mp4'),
            '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(stage/'annotated_web.mp4')])
        artifacts = {}
        for path in (source, stage/'annotated.mp4', stage/'annotated_web.mp4'):
            record = {'sha256': artifact_sha256(path), 'bytes': path.stat().st_size}
            full_video_check(path, frames=frames, fps=fps, width=width, height=height)
            _require(artifact_sha256(path) == record['sha256'] and path.stat().st_size == record['bytes'],
                     'video artifact changed during full decode')
            if path.name in VIDEO_FILES:
                artifacts[path.name] = record
        _require(_guard_exports(rebuilt, shots, forecast, exports, scratch)[3] == event_generation,
                 'physical events changed after rendering')
        _require({path.name for path in stage.iterdir()} == VIDEO_FILES, 'render artifact inventory differs')
        _object(seal_path, {**binding, 'schema_version': '1.0',
            'status': 'PRIVATE_PENDING_REPORT_AND_PUBLICATION', 'export_manifest_sha256': exports.manifest_sha256,
            'frames': frames, 'fps': fps, 'width': width, 'height': height, 'artifacts': artifacts,
            **geometry_stats,
            'clock': 'NORMALIZED_CFR; EXACT_ORIGINAL_PTS_AND_TIMEBASE_RETAINED_IN_RULE_REPLAY',
            'geometry': 'CURRENT_FRAME_PROJECTED_CALIBRATION; QUALIFIED_CONTACT_LANDINGS_ONLY',
            'prediction': 'CAUSAL_PREFIX; UNCALIBRATED; NO_CALIBRATED_PROBABILITIES_SHOWN',
            'human_review_dependent': True, 'independent_accuracy_validated': False, 'published': False})
        _require(_guard_exports(rebuilt, shots, forecast, exports, scratch)[3] == event_generation,
                 'physical events changed before render return')
        _require(all(artifact_sha256(stage/name) == record['sha256']
            and (stage/name).stat().st_size == record['bytes'] for name, record in artifacts.items()),
            'video artifact changed before render return')
        return CorrectedRenderResult(stage, MappingProxyType({name: stage/name
            for name in sorted(VIDEO_FILES | {MANIFEST_NAME})}), artifact_sha256(seal_path))
    except BaseException as error:
        try:
            seal_path.unlink(missing_ok=True)
        except BaseException as cleanup:
            error.add_note('Render seal cleanup also failed: '+str(cleanup))
        raise
