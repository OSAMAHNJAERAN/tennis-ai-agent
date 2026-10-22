"""Versioned web evidence from one analysis directory, preserving missing data.

Availability describes exported observations; it never certifies their accuracy.
Image observations and court-plane projections are not reconstructed 3D motion.
"""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import yaml

from src.api.jobs import REPO_ROOT
from src.api.media_source import validate_media_source
from src.detection.image_people import IDENTITY_SCOPE, PARTICIPANT_SEMANTICS
from src.tracking.player_motion_tracking import JOINT_NAMES, SKELETON_EDGES

CONTRACT_VERSION = '2.0'
PLAYERS = ('player_1', 'player_2')


def number(value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        return None
    return value


def point(value):
    return value if isinstance(value, list) and len(value) == 2 and all(number(v) is not None for v in value) else None


def image_point(value, width, height):
    location = point(value)
    return location if location is not None and 0 <= location[0] < width and 0 <= location[1] < height else None


def bbox(value, width, height):
    valid = isinstance(value, list) and len(value) == 4 and all(number(v) is not None for v in value)
    return value if valid and 0 <= value[0] < value[2] <= width and 0 <= value[1] < value[3] <= height else None


def matrix3(value):
    valid = isinstance(value, list) and len(value) == 3 and all(
        isinstance(row, list) and len(row) == 3 and all(number(v) is not None for v in row) for row in value)
    if not valid:
        return None
    scale = max(abs(v) for row in value for v in row)
    if scale == 0:
        return None
    a, b, c = ([v / scale for v in row] for row in value)
    determinant = (a[0] * (b[1] * c[2] - b[2] * c[1]) -
                   a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]))
    return value if abs(determinant) > 1e-12 else None


def aligned_sample(sample, index, fps):
    if not isinstance(sample, dict):
        return {}
    if 'frame_index' in sample and (number(sample['frame_index'], 0) != index or isinstance(sample['frame_index'], bool)):
        return {}
    for field in ('timestamp_seconds', 'timestamp_s'):
        if field in sample:
            timestamp = number(sample[field], 0)
            if timestamp is None or abs(timestamp - index / fps) > .01:
                return {}
    return sample


def intervals(valid, fps):
    """Half-open time intervals for contiguous available normalized frames."""
    result, start = [], None
    for index, available in enumerate([*valid, False]):
        if available and start is None:
            start = index
        elif not available and start is not None:
            result.append({'startFrame': start, 'endFrameExclusive': index,
                           'startSeconds': start / fps, 'endSeconds': index / fps})
            start = None
    return result


@lru_cache(maxsize=32)
def _file_hash(path, size, modified_ns):
    # Stat values invalidate the cache if an asset changes. Read in bounded chunks.
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def file_hash(path):
    try:
        stat = path.stat()
        return _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns) if path.is_file() else None
    except OSError:
        return None


def provenance(directory, metrics):
    config_path = next((directory / name for name in ('inference_config.yaml', 'run_config.yaml')
                        if (directory / name).is_file()), None)
    config = {}
    if config_path:
        try:
            config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
            config = config if isinstance(config, dict) else {}
        except (OSError, yaml.YAMLError):
            pass
    hashes = {}
    for role, section, field in (('players', 'player_detection', 'model'), ('ball', 'ball_detection', 'model'),
                                 ('court', 'court_detection', 'model_path'), ('pose', 'pose_estimation', 'model_path')):
        settings = config.get(section)
        settings = settings if isinstance(settings, dict) else {}
        asset = settings.get(field)
        if isinstance(asset, str):
            path = Path(asset)
            path = path if path.is_absolute() else REPO_ROOT / path
            # Only the configured local artifact is read; never download weights.
            hashes[role] = {'filename': path.name, 'sha256': file_hash(path)}
    config_hash = file_hash(config_path) if config_path else None
    # Match identity is job-specific and cannot define a production bundle.
    pipeline_settings = {key: value for key, value in config.items() if key != 'match_configuration'}
    bundle_signature = {'settings': pipeline_settings, 'checkpointHashes': hashes}
    bundle_hash = hashlib.sha256(json.dumps(bundle_signature, sort_keys=True, default=str).encode('utf-8')).hexdigest() if config else None
    bundle = config.get('production_bundle')
    bundle = bundle if isinstance(bundle, dict) else {}
    return {'bundleVersion': str(bundle['version']) if bundle.get('version') else (
                'candidate-' + bundle_hash[:12] if bundle_hash else None),
            'bundleStatus': 'UNQUALIFIED_CANDIDATE', 'productionQualified': False,
            'configSha256': config_hash, 'bundleSha256': bundle_hash, 'checkpointHashes': hashes,
            'datasetHashes': bundle.get('dataset_hashes') if isinstance(bundle.get('dataset_hashes'), dict) else None,
            'codeCommit': metrics.get('code_commit') if isinstance(metrics.get('code_commit'), str) else None,
            'pipeline': 'Phase6Pipeline', 'ballBackend': metrics.get('ball_backend'),
            'trackingStrategy': metrics.get('tracking_strategy')}


def _player_evidence(frame, key):
    return frame.get(key, frame.get(str(PLAYERS.index(key) + 1))) if isinstance(frame, dict) else None


def pose_tracking(raw, normalized_frames, count, fps, width, height):
    source = raw.get('pose_frames', raw.get('frames'))
    if not isinstance(source, list) or not source:
        return None
    records, coverage = [], {key: 0 for key in PLAYERS}
    for index in range(count):
        frame = aligned_sample(source[index], index, fps) if index < len(source) else {}
        record = {'frame_index': index, 'timestamp_seconds': index / fps}
        for key in PLAYERS:
            joints = _player_evidence(frame, key) if normalized_frames[index][key] else None
            values = []
            for name in JOINT_NAMES:
                joint = joints.get(name) if isinstance(joints, dict) else None
                confidence = number(joint.get('confidence'), 0, 1) if isinstance(joint, dict) else None
                location = image_point(joint.get('position_px'), width, height) if isinstance(joint, dict) else None
                visible = location is not None and confidence is not None and confidence > 0 and joint.get('visible') is not False
                values.append({'name': name, 'x_px': location[0] if visible else None,
                               'y_px': location[1] if visible else None, 'confidence': confidence, 'visible': visible})
            record[key] = values if isinstance(joints, dict) else None
            coverage[key] += any(joint['visible'] for joint in values)
        records.append(record)
    return {'coordinateSystem': 'normalized_video_pixels', 'jointNames': list(JOINT_NAMES),
            'boneTopology': [list(edge) for edge in SKELETON_EDGES], 'frames': records,
            'coveragePct': {key: coverage[key] / count * 100 for key in PLAYERS},
            'accuracyValidated': False, 'world3d': None,
            'source': 'PLAYER_CROP_POSE_DETECTOR', 'missingSemantics': 'Missing joints have null coordinates.'}


def racket_tracking(raw, normalized_frames, count, fps, width, height):
    source = raw.get('frames')
    if raw.get('enabled') is False or not isinstance(source, list) or not source:
        return None
    records, coverage = [], {key: 0 for key in PLAYERS}
    for index in range(count):
        frame = aligned_sample(source[index], index, fps) if index < len(source) else {}
        record = {'frame_index': index, 'timestamp_seconds': index / fps}
        for key in PLAYERS:
            evidence = _player_evidence(frame, key) if normalized_frames[index][key] else None
            evidence = evidence if isinstance(evidence, dict) else {}
            box = bbox(evidence.get('bbox_xyxy'), width, height)
            location = image_point(evidence.get('position_px'), width, height)
            confidence = number(evidence.get('confidence'), 0, 1)
            valid = (evidence.get('state') == 'DETECTED' and box is not None and location is not None and
                     confidence is not None and confidence > 0 and
                     box[0] <= location[0] <= box[2] and box[1] <= location[1] <= box[3])
            record[key] = {'state': 'DETECTED' if valid else 'MISSING', 'bbox_xyxy': box if valid else None,
                           'position_px': location if valid else None, 'confidence': confidence if valid else None,
                           'velocity_px_per_s': point(evidence.get('velocity_px_per_s')) if valid else None,
                           'motion_direction_deg': number(evidence.get('motion_direction_deg')) if valid else None,
                           'source': 'COCO_TENNIS_RACKET_PLAYER_CROP', 'ownership': 'PLAYER_CROP_ASSOCIATION',
                           'contact_event': None}
            coverage[key] += valid
        records.append(record)
    return {'coordinateSystem': 'normalized_video_pixels', 'frames': records,
            'coveragePct': {key: coverage[key] / count * 100 for key in PLAYERS},
            'accuracyValidated': False, 'world3d': None,
            'motionSemantics': 'Bounding-box translation; swing and contact are not qualified.'}


def image_people(raw, normalized_frames, count, fps, width, height, court_available):
    """Expose image observations without relying on a court or active-player ID."""
    source = raw.get('frames')
    if (not isinstance(source, list) or not source or raw.get('coordinateSystem') != 'normalized_video_pixels'
            or raw.get('identityScope') != IDENTITY_SCOPE):
        return None
    records = []
    for index in range(count):
        frame = aligned_sample(source[index], index, fps) if index < len(source) else {}
        if frame.get('frame_index') != index or number(frame.get('timestamp_seconds'), 0) is None:
            frame = {}
        proposals = frame.get('people')
        people = None
        if isinstance(proposals, list):
            people = []
            for proposal in proposals:
                if not isinstance(proposal, dict):
                    continue
                bounds = bbox(proposal.get('bbox'), width, height)
                confidence = number(proposal.get('confidence'), 0, 1)
                if bounds is None or confidence is None or confidence <= 0:
                    continue
                identity = number(proposal.get('sourceTrackId'), 0)
                identity = int(identity) if identity is not None and int(identity) == identity else None
                role = proposal.get('courtRole')
                role = role if type(role) is int and role in (1, 2) else None
                selected = normalized_frames[index].get(f'player_{role}') if role is not None else None
                # A stale candidate label cannot survive removal of its court
                # registration or selected actual box. The image box survives.
                if (not court_available or not normalized_frames[index]['court_registration_valid']
                        or proposal.get('participantStatus') != 'COURT_ROLE_CANDIDATE'
                        or selected is None or selected['bbox'] != bounds
                        or selected['source_track_id'] != identity or selected['confidence'] != confidence):
                    role = None
                people.append({'bbox': bounds, 'confidence': confidence, 'sourceTrackId': identity,
                               'courtRole': role,
                               'participantStatus': 'COURT_ROLE_CANDIDATE' if role is not None else 'UNKNOWN'})
        records.append({'frame_index': index, 'timestamp_seconds': index / fps, 'people': people})
    return {'coordinateSystem': 'normalized_video_pixels', 'identityScope': IDENTITY_SCOPE,
            'participantSemantics': PARTICIPANT_SEMANTICS, 'accuracyValidated': False,
            'missingSemantics': 'Null people means unavailable frame observations; an empty list means no valid person proposals.',
            'frames': records}


def build_result(directory: Path, job):
    warnings = []

    def read(name, required=False, warn=True):
        try:
            value = json.loads((directory / name).read_text(encoding='utf-8'),
                               parse_constant=lambda _: None)
            if not isinstance(value, dict):
                raise ValueError('Expected object')
            return value
        except (OSError, ValueError):
            if required:
                raise ValueError(f'Required analysis output is missing or invalid: {name}')
            if warn:
                warnings.append(f'Analysis unavailable for this section: {name.removesuffix(".json").replace("_", " ")}.')
            return {}

    detections = read('detections.json', True)
    metadata = detections.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Invalid video metadata in analysis output.')
    width, height = number(metadata.get('width'), 1), number(metadata.get('height'), 1)
    if width is None or height is None or int(width) != width or int(height) != height:
        raise ValueError('Invalid video dimensions in analysis output.')
    count = number(metadata.get('frames'), 1)
    fps = number(metadata.get('fps'), .1, 240)
    if count is None or int(count) != count or fps is None:
        raise ValueError('Invalid video frame count or frame rate in analysis output.')
    count = int(count)
    frames = detections.get('frames')
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError('Player detections do not cover the processed video.')
    trajectory = read('trajectories.json', True).get('ball_trajectory')
    if not isinstance(trajectory, list) or len(trajectory) != count:
        raise ValueError('Ball trajectory does not cover the processed video.')
    states = {'DETECTED', 'TRACKED', 'PREDICTED', 'INTERPOLATED', 'OCCLUDED', 'MISSING'}
    for index, sample in enumerate(trajectory):
        if not isinstance(sample, dict) or sample.get('frame_index') != index or sample.get('state') not in states:
            raise ValueError('Invalid ball trajectory frame or tracking state.')
        timestamp = number(sample.get('timestamp_seconds'), 0)
        if timestamp is None or abs(timestamp - index / fps) > .01:
            raise ValueError('Ball timestamps disagree with the processed video.')
        sample['confidence'] = number(sample.get('confidence'), 0, 1)
        for key in ('x_px', 'y_px', 'court_x_m', 'court_y_m', 'speed_kmh'):
            sample[key] = number(sample.get(key))
        valid_position = image_point([sample['x_px'], sample['y_px']], width, height) is not None
        if sample['state'] in {'MISSING', 'OCCLUDED'} or sample['confidence'] is None or not valid_position:
            sample.update(state=sample['state'] if sample['state'] in {'MISSING', 'OCCLUDED'} else 'MISSING',
                          x_px=None, y_px=None, court_x_m=None, court_y_m=None, speed_kmh=None)
        # Existing artifacts do not contain an independently qualified flight estimator.
        sample['speed_kmh'] = None
        sample['source'] = sample.get('source') if isinstance(sample.get('source'), str) else None
        sample['world_position_m'] = None
        sample['coordinate_semantics'] = 'IMAGE_OBSERVATION_WITH_OPTIONAL_COURT_PLANE_RAY_INTERSECTION'
        if sample['state'] == 'MISSING':
            sample['confidence'] = None

    camera = read('camera_registration.json', warn=False)
    camera_frames = camera.get('frames') if isinstance(camera.get('frames'), list) else []
    normalized_frames = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get('frame_index') != index:
            raise ValueError('Invalid detection frame ordering.')
        camera_sample = aligned_sample(camera_frames[index], index, fps) if index < len(camera_frames) else {}
        calibrated = frame.get('court_registration_valid') is True
        if camera.get('enabled') is True:
            calibrated = calibrated and camera_sample.get('is_valid') is True and matrix3(camera_sample.get('image_to_court')) is not None
        normalized = {'frame_index': index, 'court_registration_valid': calibrated}
        for key in ('player_1', 'player_2'):
            player = frame.get(key)
            box = bbox(player.get('bbox'), width, height) if isinstance(player, dict) else None
            confidence = number(player.get('confidence'), 0, 1) if isinstance(player, dict) else None
            valid = box is not None and confidence is not None
            track_id = number(player.get('source_track_id'), 0) if valid else None
            normalized[key] = {'bbox': box, 'confidence': confidence,
                               'source_track_id': int(track_id) if track_id is not None and int(track_id) == track_id else None} if valid else None
        sample = trajectory[index]
        normalized['ball'] = {'position_px': point([sample['x_px'], sample['y_px']]),
                              'confidence': sample['confidence'], 'state': sample['state']}
        normalized_frames.append(normalized)

    raw_court = read('court_geometry.json')
    matrix = matrix3(raw_court.get('homography_matrix'))
    residual = number(raw_court.get('reprojection_error_px'), 0)
    court = None
    if raw_court.get('is_valid') is True and matrix is not None and residual is not None:
        image_points = raw_court.get('court_keypoints_px')
        canonical_points = raw_court.get('canonical_keypoints_m')
        # Legacy dense arrays are only exposed when topology is intact. Do not
        # compact away a missing keypoint and silently shift all later indices.
        court = {'courtKeypointsPx': image_points if isinstance(image_points, list) and all(point(p) is not None for p in image_points) else [],
                 'canonicalKeypointsM': canonical_points if isinstance(canonical_points, list) and all(point(p) is not None for p in canonical_points) else [],
                 'homographyMatrix': matrix, 'reprojectionErrorPx': residual, 'isValid': True}
    else:
        warnings.append('Court mapping unavailable: calibration was not reliable enough for metric positions.')

    original_media = read('media_metadata.json', warn=False)
    mapping = original_media.get('timeMapping')
    mapping = mapping if isinstance(mapping, dict) else {}
    mapping_recorded = mapping.get('status') in {'CAPTURED_FFMPEG_FRAME_MAP', 'PARTIAL_FFMPEG_FRAME_MAP', 'FRAME_MAPPING_UNAVAILABLE'}
    timing_transitions = [False] * count
    for interval in mapping.get('validMeasurementTransitionIntervals', []) if mapping_recorded else []:
        if not isinstance(interval, dict):
            continue
        start, end = interval.get('startFrame'), interval.get('endFrameExclusive')
        if isinstance(start, int) and isinstance(end, int) and 1 <= start < end <= count:
            timing_transitions[start:end] = [True] * (end - start)
    if mapping_recorded and not all(timing_transitions[1:]):
        warnings.append('Movement timing estimates are withheld across duplicated, dropped, missing or cadence-invalid source frames.')
    motion = read('player_motion.json')
    positions, speeds, player_metrics, player_tracks = [], [], {}, {}
    for key in ('player_1', 'player_2'):
        player = motion.get(key, {})
        if not isinstance(player, dict):
            player = {}
        raw_samples = player.get('samples', [])
        raw_samples = raw_samples if isinstance(raw_samples, list) else []
        samples = [aligned_sample(raw_samples[i], i, fps) if i < len(raw_samples) else {} for i in range(count)]
        if len(raw_samples) != count or any(not sample for sample in samples):
            warnings.append(f'Analysis unavailable for this section: {key.replace("_", " ")} movement.')
        locations = [point(sample.get('ground_position_m')) if court and normalized_frames[i]['court_registration_valid']
                     and normalized_frames[i][key] is not None
                     else None for i, sample in enumerate(samples)]
        speed = [number(sample.get('speed_kmh'), 0, 43.2) if locations[i] else None for i, sample in enumerate(samples)]
        # A published window estimate must be backed by all observed positions
        # within its original window. Withhold estimates that cross a removed
        # player/camera/timestamp sample rather than bridge the new gap.
        for i, sample in enumerate(samples):
            window = number(sample.get('velocity_window_seconds'), 0)
            if speed[i] is not None and window is not None:
                start = max(0, math.ceil(i - window * fps - 1e-8))
                if start >= i or any(location is None for location in locations[start:i + 1]):
                    speed[i] = None
                elif mapping_recorded and not all(timing_transitions[start + 1:i + 1]):
                    speed[i] = None
            elif speed[i] is not None and mapping_recorded and not timing_transitions[i]:
                speed[i] = None
        positions.append(locations)
        speeds.append(speed)
        has_samples = any(value is not None for value in speed)
        summary_scope_preserved = len(raw_samples) == count and all(
            samples[i] and point(samples[i].get('ground_position_m')) == locations[i] and
            number(samples[i].get('speed_kmh'), 0, 43.2) == speed[i] for i in range(count))
        player_metrics[key] = {'total_distance_m': number(player.get('observed_distance_m'), 0) if has_samples and summary_scope_preserved else None,
                               'avg_speed_kmh': number(player.get('mean_speed_kmh'), 0, 43.2) if has_samples and summary_scope_preserved else None,
                               'max_speed_kmh': max((v for v in speed if v is not None), default=None),
                               'coverage_pct': sum(p is not None for p in locations) / count * 100}
        player_tracks[key] = {'role': 'near' if key == 'player_1' else 'far', 'name': None,
                              'identityScope': metadata.get('player_identity_scope', 'NEAR_FAR_ROLE_WITHIN_CAMERA_SEGMENT'),
                              'metrics': dict(player_metrics[key]), 'accuracyValidated': False,
                              'summaryScopePreserved': summary_scope_preserved,
                              'groundPositionSource': motion.get('ground_position_source', 'CALIBRATED_BBOX_BOTTOM_CENTER_PROXY'),
                              'positionIntervals': intervals([p is not None for p in locations], fps),
                              'speedIntervals': intervals([s is not None for s in speed], fps),
                              'uncertaintySemantics': 'Window fit residual is not independent position or speed error.',
                              'samples': [{'frame_index': i, 'timestamp_seconds': i / fps,
                                           'ground_position_m': locations[i], 'speed_kmh': speed[i],
                                           'position_fit_rmse_m': number(sample.get('position_fit_rmse_m'), 0)
                                               if locations[i] else None}
                                          for i, sample in enumerate(samples)]}
    # Retain the established non-null metric contract. A missing metric makes
    # this section unavailable; never manufacture zero distance or speed.
    if any(p['total_distance_m'] is None or p['avg_speed_kmh'] is None for p in player_metrics.values()):
        player_metrics = None
        warnings.append('Analysis unavailable for this section: player distance and average speed.')
    for index, sample in enumerate(trajectory):
        if not court or not normalized_frames[index]['court_registration_valid']:
            sample['court_x_m'] = sample['court_y_m'] = None

    metrics = read('metrics.json', True)
    # This pipeline explicitly withholds physical ball speed. Readiness is not
    # changed by making the artifacts available on the web.
    warnings.append('Ball speed is unavailable: single-camera court projection does not measure airborne speed.')
    authoritative = metrics.get('authoritative_events_enabled') is True
    events = read('match_events.json').get('events', []) if authoritative else []
    calls = read('line_calls.json').get('line_calls', []) if authoritative else []
    events = events if isinstance(events, list) else []
    calls = calls if isinstance(calls, list) else []
    if not authoritative:
        warnings.append('Bounce, shot placement, rally and scoring analysis unavailable: event measurements are not qualified.')
    base = f'/api/analyses/{job["id"]}'
    pose_raw = read('pose_tracking.json', warn=False)
    pose = pose_tracking(pose_raw, normalized_frames, count, fps, width, height) or pose_tracking(motion, normalized_frames, count, fps, width, height)
    rackets = racket_tracking(read('racket_tracking.json'), normalized_frames, count, fps, width, height)
    people = image_people(read('image_people.json', warn=False), normalized_frames, count, fps, width, height, court is not None)
    if pose is None:
        warnings.append('Body pose unavailable: no frame-aligned pose observations were exported.')
    processing_provenance = provenance(directory, metrics)
    normalized_media = {'width': width, 'height': height, 'fps': fps, 'frames': count,
                        'durationSeconds': count / fps, 'timeBase': 'frame_index / fps',
                        'coordinateSystem': 'normalized_video_pixels'}
    segments = []
    for valid in (True, False):
        for interval in intervals([court is not None and frame['court_registration_valid'] for frame in normalized_frames]
                                  if valid else [court is None or not frame['court_registration_valid'] for frame in normalized_frames], fps):
            segments.append({**interval, 'calibrationStatus': 'VALID_COURT_PLANE' if valid else 'UNAVAILABLE_COURT_PLANE',
                             'accuracyValidated': False})
    segments.sort(key=lambda segment: segment['startFrame'])
    camera_records = []
    for i in range(count):
        raw = aligned_sample(camera_frames[i], i, fps) if i < len(camera_frames) else {}
        valid = court is not None and normalized_frames[i]['court_registration_valid']
        camera_records.append({'frame_index': i, 'timestamp_seconds': i / fps, 'isValid': valid,
                               'imageToCourt': matrix3(raw.get('image_to_court')) if valid else None,
                               'anchorToFramePx': matrix3(raw.get('anchor_to_frame_px')) if valid else None,
                               'rejectionReason': raw.get('rejection_reason') if isinstance(raw.get('rejection_reason'), str) else None,
                               'registrationResidualReferencePx': number(raw.get('inlier_mean_residual_reference_px'), 0),
                               'referenceWidth': number(raw.get('reference_width'), 1),
                               'accuracyValidated': False})
    unavailable = {'ballSpeed': 'Single-camera court projection does not measure physical airborne speed.',
                   'world3d': 'No independently qualified 3D reconstruction is available.',
                   'masks': 'No aligned segmentation masks were exported.',
                   'scoring': 'Scoring measurements are not qualified.'}
    if not court:
        unavailable['court'] = 'Court calibration was not reliable enough for metric positions.'
    if pose is None:
        unavailable['pose'] = 'No frame-aligned body pose observations were exported.'
    if rackets is None:
        unavailable['racket'] = 'No frame-aligned racket observations were exported.'
    if people is None:
        unavailable['imagePeople'] = 'No raw frame-aligned person observations were exported.'
    for key in PLAYERS:
        if player_tracks[key]['metrics']['total_distance_m'] is None:
            unavailable[key + 'Summary'] = 'A complete valid measurement scope for distance and average speed was not available.'
    if not authoritative:
        unavailable['events'] = 'Event measurements are not qualified.'
    exports = {'reportJson': base + '/download/report', 'readableReport': base + '/download/readable-report',
               'csv': base + '/download/csv',
               'annotatedVideo': base + '/download/annotated' if (directory / 'annotated_web.mp4').is_file() else None,
               'originalVideo': base + '/download/original'}
    if (directory / 'source_frame_mapping.jsonl').is_file():
        exports['sourceFrameMapping'] = base + '/download/source-timing'
    summary = {'id': job['id'], 'label': job['filename'], 'phase': 'Match analysis', 'status': 'COMPLETED',
               'analyzedAt': job.get('completed_at') or job['started_at'], 'sourceVideoPath': job['filename'],
               'frames': count, 'fps': fps, 'durationSeconds': count / fps,
               'eventCount': len(events), 'lineCallCount': len(calls),
               'artifacts': sorted(p.name for p in directory.iterdir() if p.is_file() and p.suffix in {'.json', '.yaml', '.mp4'}),
               'capabilities': {'detections': True, 'trajectories': True, 'court': court is not None,
                                'players': any(track['metrics']['max_speed_kmh'] is not None for track in player_tracks.values()), 'ballMetrics': False,
                                'events': authoritative, 'lineCalls': authoritative, 'scoring': False,
                                'video': (directory / 'source.mp4').is_file(),
                                'pose': pose is not None and any(pose['coveragePct'].values()),
                                'racket': rackets is not None and any(rackets['coveragePct'].values()),
                                'imagePeople': people is not None and any(frame['people'] for frame in people['frames']),
                                'masks': False, 'world3d': False, 'report': True, 'csv': True}}
    media_source_raw = read('media_source.json', warn=False)
    media_source = validate_media_source(media_source_raw)
    if media_source_raw and media_source is None:
        warnings.append('Source attribution unavailable: the saved source record was invalid.')
    return {'contractVersion': CONTRACT_VERSION, 'summary': summary, 'mediaSource': media_source,
            'mediaMetadata': {'original': original_media.get('original'), 'normalized': normalized_media,
                              'normalization': original_media.get('normalization'),
                              'timeMapping': original_media.get('timeMapping')},
            'provenance': processing_provenance,
            'coordinateSemantics': {'image': 'Normalized video pixels, origin top-left; x right, y down.',
                                    'court': 'Meters, origin far-left doubles corner; x right, y toward near baseline.',
                                    'ballProjection': 'Viewing ray intersection with the court plane; not airborne xyz.',
                                    'playerPosition': 'Calibrated bounding-box bottom-center ground proxy.',
                                    'time': 'Normalized constant-frame-rate video presentation time.',
                                    'missing': 'Null means unavailable; zero is a measured value, never a missing sentinel.'},
            'poseTracking': pose, 'racketTracking': rackets, 'playerTracking': player_tracks, 'imagePeople': people,
            'segmentationMasks': None, 'world3d': None, 'unavailableSections': unavailable, 'exports': exports,
            'courtSegments': segments,
            'cameraTracking': {'enabled': camera.get('enabled') is True, 'frames': camera_records,
                               'coordinateSemantics': 'Court-plane registration; no reconstructed airborne geometry.',
                               'accuracyValidated': False},
            'detections': {'metadata': {k: metadata.get(k) for k in ('frames', 'fps', 'width', 'height')},
                                              'frames': normalized_frames},
            'trajectories': {'ballTrajectory': trajectory, 'player1CourtPositions': positions[0],
                             'player2CourtPositions': positions[1], 'player1Speeds': speeds[0], 'player2Speeds': speeds[1]},
            'court': court, 'playerMetrics': player_metrics, 'ballMetrics': None, 'events': events, 'lineCalls': calls,
            'pipelineMetrics': {'pipelineFps': number(metrics.get('processing_fps'), 0), 'eventCount': len(events),
                                'lineCallCount': len(calls), 'courtReprojectionErrorPx': residual,
                                'raw': {'production_qualified': False, 'authoritative_events_enabled': authoritative}},
            'matchState': None, 'scoringEvents': [], 'scoreHistory': [], 'runConfig': None,
            'videoUrl': base + '/media/source' if (directory / 'source.mp4').is_file() else None,
            'annotatedVideoUrl': base + '/media/annotated' if (directory / 'annotated_web.mp4').is_file() else None,
            'warnings': warnings}
