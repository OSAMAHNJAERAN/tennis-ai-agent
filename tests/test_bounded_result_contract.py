"""Disk-backed normalization must reproduce the exact preserved web contract.

These use actual artifact files and both real normalizers, without media or
model execution. The frozen file's hash prevents an edited reference from
silently turning a regression into a new expected result.
"""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from src.api.result_storage import ResultStorage
from src.api.results import build_result


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_baseline_2026-10-06/src/api/results.py'
BASELINE_SHA256 = 'd043172fa3674fc311176f97dc979e1660e1af8a17f8dec4f2dca34f6e33b5d7'
MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


@pytest.fixture(scope='module')
def frozen():
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA256
    spec = importlib.util.spec_from_file_location('frozen_result_contract_20261006', BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(directory, name, value):
    (directory / name).write_text(json.dumps(value), encoding='utf-8')


def read_json(directory, name):
    return json.loads((directory / name).read_text(encoding='utf-8'))


def mutate(directory, name, operation):
    value = read_json(directory, name)
    operation(value)
    write_json(directory, name, value)


def artifacts(tmp_path, count=129, fps=30):
    directory = tmp_path / 'analysis'
    directory.mkdir()
    frames, balls, samples, cameras = [], [], [], []
    for index in range(count):
        frames.append({'frame_index': index, 'court_registration_valid': True,
                       'player_1': {'bbox': [1, 1, 10, 20], 'confidence': .8, 'source_track_id': 7},
                       'player_2': {'bbox': [40, 3, 50, 30], 'confidence': .7, 'source_track_id': 8}})
        balls.append({'frame_index': index, 'timestamp_seconds': index / fps,
                      'x_px': 30., 'y_px': 50., 'court_x_m': 4., 'court_y_m': 8.,
                      'confidence': .9, 'speed_kmh': 180, 'state': 'DETECTED'})
        samples.append({'frame_index': index, 'timestamp_s': index / fps,
                        'ground_position_m': [4. + index / 1000, 8.], 'speed_kmh': 0. if index == 0 else 10.,
                        'position_fit_rmse_m': -.0})
        cameras.append({'frame_index': index, 'timestamp_seconds': index / fps,
                        'is_valid': True, 'image_to_court': MATRIX, 'anchor_to_frame_px': MATRIX,
                        'rejection_reason': None, 'inlier_mean_residual_reference_px': .125,
                        'reference_width': 960})
    write_json(directory, 'detections.json', {'metadata': {'frames': count, 'fps': fps, 'width': 100, 'height': 100}, 'frames': frames})
    write_json(directory, 'trajectories.json', {'ball_trajectory': balls})
    write_json(directory, 'court_geometry.json', {'is_valid': True, 'homography_matrix': MATRIX,
                                                'reprojection_error_px': 1.,
                                                'court_keypoints_px': [[1, 2], [3, 4]],
                                                'canonical_keypoints_m': [[0, 0], [1, 1]]})
    player = {'observed_distance_m': .5, 'mean_speed_kmh': 10., 'samples': samples}
    write_json(directory, 'player_motion.json', {'player_1': player, 'player_2': player})
    write_json(directory, 'camera_registration.json', {'enabled': True, 'frames': cameras})
    write_json(directory, 'metrics.json', {'processing_fps': 5., 'authoritative_events_enabled': False})
    job = {'id': 'match_bounded_contract', 'filename': 'Tennis é 🎾 <clip>.mp4',
           'started_at': '2026-10-06T00:00:00+00:00'}
    return directory, job


def equivalent(frozen, directory, job):
    expected = frozen.build_result(directory, job)
    expected_text = json.dumps(expected, allow_nan=False)
    scratch = directory.parent / 'result_storage'
    scratch.mkdir(exist_ok=True)
    with ResultStorage(scratch) as storage:
        bounded = build_result(directory, job, _storage=storage)
        stream = io.StringIO()
        storage.dump(bounded, stream)
        actual_text = stream.getvalue()
        # Exact bytes also pin key order, int/float distinctions, signed zero,
        # Unicode escaping, warning order and all unknown trajectory fields.
        assert actual_text.encode('utf-8') == expected_text.encode('utf-8')
        actual = json.loads(actual_text)
        assert actual == expected
    return actual


@pytest.mark.parametrize('count', [1, 64, 65, 129, 257])
def test_complete_contract_across_disk_array_boundaries(tmp_path, frozen, count):
    directory, job = artifacts(tmp_path, count=count)
    result = equivalent(frozen, directory, job)
    assert len(result['detections']['frames']) == len(result['cameraTracking']['frames']) == count
    assert len(result['trajectories']['ballTrajectory']) == count
    assert result['trajectories']['player1Speeds'][0] == 0.
    assert result['playerMetrics']['player_1']['total_distance_m'] == .5
    assert result['world3d'] is None and result['ballMetrics'] is None


@pytest.mark.parametrize('case', ['detection-count', 'detection-order', 'trajectory-count',
                                 'trajectory-order', 'trajectory-state', 'trajectory-time',
                                 'metadata-count', 'metadata-fps', 'metadata-width',
                                 'required-truncated', 'required-top-level', 'metrics-missing'])
def test_required_output_gates_match_frozen_failures(tmp_path, frozen, case):
    directory, job = artifacts(tmp_path)
    if case == 'detection-count':
        mutate(directory, 'detections.json', lambda value: value['frames'].pop())
    elif case == 'detection-order':
        mutate(directory, 'detections.json', lambda value: value['frames'][65].update(frame_index=64))
    elif case == 'trajectory-count':
        mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'].pop())
    elif case == 'trajectory-order':
        mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'][65].update(frame_index=64))
    elif case == 'trajectory-state':
        mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'][65].update(state='UNVERIFIED'))
    elif case == 'trajectory-time':
        mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'][65].update(timestamp_seconds=65 / 30 + .011))
    elif case.startswith('metadata-'):
        key = case.removeprefix('metadata-')
        key = 'frames' if key == 'count' else key
        invalid = {'frames': 129.5, 'fps': 0, 'width': True}[key]
        mutate(directory, 'detections.json', lambda value: value['metadata'].update({key: invalid}))
    elif case == 'required-truncated':
        path = directory / 'trajectories.json'
        path.write_text(path.read_text(encoding='utf-8')[:-2], encoding='utf-8')
    elif case == 'required-top-level':
        (directory / 'detections.json').write_text('[]', encoding='utf-8')
    else:
        (directory / 'metrics.json').unlink()
    with pytest.raises(ValueError) as expected:
        frozen.build_result(directory, job)
    scratch = tmp_path / 'result_storage'
    scratch.mkdir()
    with ResultStorage(scratch) as storage:
        with pytest.raises(ValueError) as actual:
            build_result(directory, job, _storage=storage)
        assert str(actual.value) == str(expected.value)


def optional_value(directory, name, count):
    if name in {'player_motion.json', 'camera_registration.json'}:
        return read_json(directory, name)
    if name == 'pose_tracking.json':
        return {'frames': [{'1': {'left_wrist': {'position_px': [4., 9.], 'confidence': .8}}}
                           for _ in range(count)]}
    if name == 'racket_tracking.json':
        return {'enabled': True, 'frames': [{'1': {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.],
                                                   'position_px': [6., 7.], 'confidence': .7}}
                                          for _ in range(count)]}
    if name == 'image_people.json':
        return {'coordinateSystem': 'normalized_video_pixels',
                'identityScope': 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
                'frames': [{'frame_index': index, 'timestamp_seconds': index / 30,
                            'people': [{'bbox': [1, 1, 10, 20], 'confidence': .8, 'sourceTrackId': 7,
                                        'courtRole': 1, 'participantStatus': 'COURT_ROLE_CANDIDATE'}]}
                           for index in range(count)]}
    raise AssertionError(name)


@pytest.mark.parametrize('name', ['player_motion.json', 'camera_registration.json', 'pose_tracking.json',
                                 'racket_tracking.json', 'image_people.json'])
@pytest.mark.parametrize('suffix', ['truncated-container', 'invalid-member', 'trailing-data'])
def test_valid_optional_prefix_with_corrupt_tail_is_wholly_unavailable(tmp_path, frozen, name, suffix):
    directory, job = artifacts(tmp_path)
    valid = json.dumps(optional_value(directory, name, 129))
    malformed = (valid[:-1] if suffix == 'truncated-container' else
                 valid[:-1] + ', "broken": }' if suffix == 'invalid-member' else valid + ' trailing garbage')
    (directory / name).write_text(malformed, encoding='utf-8')
    corrupt_result = equivalent(frozen, directory, job)
    # A successfully parsed prefix must never survive: the legacy whole-file
    # read returns {} on any suffix error, including when warning=False.
    # A wrong top-level type follows the same warning policy as a syntax
    # failure, unlike a valid empty object in sections that warn on read errors.
    write_json(directory, name, [])
    absent_result = equivalent(frozen, directory, job)
    assert corrupt_result == absent_result
    if name == 'racket_tracking.json':
        assert corrupt_result['racketTracking'] is None
    elif name == 'image_people.json':
        assert corrupt_result['imagePeople'] is None
    elif name == 'player_motion.json':
        assert corrupt_result['playerMetrics'] is None
        assert corrupt_result['trajectories']['player1CourtPositions'] == [None] * 129
    elif name == 'camera_registration.json':
        assert corrupt_result['cameraTracking']['enabled'] is False
        assert corrupt_result['detections']['frames'][0]['court_registration_valid'] is True
    else:
        assert corrupt_result['poseTracking'] is None


def test_duplicate_members_last_win_and_unknown_ball_payload_is_exact(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    points = read_json(directory, 'trajectories.json')['ball_trajectory']
    points[0].update(extra={'ints': [0, 1, -5], 'floats': [0., -.0, 1e-12],
                            'nested': {'text': 'é 🎾 \\ quoted " value', 'empty': [], 'null': None}},
                     legacy_field=True)
    # Last repeated object member wins, at both root-array and frame depth.
    encoded = json.dumps(points).replace('"legacy_field": true', '"legacy_field": false, "legacy_field": true')
    (directory / 'trajectories.json').write_text('{"ball_trajectory": [], "ball_trajectory": ' + encoded + '}', encoding='utf-8')
    camera = read_json(directory, 'camera_registration.json')
    (directory / 'camera_registration.json').write_text('{"enabled": false, "enabled": true, "frames": [], "frames": ' +
                                                       json.dumps(camera['frames']) + '}', encoding='utf-8')
    result = equivalent(frozen, directory, job)
    first = result['trajectories']['ballTrajectory'][0]
    assert first['legacy_field'] is True and first['extra']['nested']['empty'] == []
    assert type(first['extra']['ints'][0]) is int and type(first['extra']['floats'][0]) is float
    assert first['extra']['floats'][1].hex() == '-0x0.0p+0'
    assert result['cameraTracking']['enabled'] is True


def test_literal_nonfinite_values_become_null_without_dropping_unknown_keys(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    path = directory / 'trajectories.json'
    value = path.read_text(encoding='utf-8')
    value = value.replace('"frame_index": 0,', '"private": [NaN, Infinity, -Infinity], "frame_index": 0,', 1)
    path.write_text(value, encoding='utf-8')
    result = equivalent(frozen, directory, job)
    assert result['trajectories']['ballTrajectory'][0]['private'] == [None, None, None]


def test_exponent_overflow_in_unknown_ball_field_is_not_silently_sanitized(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    path = directory / 'trajectories.json'
    path.write_text(path.read_text(encoding='utf-8').replace('"frame_index": 0,',
                    '"private": 1e999, "frame_index": 0,', 1), encoding='utf-8')
    expected = frozen.build_result(directory, job)
    with pytest.raises(ValueError, match='Out of range float values'):
        json.dumps(expected, allow_nan=False)
    scratch = tmp_path / 'result_storage'
    scratch.mkdir()
    with ResultStorage(scratch) as storage:
        bounded = build_result(directory, job, _storage=storage)
        with pytest.raises(ValueError, match='Out of range float values'):
            storage.dump(bounded, io.StringIO())


def test_camera_rejection_and_long_velocity_windows_preserve_gap_withholding(tmp_path, frozen):
    directory, job = artifacts(tmp_path, count=257)
    mutate(directory, 'camera_registration.json', lambda value: value['frames'][65].update(
        is_valid=False, rejection_reason='CAMERA_CUT'))
    def windows(value):
        for key in ('player_1', 'player_2'):
            for index, sample in enumerate(value[key]['samples']):
                sample['velocity_window_seconds'] = index / 30
    mutate(directory, 'player_motion.json', windows)
    result = equivalent(frozen, directory, job)
    assert result['trajectories']['player1CourtPositions'][65] is None
    assert result['trajectories']['player1Speeds'][:66] == [None] + [10.] * 64 + [None]
    assert result['trajectories']['player1Speeds'][66:] == [None] * 191
    assert result['trajectories']['ballTrajectory'][65]['x_px'] == 30.
    assert result['trajectories']['ballTrajectory'][65]['court_x_m'] is None
    assert result['playerTracking']['player_1']['summaryScopePreserved'] is False
    assert [row['calibrationStatus'] for row in result['courtSegments']] == [
        'VALID_COURT_PLANE', 'UNAVAILABLE_COURT_PLANE', 'VALID_COURT_PLANE']


def test_recorded_source_timing_gaps_and_boolean_interval_edges_match(tmp_path, frozen):
    directory, job = artifacts(tmp_path, count=257)
    write_json(directory, 'media_metadata.json', {'timeMapping': {
        'status': 'CAPTURED_FFMPEG_FRAME_MAP',
        'validMeasurementTransitionIntervals': [{'startFrame': True, 'endFrameExclusive': 65},
                                               {'startFrame': 66, 'endFrameExclusive': 257}]}})
    mutate(directory, 'player_motion.json', lambda value: value['player_1']['samples'][256].update(
        velocity_window_seconds=256 / 30))
    result = equivalent(frozen, directory, job)
    assert result['trajectories']['player1Speeds'][65] is None
    assert result['trajectories']['player1Speeds'][256] is None
    assert result['trajectories']['player2Speeds'][256] == 10.
    assert result['trajectories']['player1CourtPositions'][65] is not None
    assert any('Movement timing estimates are withheld' in warning for warning in result['warnings'])


def test_one_bad_motion_sample_does_not_erase_the_other_player(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    mutate(directory, 'player_motion.json', lambda value: value['player_1']['samples'].__setitem__(65, None))
    result = equivalent(frozen, directory, job)
    assert result['trajectories']['player1CourtPositions'][65] is None
    assert result['trajectories']['player2CourtPositions'][65] is not None
    assert result['playerMetrics'] is None
    assert result['playerTracking']['player_2']['metrics']['total_distance_m'] == .5


def test_zero_motion_and_fractional_edge_occlusion_semantics_are_preserved(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    def zero(value):
        for key in ('player_1', 'player_2'):
            value[key].update(observed_distance_m=0, mean_speed_kmh=0.)
            for sample in value[key]['samples']:
                sample['speed_kmh'] = 0
    mutate(directory, 'player_motion.json', zero)
    mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'][0].update(x_px=99.5))
    mutate(directory, 'trajectories.json', lambda value: value['ball_trajectory'][65].update(state='OCCLUDED', confidence=.5))
    result = equivalent(frozen, directory, job)
    assert result['playerMetrics']['player_1']['total_distance_m'] == 0
    assert result['playerMetrics']['player_2']['avg_speed_kmh'] == 0.
    assert result['trajectories']['ballTrajectory'][0]['x_px'] == 99.5
    assert result['trajectories']['ballTrajectory'][65]['state'] == 'OCCLUDED'
    assert result['trajectories']['ballTrajectory'][65]['confidence'] == .5
    assert result['trajectories']['ballTrajectory'][65]['x_px'] is None
    assert all(point['speed_kmh'] is None for point in result['trajectories']['ballTrajectory'])


@pytest.mark.parametrize('pose_source', ['empty', 'corrupt-tail', 'short-valid'])
def test_pose_fallback_and_racket_box_validation_match(tmp_path, frozen, pose_source):
    directory, job = artifacts(tmp_path)
    def pose(value):
        value['pose_frames'] = [{'1': {'left_wrist': {'position_px': [4., 9.], 'confidence': .8},
                                             'right_wrist': {'position_px': [100., 2.], 'confidence': .9}}}
                                for _ in range(129)]
    mutate(directory, 'player_motion.json', pose)
    source = optional_value(directory, 'pose_tracking.json', 1)
    if pose_source == 'empty':
        source['frames'] = []
    encoded = json.dumps(source)
    (directory / 'pose_tracking.json').write_text(encoded + (' broken' if pose_source == 'corrupt-tail' else ''), encoding='utf-8')
    rackets = optional_value(directory, 'racket_tracking.json', 129)
    rackets['frames'][65]['1']['position_px'] = [20., 7.]
    write_json(directory, 'racket_tracking.json', rackets)
    result = equivalent(frozen, directory, job)
    first_joints = {joint['name']: joint for joint in result['poseTracking']['frames'][0]['player_1']}
    assert first_joints['left_wrist']['visible'] is True
    assert result['racketTracking']['frames'][65]['player_1']['state'] == 'MISSING'
    if pose_source == 'short-valid':
        assert result['poseTracking']['frames'][65]['player_1'] is None
    else:
        joints = {joint['name']: joint for joint in result['poseTracking']['frames'][65]['player_1']}
        assert joints['left_wrist']['visible'] is True
        assert joints['right_wrist']['visible'] is False and joints['right_wrist']['confidence'] == .9


def test_people_null_empty_unknown_and_role_evidence_remain_distinct(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    people = optional_value(directory, 'image_people.json', 129)
    people['frames'][0]['people'] = None
    people['frames'][1]['people'] = []
    people['frames'][65]['people'][0]['sourceTrackId'] = 70
    people['frames'][66]['people'][0]['confidence'] = .75
    people['frames'][67]['timestamp_seconds'] = 20
    people['frames'][68]['people'].extend([None, {'bbox': [1, 1, 101, 20], 'confidence': .8}])
    mutate(directory, 'camera_registration.json', lambda value: value['frames'][69].update(is_valid=False))
    write_json(directory, 'image_people.json', people)
    result = equivalent(frozen, directory, job)
    frames = result['imagePeople']['frames']
    assert frames[0]['people'] is None and frames[1]['people'] == []
    assert frames[2]['people'][0]['courtRole'] == 1
    assert frames[65]['people'][0]['participantStatus'] == 'UNKNOWN'
    assert frames[66]['people'][0]['participantStatus'] == 'UNKNOWN'
    assert frames[67]['people'] is None and len(frames[68]['people']) == 1
    assert frames[69]['people'][0]['courtRole'] is None
    assert result['summary']['capabilities']['imagePeople'] is True


def test_missing_court_topology_and_singular_matrix_are_not_compacted(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    mutate(directory, 'court_geometry.json', lambda value: value.update(court_keypoints_px=[[1, 2], None, [3, 4]]))
    result = equivalent(frozen, directory, job)
    assert result['court']['courtKeypointsPx'] == []
    mutate(directory, 'court_geometry.json', lambda value: value.update(homography_matrix=[[0, 0, 0]] * 3))
    result = equivalent(frozen, directory, job)
    assert result['court'] is None and result['playerMetrics'] is None
    assert result['trajectories']['player1CourtPositions'] == [None] * 129


def test_authoritative_event_and_call_arrays_beyond_small_array_threshold(tmp_path, frozen):
    directory, job = artifacts(tmp_path)
    mutate(directory, 'metrics.json', lambda value: value.update(authoritative_events_enabled=True))
    events = [{'event_type': 'UNKNOWN', 'timestamp_s': index / 30, 'confidence': None, 'extra': -.0}
              for index in range(129)]
    calls = [{'frame_index': index, 'decision': None, 'extra': [index, index + .25]} for index in range(129)]
    write_json(directory, 'match_events.json', {'events': events})
    write_json(directory, 'line_calls.json', {'line_calls': calls})
    result = equivalent(frozen, directory, job)
    assert result['events'] == events and result['lineCalls'] == calls
    assert result['summary']['eventCount'] == result['summary']['lineCallCount'] == 129

