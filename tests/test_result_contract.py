"""Web contract evidence stays aligned, nullable and isolated by analysis."""

import json

import pytest

from src.api.results import build_result, provenance
from tests.test_product_api import write_artifacts


def update_artifact(directory, name, mutate):
    path = directory / name
    artifact = json.loads(path.read_text(encoding='utf-8'))
    mutate(artifact)
    path.write_text(json.dumps(artifact), encoding='utf-8')


def complete_player_frames(directory):
    def prepare(artifact):
        for frame in artifact['frames']:
            frame['court_registration_valid'] = True
            frame['player_2'] = dict(frame['player_1'])
    update_artifact(directory, 'detections.json', prepare)


def test_one_malformed_optional_player_sample_preserves_other_samples_and_player(tmp_path):
    job = write_artifacts(tmp_path)
    complete_player_frames(tmp_path)
    update_artifact(tmp_path, 'player_motion.json', lambda artifact: artifact['player_1']['samples'].__setitem__(1, None))
    result = build_result(tmp_path, job)
    assert result['trajectories']['player1CourtPositions'] == [[4, 8], None, [4, 8]]
    assert result['trajectories']['player2CourtPositions'] == [[4, 8]] * 3
    assert result['playerMetrics'] is None
    assert result['playerTracking']['player_1']['metrics']['total_distance_m'] is None
    assert result['playerTracking']['player_2']['metrics']['total_distance_m'] == .5
    assert result['summary']['capabilities']['players'] is True


def test_zero_motion_is_measured_and_not_converted_to_missing(tmp_path):
    job = write_artifacts(tmp_path)
    complete_player_frames(tmp_path)
    def zero(artifact):
        for player in ('player_1', 'player_2'):
            artifact[player]['observed_distance_m'] = 0
            artifact[player]['mean_speed_kmh'] = 0
            for sample in artifact[player]['samples']:
                sample['speed_kmh'] = 0
    update_artifact(tmp_path, 'player_motion.json', zero)
    result = build_result(tmp_path, job)
    assert result['playerMetrics']['player_1']['total_distance_m'] == 0
    assert result['playerMetrics']['player_2']['avg_speed_kmh'] == 0
    assert result['playerTracking']['player_1']['summaryScopePreserved'] is True


def test_motion_timestamp_mismatch_withholds_cross_gap_window_and_stale_summary(tmp_path):
    job = write_artifacts(tmp_path)
    complete_player_frames(tmp_path)
    def timestamps(artifact):
        for index, sample in enumerate(artifact['player_1']['samples']):
            sample.update(timestamp_s=index / 30, velocity_window_seconds=index / 30)
        artifact['player_1']['samples'][1]['timestamp_s'] = 10
    update_artifact(tmp_path, 'player_motion.json', timestamps)
    result = build_result(tmp_path, job)
    assert result['trajectories']['player1CourtPositions'][1] is None
    assert result['trajectories']['player1Speeds'] == [None, None, None]
    assert result['playerTracking']['player_1']['metrics']['total_distance_m'] is None
    assert result['playerTracking']['player_2']['metrics']['total_distance_m'] == .5


def test_camera_rejection_overrides_stale_detection_flag_and_preserves_other_evidence(tmp_path):
    job = write_artifacts(tmp_path)
    complete_player_frames(tmp_path)
    matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    camera = {'enabled': True, 'frames': [
        {'frame_index': index, 'is_valid': index != 1, 'image_to_court': matrix,
         'anchor_to_frame_px': matrix, 'rejection_reason': 'view rejected' if index == 1 else None}
        for index in range(3)]}
    (tmp_path / 'camera_registration.json').write_text(json.dumps(camera))
    result = build_result(tmp_path, job)
    assert result['detections']['frames'][1]['court_registration_valid'] is False
    assert result['trajectories']['player1CourtPositions'][1] is None
    assert result['trajectories']['ballTrajectory'][1]['court_x_m'] is None
    assert result['trajectories']['ballTrajectory'][1]['x_px'] == 30
    assert result['cameraTracking']['frames'][1]['imageToCourt'] is None
    assert result['cameraTracking']['frames'][1]['rejectionReason'] == 'view rejected'
    assert [segment['calibrationStatus'] for segment in result['courtSegments']] == [
        'VALID_COURT_PLANE', 'UNAVAILABLE_COURT_PLANE', 'VALID_COURT_PLANE']


def test_optional_pose_fallback_uses_actual_joints_and_rejects_outside_image(tmp_path):
    job = write_artifacts(tmp_path)
    (tmp_path / 'pose_tracking.json').write_text('{"frames": []}')
    def poses(artifact):
        artifact['pose_frames'] = [{'1': {'left_wrist': {'position_px': [4., 9.], 'confidence': .8},
                                        'right_wrist': {'position_px': [100., 2.], 'confidence': .9}}}] * 3
    update_artifact(tmp_path, 'player_motion.json', poses)
    result = build_result(tmp_path, job)
    joints = {joint['name']: joint for joint in result['poseTracking']['frames'][0]['player_1']}
    assert joints['left_wrist']['visible'] is True
    assert joints['right_wrist']['visible'] is False and joints['right_wrist']['x_px'] is None
    assert joints['right_wrist']['confidence'] == .9


def test_racket_center_outside_its_detector_box_stays_missing(tmp_path):
    job = write_artifacts(tmp_path)
    racket = {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.], 'position_px': [20., 7.], 'confidence': .7}
    (tmp_path / 'racket_tracking.json').write_text(json.dumps({'enabled': True, 'frames': [{'1': racket}] * 3}))
    result = build_result(tmp_path, job)
    assert result['racketTracking']['coveragePct']['player_1'] == 0
    assert result['racketTracking']['frames'][0]['player_1']['position_px'] is None
    assert result['summary']['capabilities']['racket'] is False


def test_fractional_edge_ball_is_valid_occlusion_preserved_and_speed_unqualified(tmp_path):
    job = write_artifacts(tmp_path)
    def ball(artifact):
        artifact['ball_trajectory'][0].update(x_px=99.5, speed_kmh=180)
        artifact['ball_trajectory'][1].update(state='OCCLUDED', confidence=.5)
    update_artifact(tmp_path, 'trajectories.json', ball)
    result = build_result(tmp_path, job)
    points = result['trajectories']['ballTrajectory']
    assert points[0]['state'] == 'DETECTED' and points[0]['x_px'] == 99.5
    assert points[0]['speed_kmh'] is None
    assert points[1]['state'] == 'OCCLUDED' and points[1]['x_px'] is None
    assert result['videoUrl'] is None and result['annotatedVideoUrl'] is None


def test_missing_court_keypoint_never_shifts_legacy_topology_and_singular_matrix_rejected(tmp_path):
    job = write_artifacts(tmp_path)
    update_artifact(tmp_path, 'court_geometry.json', lambda artifact: artifact.update(court_keypoints_px=[[1, 2], None, [3, 4]]))
    result = build_result(tmp_path, job)
    assert result['court']['courtKeypointsPx'] == []
    update_artifact(tmp_path, 'court_geometry.json', lambda artifact: artifact.update(homography_matrix=[[0, 0, 0]] * 3))
    result = build_result(tmp_path, job)
    assert result['court'] is None
    assert result['trajectories']['player1CourtPositions'] == [None] * 3


def test_bundle_version_excludes_analysis_identity_and_changes_with_local_weights(tmp_path):
    import yaml
    weight = tmp_path / 'local.pt'
    weight.write_bytes(b'first local checkpoint')
    config = {'player_detection': {'model': str(weight)}, 'match_configuration': {'match_id': 'first'}}
    path = tmp_path / 'inference_config.yaml'
    path.write_text(yaml.safe_dump(config))
    first = provenance(tmp_path, {})
    config['match_configuration']['match_id'] = 'second'
    path.write_text(yaml.safe_dump(config))
    second = provenance(tmp_path, {})
    assert first['bundleVersion'] == second['bundleVersion']
    assert first['configSha256'] != second['configSha256']
    weight.write_bytes(b'updated local checkpoint')
    third = provenance(tmp_path, {})
    assert third['bundleVersion'] != second['bundleVersion']
    assert third['productionQualified'] is False


@pytest.mark.parametrize('malformed', ['{broken', '[]'])
def test_invalid_optional_racket_json_preserves_ball_and_player_evidence(tmp_path, malformed):
    job = write_artifacts(tmp_path)
    complete_player_frames(tmp_path)
    (tmp_path / 'racket_tracking.json').write_text(malformed)
    result = build_result(tmp_path, job)
    assert result['racketTracking'] is None
    assert result['playerMetrics']['player_1']['total_distance_m'] == .5
    assert result['trajectories']['ballTrajectory'][0]['state'] == 'DETECTED'
    json.dumps(result, allow_nan=False)
