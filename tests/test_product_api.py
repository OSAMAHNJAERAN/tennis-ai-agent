import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import JobStore, Settings
from src.api.results import build_result


@pytest.fixture
def service(tmp_path):
    app = create_app(Settings(tmp_path, Path('config.yaml'), max_upload_bytes=64), start_worker=False)
    with TestClient(app) as client:
        yield client, app.state.store


def upload(client, filename='match.mp4', body=b'\x00\x00\x00\x18ftypisom' + b'\0' * 20):
    return client.post('/api/analyses', params={'filename': filename}, content=body)


def test_upload_is_unique_persisted_and_path_safe(service):
    client, store = service
    first = upload(client, '../../same.mp4')
    second = upload(client, 'same.mp4')
    assert first.status_code == second.status_code == 202
    a, b = first.json(), second.json()
    assert a['id'] != b['id']
    assert a['filename'] == 'same.mp4'
    assert a['status'] == 'UPLOADED'
    assert 'config_path' not in a
    assert (store.upload_dir(a['id']) / 'original.mp4').is_file()
    assert JobStore(store.root).get(a['id'])['size_bytes'] == 32
    assert len(client.get('/api/analyses').json()) == 2
    assert client.get(f'/api/analyses/{a["id"]}/result').status_code == 409


@pytest.mark.parametrize('filename,body,extension', [
    ('match.MP4', b'\x00\x00\x00\x18ftypisom', '.mp4'),
    ('match.mov', b'\x00\x00\x00\x18ftypqt  ', '.mov'),
    ('match.avi', b'RIFF\x04\x00\x00\x00AVI ', '.avi'),
])
def test_supported_container_uploads_are_queued(service, filename, body, extension):
    client, store = service
    response = upload(client, filename, body)
    assert response.status_code == 202
    job = response.json()
    assert job['status'] == 'UPLOADED'
    assert (store.upload_dir(job['id']) / ('original' + extension)).read_bytes() == body


@pytest.mark.parametrize('filename,body,status', [
    ('bad.exe', b'RIFFxxxxAVI ', 415), ('bad.mp4', b'not a video', 415),
    ('big.mp4', b'\0' * 65, 413), ('empty.mov', b'', 415),
    ('bad.avi', b'\0\0\0\x18ftypisom', 415),
])
def test_invalid_uploads_leave_no_job_or_partial_file(service, filename, body, status):
    client, store = service
    assert upload(client, filename, body).status_code == status
    assert store.list() == []
    assert list(store.root.glob('uploads/*/*')) == []


def test_chunked_upload_limit_is_enforced_without_content_length(service):
    client, store = service
    response = client.post('/api/analyses?filename=chunked.mp4',
                           content=iter([b'\0\0\0\x18ftypisom', b'x' * 64]))
    assert response.status_code == 413
    assert store.list() == []


def test_queue_claim_and_restart_recovery(service):
    client, store = service
    first, second = upload(client).json(), upload(client).json()
    job = store.claim()
    assert job['id'] == first['id'] and job['started_at']
    store.stage(job['id'], 'Tracking players', 3)
    assert client.get(f'/api/analyses/{job["id"]}').json()['step'] == 3
    store.recover_interrupted()
    failed = store.get(first['id'])
    assert failed['status'] == 'FAILED' and failed['completed_at'] and failed['error']
    assert store.claim()['id'] == second['id']
    store.finish(second['id'])
    assert store.get(second['id'])['status'] == 'COMPLETED'


def test_only_completed_media_is_served_with_ranges(service):
    client, store = service
    job = upload(client).json()
    base = f'/api/analyses/{job["id"]}'
    output = store.output_dir(job['id'])
    output.mkdir(parents=True)
    (output / 'source.mp4').write_bytes(bytes(range(64)))
    assert client.get(base + '/media/source').status_code == 404
    store.finish(job['id'])
    response = client.get(base + '/media/source', headers={'Range': 'bytes=8-15'})
    assert response.status_code == 206 and response.content == bytes(range(8, 16))
    assert client.get(base + '/media/processing.log').status_code == 404
    assert client.get('/api/analyses/unknown/media/source').status_code == 404
    assert client.get(base + '/result').status_code == 503


def write_artifacts(directory):
    points = [{'frame_index': i, 'timestamp_seconds': i / 30, 'x_px': 30., 'y_px': 50.,
               'court_x_m': 4., 'court_y_m': 8., 'confidence': .9, 'speed_kmh': None, 'state': 'DETECTED'} for i in range(3)]
    player = {'observed_distance_m': .5, 'mean_speed_kmh': 10.,
              'samples': [{'ground_position_m': [4., 8.], 'speed_kmh': 10.} for _ in range(3)]}
    artifacts = {
        'detections.json': {'metadata': {'frames': 3, 'fps': 30, 'width': 100, 'height': 100},
                            'frames': [{'frame_index': i, 'court_registration_valid': i != 1,
                                        'player_1': {'bbox': [1, 1, 10, 20], 'confidence': .8}, 'player_2': None} for i in range(3)]},
        'trajectories.json': {'ball_trajectory': points},
        'court_geometry.json': {'is_valid': True, 'homography_matrix': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'reprojection_error_px': 1},
        'player_motion.json': {'player_1': player, 'player_2': player},
        'metrics.json': {'processing_fps': 5, 'authoritative_events_enabled': False},
    }
    for name, data in artifacts.items():
        (directory / name).write_text(json.dumps(data), encoding='utf-8')
    return {'id': 'match_test', 'filename': 'test.mp4', 'started_at': '2026-09-19T00:00:00+00:00'}


def test_adapter_preserves_camera_gaps_and_unavailable_measurements(tmp_path):
    result = build_result(tmp_path, write_artifacts(tmp_path))
    assert result['trajectories']['player1CourtPositions'] == [[4, 8], None, [4, 8]]
    assert result['trajectories']['ballTrajectory'][1]['court_x_m'] is None
    assert result['ballMetrics'] is None
    assert result['matchState'] is None and result['events'] == []
    assert result['playerMetrics'] is None
    assert result['playerTracking']['player_1']['metrics']['max_speed_kmh'] == 10
    assert result['playerTracking']['player_1']['metrics']['total_distance_m'] is None
    assert result['trajectories']['player2CourtPositions'] == [None, None, None]
    assert result['playerTracking']['player_2']['metrics']['max_speed_kmh'] is None
    assert result['detections']['frames'][0]['player_1']['confidence'] == .8


def test_required_output_failure_and_optional_section_absence(tmp_path):
    job = write_artifacts(tmp_path)
    (tmp_path / 'player_motion.json').unlink()
    assert build_result(tmp_path, job)['playerMetrics'] is None
    (tmp_path / 'trajectories.json').unlink()
    with pytest.raises(ValueError, match='Required analysis output'):
        build_result(tmp_path, job)


def test_worker_missing_weights_finishes_with_actionable_error(tmp_path, monkeypatch):
    from src.api.worker import process_job
    store = JobStore(tmp_path)
    job = store.create(store.new_id(), 'video.mp4', '.mp4', 12, tmp_path / 'missing.yaml')
    job = store.claim()
    process_job(store, job, Settings(tmp_path, tmp_path / 'missing.yaml'))
    result = store.get(job['id'])
    assert result['status'] == 'FAILED' and 'configuration is missing' in result['error']
    assert (store.output_dir(job['id']) / 'processing.log').is_file()


@pytest.mark.parametrize('failure,expected', [
    (MemoryError('allocation failed'), 'Insufficient processing memory'),
    (RuntimeError('CUDA out of memory'), 'Insufficient processing memory'),
    (RuntimeError('private server detail'), 'analysis pipeline failed'),
])
def test_worker_resource_and_inference_failures_are_persisted(tmp_path, monkeypatch, failure, expected):
    from src.api import worker
    store = JobStore(tmp_path)
    store.create(store.new_id(), 'video.mp4', '.mp4', 12, tmp_path / 'config.yaml')
    job = store.claim()
    def fail(_):
        raise failure
    monkeypatch.setattr(worker, 'check_models', fail)
    worker.process_job(store, job, Settings(tmp_path, tmp_path / 'config.yaml'))
    failed = store.get(job['id'])
    assert failed['status'] == 'FAILED' and expected in failed['error']
    assert failed['completed_at'] and 'private server detail' not in failed['error']
    assert not (store.output_dir(job['id']) / 'result.json').exists()


def test_missing_weight_check_does_not_load_or_download_models(tmp_path):
    from src.api.worker import check_models, AnalysisError
    import yaml
    config = {'court_detection': {'model_path': str(tmp_path / 'court.pth')},
              'player_detection': {'model': str(tmp_path / 'player.pt')},
              'ball_detection': {'model': str(tmp_path / 'ball.pt')}}
    path = tmp_path / 'pipeline.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(AnalysisError, match='model weights or inference source files are missing'):
        check_models(path)
    assert not list(tmp_path.glob('*.pt'))


def test_adapter_rejects_out_of_bounds_ball_and_handles_null_optional_fields(tmp_path):
    job = write_artifacts(tmp_path)
    path = tmp_path / 'trajectories.json'
    data = json.loads(path.read_text())
    data['ball_trajectory'][0]['x_px'] = 1000
    path.write_text(json.dumps(data))
    court = tmp_path / 'court_geometry.json'
    data = json.loads(court.read_text())
    data['court_keypoints_px'] = None
    court.write_text(json.dumps(data))
    result = build_result(tmp_path, job)
    assert result['court']['courtKeypointsPx'] == []
    assert result['trajectories']['ballTrajectory'][0]['state'] == 'MISSING'
    assert result['detections']['frames'][0]['ball']['position_px'] is None


def test_malformed_optional_motion_cannot_publish_stale_metrics(tmp_path):
    job = write_artifacts(tmp_path)
    motion = tmp_path / 'player_motion.json'
    data = json.loads(motion.read_text())
    data['player_1']['samples'] = [None] * 3
    motion.write_text(json.dumps(data))
    assert build_result(tmp_path, job)['playerMetrics'] is None


def test_v2_pose_racket_evidence_preserves_confidence_and_missing_players(tmp_path):
    job = write_artifacts(tmp_path)
    path = tmp_path / 'player_motion.json'
    motion = json.loads(path.read_text())
    motion['pose_frames'] = [{'1': {'left_wrist': {'position_px': [4., 9.], 'confidence': .8},
                                  'right_wrist': {'position_px': None, 'confidence': .2}},
                              '2': {'left_wrist': {'position_px': [6., 8.], 'confidence': .9}}} for _ in range(3)]
    path.write_text(json.dumps(motion))
    (tmp_path / 'racket_tracking.json').write_text(json.dumps({'enabled': True, 'frames': [
        {'1': {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.], 'position_px': [6., 7.], 'confidence': .7},
         '2': {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.], 'position_px': [6., 7.], 'confidence': .7}} for _ in range(3)]}))
    result = build_result(tmp_path, job)
    assert result['contractVersion'] == '2.0'
    assert result['summary']['capabilities']['pose'] is True
    joints = {joint['name']: joint for joint in result['poseTracking']['frames'][0]['player_1']}
    assert joints['left_wrist']['x_px'] == 4 and joints['left_wrist']['visible'] is True
    assert joints['right_wrist']['x_px'] is None and joints['right_wrist']['confidence'] == .2
    assert result['poseTracking']['frames'][0]['player_2'] is None
    assert result['racketTracking']['frames'][0]['player_1']['state'] == 'DETECTED'
    assert result['racketTracking']['frames'][0]['player_2']['state'] == 'MISSING'
    assert result['world3d'] is None and result['summary']['capabilities']['world3d'] is False


def test_reports_escape_content_preserve_nulls_and_belong_to_requested_id(service):
    from src.api.reporting import readable_report, telemetry_csv
    import csv
    import io
    client, store = service
    first, second = upload(client).json(), upload(client).json()
    directory = store.output_dir(first['id'])
    directory.mkdir(parents=True)
    job = write_artifacts(directory)
    job.update(id=first['id'], filename='<script>alert(1)</script>.mp4')
    saved = build_result(directory, job)
    (directory / 'result.json').write_text(json.dumps(saved))
    store.finish(first['id'])
    html = readable_report(saved)
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert first['id'] in html and second['id'] not in html
    rows = list(csv.DictReader(io.StringIO(telemetry_csv(saved))))
    assert len(rows) == 3 and rows[1]['player1_speed_kmh'] == ''
    assert rows[1]['ball_projected_court_x_m'] == ''
    base = f'/api/analyses/{first["id"]}'
    response = client.get(base + '/download/readable-report')
    assert response.status_code == 200 and 'text/html' in response.headers['content-type']
    assert first['id'] in response.text
    response = client.get(base + '/download/csv')
    assert response.status_code == 200 and 'text/csv' in response.headers['content-type']
    assert client.get(f'/api/analyses/{second["id"]}/download/readable-report').status_code == 404
