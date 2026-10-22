"""Real queued semantic corrections and complete delivery on synthetic media.

No inference runtime is admitted by these software tests. The configured crop
provider's identity and resource refusal are tested before any model import.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from src.api import correction_worker, worker
from src.api.analysis_generation import resolve_generation
from src.api.corrections import CorrectionStore
from src.api.jobs import JobStore, Settings
from src.api.result_document import ResultDocument
from tests.test_corrected_preparation import build_complete_base
from tests.test_corrected_shot_rebuild import no_model_imports, advance_equivalent_generation


@pytest.fixture
def queued_correction(tmp_path):
    fixture_root = tmp_path/'native'
    fixture_root.mkdir()
    fixture = build_complete_base(fixture_root)
    settings = Settings(tmp_path/'service', tmp_path/'unused.yaml', require_auth=True)
    jobs = JobStore(settings.root)
    jobs.create('fixture', fixture.job['filename'], '.mp4', 1, settings.config,
                owner_id=fixture.job['owner_id'])
    jobs.claim()
    jobs.finish('fixture')
    output = jobs.output_dir('fixture')
    shutil.copytree(fixture.directory, output)
    fixture.directory = output
    fixture.snapshot = resolve_generation(output)
    commands = CorrectionStore(jobs)
    payload = {key: copy.deepcopy(value) for key, value in fixture.request.items()
               if key not in {'analysis_id', 'reviewer_id'}}
    submitted = commands.submit('fixture', fixture.job['owner_id'], payload,
                                authorization_method='AUTHENTICATED_OWNER')
    claimed = commands.claim()
    assert claimed['id'] == submitted['id'] and claimed['status'] == 'PROCESSING'
    return SimpleNamespace(fixture=fixture, settings=settings, jobs=jobs,
                           commands=commands, claimed=claimed, payload=payload)


def assert_original_unchanged(case):
    fixture = case.fixture
    assert case.jobs.get('fixture')['status'] == 'COMPLETED'
    assert (fixture.directory/'.result/current.store').read_bytes() == fixture.marker
    for name, record in fixture.snapshot._files.items():
        path = fixture.snapshot.resolve_file(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']


@pytest.mark.parametrize('lost_acknowledgement', [False, True])
def test_queued_owner_command_rebuilds_renders_and_publishes_complete_generation(
        queued_correction, monkeypatch, lost_acknowledgement):
    case = queued_correction
    from src.scoring import corrected_publication
    real_publish = corrected_publication.publish_corrected_result
    commits = []

    def publish(*args, **kwargs):
        token = real_publish(*args, **kwargs)
        commits.append(token)
        if lost_acknowledgement:
            raise OSError('Synthetic acknowledgement loss after actual pointer commit')
        return token

    monkeypatch.setattr(corrected_publication, 'publish_corrected_result', publish)
    correction_worker.process_correction(case.jobs, case.claimed, case.settings)
    completed = case.commands.get(case.claimed['id'], internal=True)
    assert completed['status'] == 'COMPLETED' and completed['error'] is None
    assert completed['generation'] == commits[0] and len(commits) == 1
    assert completed['step'] == 11 and completed['attempt'] == 1
    assert case.jobs.get('fixture')['status'] == 'COMPLETED'
    assert case.commands.committed_generation(completed) == commits[0]
    assert correction_worker.committed_correction_generation(case.jobs, completed) == commits[0]
    snapshot = resolve_generation(case.fixture.directory)
    assert snapshot.generation == commits[0] and snapshot.analysis_revision == 1
    with ResultDocument(case.fixture.directory, 'fixture') as document:
        result = document.value
        assert result['pointAnalysis']['summary']['totalPointsPlayed'] == 1
        assert result['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 1
        review = result['analysisCorrection']
        assert review['authorizationStatus'] == 'OWNER_AUTHORIZED'
        assert review['authorizationReceipt'] == case.claimed['authorization']
        assert review['execution']['pose_provider_calls'] == 0
        assert review['execution']['pose_prediction_calls'] == 0
        assert review['execution']['new_vision_inference'] is False
        assert review['execution']['selected_player_checkpoint_sha256'] is None
        assert result['nativeForecast']['current']['sample_count'] == 1
        assert result['nativeForecast']['current']['serve_estimates'][0]['probability'] == .4
        assert result['provenance']['productionQualified'] is False
        presented = document.presented('fixture')
        assert presented['annotatedVideoUrl'].endswith('?generation='+commits[0])
        assert all('?generation='+commits[0] in url for url in presented['exports'].values())
    with ResultDocument(case.fixture.directory, 'fixture',
                        generation=case.fixture.snapshot.generation) as previous:
        assert previous.value['analysisRevision'] == 0
        assert previous.value['pointAnalysis']['summary']['totalPointsPlayed'] == 0
    html = snapshot.resolve_file('readable_report.html').read_text(encoding='utf-8')
    assert 'Owner-authorized correction review' in html
    assert snapshot.resolve_file('annotated_web.mp4').stat().st_size > 0
    retried = case.commands.submit('fixture', case.fixture.job['owner_id'], case.payload,
                                  authorization_method='AUTHENTICATED_OWNER')
    assert retried['status'] == 'COMPLETED' and retried['generation'] == commits[0]
    assert case.commands.claim() is None


@pytest.mark.parametrize('failure', ['exception', 'cancel', 'stale'])
def test_interrupted_native_work_keeps_completed_analysis_available(queued_correction, monkeypatch, failure):
    case = queued_correction
    from src.scoring import corrected_exports
    original = corrected_exports.write_corrected_exports

    def interrupt(*args, **kwargs):
        exports = original(*args, **kwargs)
        if failure == 'exception':
            raise OSError('Synthetic failure after actual native exports')
        if failure == 'cancel':
            case.commands.cancel(case.claimed['id'], case.fixture.job['owner_id'])
        else:
            advance_equivalent_generation(case.fixture)
        return exports

    monkeypatch.setattr(corrected_exports, 'write_corrected_exports', interrupt)
    correction_worker.process_correction(case.jobs, case.claimed, case.settings)
    final = case.commands.get(case.claimed['id'], internal=True)
    assert final['status'] == {'exception': 'FAILED', 'cancel': 'CANCELLED', 'stale': 'CONFLICT'}[failure]
    assert final['error'] and final['generation'] is None
    assert case.jobs.get('fixture')['status'] == 'COMPLETED'
    if failure != 'stale':
        assert_original_unchanged(case)
    with ResultDocument(case.fixture.directory, 'fixture',
                        generation=case.fixture.snapshot.generation) as previous:
        assert previous.value['analysisRevision'] == 0
    assert not list((case.settings.root/'corrections').rglob('corrected_delivery_manifest.json'))


def test_fixed_cpu_resource_refusal_is_actionable_and_preserves_base(queued_correction, monkeypatch):
    from scripts.train import train_yolo26_author_ball_repair as resources
    case = queued_correction
    stages = []
    original_guard = resources.resource_guard

    def snapshot(_directory, torch=None):
        assert torch is None
        return {'ram': 1536*1024**2-1, 'disk': 2*1024**3, 'rss': 0, 'peak_rss': 0, 'vram': None}

    def guard(values, stage):
        stages.append(stage)
        return original_guard(values, stage)

    monkeypatch.setattr(resources, 'resource_snapshot', snapshot)
    monkeypatch.setattr(resources, 'resource_guard', guard)
    correction_worker.process_correction(case.jobs, case.claimed, case.settings)
    final = case.commands.get(case.claimed['id'], internal=True)
    assert final['status'] == 'FAILED' and 'Insufficient free processing memory' in final['error']
    assert stages == ['cpu']
    assert_original_unchanged(case)


def test_cancellation_before_processing_does_not_rebuild_or_publish(queued_correction):
    case = queued_correction
    case.commands.cancel(case.claimed['id'], case.fixture.job['owner_id'])
    correction_worker.process_correction(case.jobs, case.claimed, case.settings)
    assert case.commands.get(case.claimed['id'], internal=True)['status'] == 'CANCELLED'
    assert_original_unchanged(case)


def test_scratch_creation_failure_does_not_kill_worker_or_change_original(queued_correction, monkeypatch):
    case = queued_correction

    def fail_scratch(**_kwargs):
        raise PermissionError('Synthetic inaccessible correction scratch')

    monkeypatch.setattr(correction_worker.tempfile, 'mkdtemp', fail_scratch)
    correction_worker.process_correction(case.jobs, case.claimed, case.settings)
    final = case.commands.get(case.claimed['id'], internal=True)
    assert final['status'] == 'FAILED' and final['generation'] is None
    assert_original_unchanged(case)


@pytest.mark.parametrize('wrong_checkpoint_hash', [False, True])
def test_crop_provider_checks_original_checkpoint_and_fixed_pre_torch_floor(
        queued_correction, monkeypatch, wrong_checkpoint_hash):
    import yaml
    from scripts.train import train_yolo26_author_ball_repair as resources
    from src.api.analysis_generation import publish_analysis_generation
    from src.api.result_storage import ResultStorage
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import CorrectedPoseRequest
    case = queued_correction
    fixture = case.fixture
    checkpoint = fixture.scratch/'synthetic-checkpoint-identity-only.pt'
    checkpoint.write_bytes(b'EXPLICIT SYNTHETIC BYTES: NEVER LOAD AS A MODEL')
    config = yaml.safe_load(fixture.snapshot.resolve_file('run_config.yaml').read_text(encoding='utf-8'))
    config['model_system'].update(ultralytics_version='8.4.36', inference_head='one_to_one', end2end=True, nms=False)
    config['player_detection']['model'] = str(checkpoint)
    config['pose_estimation']['model_path'] = str(checkpoint)
    run_config = fixture.scratch/'updated-run-config.yaml'
    run_config.write_text(yaml.safe_dump(config), encoding='utf-8')
    with ResultDocument(fixture.directory, 'fixture') as document:
        result = json.loads(''.join(document.storage.json_chunks(document.value)))
    result['provenance']['checkpointHashes']['players'] = {'filename': checkpoint.name,
        'sha256': 'f'*64 if wrong_checkpoint_hash else hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    result['provenance']['configSha256'] = hashlib.sha256(run_config.read_bytes()).hexdigest()
    files = {name: fixture.snapshot.resolve_file(name) for name in fixture.snapshot._files
             if name not in {'result.json', 'summary.json'}}
    files.update({'run_config.yaml': run_config, 'inference_config.yaml': run_config})
    with ResultStorage(fixture.directory) as storage:
        token = publish_analysis_generation(storage, result, fixture.directory, files=files,
            analysis_revision=0, expected_generation=fixture.snapshot.generation)
    fixture.request['base_generation'] = token
    fixture.request['base_rule_replay_sha256'] = resolve_generation(fixture.directory).file_sha256('native_rule_replay.json')
    admitted = []

    def snapshot(_directory, torch=None):
        assert torch is None
        return {'ram': 4608*1024**2-1, 'disk': 2*1024**3, 'rss': 0, 'peak_rss': 0, 'vram': None}

    original_guard = resources.resource_guard

    def guard(values, stage):
        admitted.append(stage)
        return original_guard(values, stage)

    monkeypatch.setattr(resources, 'resource_snapshot', snapshot)
    monkeypatch.setattr(resources, 'resource_guard', guard)
    monkeypatch.setattr(correction_worker.importlib.metadata, 'version', lambda _name: '8.4.36')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        provider = correction_worker.SelectedCorrectionPose(rebuilt, fixture.scratch)
        request = CorrectedPoseRequest('fixture', 1, token, fixture.source_sha,
            rebuilt.snapshot.file_sha256('run_config.yaml'), str(checkpoint), (), (), 8, 2, None, 1.)
        try:
            expected = 'differs from original inference' if wrong_checkpoint_hash else 'Insufficient free processing memory'
            with pytest.raises((ValueError, correction_worker.CorrectionResourceError), match=expected):
                provider(request)
            record = provider.record()
            assert record['new_vision_inference'] is False and record['pose_provider_calls'] == 0
            assert record['pose_prediction_calls'] == 0
            assert record['selected_player_checkpoint_sha256'] is None and record['runtime_versions'] is None
            assert admitted == ([] if wrong_checkpoint_hash else ['before_gpu'])
        finally:
            provider.close()
        with pytest.raises(ValueError, match='provider is closed'):
            provider.record()


def test_queues_alternate_and_fall_back_without_starving_pending_work(monkeypatch):
    calls = []
    uploads, corrections = [1, 2], [3, 4]
    jobs = SimpleNamespace(claim=lambda: uploads.pop(0) if uploads else None)
    commands = SimpleNamespace(claim=lambda: corrections.pop(0) if corrections else None)
    settings = object()
    monkeypatch.setattr(worker, 'process_job', lambda store, item, config:
                        calls.append(('upload', store is jobs, item, config is settings)))
    monkeypatch.setattr(correction_worker, 'process_correction', lambda store, item, config:
                        calls.append(('correction', store is jobs, item, config is settings)))
    preference = False
    for _ in range(4):
        processed, preference = worker.process_next_work(jobs, commands, settings, prefer_correction=preference)
        assert processed
    assert calls == [('upload', True, 1, True), ('correction', True, 3, True),
                     ('upload', True, 2, True), ('correction', True, 4, True)]
    corrections.append(5)
    processed, preference = worker.process_next_work(jobs, commands, settings, prefer_correction=False)
    assert processed and preference is False and calls[-1] == ('correction', True, 5, True)
    assert worker.process_next_work(jobs, commands, settings, prefer_correction=preference) == (False, False)


def test_main_recovers_both_queues_only_under_exclusive_lock(monkeypatch, tmp_path):
    from src.api import corrections
    events = []
    settings = SimpleNamespace(root=tmp_path)
    jobs = SimpleNamespace(recover_interrupted=lambda callback: events.append('upload recovery'))
    commands = SimpleNamespace(recover_interrupted=lambda: events.append('correction recovery'))

    class StopLoop(BaseException):
        pass

    class Lock:
        def __init__(self, path, timeout):
            assert Path(path) == tmp_path/'worker.lock' and timeout == 0

        def __enter__(self):
            events.append('locked')

        def __exit__(self, *_args):
            events.append('unlocked')

    def next_work(store, retained, config, **options):
        assert store is jobs and retained is commands and config is settings
        assert options == {'prefer_correction': False}
        events.append('queue processing')
        raise StopLoop()

    monkeypatch.setattr(worker.Settings, 'from_env', lambda: settings)
    monkeypatch.setattr(worker, 'JobStore', lambda root: jobs)
    monkeypatch.setattr(corrections, 'CorrectionStore', lambda store: commands)
    monkeypatch.setattr(worker, 'FileLock', Lock)
    monkeypatch.setattr(worker, 'process_next_work', next_work)
    with pytest.raises(StopLoop):
        worker.main()
    assert events == ['locked', 'upload recovery', 'correction recovery', 'queue processing', 'unlocked']


@pytest.mark.parametrize('browser_event_amendment', [False, True])
def test_cookie_owner_http_command_through_worker_to_matching_media_and_reports(tmp_path, browser_event_amendment):
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    fixture_root = tmp_path/'native'
    fixture_root.mkdir()
    fixture = build_complete_base(fixture_root)
    settings = Settings(tmp_path/'service', tmp_path/'unused.yaml', require_auth=True)
    app = create_app(settings, start_worker=False, demos_manifest=tmp_path/'no-public-demos.json')
    jobs = app.state.store
    with TestClient(app) as owner, TestClient(app) as stranger, TestClient(app) as guest:
        registered = owner.post('/api/auth/register', json={
            'username': 'correction_owner', 'password': 'explicit-worker-test-password-123'})
        assert registered.status_code == 201, registered.text
        owner_id = registered.json()['user']['id']
        assert stranger.post('/api/auth/register', json={
            'username': 'other_review_owner', 'password': 'different-worker-test-password-456'}).status_code == 201
        jobs.create('fixture', fixture.job['filename'], '.mp4', 1, settings.config, owner_id=owner_id)
        jobs.finish('fixture')
        output = jobs.output_dir('fixture')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(fixture.directory, output)
        base = '/api/analyses/fixture'
        before = owner.get(base+'/result').json()
        assert before['analysisGeneration'] == fixture.snapshot.generation
        source = owner.get(base+'/review-source', params={'generation': before['analysisGeneration']})
        assert source.status_code == 200 and source.json()['is_current'] is True
        assert guest.get(base+'/review-source').status_code == 401
        assert stranger.get(base+'/review-source').status_code == 404
        payload = {key: copy.deepcopy(value) for key, value in fixture.request.items()
                   if key not in {'analysis_id', 'reviewer_id'}}
        if browser_event_amendment:
            from tests.test_corrected_rule_rebuild import browser_number_encoding
            reviewed = source.json()['inputs'][0]
            replacement = copy.deepcopy(reviewed['source_input'])
            replacement['context']['binding']['evidence_ref'] = 'SYNTHETIC_BROWSER_BINDING_REVIEW'
            payload['input_amendments'] = [{'event_id': replacement['event_id'],
                'expected_input_fingerprint': reviewed['input_fingerprint'],
                'replacement_source_input': replacement}]
            payload = browser_number_encoding(payload)
        response = owner.post(base+'/corrections', json=payload)
        assert response.status_code == 202, response.text
        correction_id = response.json()['id']
        commands = CorrectionStore(jobs)
        assert commands.get(correction_id, internal=True)['command']['reviewer_id'] == owner_id
        # Admission and pending processing retain the complete old analysis.
        assert owner.get(base+'/result').json()['analysisGeneration'] == before['analysisGeneration']
        assert worker.process_next_work(jobs, commands, settings, prefer_correction=True) == (True, False)
        status = owner.get(base+'/corrections/'+correction_id)
        assert status.status_code == 200 and status.json()['status'] == 'COMPLETED', status.text
        current = owner.get(base+'/result')
        assert current.status_code == 200 and current.headers['cache-control'] == 'private, no-store'
        result = current.json()
        token = result['analysisGeneration']
        assert token == status.json()['generation'] and result['analysisRevision'] == 1
        assert result['analysisCorrection']['authorizationReceipt']['owner_id'] == owner_id
        assert result['pointAnalysis']['summary']['totalPointsPlayed'] == 1
        assert result['nativeForecast']['current']['sample_count'] == 1
        if browser_event_amendment:
            after = owner.get(base+'/review-source', params={'generation': token}).json()['inputs'][0]
            assert after['source_input']['context']['binding']['evidence_ref'] == 'SYNTHETIC_BROWSER_BINDING_REVIEW'
            for key in ('timestamp_s', 'confidence', 'ball_position_px', 'court_position_m', 'evidence'):
                assert json.dumps(after['source_input'][key], sort_keys=True) == json.dumps(
                    reviewed['source_input'][key], sort_keys=True)
            assert after['input_fingerprint'] != reviewed['input_fingerprint']
        assert owner.get(base+'/result', params={'generation': before['analysisGeneration']}).json() == before
        report = owner.get(result['exports']['reportJson'])
        assert report.status_code == 200 and report.json() == result
        html = owner.get(result['exports']['readableReport'])
        assert html.status_code == 200 and 'Owner-authorized correction review' in html.text
        assert owner.get(result['exports']['csv']).status_code == 200
        video = owner.get(result['annotatedVideoUrl'])
        snapshot = resolve_generation(output)
        assert video.status_code == 200 and video.headers['x-analysis-generation'] == token
        assert hashlib.sha256(video.content).hexdigest() == snapshot.file_sha256('annotated_web.mp4')
        ranged = owner.get(result['annotatedVideoUrl'], headers={'Range': 'bytes=0-63'})
        assert ranged.status_code == 206 and ranged.content == video.content[:64]
        for client, expected in ((guest, 401), (stranger, 404)):
            for url in (base+'/result', base+'/review-source', base+'/corrections/'+correction_id,
                        result['annotatedVideoUrl'], *result['exports'].values()):
                assert client.get(url).status_code == expected, url
        assert jobs.get('fixture')['status'] == 'COMPLETED'


def test_strict_crop_proxy_retains_swallowed_model_failure_and_forces_selected_options():
    supplied = []
    model = SimpleNamespace(predict=lambda image, **options: supplied.append((image, options)))
    options = correction_worker._crop_options({'imgsz': 640})
    proxy = correction_worker._StrictPoseModel(model, options, lambda: None)
    assert proxy.prediction_calls == 0
    proxy.predict('crop', device='cpu', imgsz=1)
    assert supplied == [('crop', options)]
    assert proxy.prediction_calls == 1

    def failed(*_args, **_kwargs):
        raise RuntimeError('actual crop failure')

    model.predict = failed
    with pytest.raises(RuntimeError, match='actual crop failure'):
        proxy.predict('crop')
    with pytest.raises(RuntimeError, match='Earlier selected-player'):
        proxy.predict('next crop')
    assert str(proxy.failure) == 'Earlier selected-player crop inference failed'
    assert proxy.prediction_calls == 2


@pytest.mark.parametrize('imgsz', [640, [480, 640], (960, 1280)])
def test_crop_defaults_follow_checkpoint_predict_semantics(imgsz):
    # Prior full-frame predictor settings are deliberately different. Pinned
    # predict merges checkpoint overrides and method defaults before kwargs.
    options = correction_worker._crop_options({'imgsz': imgsz, 'task': 'pose', 'single_cls': False})
    assert options['imgsz'] == (imgsz if type(imgsz) is int else list(imgsz))
    assert options['conf'] == .25 and options['batch'] == 1 and options['rect'] is True
    assert options['half'] is False and options['max_det'] == 300 and options['iou'] == .7
    assert options['classes'] is None and options['augment'] is False
    assert options['end2end'] is True and options['nms'] is False and options['device'] == 0
    assert correction_worker._crop_options({})['imgsz'] == 640


@pytest.mark.parametrize('imgsz', [True, 0, 4097, [], [640, True], [640, 640, 640]])
def test_invalid_checkpoint_crop_shape_is_rejected(imgsz):
    with pytest.raises(ValueError, match='crop image size is invalid'):
        correction_worker._crop_options({'imgsz': imgsz})
