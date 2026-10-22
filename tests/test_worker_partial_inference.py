"""Real worker/queue/seals with explicitly injected media and model boundaries.

These tests verify worker routing and immutable source reuse. They do not run
models or prove actual pipeline/process-loss inference recovery.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from src.api import worker
from src.api.jobs import JobStore, Settings
from src.api.pipeline_checkpoint import write_pipeline_checkpoint
from src.api.result_document import ResultDocument
from src.utils.ball_inference_journal import BallInferenceJournal
from src.utils.native_inference_session import (SOURCE_SEAL_NAME, ball_journal_identity,
    inference_identity, load_normalized_source, seal_normalized_source, validate_partial_inference)
from tests.test_native_inference_prefix import audits, write as write_prefix, video_metadata


SCOPE = 'SYNTHETIC_WORKER_ROUTING_ONLY_NO_MEDIA_OR_MODELS'
FRAMES = 3


@pytest.fixture
def partial_worker(tmp_path, monkeypatch):
    config = {'model_system': {'family': 'yolo26'},
              'ball_detection': {'backend': 'yolo26', 'model': 'selected-ball.pt', 'imgsz': 1280},
              'scope': SCOPE}
    config_path = tmp_path / 'candidate.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    settings = Settings(tmp_path / 'storage', config_path)
    store = JobStore(settings.root)
    store.create(store.new_id(), 'synthetic.mp4', '.mp4', 12, config_path, owner_id='partial-owner')
    job = store.claim()
    output = store.output_dir(job['id'])
    output.mkdir(parents=True)
    original = store.upload_dir(job['id']) / 'original.mp4'
    original.parent.mkdir(parents=True)
    original.write_bytes((SCOPE + ':original').encode())
    calls = []
    pipeline_modes = []

    def effective():
        value = deepcopy(config)
        value.setdefault('match_configuration', {})['match_id'] = job['id']
        return value

    def base_identity(claimed=job, current=None, uploaded=original):
        current = effective() if current is None else current
        return {'analysis_id': claimed['id'], 'owner_id': claimed['owner_id'],
                'original_sha256': hashlib.sha256(uploaded.read_bytes()).hexdigest(),
                'effective_config': deepcopy(current), 'scope': SCOPE}

    def identity(claimed, current, uploaded):
        return base_identity(claimed, current, uploaded)

    def execute(args, timeout=7200):
        calls.append(('execute', list(args)))
        if args[0] == 'ffprobe':
            if '-count_frames' in args:
                return json.dumps({'streams': [{'nb_read_frames': str(FRAMES)}]})
            assert timeout == 60 and Path(args[-1]) == original
            return json.dumps({'streams': [{'width': 1920, 'height': 1080,
                'avg_frame_rate': '60000/1001', 'r_frame_rate': '60000/1001',
                'duration': str(FRAMES * 1001 / 60000), 'nb_frames': str(FRAMES),
                'start_time': '0', 'time_base': '1/60000', 'codec_name': 'SYNTHETIC'}]})
        assert args[0] == 'ffmpeg'
        destination = Path(args[-1])
        assert destination.parent == output
        if destination.name == 'source.mp4':
            calls.append(('normalize', job['id']))
            Path(args[args.index('-stats_enc_pre') + 1]).write_text('SYNTHETIC_STATS', encoding='utf-8')
        else:
            assert destination.name == 'annotated_web.mp4'
        destination.write_bytes((SCOPE + ':' + destination.name).encode())
        return ''

    def mapping(stats, destination, *_args):
        assert Path(stats) == output / 'normalization_frame_stats.txt'
        with destination.open('w') as stream:
            for index in range(FRAMES):
                stream.write(json.dumps({'frame_index': index, 'source_pts': str(index * 1001)}) + '\n')
        return {'frames': FRAMES, 'status': 'SYNTHETIC_FRAME_MAP',
                'validMeasurementTransitionIntervals': []}

    def normalized_snapshot():
        return {name: (output / name).read_bytes() for name in
            ('source.mp4', 'media_metadata.json', 'inference_config.yaml', 'source_frame_mapping.jsonl',
             'normalization_frame_stats.txt', SOURCE_SEAL_NAME)}

    def make_partial(*, journal_frames=1):
        source_seal = load_normalized_source(output, base_identity())
        if source_seal is None:
            (output / 'source.mp4').write_bytes((SCOPE + ':source.mp4').encode())
            (output / 'media_metadata.json').write_text(json.dumps({'timeMapping': {'frames': FRAMES}}))
            (output / 'inference_config.yaml').write_text(yaml.safe_dump(effective()))
            (output / 'source_frame_mapping.jsonl').write_text('SYNTHETIC_MAPPING\n')
            (output / 'normalization_frame_stats.txt').write_text('SYNTHETIC_STATS\n')
            source_seal = seal_normalized_source(output, base_identity())
        derived = inference_identity(source_seal)
        audits(output, FRAMES)
        write_prefix(output, count=FRAMES, identity=derived)
        if journal_frames is not None:
            journal_id = ball_journal_identity(output, derived, effective()['ball_detection'], 'cuda:2')
            with BallInferenceJournal.create(output, journal_id, FRAMES) as journal:
                for index in range(journal_frames):
                    journal.append(index, [])
        return validate_partial_inference(output, base_identity(), effective()['ball_detection'])

    def complete_outputs():
        for name in worker.REQUIRED_PIPELINE_FILES:
            path = output / name
            if not path.exists():
                path.write_text(json.dumps({'scope': SCOPE, 'artifact': name}), encoding='utf-8')

    class Pipeline:
        def __init__(self, path, *, inference_identity):
            assert Path(path) == output / 'inference_config.yaml'
            self.identity = inference_identity
            calls.append(('pipeline_init', deepcopy(inference_identity)))

        def run(self, source, destination):
            assert Path(source) == output / 'source.mp4' and Path(destination) == output
            saved = load_normalized_source(output, base_identity())
            assert saved is not None and inference_identity(saved) == self.identity
            mode = pipeline_modes.pop(0) if pipeline_modes else 'complete'
            calls.append(('pipeline_run', mode))
            if mode == 'interrupt_after_prefix':
                partial = make_partial(journal_frames=1)
                assert partial['inference_identity'] == self.identity and partial['next_frame'] == 1
                raise KeyboardInterrupt('Injected worker loss after complete player prefix and one raw frame')
            if mode == 'ordinary_failure':
                raise RuntimeError('Injected ordinary model-boundary failure')
            partial = validate_partial_inference(output, base_identity(), effective()['ball_detection'])
            if partial is not None:
                calls.append(('resumed_frontier', partial['next_frame']))
            complete_outputs()
            if mode == 'mutate_source':
                with (output / 'source.mp4').open('ab') as stream:
                    stream.write(b'changed after validation')

    pipeline_module = ModuleType('src.pipeline.phase6_pipeline')
    pipeline_module.Phase6Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, pipeline_module.__name__, pipeline_module)
    monkeypatch.setattr(worker, 'check_models', lambda path: deepcopy(config))
    monkeypatch.setattr(worker, '_checkpoint_identity', identity)
    monkeypatch.setattr(worker, 'binary', lambda name: name)
    monkeypatch.setattr(worker, 'execute', execute)
    monkeypatch.setattr(worker, 'frame_mapping', mapping)

    def result(directory, claimed, *, _storage):
        assert directory == output and claimed['id'] == job['id'] and _storage is not None
        calls.append(('build_result', claimed['attempt']))
        return {'contractVersion': '2.0', 'scope': SCOPE, 'accuracyQualified': False,
                'summary': {'id': claimed['id'], 'frames': FRAMES, 'fps': video_metadata()['fps'],
                            'durationSeconds': FRAMES / video_metadata()['fps'], 'status': 'COMPLETED'}}

    monkeypatch.setattr(worker, 'build_result', result)
    return SimpleNamespace(store=store, job=job, output=output, settings=settings, config=config,
        original=original, calls=calls, pipeline_modes=pipeline_modes, effective=effective,
        base_identity=base_identity, make_partial=make_partial, complete_outputs=complete_outputs,
        normalized_snapshot=normalized_snapshot)


def test_same_job_recovery_reuses_sealed_source_and_passes_partial_identity(partial_worker, monkeypatch):
    fixture = partial_worker
    fixture.pipeline_modes.append('interrupt_after_prefix')
    with pytest.raises(KeyboardInterrupt, match='Injected worker loss'):
        worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'PROCESSING'
    assert not (fixture.output / '.pipeline_checkpoint.json').exists()
    assert not (fixture.output / '.result/current.store').exists()
    snapshot = fixture.normalized_snapshot()
    journal_bytes = (fixture.output / '.ball_inference.sqlite3').read_bytes()
    restarted = JobStore(fixture.settings.root)
    restarted.recover_interrupted(lambda job: worker.recoverable_completed_job(restarted, job))
    resumed = restarted.claim()
    assert resumed['id'] == fixture.job['id'] and resumed['attempt'] == 2
    monkeypatch.setattr(worker, '_prepare_inference', lambda *args, **kwargs: pytest.fail('Resume normalized again'))
    worker.process_job(restarted, resumed, fixture.settings)
    assert restarted.get(resumed['id'])['status'] == 'COMPLETED'
    assert sum(call[0] == 'normalize' for call in fixture.calls) == 1
    assert ('resumed_frontier', 1) in fixture.calls
    assert fixture.normalized_snapshot() == snapshot
    assert (fixture.output / '.ball_inference.sqlite3').read_bytes() == journal_bytes
    assert len([call for call in fixture.calls if call[0] == 'pipeline_init']) == 2
    assert (fixture.output / '.pipeline_checkpoint.json').is_file()
    with ResultDocument(fixture.output, resumed['id']) as document:
        assert document.value['summary']['frames'] == FRAMES and document.snapshot is not None
        assert document.value['accuracyQualified'] is False


def test_valid_prefix_without_raw_journal_starts_at_zero_without_renormalizing(partial_worker, monkeypatch):
    fixture = partial_worker
    partial = fixture.make_partial(journal_frames=None)
    assert partial['next_frame'] == 0
    snapshot = fixture.normalized_snapshot()
    monkeypatch.setattr(worker, '_prepare_inference', lambda *args, **kwargs: pytest.fail('Prefix resume normalized again'))
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'COMPLETED'
    assert fixture.normalized_snapshot() == snapshot
    assert ('resumed_frontier', 0) in fixture.calls
    assert not any(call[0] == 'normalize' for call in fixture.calls)


def test_complete_marker_has_precedence_over_damaged_partial_state(partial_worker, monkeypatch):
    fixture = partial_worker
    fixture.make_partial()
    fixture.complete_outputs()
    write_pipeline_checkpoint(fixture.output, fixture.base_identity(),
        required_files=worker.REQUIRED_PIPELINE_FILES, optional_files=worker.OPTIONAL_PIPELINE_FILES)
    (fixture.output / '.native_inference_prefix.json').write_text('damaged unused partial marker')
    monkeypatch.setattr(worker, '_load_partial_inference', lambda *args: pytest.fail('Completed run inspected partial state'))
    monkeypatch.setattr(worker, '_prepare_inference', lambda *args, **kwargs: pytest.fail('Completed run repeated inference'))
    assert worker.recoverable_completed_job(fixture.store, fixture.job) is True
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'COMPLETED'
    assert not any(call[0] == 'pipeline_run' for call in fixture.calls)


def test_corrupt_complete_marker_never_falls_back_to_valid_partial_prefix(partial_worker, monkeypatch):
    fixture = partial_worker
    fixture.make_partial()
    (fixture.output / '.pipeline_checkpoint.json').write_text('{}')
    monkeypatch.setattr(worker, '_load_partial_inference', lambda *args: pytest.fail('Corrupt completion fell back'))
    assert worker.recoverable_completed_job(fixture.store, fixture.job) is False
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)


@pytest.mark.parametrize('damage', ['source', 'configuration', 'prefix', 'journal'])
def test_changed_partial_inputs_fail_recovery_without_rewrites(partial_worker, damage):
    fixture = partial_worker
    fixture.make_partial()
    if damage == 'source':
        (fixture.output / 'source.mp4').write_bytes(b'changed retained source')
    elif damage == 'configuration':
        fixture.config['ball_detection']['imgsz'] = 640
    elif damage == 'prefix':
        (fixture.output / '.native_inference_prefix.json').write_text('{}')
    else:
        with sqlite3.connect(fixture.output / '.ball_inference.sqlite3') as connection:
            connection.execute("UPDATE raw_frames SET payload='[ ]' WHERE frame_index=0")
    observed = {path.name: path.read_bytes() for path in fixture.output.iterdir() if path.is_file()}
    assert worker.recoverable_completed_job(fixture.store, fixture.job) is False
    fixture.store.recover_interrupted(lambda job: worker.recoverable_completed_job(fixture.store, job))
    assert fixture.store.claim() is None
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)
    assert all((fixture.output / name).read_bytes() == data for name, data in observed.items()
               if name != 'processing.log')


def test_source_change_during_resumed_pipeline_cannot_seal_or_publish(partial_worker):
    fixture = partial_worker
    fixture.make_partial()
    fixture.pipeline_modes.append('mutate_source')
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not (fixture.output / '.pipeline_checkpoint.json').exists()
    assert not (fixture.output / '.result/current.store').exists()


def test_source_change_between_post_pipeline_check_and_completed_hashing_cannot_publish(partial_worker, monkeypatch):
    from src.api import pipeline_checkpoint
    fixture = partial_worker
    fixture.make_partial()
    original_write = pipeline_checkpoint.write_pipeline_checkpoint
    mutations = []

    def mutate_before_hashing(directory, identity, **kwargs):
        assert Path(directory) == fixture.output
        # The post-pipeline source-seal check already succeeded. The completed
        # checkpoint will deliberately bind these changed bytes, exercising
        # the separate normalized-source guard at final publication.
        with (fixture.output / 'source.mp4').open('ab') as stream:
            stream.write(b'changed between source verification and completed hashing')
        mutations.append(True)
        return original_write(directory, identity, **kwargs)

    monkeypatch.setattr(pipeline_checkpoint, 'write_pipeline_checkpoint', mutate_before_hashing)
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert mutations == [True]
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert (fixture.output / '.pipeline_checkpoint.json').is_file()
    assert not (fixture.output / '.result/current.store').exists()
    assert ('build_result', fixture.job['attempt']) in fixture.calls


@pytest.mark.parametrize('change', ['modify', 'remove'])
def test_completed_native_recovery_rechecks_its_existing_source_seal_at_publication(partial_worker, monkeypatch, change):
    fixture = partial_worker
    fixture.make_partial()
    fixture.complete_outputs()
    write_pipeline_checkpoint(fixture.output, fixture.base_identity(),
        required_files=worker.REQUIRED_PIPELINE_FILES, optional_files=worker.OPTIONAL_PIPELINE_FILES)
    original_builder = worker.build_result
    mutations = []

    def mutate_seal_after_completed_admission(*args, **kwargs):
        result = original_builder(*args, **kwargs)
        marker = fixture.output / SOURCE_SEAL_NAME
        if change == 'modify':
            marker.write_text('{}')
        else:
            marker.rename(fixture.output / 'retained_source_seal_before_publication.json')
        mutations.append(True)
        return result

    monkeypatch.setattr(worker, 'build_result', mutate_seal_after_completed_admission)
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert mutations == [True]
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not (fixture.output / '.result/current.store').exists()
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)


def test_older_completed_native_checkpoint_without_source_seal_keeps_completed_only_contract(partial_worker, monkeypatch):
    fixture = partial_worker
    fixture.make_partial()
    fixture.complete_outputs()
    write_pipeline_checkpoint(fixture.output, fixture.base_identity(),
        required_files=worker.REQUIRED_PIPELINE_FILES, optional_files=worker.OPTIONAL_PIPELINE_FILES)
    (fixture.output / SOURCE_SEAL_NAME).rename(fixture.output / 'retained_unrequired_source_seal.json')
    monkeypatch.setattr(worker, '_load_partial_inference', lambda *args: pytest.fail('Old completion inspected partial state'))
    monkeypatch.setattr(worker, '_prepare_inference', lambda *args, **kwargs: pytest.fail('Old completion repeated inference'))
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'COMPLETED'
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)


def test_missing_source_seal_with_retained_prefix_does_not_restart_normalization(partial_worker, monkeypatch):
    fixture = partial_worker
    fixture.make_partial()
    (fixture.output / SOURCE_SEAL_NAME).rename(fixture.output / 'retained_source_seal.json')
    monkeypatch.setattr(worker, '_prepare_inference', lambda *args, **kwargs: pytest.fail('Invalid prefix normalized again'))
    assert worker.recoverable_completed_job(fixture.store, fixture.job) is False
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)


@pytest.mark.parametrize('reason', ['cancelled', 'cancel_during_validation', 'attempt_limit'])
def test_partial_recovery_preserves_real_queue_cancellation_and_attempt_limit(partial_worker, monkeypatch, reason):
    fixture = partial_worker
    fixture.make_partial()
    validations = []
    original = worker._load_partial_inference

    def validate(*args):
        validations.append(True)
        if reason == 'cancel_during_validation':
            fixture.store.cancel(fixture.job['id'], 'partial-owner')
        return original(*args)

    monkeypatch.setattr(worker, '_load_partial_inference', validate)
    if reason == 'cancelled':
        fixture.store.cancel(fixture.job['id'], 'partial-owner')
    elif reason == 'attempt_limit':
        with fixture.store.connect() as connection:
            connection.execute('UPDATE analyses SET attempt=3 WHERE id=?', (fixture.job['id'],))
    fixture.store.recover_interrupted(lambda job: worker.recoverable_completed_job(fixture.store, job))
    assert fixture.store.claim() is None
    saved = fixture.store.get(fixture.job['id'])
    assert saved['status'] == 'FAILED'
    assert bool(validations) is (reason == 'cancel_during_validation')
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)


def test_ordinary_pipeline_exception_is_failed_without_automatic_recovery(partial_worker):
    fixture = partial_worker
    fixture.make_partial()
    fixture.pipeline_modes.append('ordinary_failure')
    worker.process_job(fixture.store, fixture.job, fixture.settings)
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    fixture.store.recover_interrupted(lambda job: pytest.fail('FAILED job was reconsidered for restart'))
    assert fixture.store.claim() is None
    assert not (fixture.output / '.pipeline_checkpoint.json').exists()


@pytest.mark.parametrize('config,enabled', [
    ({'model_system': {'family': 'yolo26'}, 'ball_detection': {'backend': 'yolo26'}}, True),
    ({'model_system': {'family': 'yolo26'}}, False),
    ({'model_system': {'family': 'yolo26'}, 'ball_detection': {'backend': 'wasb'}}, False),
    ({'model_system': {'family': 'historical'}, 'ball_detection': {'backend': 'yolo26'}}, False),
])
def test_partial_contract_is_enabled_only_for_native_ball_pipeline(config, enabled):
    assert worker._partial_inference_enabled(config) is enabled


def test_loss_before_completed_player_prefix_is_not_recoverable(partial_worker):
    fixture = partial_worker
    (fixture.output / 'source.mp4').write_bytes(b'unsealed-prefix-source')
    (fixture.output / 'media_metadata.json').write_text(json.dumps({'timeMapping': {'frames': FRAMES}}))
    (fixture.output / 'inference_config.yaml').write_text(yaml.safe_dump(fixture.effective()))
    seal_normalized_source(fixture.output, fixture.base_identity())
    assert worker.recoverable_completed_job(fixture.store, fixture.job) is False
    fixture.store.recover_interrupted(lambda job: worker.recoverable_completed_job(fixture.store, job))
    assert fixture.store.get(fixture.job['id'])['status'] == 'FAILED'
    assert not any(call[0] in ('normalize', 'pipeline_run') for call in fixture.calls)
