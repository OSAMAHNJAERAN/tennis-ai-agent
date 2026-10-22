"""Private pilot inference does not weaken the production qualification gate."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from src.detection import yolo26_bundle as bundle
from scripts.evaluate import run_yolo26_candidate as runner


@pytest.fixture
def private_assets(tmp_path):
    config = yaml.safe_load(Path('configs/production/yolo26.yaml').read_text())
    manifest = copy.deepcopy(bundle.validate_yolo26_config(config))
    manifest.update(status='UNQUALIFIED_PRIVATE_FIXTURE', use_scope='PRIVATE_OFFLINE_EVALUATION',
                    evaluation_only=True, public_redistribution_allowed=False, qualification_receipt=None)
    for role, record in manifest['models'].items():
        weight = tmp_path / f'{role}.pt'
        weight.write_bytes(f'isolated {role} asset fixture'.encode())
        record.update(path=str(weight), sha256=bundle.sha256_file(weight), independent_accuracy_validated=False)
        provenance = tmp_path / f'{role}_provenance.json'
        if role in {'ball', 'court'}:
            document = {'role': role, 'status': 'TRAINED_UNQUALIFIED', 'data_audit': {'final_test_used': False},
                        'checkpoints': [{'path': str(weight), 'sha256': record['sha256']}]}
            key = 'training_provenance'
        else:
            document = {'artifacts': [{'file': str(weight), 'sha256': record['sha256']}]}
            key = 'origin_provenance'
        provenance.write_text(json.dumps(document))
        record[key] = {'path': str(provenance), 'sha256': bundle.sha256_file(provenance)}
    source_video = tmp_path / 'authorized-development.mp4'
    source_video.write_bytes(b'private source fixture')
    source = {'source_id': 'dev', 'source_path': str(source_video), 'sha256': bundle.sha256_file(source_video),
              'training_allowed': True, 'public_redistribution_allowed': False,
              'use_scope': 'PRIVATE_LOCAL_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT_TRAINING',
              'development_split': 'train', 'qualification_evidence': False}
    source_manifest = tmp_path / 'sources.json'
    source_manifest.write_text(json.dumps({'sources': [source]}))
    manifest['evaluation_sources'] = {'path': str(source_manifest), 'sha256': bundle.sha256_file(source_manifest)}
    config['ball_detection']['model'] = manifest['models']['ball']['path']
    config['player_detection']['model'] = manifest['models']['players']['path']
    config['court_detection']['model_path'] = manifest['models']['court']['path']
    manifest_path = tmp_path / 'private-manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    config['model_system']['artifact_manifest'] = str(manifest_path)
    config['event_detection']['authoritative_enabled'] = False
    return config, manifest, manifest_path


def test_asset_integrity_can_be_evaluated_but_default_production_still_rejects_private_weights(private_assets):
    config, manifest, _ = private_assets
    checked = bundle.validate_evaluation_config(config)
    assert 'verified_qualification' not in checked
    assert checked['models']['ball']['sha256'] == manifest['models']['ball']['sha256']
    with pytest.raises(ValueError, match='independent tennis qualification'):
        bundle.validate_yolo26_config(config, check_assets=True)
    with pytest.raises(ValueError, match='independent tennis qualification'):
        bundle.build_yolo26_bundle(config)


def test_asset_hash_mismatch_is_rejected_even_in_explicit_evaluation(private_assets):
    config, manifest, _ = private_assets
    Path(manifest['models']['ball']['path']).write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='SHA256'):
        bundle.validate_evaluation_config(config)


def test_retained_training_receipt_must_bind_the_exact_checkpoint(private_assets):
    config, manifest, path = private_assets
    ref = manifest['models']['court']['training_provenance']
    evidence_path = Path(ref['path'])
    evidence = json.loads(evidence_path.read_text())
    evidence['checkpoints'][0]['sha256'] = '0' * 64
    evidence_path.write_text(json.dumps(evidence))
    ref['sha256'] = bundle.sha256_file(evidence_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not bound'):
        bundle.validate_evaluation_config(config)


@pytest.mark.parametrize('change', ['public', 'qualified', 'authority', 'receipt', 'missing_scope'])
def test_evaluation_rejects_public_or_authoritative_masquerading(private_assets, change):
    config, manifest, path = private_assets
    if change == 'public':
        manifest['public_redistribution_allowed'] = True
    elif change == 'qualified':
        manifest['models']['ball']['independent_accuracy_validated'] = True
    elif change == 'authority':
        config['event_detection']['authoritative_enabled'] = True
    elif change == 'receipt':
        manifest['qualification_receipt'] = {'path': 'forged.json'}
    else:
        manifest.pop('use_scope')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.build_evaluation_bundle(config)


def test_web_worker_never_loads_private_pilot_manifest(private_assets, tmp_path):
    from src.api.worker import check_models, AnalysisError
    config, _, _ = private_assets
    config['model_system']['evaluation_only'] = True  # A config flag cannot activate the constructor argument.
    path = tmp_path / 'worker.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(AnalysisError):
        check_models(path)


def test_pipeline_dispatch_requires_explicit_evaluation_constructor_argument(private_assets, tmp_path, monkeypatch):
    from src.pipeline.phase6_pipeline import Phase6Pipeline
    config, _, _ = private_assets
    path = tmp_path / 'pipeline.yaml'
    path.write_text(yaml.safe_dump(config))
    player = SimpleNamespace(model=object())
    model_bundle = SimpleNamespace(player_detector=player, court_detector=object(), ball_detector=object())
    calls = []
    monkeypatch.setattr(bundle, 'build_yolo26_bundle', lambda config: calls.append('production') or model_bundle)
    monkeypatch.setattr(bundle, 'build_evaluation_bundle', lambda config: calls.append('private') or model_bundle)
    Phase6Pipeline(str(path))
    Phase6Pipeline(str(path), evaluation_only=True)
    assert calls == ['production', 'private']


def test_cli_preflight_preserves_private_assets_and_never_constructs_models(private_assets, tmp_path, monkeypatch):
    config, manifest, path = private_assets
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / 'private_preflight'
    before = bundle.sha256_file(path)
    monkeypatch.setattr(runner, 'resource_guard', lambda output, device: {'device': device, 'free_ram_bytes': 2 * 1024 ** 3})
    monkeypatch.setattr(bundle, '_construct_bundle', lambda *args: pytest.fail('Preflight constructed vision models'))
    result = runner.main(['--config', str(config_path), '--source-id', 'dev', '--output', str(output)])
    assert result['status'] == 'PREFLIGHT_ONLY_UNQUALIFIED'
    assert bundle.sha256_file(path) == before
    assert not (output / 'source.mp4').exists()
    with pytest.raises(ValueError, match='new private run'):
        runner.main(['--config', str(config_path), '--source-id', 'dev', '--output', str(output)])


def test_cli_records_actual_resource_guard_failure(private_assets, tmp_path, monkeypatch):
    config, _, _ = private_assets
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / 'blocked_private_preflight'
    measured = {'free_ram_bytes': 123, 'free_disk_bytes': 234, 'free_vram_bytes': None}
    def blocked(output, device):
        raise runner.ResourceGuardError('fixture resource floor', measured)
    monkeypatch.setattr(runner, 'resource_guard', blocked)
    with pytest.raises(runner.ResourceGuardError):
        runner.main(['--config', str(config_path), '--source-id', 'dev', '--output', str(output)])
    receipt = json.loads((output / 'evaluation_receipt.json').read_text())
    assert receipt['status'] == 'RESOURCE_GUARD_BLOCKED_PRIVATE_UNQUALIFIED'
    assert receipt['resources'] == measured


def test_gpu_guard_rechecks_ram_after_runtime_initialization_and_retains_vram(tmp_path, monkeypatch):
    import psutil
    import torch
    values = iter([2 * 1024 ** 3, 1000 * 1024 ** 2])
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=next(values)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (3 * 1024 ** 3, 6 * 1024 ** 3))
    with pytest.raises(runner.ResourceGuardError, match='1.5 GiB') as blocked:
        runner.resource_guard(tmp_path, '0')
    assert blocked.value.resources['free_ram_bytes'] == 1000 * 1024 ** 2
    assert blocked.value.resources['free_vram_bytes'] == 3 * 1024 ** 3


def test_private_source_manifest_rejects_final_test_relabeling(private_assets):
    _, manifest, _ = private_assets
    path = Path(manifest['evaluation_sources']['path'])
    document = json.loads(path.read_text())
    document['sources'][0]['development_split'] = 'test'
    path.write_text(json.dumps(document))
    manifest['evaluation_sources']['sha256'] = bundle.sha256_file(path)
    with pytest.raises(ValueError, match='private development'):
        runner.evaluation_source(manifest, 'dev')


def test_private_source_selection_rejects_another_id_and_changed_video(private_assets):
    _, manifest, _ = private_assets
    with pytest.raises(ValueError, match='source_id'):
        runner.evaluation_source(manifest, 'another-match')
    source, video = runner.evaluation_source(manifest, 'dev')
    assert source['development_split'] == 'train'
    video.write_bytes(b'different footage')
    with pytest.raises(ValueError, match='video path or SHA256'):
        runner.evaluation_source(manifest, 'dev')


def test_private_source_registry_changes_require_an_explicit_new_pin(private_assets):
    _, manifest, _ = private_assets
    path = Path(manifest['evaluation_sources']['path'])
    document = json.loads(path.read_text())
    document['sources'][0]['source_id'] = 'different-match'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='manifest path or SHA256'):
        runner.evaluation_source(manifest, 'different-match')


def test_runtime_source_binding_records_recursive_hashes_and_detects_all_changes(tmp_path):
    source = tmp_path / 'src'
    (source / 'nested').mkdir(parents=True)
    (source / 'one.py').write_text('value = 1')
    (source / 'nested' / 'two.py').write_text('value = 2')
    before = runner.runtime_source_binding(tmp_path)
    assert set(before) == {'src/one.py', 'src/nested/two.py'}
    assert before['src/one.py'] == bundle.sha256_file(source / 'one.py')
    (source / 'one.py').write_text('value = 3')
    (source / 'nested' / 'two.py').unlink()
    (source / 'three.py').write_text('value = 4')
    changes = runner.binding_changes(before, runner.runtime_source_binding(tmp_path))
    assert changes == {'changed': ['src/one.py'], 'added': ['src/three.py'], 'removed': ['src/nested/two.py']}


def test_owned_resource_sampler_bounds_aggregates_and_stops_after_exception(monkeypatch):
    snapshots = iter([{'available_ram_bytes': 100, 'process_rss_bytes': 20, 'free_vram_bytes': 80},
                      {'available_ram_bytes': 90, 'process_rss_bytes': 30, 'free_vram_bytes': 70}])
    monkeypatch.setattr(runner, 'resource_snapshot', lambda device: next(snapshots))
    sampler = runner.OwnedResourceSampler('0', interval_seconds=60)
    with pytest.raises(LookupError, match='pipeline failure'):
        with sampler:
            raise LookupError('pipeline failure')
    assert not sampler._thread.is_alive()
    assert sampler.summary['samples'] == 2
    assert sampler.summary['fields']['process_rss_bytes'] == {'minimum': 20, 'maximum': 30, 'last': 30}
    assert sampler.summary['fields']['available_ram_bytes']['minimum'] == 90
    assert sampler.summary['fields']['free_vram_bytes']['minimum'] == 70
    assert sampler.summary['observed_peaks_are_sampling_lower_bounds'] is True
    def fail(device):
        raise RuntimeError('sampler unavailable')
    monkeypatch.setattr(runner, 'resource_snapshot', fail)
    for _ in range(100):
        sampler._sample()
    assert sampler.summary['sampling_errors_count'] == 100
    assert len(sampler.summary['sampling_errors']) == 8


def test_native_prefix_preserves_distinct_pts_and_decodes_the_complete_bounded_artifact(tmp_path):
    if not runner.shutil.which('ffmpeg') or not runner.shutil.which('ffprobe'):
        pytest.skip('FFmpeg unavailable')
    video = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 25., (320, 240))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((240, 320, 3), index * 20, np.uint8))
    writer.release()
    output = tmp_path / 'prefix'
    output.mkdir()
    prefix, metadata = runner.native_prefix(video, output, 5)
    assert runner.decoded_count(prefix) == 5
    records = [json.loads(line) for line in (output / 'source_frame_mapping.jsonl').read_text().splitlines()]
    assert [record['source_frame_index'] for record in records] == list(range(5))
    assert [record['source_timestamp_seconds'] for record in records] == [index / 25 for index in range(5)]
    assert metadata['full_source_decode_qualified'] is False


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '1e999'])
def test_private_artifact_check_rejects_nonfinite_json_numbers(tmp_path, value):
    path = tmp_path / 'invalid.json'
    path.write_text('{"value":' + value + '}')
    with pytest.raises(ValueError, match='Nonfinite'):
        runner._finite_json(path)
