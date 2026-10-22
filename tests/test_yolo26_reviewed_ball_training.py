"""Pure training-boundary checks; synthetic fixtures are never model evidence.

Run with --noconftest and plugin autoload disabled to avoid the repository's
image fixtures. No test decodes an image, initializes CUDA or imports ML code.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.train import train_yolo26_reviewed_ball as training


FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'PIL', 'psutil'}
EXACT_COUNTS = {'images': 79, 'positive_boxes': 75, 'negative_tiles': 4,
                'train': 58, 'val': 21, 'test': 0,
                'positive_split_counts': {'train': 55, 'development': 20}}
SOURCE = Path(training.__file__)
GENERIC_SOURCE = training.ROOT / 'scripts/train/train_yolo26_tennis.py'


@pytest.fixture
def export_boundary(tmp_path, monkeypatch):
    """Stub only the deep verifier; the trainer still rehashes manifest bytes."""
    from scripts.data import export_yolo26_reviewed_ball_wave as exporter

    root = tmp_path / 'workspace'
    root.mkdir()
    monkeypatch.setattr(training, 'ROOT', root)
    path = root / 'data/source_manifest.json'
    path.parent.mkdir()
    path.write_bytes(b'{"synthetic_export":true}\n')
    weights = root / 'official/yolo26m.pt'
    weights.parent.mkdir()
    weights.write_bytes(b'synthetic initializer; never load as a model')
    monkeypatch.setattr(training, 'OFFICIAL_WEIGHTS', weights)
    monkeypatch.setattr(training, 'OFFICIAL_WEIGHTS_SHA256', training.digest(weights))
    samples = [{'sample_id': f'synthetic_{index:03}',
                'development_split': 'train' if index < 58 else 'val',
                'training_allowed': True, 'match_id': None, 'session_id': None}
               for index in range(79)]
    result = {'manifest': {'status': 'EXPORTED_PRIVATE_AGENT_REVIEWED_BALL_DEVELOPMENT_NOT_QUALIFICATION',
                          'counts': deepcopy(EXACT_COUNTS), 'qualification_evidence': False,
                          'final_test_used': False, 'independent_human_review': False,
                          'source_independence_verified': False},
              'counts': deepcopy(EXACT_COUNTS), 'samples': samples,
              'receipt_path': path.parent / 'export_receipt.json', 'receipt_sha256': '1' * 64,
              'data_yaml_path': path.parent / 'ball/data.yaml', 'data_yaml_sha256': '2' * 64}
    calls = []

    def verify(output, *, expected_manifest_sha256):
        calls.append((output, expected_manifest_sha256))
        return deepcopy(result)

    monkeypatch.setattr(exporter, 'verify_export', verify)
    monkeypatch.setattr(training.importlib.metadata, 'version', lambda name: training.PINNED_RUNTIME[name])
    return SimpleNamespace(root=root, manifest=path, manifest_sha=training.digest(path),
                           weights=weights, result=result, calls=calls,
                           output=root / 'outputs/fresh_run')


def test_import_is_stdlib_only_even_when_heavy_runtime_is_forbidden(monkeypatch):
    actual_import = builtins.__import__
    attempted = []

    def guard(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            attempted.append(name)
            raise AssertionError(f'Unexpected heavy import {name}')
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    spec = importlib.util.spec_from_file_location('synthetic_pure_ball_trainer_import', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert attempted == []
    assert module.EXPECTED_COUNTS == EXACT_COUNTS


def test_audit_uses_exact_deep_export_verifier_and_keeps_unknown_sessions(export_boundary):
    case = export_boundary
    samples, audit = training.audit_export(case.manifest, case.manifest_sha)
    assert case.calls == [(case.manifest.parent, case.manifest_sha)]
    assert len(samples) == 79
    assert all(row['match_id'] is None and row['session_id'] is None for row in samples)
    assert audit['counts'] == EXACT_COUNTS
    assert audit['usage_scope'] == 'PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT_ONLY'
    assert audit['raw_redistribution_cleared'] is False
    assert 'NOT_FULL_FRAME_ACCURACY' in audit['validation_scope']
    for key in ('qualification_evidence', 'final_test_used', 'independent_human_review',
                'source_independence_verified'):
        assert audit[key] is False


@pytest.mark.parametrize('token', [None, '', 'a' * 63, 'a' * 65, 'A' * 64, 'g' * 64, 123, True])
def test_manifest_sha_requires_exact_lowercase_token_before_verifier(export_boundary, token):
    with pytest.raises(ValueError, match='SHA256'):
        training.audit_export(export_boundary.manifest, token)
    assert export_boundary.calls == []


@pytest.mark.parametrize('change', ['bytes', 'filename'])
def test_changed_or_renamed_manifest_never_reaches_deep_verifier(export_boundary, change):
    case = export_boundary
    path = case.manifest
    if change == 'bytes':
        path.write_bytes(path.read_bytes() + b'changed after review')
        expected = case.manifest_sha
    else:
        path = path.with_name('unreviewed.json')
        path.write_bytes(case.manifest.read_bytes())
        expected = training.digest(path)
    with pytest.raises(ValueError, match='exact final ROOT-reviewed'):
        training.audit_export(path, expected)
    assert case.calls == []


@pytest.mark.parametrize(('section', 'field', 'value'), [
    ('manifest', 'status', 'QUALIFIED_PRODUCTION'),
    ('manifest', 'qualification_evidence', True),
    ('manifest', 'qualification_evidence', None),
    ('manifest', 'final_test_used', True),
    ('manifest', 'independent_human_review', True),
    ('manifest', 'source_independence_verified', True),
    ('counts', 'test', 1), ('counts', 'images', 80), ('counts', 'negative_tiles', 5),
    ('counts', 'train', 57), ('counts', 'val', 22), ('counts', 'positive_boxes', 74),
])
def test_wrapper_rejects_widened_claims_and_sample_counts(export_boundary, section, field, value):
    case = export_boundary
    case.result[section][field] = value
    with pytest.raises(ValueError, match='exact79sample'):
        training.audit_export(case.manifest, case.manifest_sha)


@pytest.mark.parametrize(('field', 'value'), [('development_split', 'test'),
                                           ('development_split', 'development'),
                                           ('training_allowed', False),
                                           ('training_allowed', 1)])
def test_wrapper_cannot_admit_test_or_unapproved_rows(export_boundary, field, value):
    case = export_boundary
    case.result['samples'][0][field] = value
    with pytest.raises(ValueError, match='authorized train/DEV'):
        training.audit_export(case.manifest, case.manifest_sha)


def test_missing_sample_rejected_even_when_reported_counts_are_unchanged(export_boundary):
    case = export_boundary
    case.result['samples'].pop()
    with pytest.raises(ValueError, match='authorized train/DEV'):
        training.audit_export(case.manifest, case.manifest_sha)


def test_main_default_rehashes_official_weights_and_writes_nothing(export_boundary, monkeypatch, capsys):
    case = export_boundary
    actual_import = builtins.__import__
    attempts = []

    def guard(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            attempts.append(name)
            pytest.fail(f'Pure preflight attempted {name}')
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    assert training.main(['--manifest', str(case.manifest), '--manifest-sha256', case.manifest_sha,
                          '--output', str(case.output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'PURE_DATA_PREFLIGHT_PASSED_NOT_TRAINED'
    assert report['resource_guard_pending'] is True
    assert report['output_files_written'] == 0
    assert report['initial_weights']['sha256'] == training.digest(case.weights)
    assert report['training_arguments']['epochs'] == 20
    assert report['qualification_evidence'] is False
    assert report['final_test_used'] is False
    assert not case.output.exists()
    assert attempts == []


def test_changed_initializer_cannot_reach_execution(export_boundary, monkeypatch):
    case = export_boundary
    case.weights.write_bytes(b'old pilot or substituted initializer')
    monkeypatch.setattr(training, 'execute_training', lambda *args: pytest.fail('Invalid initializer executed'))
    with pytest.raises(ValueError, match='official YOLO26m'):
        training.main(['--manifest', str(case.manifest), '--manifest-sha256', case.manifest_sha,
                       '--output', str(case.output), '--execute'])
    assert not case.output.exists()


@pytest.mark.parametrize('extra', [['--epochs', '0'], ['--epochs', '21'], ['--epochs', '100'],
                                 ['--seed', '-1'], ['--batch', '2'], ['--image-size', '1280'],
                                 ['--device', 'cpu'], ['--weights', 'other.pt'], ['--role', 'court']])
def test_invalid_cli_cannot_touch_data_or_start_runtime(export_boundary, monkeypatch, extra):
    monkeypatch.setattr(training, 'audit_export', lambda *args: pytest.fail('Invalid CLI audited data'))
    with pytest.raises((SystemExit, ValueError)):
        training.main(['--manifest-sha256', '0' * 64, '--output', str(export_boundary.output), *extra])
    assert not export_boundary.output.exists()


@pytest.mark.parametrize('epochs', [False, True, 1.0, None, -1, 0, 21])
def test_initial_epoch_ceiling_cannot_be_bypassed_with_noninteger_values(tmp_path, epochs):
    with pytest.raises(ValueError, match='1-20'):
        training.training_settings(tmp_path, epochs=epochs)


@pytest.mark.parametrize('seed', [False, True, 1.0, None, -1])
def test_seed_rejects_ambiguous_values(tmp_path, seed):
    with pytest.raises(ValueError, match='seed'):
        training.training_settings(tmp_path, seed=seed)


def test_fixed_settings_preserve_native_pixels_and_disable_implicit_augmentation(tmp_path):
    settings = training.training_settings(tmp_path)
    for key in ('mosaic', 'mixup', 'cutmix', 'copy_paste', 'scale', 'translate', 'degrees', 'shear',
                'perspective', 'fliplr', 'flipud', 'bgr', 'multi_scale', 'erasing', 'warmup_epochs'):
        assert settings[key] == 0
    assert settings['augmentations'] == []
    assert settings['auto_augment'] is None
    assert settings['rect'] is False
    assert settings['imgsz'] == 640 and settings['batch'] == 1
    assert settings['workers'] == 0 and settings['cache'] is False and settings['amp'] is False
    assert settings['nbs'] == 1 and settings['optimizer'] == 'AdamW' and settings['lr0'] == .0005
    assert settings['save'] is True and settings['save_period'] == 5 and settings['exist_ok'] is False


@pytest.mark.parametrize('path_kind', ['outside', 'outputs_root', 'sibling_prefix'])
def test_output_cannot_escape_fixed_workspace_root(export_boundary, path_kind):
    case = export_boundary
    candidate = {'outside': case.root / 'data/overwrite', 'outputs_root': case.root / 'outputs',
                 'sibling_prefix': case.root / 'outputs_backup/run'}[path_kind]
    with pytest.raises(ValueError, match='outputs root'):
        training.validate_fresh_output(candidate)


def test_existing_empty_output_is_preserved_including_resume(export_boundary):
    export_boundary.output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='including for resume'):
        training.validate_fresh_output(export_boundary.output)
    assert list(export_boundary.output.iterdir()) == []


@pytest.mark.parametrize('package', ['torch', 'ultralytics'])
def test_runtime_pin_uses_metadata_and_refuses_version_drift(monkeypatch, package):
    monkeypatch.setattr(training.importlib.metadata, 'version',
                        lambda name: 'changed' if name == package else training.PINNED_RUNTIME[name])
    with pytest.raises(ValueError, match='pinned installed runtime'):
        training.runtime_versions()


@pytest.fixture
def resume_boundary(tmp_path):
    checkpoint = tmp_path / 'prior_run/training/weights/epoch5.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'synthetic optimizer-bearing checkpoint placeholder')
    settings = training.training_settings(tmp_path / 'fresh_run')
    data = {'manifest_path': str(tmp_path / 'source_manifest.json'), 'manifest_sha256': 'a' * 64}
    metadata = {'mode': training.MODE, 'role': 'ball', 'data_audit': deepcopy(data),
                'runtime_versions': deepcopy(training.PINNED_RUNTIME), 'final_test_used': False,
                'qualification_evidence': False, 'training_arguments': deepcopy(settings),
                'checkpoints': [{'path': str(checkpoint), 'sha256': training.digest(checkpoint)}]}
    metadata['training_arguments'].update(data='old_absolute_dataset.yaml', project='old_output', name='old_name')
    metadata_path = checkpoint.parents[2] / 'run_metadata.json'

    def save():
        metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
        return metadata_path

    save()
    return SimpleNamespace(checkpoint=checkpoint, settings=settings, data=data, metadata=metadata,
                           save=save, metadata_path=metadata_path)


def test_resume_accepts_same_epoch_budget_into_fresh_output_without_loading_model(resume_boundary):
    case = resume_boundary
    result = training.audit_resume(case.checkpoint, training.digest(case.checkpoint), case.settings, case.data)
    assert result['checkpoint_sha256'] == training.digest(case.checkpoint)
    assert result['metadata_sha256'] == training.digest(case.metadata_path)
    assert result['metadata'] == case.metadata


@pytest.mark.parametrize(('key', 'value'), [('epochs', 21), ('imgsz', 1280), ('seed', 710),
                                         ('lr0', .001), ('batch', 2), ('augmentations', None),
                                         ('resume', True)])
def test_resume_cannot_change_training_or_extend_total_epochs(resume_boundary, key, value):
    case = resume_boundary
    case.settings[key] = value
    with pytest.raises(ValueError, match='same total epochs'):
        training.audit_resume(case.checkpoint, training.digest(case.checkpoint), case.settings, case.data)


@pytest.mark.parametrize(('key', 'value'), [('mode', 'QUALIFIED'), ('role', 'court'),
                                         ('final_test_used', True), ('qualification_evidence', True),
                                         ('runtime_versions', {'ultralytics': 'other'})])
def test_resume_preserves_private_mode_role_runtime_and_no_test_claim(resume_boundary, key, value):
    case = resume_boundary
    case.metadata[key] = value
    case.save()
    with pytest.raises(ValueError, match='same exact reviewed export'):
        training.audit_resume(case.checkpoint, training.digest(case.checkpoint), case.settings, case.data)


@pytest.mark.parametrize('field', ['manifest_path', 'manifest_sha256'])
def test_resume_cannot_substitute_reviewed_dataset(resume_boundary, field):
    case = resume_boundary
    case.metadata['data_audit'][field] = 'changed'
    case.save()
    with pytest.raises(ValueError, match='same exact reviewed export'):
        training.audit_resume(case.checkpoint, training.digest(case.checkpoint), case.settings, case.data)


@pytest.mark.parametrize('change', ['unregistered', 'registration_hash', 'changed_bytes', 'wrong_layout'])
def test_resume_requires_original_hash_registered_checkpoint(resume_boundary, change):
    case = resume_boundary
    expected = training.digest(case.checkpoint)
    checkpoint = case.checkpoint
    if change == 'unregistered':
        case.metadata['checkpoints'] = []
    elif change == 'registration_hash':
        case.metadata['checkpoints'][0]['sha256'] = '0' * 64
    elif change == 'changed_bytes':
        checkpoint.write_bytes(b'checkpoint changed since registration')
    else:
        checkpoint = checkpoint.parents[2] / 'unregistered.pt'
        checkpoint.write_bytes(case.checkpoint.read_bytes())
    case.save()
    with pytest.raises(ValueError, match='(registered|SHA256 differs)'):
        training.audit_resume(checkpoint, expected, case.settings, case.data)


def _checkpoint():
    return {'epoch': 5, 'optimizer': {'synthetic_state': True}, 'ema': object(), 'updates': 348,
            'train_args': {'epochs': 20}}


def test_resume_checkpoint_proves_unfinished_optimizer_and_ema_state_without_loading_it():
    assert training.validate_checkpoint_state(_checkpoint(), 20) == 6


@pytest.mark.parametrize(('key', 'value'), [('epoch', -1), ('epoch', True), ('epoch', 19), ('epoch', 20),
                                         ('optimizer', None), ('ema', None), ('updates', -1),
                                         ('updates', True), ('updates', 3.5),
                                         ('train_args', {'epochs': 21})])
def test_stripped_completed_or_inconsistent_resume_state_is_rejected(key, value):
    state = _checkpoint()
    state[key] = value
    with pytest.raises(ValueError, match='unfinished unstripped'):
        training.validate_checkpoint_state(state, 20)


def _specialist():
    return SimpleNamespace(task='detect', model=SimpleNamespace(
        yaml={'scale': 'm', 'nc': 1, 'yaml_file': 'yolo26m-tennis-ball.yaml'},
        end2end=True, names={0: 'tennis-ball'}))


def test_resume_model_is_selected_yolo26m_specialist():
    training.validate_resume_model(_specialist())


@pytest.mark.parametrize(('field', 'value'), [('task', 'pose'), ('scale', 's'), ('nc', 2), ('nc', True),
                                           ('yaml_file', 'yolo11m.yaml'), ('yaml_file', 'yolo26s.yaml'),
                                           ('yaml_file', None), ('end2end', False), ('end2end', 1),
                                           ('names', {0: 'person'}), ('names', {0: 'tennis-ball', 1: 'racket'})])
def test_resume_does_not_accept_foreign_task_architecture_head_or_classes(field, value):
    model = _specialist()
    if field == 'task':
        model.task = value
    elif field in {'scale', 'nc', 'yaml_file'}:
        model.model.yaml[field] = value
    else:
        setattr(model.model, field, value)
    with pytest.raises(ValueError, match='end-to-end YOLO26m'):
        training.validate_resume_model(model)


def test_resume_trainer_restores_new_output_after_library_resets_arguments(tmp_path, monkeypatch):
    class FakeDetectionTrainer:
        def check_resume(self, overrides):
            self.args = SimpleNamespace(project='old', name='old', save_dir='old/training', epochs=20,
                                        resume='old/checkpoint.pt')
            self.resume = True

    fake = ModuleType('ultralytics.models.yolo.detect')
    fake.DetectionTrainer = FakeDetectionTrainer
    monkeypatch.setitem(sys.modules, 'ultralytics.models.yolo.detect', fake)
    settings = training.training_settings(tmp_path / 'fresh')
    trainer = training._resume_trainer(settings)()
    trainer.check_resume({})
    assert trainer.args.save_dir == str(tmp_path / 'fresh/training')
    assert trainer.args.project == str(tmp_path / 'fresh')
    assert trainer.args.epochs == 20 and trainer.resume is True
    assert trainer.args.resume == 'old/checkpoint.pt'


def _write_source_tree(root):
    for relative in ('scripts/train/train_yolo26_reviewed_ball.py', 'scripts/train/train_yolo26_tennis.py',
                     'scripts/data/export_yolo26_reviewed_ball_wave.py',
                     'scripts/data/prepare_yolo26_reviewed_ball_wave.py',
                     'scripts/data/prepare_yolo26_ball_review_wave.py', 'src/later/not_imported.py'):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source fixture\n', encoding='utf-8')


def test_source_snapshot_includes_code_that_has_not_been_imported(tmp_path, monkeypatch):
    _write_source_tree(tmp_path)
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    before = training._source_bindings()
    late = tmp_path / 'src/later/not_imported.py'
    assert str(late) in before
    assert 'src.later.not_imported' not in sys.modules
    late.write_text('# changed before its future import\n', encoding='utf-8')
    added = tmp_path / 'src/added.py'
    added.write_text('# added during training\n', encoding='utf-8')
    changes = training._binding_changes(before, training._source_bindings())
    assert changes == {'changed': [str(late)], 'added': [str(added)], 'removed': []}


def test_runtime_snapshot_covers_unimported_code_yaml_and_detects_removed_files(tmp_path):
    python = tmp_path / 'engine/later.py'
    python.parent.mkdir()
    python.write_bytes(b'# source')
    yaml = tmp_path / 'model.yaml'
    yaml.write_bytes(b'nc: 1\n')
    (tmp_path / 'irrelevant.bin').write_bytes(b'not a runtime source/configuration')
    before = training._runtime_source_bindings(tmp_path)
    assert set(before) == {str(python), str(yaml)}
    python.unlink()
    yaml.write_bytes(b'nc: 2\n')
    added = tmp_path / 'added.yaml'
    added.write_bytes(b'scale: s\n')
    changes = training._binding_changes(before, training._runtime_source_bindings(tmp_path))
    assert changes == {'changed': [str(yaml)], 'added': [str(added)], 'removed': [str(python)]}


@pytest.mark.parametrize('missing', [False, True])
def test_absent_runtime_source_tree_cannot_claim_integrity(tmp_path, missing):
    with pytest.raises(ValueError, match='(missing|no code/configuration)'):
        training._runtime_source_bindings(tmp_path / 'missing' if missing else tmp_path)


def test_copy_keeps_empty_negative_labels_and_cannot_overwrite_prior_copies(tmp_path):
    samples = []
    for ident, split, label_bytes in [('positive', 'train', b'0 .5 .5 .01 .01\n'), ('negative', 'val', b'')]:
        image = tmp_path / (ident + '.png')
        label = tmp_path / (ident + '.txt')
        image.write_bytes(b'opaque native image bytes; no decoder used')
        label.write_bytes(label_bytes)
        samples.append({'sample_id': ident, 'development_split': split, 'image_path': str(image),
                        'image_source_sha256': training.digest(image), 'label_path': str(label),
                        'label_sha256': training.digest(label), 'negative_visible_absence_only': not label_bytes})
    output = tmp_path / 'new_run'
    records, bindings = training._copy_training_inputs(samples, output)
    assert [row['instances'] for row in records] == [1, 0]
    assert len(bindings) == 4
    for row in bindings:
        assert Path(row['path']).read_bytes() == Path(row['source_path']).read_bytes()
    with pytest.raises(FileExistsError):
        training._copy_training_inputs(samples, output)


def test_copy_rejects_changed_export_before_creating_target(tmp_path):
    image = tmp_path / 'reviewed.png'
    image.write_bytes(b'changed image')
    sample = {'sample_id': 'positive', 'development_split': 'train', 'image_path': str(image),
              'image_source_sha256': '0' * 64, 'label_path': str(tmp_path / 'label.txt'),
              'label_sha256': '1' * 64, 'negative_visible_absence_only': False}
    with pytest.raises(ValueError, match='Exported source changed'):
        training._copy_training_inputs([sample], tmp_path / 'new_run')
    assert not (tmp_path / 'new_run/training_inputs/ball/images/train/positive.png').exists()


@pytest.fixture
def synthetic_execution(tmp_path, monkeypatch):
    """Exercise execution control flow using inert stand-ins, never real ML/CUDA."""
    root = tmp_path / 'workspace'
    _write_source_tree(root)
    monkeypatch.setattr(training, 'ROOT', root)
    output = root / 'outputs/synthetic_run'
    weights = root / 'models/yolo26/yolo26m.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'synthetic initial weights')
    monkeypatch.setattr(training, 'OFFICIAL_WEIGHTS', weights)
    monkeypatch.setattr(training, 'OFFICIAL_WEIGHTS_SHA256', training.digest(weights))
    events = []
    state = SimpleNamespace(available=1024 ** 3, vram=2 * 1024 ** 3, hook=None, cuda_available=True)
    runtime_root = tmp_path / 'synthetic_ultralytics'
    runtime_root.mkdir()
    (runtime_root / '__init__.py').write_bytes(b'# synthetic inert runtime')
    (runtime_root / 'model.yaml').write_bytes(b'nc: 1\n')

    def prepare(records, role, directory, *, pilot):
        events.append('prepare_dataset')
        assert role == 'ball' and pilot is True
        for split in ('train', 'val'):
            (directory / f'{split}.txt').write_text('\n'.join(row['image'] for row in records if row['split'] == split), encoding='utf-8')
        data = directory / 'dataset.yaml'
        data.write_text('train: train.txt\nval: val.txt\nnames: [tennis-ball]\n', encoding='utf-8')
        return data

    class FakeModel:
        def __init__(self):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def train(self, **arguments):
            events.append('model_train')
            assert arguments['augmentations'] == [] and arguments['amp'] is False
            weights_dir = output / 'training/weights'
            weights_dir.mkdir(parents=True)
            last = weights_dir / 'last.pt'
            trainer = SimpleNamespace(epoch=0, epochs=1, stop=False, wdir=weights_dir, last=last,
                                      optimizer=SimpleNamespace(state={1: {'step': 58}}),
                                      train_loader=[None] * 58, accumulate=1,
                                      args=SimpleNamespace(warmup_epochs=0.))
            self.callbacks['on_train_batch_end'](trainer)
            self.callbacks['on_train_epoch_end'](trainer)
            last.write_bytes(b'synthetic unstripped optimizer-bearing bytes')
            self.callbacks['on_model_save'](trainer)
            last.write_bytes(b'synthetic stripped final bytes')
            (output / 'training/results.csv').write_bytes(b'epoch,train/box_loss\n1,1\n')
            if state.hook:
                state.hook()

    def build(role, initializer, directory):
        events.append('construct_model')
        assert role == 'ball' and initializer == weights and directory == output
        return FakeModel(), {'synthetic_only': True}

    # Compile the actual unchanged guard function alone, avoiding the generic
    # module's top-level NumPy dependency. This does not replace guard logic.
    guard_node = next(node for node in ast.parse(GENERIC_SOURCE.read_text(encoding='utf-8')).body
                      if isinstance(node, ast.FunctionDef) and node.name == 'check_resource_guard')
    namespace = {}
    exec(compile(ast.Module(body=[guard_node], type_ignores=[]), str(GENERIC_SOURCE), 'exec'), namespace)

    def guard(resources, **kwargs):
        events.append('resource_guard')
        namespace['check_resource_guard'](resources, **kwargs)

    generic = ModuleType('scripts.train.train_yolo26_tennis')
    generic.build_training_model = build
    generic.check_resource_guard = guard
    generic.prepare_dataset = prepare
    monkeypatch.setitem(sys.modules, generic.__name__, generic)
    psutil = ModuleType('psutil')
    psutil.Process = lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=64 * 1024 ** 2))
    psutil.virtual_memory = lambda: SimpleNamespace(available=state.available)
    monkeypatch.setitem(sys.modules, 'psutil', psutil)
    torch = ModuleType('torch')
    torch.__version__ = 'SYNTHETIC_NO_MODEL_OR_GPU_EXECUTED'
    torch.version = SimpleNamespace(cuda='SYNTHETIC')
    torch.cuda = SimpleNamespace(is_available=lambda: state.cuda_available,
        set_device=lambda index: events.append('cuda_set_device'), init=lambda: events.append('cuda_initialized'),
        get_device_name=lambda index: 'SYNTHETIC_NO_GPU', mem_get_info=lambda index: (state.vram, 6 * 1024 ** 3),
        reset_peak_memory_stats=lambda index: events.append('synthetic_reset_peak'),
        max_memory_allocated=lambda index: 0, max_memory_reserved=lambda index: 0)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    ultralytics = ModuleType('ultralytics')
    ultralytics.__file__ = str(runtime_root / '__init__.py')
    monkeypatch.setitem(sys.modules, 'ultralytics', ultralytics)
    monkeypatch.setattr(training.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='synthetic_revision'))
    monkeypatch.setattr(training.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024 ** 3))
    monkeypatch.setattr(training, 'audit_export', lambda *args: ([], {'synthetic_reaudit_only': True}))
    samples = []
    for index, split in enumerate(('train', 'val')):
        image = root / f'synthetic_source_{index}.png'
        label = root / f'synthetic_source_{index}.txt'
        image.write_bytes(b'opaque native crop bytes')
        label.write_bytes(b'0 .5 .5 .01 .01\n' if index == 0 else b'')
        samples.append({'sample_id': f'synthetic_{index}', 'development_split': split, 'image_path': str(image),
                        'image_source_sha256': training.digest(image), 'label_path': str(label),
                        'label_sha256': training.digest(label), 'negative_visible_absence_only': index == 1})
    data = {'manifest_path': str(root / 'synthetic_manifest.json'), 'manifest_sha256': 'a' * 64}

    def run():
        return training.execute_training(samples, data, output, training.training_settings(output, epochs=1),
                                         deepcopy(training.PINNED_RUNTIME))

    return SimpleNamespace(root=root, output=output, weights=weights, runtime_root=runtime_root,
                           events=events, state=state, run=run,
                           metadata=lambda: json.loads((output / 'run_metadata.json').read_text(encoding='utf-8')))


def test_synthetic_execution_records_updates_and_retains_unstripped_final_checkpoint(synthetic_execution):
    case = synthetic_execution
    metadata = case.run()
    assert case.events.index('cuda_initialized') < case.events.index('resource_guard') < case.events.index('construct_model')
    assert metadata['resource_guard']['minimum_available_ram_bytes'] == 1024 ** 3
    assert metadata['resource_guard']['minimum_free_vram_bytes'] == 2 * 1024 ** 3
    assert metadata['optimizer_update_receipts'] == [{'epoch': 1, 'optimizer_step_values': [58.0],
        'batches_per_epoch': 58, 'accumulation': 1, 'warmup_epochs': 0.}]
    preserved = Path(metadata['unstripped_resume_checkpoint']['path'])
    assert preserved.read_bytes() == b'synthetic unstripped optimizer-bearing bytes'
    assert (preserved.parent / 'last.pt').read_bytes() == b'synthetic stripped final bytes'
    assert metadata['unstripped_resume_checkpoint']['sha256'] == training.digest(preserved)
    assert metadata['source_integrity_changes'] == {'changed': [], 'added': [], 'removed': []}
    assert metadata['runtime_integrity_changes'] == {'changed': [], 'added': [], 'removed': []}
    bound = {Path(row['path']).name for row in metadata['working_input_bindings']}
    assert {'dataset.yaml', 'train.txt', 'val.txt', 'yolo26m.pt'} <= bound
    assert metadata['production_promoted'] is False and metadata['qualification_evidence'] is False
    assert metadata['final_test_used'] is False and metadata['independent_human_review'] is False


@pytest.mark.parametrize('resource', ['ram', 'vram'])
def test_exact_post_cuda_floor_refuses_one_byte_short_before_copy_or_model(synthetic_execution, resource):
    case = synthetic_execution
    if resource == 'ram':
        case.state.available -= 1
    else:
        case.state.vram -= 1
    with pytest.raises(ValueError, match='(1 GiB|2 GiB)'):
        case.run()
    assert 'cuda_initialized' in case.events and 'resource_guard' in case.events
    assert 'construct_model' not in case.events and 'prepare_dataset' not in case.events
    assert not (case.output / 'training_inputs').exists()
    assert case.metadata()['status'] == 'NOT_STARTED_RESOURCE_OR_RUNTIME_GUARD_FAILURE'


def test_absent_cuda_is_retained_as_not_started_without_cpu_fallback(synthetic_execution):
    case = synthetic_execution
    case.state.cuda_available = False
    with pytest.raises(ValueError, match='CPU training is not selected'):
        case.run()
    assert 'construct_model' not in case.events
    assert case.metadata()['status'] == 'NOT_STARTED_RESOURCE_OR_RUNTIME_GUARD_FAILURE'


@pytest.mark.parametrize('target', ['repository_changed', 'repository_added', 'runtime_changed', 'runtime_added',
                                  'runtime_removed', 'yaml', 'train_list', 'val_list', 'initializer'])
def test_end_snapshot_detects_code_runtime_split_or_initializer_mutations(synthetic_execution, target):
    case = synthetic_execution

    def mutate():
        path = {'repository_changed': case.root / 'src/later/not_imported.py',
                'repository_added': case.root / 'src/late_addition.py',
                'runtime_changed': case.runtime_root / 'model.yaml',
                'runtime_added': case.runtime_root / 'late_addition.py',
                'runtime_removed': case.runtime_root / 'model.yaml',
                'yaml': case.output / 'training_inputs/dataset.yaml',
                'train_list': case.output / 'training_inputs/train.txt',
                'val_list': case.output / 'training_inputs/val.txt', 'initializer': case.weights}[target]
        if target == 'runtime_removed':
            path.unlink()
        else:
            path.write_bytes(b'changed during synthetic training; never real model evidence')

    case.state.hook = mutate
    with pytest.raises(ValueError, match='bytes changed'):
        case.run()
    metadata = case.metadata()
    assert metadata['status'] == 'TRAINING_FAILED_UNQUALIFIED'
    assert metadata['input_or_source_files_changed']
    assert metadata['qualification_evidence'] is False and metadata['production_promoted'] is False


def test_atomic_metadata_write_rejects_nonfinite_claims_and_keeps_prior_receipt(tmp_path):
    path = tmp_path / 'run_metadata.json'
    training._write_metadata(path, {'status': 'BEFORE'})
    with pytest.raises(ValueError):
        training._write_metadata(path, {'metric': float('nan')})
    assert json.loads(path.read_text()) == {'status': 'BEFORE'}


def test_atomic_metadata_retries_transient_windows_lock_and_publishes_exact_json(tmp_path, monkeypatch):
    path = tmp_path / 'run_metadata.json'
    training._write_metadata(path, {'status': 'BEFORE'})
    actual_replace = training.os.replace
    calls, delays = [], []

    def briefly_locked(source, destination):
        calls.append((source, destination))
        if len(calls) <= 2:
            raise PermissionError('Synthetic transient Windows scanner lock')
        actual_replace(source, destination)

    monkeypatch.setattr(training.os, 'replace', briefly_locked)
    monkeypatch.setattr(training.time, 'sleep', delays.append)
    payload = {'status': 'AFTER', 'qualification_evidence': False, 'value': 1.25}
    training._write_metadata(path, payload)
    assert delays == [.01, .025] and len(calls) == 3
    assert path.read_text(encoding='utf-8') == json.dumps(payload, indent=2, allow_nan=False) + '\n'
    assert not Path(str(path) + '.tmp').exists()


def test_atomic_metadata_retry_exhaustion_raises_and_preserves_prior_receipt(tmp_path, monkeypatch):
    path = tmp_path / 'run_metadata.json'
    training._write_metadata(path, {'status': 'BEFORE'})
    calls, delays = [], []

    def permanently_locked(source, destination):
        calls.append((source, destination))
        raise PermissionError('Synthetic permanent permission failure')

    monkeypatch.setattr(training.os, 'replace', permanently_locked)
    monkeypatch.setattr(training.time, 'sleep', delays.append)
    with pytest.raises(PermissionError, match='permanent permission failure'):
        training._write_metadata(path, {'status': 'MUST_NOT_BE_PUBLISHED'})
    assert delays == [.01, .025, .05, .1] and len(calls) == 5
    assert json.loads(path.read_text()) == {'status': 'BEFORE'}
    assert Path(str(path) + '.tmp').is_file()


def test_atomic_metadata_does_not_retry_or_hide_other_io_errors(tmp_path, monkeypatch):
    path = tmp_path / 'run_metadata.json'
    training._write_metadata(path, {'status': 'BEFORE'})
    calls, delays = [], []

    def io_failure(source, destination):
        calls.append((source, destination))
        raise OSError('Synthetic non-permission IO failure')

    monkeypatch.setattr(training.os, 'replace', io_failure)
    monkeypatch.setattr(training.time, 'sleep', delays.append)
    with pytest.raises(OSError, match='non-permission IO failure'):
        training._write_metadata(path, {'status': 'MUST_NOT_BE_PUBLISHED'})
    assert len(calls) == 1 and delays == []
    assert json.loads(path.read_text()) == {'status': 'BEFORE'}


def test_actual_exact_export_audits_without_any_image_model_or_gpu_import(monkeypatch):
    """Actual retained79sample data integrity; this is not vision accuracy."""
    actual_import = builtins.__import__
    attempted = []

    def guard(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            attempted.append(name)
            pytest.fail(f'Actual byte-only export audit attempted {name}')
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guard)
    manifest = training.DEFAULT_MANIFEST
    samples, audit = training.audit_export(
        manifest, 'b3fa03c171c797db9df52038fe1b984cd6aef6aafb942e6885207430832a25ab')
    assert len(samples) == 79 and audit['counts'] == EXACT_COUNTS
    assert sum(row['development_split'] == 'train' for row in samples) == 58
    assert sum(row['development_split'] == 'val' for row in samples) == 21
    assert sum(row['negative_visible_absence_only'] for row in samples) == 4
    assert not any(row.get('match_id') or row.get('session_id') for row in samples)
    assert audit['source_independence_verified'] is False
    assert audit['qualification_evidence'] is False
    assert attempted == []

