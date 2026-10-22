"""Exact-export native640 private ball development; default is pure preflight.

No Torch, Ultralytics, NumPy, image or video runtime is imported before execute.
This dedicated wrapper does not relax the generic trainer's qualification audit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / 'data/yolo26/ball_reviewed_development_v1/source_manifest.json'
OFFICIAL_WEIGHTS = ROOT / 'models/yolo26/yolo26m.pt'
OFFICIAL_WEIGHTS_SHA256 = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
PINNED_RUNTIME = {'ultralytics': '8.4.36', 'torch': '2.13.0+cu126'}
MODE = 'REVIEWED_NATIVE_BALL_DEVELOPMENT'
EXPECTED_COUNTS = {'images': 79, 'positive_boxes': 75, 'negative_tiles': 4, 'train': 58, 'val': 21, 'test': 0,
                   'positive_split_counts': {'train': 55, 'development': 20}}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def sha256_token(value, name):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} needs an exact lowercase SHA256 token')
    return value


def audit_export(manifest_path, expected_manifest_sha256):
    """Reuse the exporter's pure, deep source/review/label verification."""
    from scripts.data.export_yolo26_reviewed_ball_wave import verify_export

    path = Path(manifest_path).resolve()
    expected = sha256_token(expected_manifest_sha256, 'Reviewed manifest')
    if path.name != 'source_manifest.json' or digest(path) != expected:
        raise ValueError('Require exact final ROOT-reviewed exported manifest bytes')
    result = verify_export(path.parent, expected_manifest_sha256=expected)
    manifest = result['manifest']
    if (result['counts'] != EXPECTED_COUNTS or manifest.get('counts') != EXPECTED_COUNTS
            or manifest.get('status') != 'EXPORTED_PRIVATE_AGENT_REVIEWED_BALL_DEVELOPMENT_NOT_QUALIFICATION'
            or any(manifest.get(key) is not False for key in ('qualification_evidence', 'final_test_used',
                                                            'independent_human_review', 'source_independence_verified'))):
        raise ValueError('Only the exact79sample private development export is eligible; no TEST or qualification claims')
    samples = result['samples']
    if (len(samples) != 79 or any(row.get('development_split') not in ('train', 'val') for row in samples)
            or any(row.get('training_allowed') is not True for row in samples)):
        raise ValueError('Require79 authorized train/DEV samples and no additional role or split')
    audit = {'manifest_path': str(path), 'manifest_sha256': expected,
             'export_receipt_path': str(result['receipt_path']), 'export_receipt_sha256': result['receipt_sha256'],
             'data_yaml_path': str(result['data_yaml_path']), 'data_yaml_sha256': result['data_yaml_sha256'],
             'counts': result['counts'], 'final_test_used': False, 'qualification_evidence': False,
             'independent_human_review': False, 'source_independence_verified': False,
             'validation_scope': 'FIXED_NATIVE640_AGENT_REVIEWED_DEV_CROPS_NOT_FULL_FRAME_ACCURACY',
             'usage_scope': 'PRIVATE_LOCAL_ACADEMIC_DEVELOPMENT_ONLY', 'raw_redistribution_cleared': False}
    return samples, audit


def training_settings(output, *, epochs=20, seed=709):
    if type(epochs) is not int or not 1 <= epochs <= 20:
        raise ValueError('Initial reviewed-ball training is limited to1-20 total epochs; no extension is implemented')
    if type(seed) is not int or seed < 0:
        raise ValueError('Training seed must be a nonnegative integer')
    output = Path(output).resolve()
    return dict(data=str(output / 'training_inputs/dataset.yaml'), epochs=epochs, imgsz=640, batch=1,
                device='0', seed=seed, deterministic=True, patience=15, save=True, save_period=5,
                amp=False, workers=0, cache=False, optimizer='AdamW', lr0=.0005, nbs=1, warmup_epochs=0.,
                project=str(output), name='training', exist_ok=False, val=True, plots=False,
                mosaic=0., close_mosaic=0, mixup=0., cutmix=0., copy_paste=0., scale=0., translate=0.,
                degrees=0., shear=0., perspective=0., fliplr=0., flipud=0., bgr=0., multi_scale=0., rect=False,
                auto_augment=None, augmentations=[], erasing=0., hsv_h=.01, hsv_s=.15, hsv_v=.15)


def validate_fresh_output(output):
    output = Path(output).resolve()
    allowed = (ROOT / 'outputs').resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Choose a fresh run directory inside the workspace outputs root')
    if output.exists():
        raise FileExistsError('Preserve prior evidence; the output directory must not already exist, including for resume')
    return output


def runtime_versions():
    versions = {name: importlib.metadata.version(name) for name in PINNED_RUNTIME}
    if versions != PINNED_RUNTIME:
        raise ValueError(f'Require the pinned installed runtime {PINNED_RUNTIME}; observed {versions}')
    return versions


def audit_resume(checkpoint, checkpoint_sha256, settings, data_audit):
    checkpoint = Path(checkpoint).resolve()
    expected = sha256_token(checkpoint_sha256, 'Resume checkpoint')
    if digest(checkpoint) != expected:
        raise ValueError('Resume checkpoint SHA256 differs')
    if checkpoint.parent.name != 'weights' or checkpoint.parent.parent.name != 'training':
        raise ValueError('Resume only a registered checkpoint from a retained reviewed-ball training output')
    metadata_path = checkpoint.parents[2] / 'run_metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if (metadata.get('mode') != MODE or metadata.get('role') != 'ball'
            or metadata.get('data_audit', {}).get('manifest_sha256') != data_audit['manifest_sha256']
            or metadata.get('data_audit', {}).get('manifest_path') != data_audit['manifest_path']
            or metadata.get('runtime_versions') != PINNED_RUNTIME
            or metadata.get('final_test_used') is not False or metadata.get('qualification_evidence') is not False):
        raise ValueError('Resume must retain the same exact reviewed export, role, private mode and pinned runtime')
    prior = metadata.get('training_arguments', {})
    ignored = {'data', 'project', 'name'}  # Fresh byte-identical working copy and output location only.
    if {key: value for key, value in prior.items() if key not in ignored} != {
            key: value for key, value in settings.items() if key not in ignored}:
        raise ValueError('Interrupted resume must retain all training settings and the same total epochs; extensions are forbidden')
    registered = list(metadata.get('checkpoints', []))
    if metadata.get('unstripped_resume_checkpoint'):
        registered.append(metadata['unstripped_resume_checkpoint'])
    if not any(Path(row.get('path', '')).resolve() == checkpoint and row.get('sha256') == expected for row in registered):
        raise ValueError('Resume checkpoint must be hash-registered in its original run metadata')
    return {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': expected,
            'metadata_path': str(metadata_path), 'metadata_sha256': digest(metadata_path), 'metadata': metadata}


def validate_checkpoint_state(checkpoint_state, total_epochs):
    epoch = checkpoint_state.get('epoch') if isinstance(checkpoint_state, dict) else None
    if (type(epoch) is not int or epoch < 0 or epoch + 1 >= total_epochs
            or checkpoint_state.get('optimizer') is None or checkpoint_state.get('ema') is None
            or type(checkpoint_state.get('updates')) is not int or checkpoint_state['updates'] < 0
            or checkpoint_state.get('train_args', {}).get('epochs') != total_epochs):
        raise ValueError('Resume requires an unfinished unstripped checkpoint with optimizer/EMA state and identical total epochs')
    return epoch + 1


def validate_resume_model(model):
    """Reject another architecture even when its task, class count and scale match."""
    core = getattr(model, 'model', None)
    architecture = getattr(core, 'yaml', {})
    yaml_file = architecture.get('yaml_file') if isinstance(architecture, dict) else None
    if (getattr(model, 'task', None) != 'detect' or not isinstance(architecture, dict)
            or architecture.get('scale') != 'm' or type(architecture.get('nc')) is not int
            or architecture['nc'] != 1 or not isinstance(yaml_file, str)
            or Path(yaml_file).name not in {'yolo26m.yaml', 'yolo26m-tennis-ball.yaml'}
            or getattr(core, 'end2end', None) is not True
            or getattr(core, 'names', None) != {0: 'tennis-ball'}):
        raise ValueError('Resume checkpoint is not the selected end-to-end YOLO26m one-class tennis-ball Detect specialist')


def _resume_trainer(settings):
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ReviewedBallResumeTrainer(DetectionTrainer):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            for key, value in settings.items():
                setattr(self.args, key, value)
            self.args.save_dir = str(Path(settings['project']) / settings['name'])

    return ReviewedBallResumeTrainer


def _write_metadata(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Windows scanners may briefly hold the previous receipt open. Retry only
    # that replace boundary, retaining atomic publication and permanent errors.
    delays = iter((.01, .025, .05, .1))
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            delay = next(delays, None)
            if delay is None:
                raise
            time.sleep(delay)


def _copy_training_inputs(samples, output):
    """Keep Ultralytics label-scan caches away from immutable reviewed data."""
    records, bindings = [], []
    working = output / 'training_inputs'
    for row in samples:
        split, ident = row['development_split'], row['sample_id']
        image = working / 'ball/images' / split / (ident + '.png')
        label = working / 'ball/labels' / split / (ident + '.txt')
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        for source, target, expected in ((row['image_path'], image, row['image_source_sha256']),
                                         (row['label_path'], label, row['label_sha256'])):
            if digest(source) != expected:
                raise ValueError('Exported source changed before training working-copy creation')
            with Path(source).open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if digest(target) != expected:
                raise ValueError('Training working copy differs from immutable reviewed export')
            bindings.append({'path': str(target), 'sha256': expected, 'source_path': str(source)})
        records.append({'image': str(image), 'label': str(label), 'split': split,
                        'instances': 0 if row['negative_visible_absence_only'] else 1})
    return records, bindings


def _source_bindings():
    # Include modules that training will import later, not only sys.modules at
    # the instant this snapshot is taken. Restrict traversal to the source tree.
    scripts = ('scripts/train/train_yolo26_reviewed_ball.py', 'scripts/train/train_yolo26_tennis.py',
               'scripts/data/export_yolo26_reviewed_ball_wave.py',
               'scripts/data/prepare_yolo26_reviewed_ball_wave.py',
               'scripts/data/prepare_yolo26_ball_review_wave.py')
    paths = {ROOT / relative for relative in scripts}
    paths.update((ROOT / 'src').rglob('*.py'))
    return {str(path.resolve()): digest(path) for path in sorted(paths)}


def _runtime_source_bindings(package_root):
    """Snapshot installed Ultralytics code/configuration without importing it."""
    root = Path(package_root).resolve()
    if not root.is_dir():
        raise ValueError('Installed Ultralytics package source directory is missing')
    paths = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix in {'.py', '.yaml'})
    if not paths:
        raise ValueError('Installed Ultralytics package has no code/configuration to bind')
    return {str(path): digest(path) for path in paths}


def _binding_changes(before, after):
    """Keep additions and removals visible, including files never imported."""
    return {'changed': sorted(path for path in before.keys() & after.keys() if before[path] != after[path]),
            'added': sorted(after.keys() - before.keys()), 'removed': sorted(before.keys() - after.keys())}


def execute_training(samples, data_audit, output, settings, versions, resume=None):
    output.mkdir(parents=True, exist_ok=False)
    metadata_path = output / 'run_metadata.json'
    metadata = {'mode': MODE, 'role': 'ball', 'status': 'PREFLIGHT_PASSED_NOT_TRAINED',
                'started_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
                'python_executable': sys.executable, 'runtime_versions': versions, 'data_audit': data_audit,
                'training_arguments': settings, 'final_test_used': False, 'qualification_evidence': False,
                'independent_human_review': False, 'production_promoted': False,
                'initial_weights': {'path': str(OFFICIAL_WEIGHTS), 'sha256': OFFICIAL_WEIGHTS_SHA256},
                'maximum_total_epochs': 20, 'extension_policy': 'NO_EXTENSION_IMPLEMENTED_REQUIRES_FUTURE_REVIEW',
                'best_checkpoint_policy': 'PINNED_ULTRALYTICS_DEV_MAP50_95_FITNESS_FOR_ENGINEERING_REVIEW_ONLY',
                'validation_policy': 'FIXED_NATIVE640_AGENT_REVIEWED_DEV_CROPS_NOT_FULL_FRAME_OR_INDEPENDENT_QUALIFICATION',
                'resume_policy': 'SAME_EXACT_DATA_SETTINGS_TOTAL_EPOCHS_FRESH_OUTPUT_UNSTRIPPED_REGISTERED_CHECKPOINT',
                'resume_limits': 'Pinned Ultralytics saves EMA-derived FP16 model and FP16-converted optimizer state; RNG stream is not checkpointed. Resume is not bitwise continuation.',
                'resume': resume, 'resource_guard_pending': True, 'checkpoints': []}
    _write_metadata(metadata_path, metadata)
    training_started = None
    try:
        # These imports are deliberately after pure preflight and never run
        # merely for --help, configuration inspection or source/data audit.
        from scripts.train.train_yolo26_tennis import build_training_model, check_resource_guard, prepare_dataset
        import psutil
        import torch
        import ultralytics
        if not torch.cuda.is_available():
            raise ValueError('Reviewed-ball execution requires the declared CUDA GPU0; CPU training is not selected')
        torch.cuda.set_device(0)
        torch.cuda.init()
        metadata.update(cuda=torch.version.cuda, torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                        free_vram_bytes=torch.cuda.mem_get_info(0)[0], free_ram_bytes=psutil.virtual_memory().available,
                        free_disk_bytes=shutil.disk_usage(output).free, owned_rss_bytes=psutil.Process().memory_info().rss,
                        source_bindings=_source_bindings())
        runtime_root = Path(ultralytics.__file__).parent
        metadata['runtime_source_bindings'] = _runtime_source_bindings(runtime_root)
        check_resource_guard(metadata, gpu_requested=True, role='ball', image_size=640)
        metadata['resource_guard_pending'] = False
        metadata['resource_guard'] = {'measurement_point': 'AFTER_TORCH_AND_CUDA_INITIALIZATION_BEFORE_MODEL',
                                      'minimum_available_ram_bytes': 1024 ** 3, 'minimum_free_vram_bytes': 2 * 1024 ** 3,
                                      'minimum_free_disk_bytes': 1024 ** 3}
        if resume:
            prior = resume['metadata'].get('source_bindings', {})
            if prior != metadata['source_bindings'] or resume['metadata'].get('runtime_source_bindings') != metadata['runtime_source_bindings']:
                raise ValueError('Resume requires unchanged training implementation/runtime source bindings')
        records, input_bindings = _copy_training_inputs(samples, output)
        data = prepare_dataset(records, 'ball', output / 'training_inputs', pilot=True)
        if str(data) != settings['data']:
            raise ValueError('Prepared train/DEV YAML differs from fixed training arguments')
        metadata['working_input_bindings'] = input_bindings
        metadata['dataset_yaml_sha256'] = digest(data)
        # Split membership and YAML are inputs too; image/label hashes alone
        # cannot prove that the intended train/DEV lists remained in use.
        for path in (data, data.parent / 'train.txt', data.parent / 'val.txt', OFFICIAL_WEIGHTS):
            input_bindings.append({'path': str(path), 'sha256': digest(path)})
        if digest(OFFICIAL_WEIGHTS) != OFFICIAL_WEIGHTS_SHA256:
            raise ValueError('Official YOLO26m initialization bytes changed before model construction')
        metadata['training_arguments_sha256'] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        metadata['git_revision'] = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', 'rev-parse', 'HEAD'],
                                                 cwd=ROOT, capture_output=True, text=True).stdout.strip()
        if resume:
            from ultralytics import YOLO
            model = YOLO(resume['checkpoint_path'])
            metadata['resume_start_epoch'] = validate_checkpoint_state(model.ckpt, settings['epochs'])
            validate_resume_model(model)
            arguments = dict(settings, resume=resume['checkpoint_path'], trainer=_resume_trainer(settings))
        else:
            model, transfer = build_training_model('ball', OFFICIAL_WEIGHTS, output)
            metadata['transfer'] = transfer
            arguments = settings
        process = psutil.Process()
        usage = {'peak_owned_rss_bytes': process.memory_info().rss,
                 'minimum_available_ram_bytes': psutil.virtual_memory().available}

        def sample_usage(trainer):
            usage['peak_owned_rss_bytes'] = max(usage['peak_owned_rss_bytes'], process.memory_info().rss)
            usage['minimum_available_ram_bytes'] = min(usage['minimum_available_ram_bytes'], psutil.virtual_memory().available)

        def record_updates(trainer):
            sample_usage(trainer)
            steps = sorted({float(state['step']) for state in trainer.optimizer.state.values() if 'step' in state})
            metadata.setdefault('optimizer_update_receipts', []).append({
                'epoch': int(trainer.epoch) + 1, 'optimizer_step_values': steps,
                'batches_per_epoch': len(trainer.train_loader), 'accumulation': int(trainer.accumulate),
                'warmup_epochs': float(trainer.args.warmup_epochs)})
            _write_metadata(metadata_path, metadata)

        def record_checkpoints(trainer):
            epoch = int(trainer.epoch) + 1
            if (epoch == int(trainer.epochs) or trainer.stop) and Path(trainer.last).is_file():
                preserved = Path(trainer.wdir) / f'development_epoch{epoch}.pt'
                with Path(trainer.last).open('rb') as incoming, preserved.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                metadata['unstripped_resume_checkpoint'] = {'path': str(preserved), 'sha256': digest(preserved), 'epoch': epoch}
            metadata['checkpoints'] = [{'path': str(path), 'sha256': digest(path)} for path in sorted(Path(trainer.wdir).glob('*.pt'))]
            _write_metadata(metadata_path, metadata)

        model.add_callback('on_train_batch_end', sample_usage)
        model.add_callback('on_train_epoch_end', record_updates)
        model.add_callback('on_model_save', record_checkpoints)
        metadata['status'] = 'TRAINING_STARTED_UNQUALIFIED'
        _write_metadata(metadata_path, metadata)
        torch.cuda.reset_peak_memory_stats(0)
        training_started = time.perf_counter()
        model.train(**arguments)
        metadata.update(status='TRAINED_UNQUALIFIED', completed_utc=datetime.now(timezone.utc).isoformat(),
                        training_elapsed_seconds=time.perf_counter() - training_started,
                        peak_torch_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        peak_torch_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0), resource_usage=usage)
        metadata['checkpoints'] = [{'path': str(path), 'sha256': digest(path)} for path in sorted((output / 'training/weights').glob('*.pt'))]
        csv_path = output / 'training/results.csv'
        if csv_path.is_file():
            metadata['metrics_csv'] = {'path': str(csv_path), 'sha256': digest(csv_path)}
        source_changes = _binding_changes(metadata['source_bindings'], _source_bindings())
        runtime_changes = _binding_changes(metadata['runtime_source_bindings'], _runtime_source_bindings(runtime_root))
        metadata['source_integrity_changes'] = source_changes
        metadata['runtime_integrity_changes'] = runtime_changes
        changed = [path for changes in (source_changes, runtime_changes) for paths in changes.values() for path in paths]
        changed.extend(row['path'] for row in input_bindings
                       if not Path(row['path']).is_file() or digest(row['path']) != row['sha256'])
        audit_export(data_audit['manifest_path'], data_audit['manifest_sha256'])
        metadata['input_or_source_files_changed'] = changed
        if changed:
            raise ValueError('Source or training image/label bytes changed during execution')
        _write_metadata(metadata_path, metadata)
    except BaseException as error:
        metadata.update(status=('TRAINING_FAILED_UNQUALIFIED' if training_started is not None else 'NOT_STARTED_RESOURCE_OR_RUNTIME_GUARD_FAILURE'),
                        failure_type=type(error).__name__, failure_reason=str(error), failed_utc=datetime.now(timezone.utc).isoformat())
        _write_metadata(metadata_path, metadata)
        raise
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--manifest-sha256', required=True, help='Exact final exported manifest digest separately reviewed by ROOT')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=709)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--resume-sha256')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    output = validate_fresh_output(args.output)
    settings = training_settings(output, epochs=args.epochs, seed=args.seed)
    samples, data_audit = audit_export(args.manifest, args.manifest_sha256)
    versions = runtime_versions()
    if digest(OFFICIAL_WEIGHTS) != OFFICIAL_WEIGHTS_SHA256:
        raise ValueError('Require the exact locally retained official YOLO26m checkpoint, never an old pilot')
    if bool(args.resume) != bool(args.resume_sha256):
        raise ValueError('Resume requires both its exact checkpoint path and separately recorded SHA256')
    resume = audit_resume(args.resume, args.resume_sha256, settings, data_audit) if args.resume else None
    if args.execute:
        result = execute_training(samples, data_audit, output, settings, versions, resume)
        print(json.dumps({'status': result['status'], 'metadata': str(output / 'run_metadata.json'), 'data_audit': data_audit}))
    else:
        print(json.dumps({'status': 'PURE_DATA_PREFLIGHT_PASSED_NOT_TRAINED', 'mode': MODE, 'role': 'ball',
                          'data_audit': data_audit, 'training_arguments': settings, 'runtime_versions': versions,
                          'initial_weights': {'path': str(OFFICIAL_WEIGHTS), 'sha256': OFFICIAL_WEIGHTS_SHA256},
                          'resource_guard_pending': True, 'output_files_written': 0, 'torch_imported': 'torch' in sys.modules,
                          'qualification_evidence': False, 'final_test_used': False, 'resume': resume}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
