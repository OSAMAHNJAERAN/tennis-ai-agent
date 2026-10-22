"""Exclusive-worker correction orchestration, separate from original job state.

Native owners finish before complete-generation publication. Only a reachable
configured crop-pose branch loads the original selected player checkpoint;
semantic processing never silently substitutes another model or device.
"""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import gc
import importlib.metadata
from pathlib import Path
import re
import tempfile
import traceback

import yaml

from src.api.jobs import JobCancelled, REPO_ROOT
from src.api.analysis_generation import resolve_generation, _safe_path
from src.api.result_document import ResultDocument
from src.scoring.corrected_rule_rebuild import MAX_REQUEST_BYTES, _digest, _request
from src.utils.point_timeline_artifact import artifact_sha256


def _require(condition, message):
    if not condition:
        raise ValueError('Correction worker: '+message)


class CorrectionResourceError(RuntimeError):
    """Host/device admission or the owner's stored-byte budget refused work."""


def _resources(directory, stage, torch=None):
    # Share the predeclared host/device floors without importing an inference
    # runtime merely to admit semantic/CPU work.
    from scripts.train.train_yolo26_author_ball_repair import resource_snapshot, resource_guard
    values = resource_snapshot(directory, torch)
    try:
        resource_guard(values, stage)
    except ValueError as error:
        raise CorrectionResourceError('Insufficient free processing memory or disk space. '
            'The correction was stopped before publication; retry when resources are available.') from error
    return values


class _StrictPoseModel:
    """Retain a crop failure even when the legacy extractor catches it."""
    def __init__(self, model, options, check):
        self.model, self.options, self.check = model, options, check
        self.failure = None
        self.prediction_calls = 0

    def predict(self, image, **_arguments):
        try:
            if self.failure is not None:
                raise RuntimeError('Earlier selected-player crop inference failed') from self.failure
            self.check()
            self.prediction_calls += 1
            return self.model.predict(image, **self.options)
        except Exception as error:
            self.failure = error
            raise


def _crop_options(overrides):
    """Resolve pinned native crop defaults, separately from full-frame tracking.

    Ultralytics 8.4.36 resets predict confidence to .25 and takes image size
    from checkpoint overrides on each crop call. A prior full-frame .track
    call's .4/960 settings do not define the crop preprocessing.
    """
    _require(type(overrides) is dict, 'selected model checkpoint overrides are unavailable')
    imgsz = overrides.get('imgsz', 640)
    dimensions = [imgsz] if type(imgsz) is int else imgsz
    _require(type(dimensions) in (list, tuple) and 1 <= len(dimensions) <= 2
             and all(type(value) is int and 32 <= value <= 4096 for value in dimensions),
             'selected checkpoint crop image size is invalid')
    _require(type(overrides.get('single_cls', False)) is bool,
             'selected checkpoint class configuration is invalid')
    return {'imgsz': imgsz if type(imgsz) is int else list(imgsz), 'conf': .25,
        'verbose': False, 'device': 0, 'end2end': True, 'nms': False,
        'batch': 1, 'rect': True, 'half': False, 'max_det': 300, 'iou': .7,
        'augment': False, 'agnostic_nms': False, 'classes': None,
        'single_cls': overrides.get('single_cls', False)}


class SelectedCorrectionPose:
    """Lazy original-checkpoint admission for the actual native crop extractor."""
    def __init__(self, rebuilt, scratch):
        self.rebuilt, self.snapshot, self.scratch = rebuilt, rebuilt.snapshot, scratch
        self.calls = 0
        self.extractor = self.proxy = self.torch = None
        self.checkpoint = self.checkpoint_sha = None
        self.runtime = self.options = self.architecture = None
        self.closed = False

    def _current(self):
        from src.scoring.corrected_shot_rebuild import _current
        _require(not self.closed, 'selected crop provider is closed')
        _current(self.rebuilt)
        if self.checkpoint is not None:
            _require(artifact_sha256(self.checkpoint) == self.checkpoint_sha,
                     'original selected player checkpoint changed during correction')

    def _load(self, request):
        path = self.snapshot.resolve_file('run_config.yaml')
        _require(path is not None and path.stat().st_size <= MAX_REQUEST_BYTES,
                 'bounded original configuration is required for crop inference')
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        _require(type(config) is dict, 'original crop configuration must be an object')
        system, player, pose = (config.get(name, {}) for name in
                                ('model_system', 'player_detection', 'pose_estimation'))
        _require(all(type(value) is dict for value in (system, player, pose))
                 and system.get('family') == 'yolo26' and system.get('vision_model_count') == 3
                 and system.get('ultralytics_version') == '8.4.36'
                 and system.get('inference_head') == 'one_to_one'
                 and system.get('end2end') is True and system.get('nms') is False
                 and pose.get('source') == 'player_model'
                 and player.get('model') == request.selected_player_model_path
                 and pose.get('model_path') in (None, player.get('model')),
                 'crop inference must reuse the pinned original three-model player role')
        selected = Path(player['model'])
        selected = _safe_path(selected if selected.is_absolute() else REPO_ROOT/selected)
        _require(selected.is_file(), 'the original selected player checkpoint is missing')
        with ResultDocument(self.snapshot.files_directory.parent.parent, self.snapshot.analysis_id,
                            generation=self.snapshot.generation) as document:
            original = document.value.get('provenance')
            hashes = original.get('checkpointHashes') if isinstance(original, dict) else None
            retained = hashes.get('players') if isinstance(hashes, dict) else None
            expected = retained.get('sha256') if isinstance(retained, dict) else None
            _require(type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected)
                     and retained.get('filename') == selected.name,
                     'captured original player checkpoint identity is unavailable')
        _require(artifact_sha256(selected) == expected,
                 'configured player checkpoint differs from original inference')
        _require(importlib.metadata.version('ultralytics') == '8.4.36',
                 'the pinned original pose runtime is unavailable')
        # No Torch/CUDA import or model load can precede this unchanged floor.
        _resources(self.scratch, 'before_gpu')
        import torch
        self.torch = torch
        _require(torch.cuda.is_available(), 'selected crop inference requires CUDA; no CPU fallback')
        _resources(self.scratch, 'after_cuda', torch)
        from src.detection.yolo26_adapters import _load
        from src.shot_analysis.pose_feature_extractor import PoseFeatureExtractor
        model = _load(selected, 'pose', 's', 17, device=0)
        self.checkpoint, self.checkpoint_sha = selected, expected
        self.options = _crop_options(model.overrides)
        self.proxy = _StrictPoseModel(model, self.options, self._current)
        self.extractor = PoseFeatureExtractor(str(selected), device='cuda:0', model=self.proxy)
        self.runtime = {'ultralytics': '8.4.36', 'torch': torch.__version__, 'cuda': torch.version.cuda}
        self.architecture = {'family': 'yolo26s', 'task': 'pose', 'keypoint_shape': [17, 3],
                             'role': 'original_selected_players', 'device': 'cuda:0'}
        self._current()
        _resources(self.scratch, 'after_cuda', torch)

    def __call__(self, request):
        from src.scoring.corrected_shot_rebuild import CorrectedPoseRequest
        self._current()
        _require(isinstance(request, CorrectedPoseRequest)
                 and request.analysis_id == self.snapshot.analysis_id
                 and request.analysis_revision == self.snapshot.analysis_revision+1
                 and request.base_generation == self.snapshot.generation
                 and request.source_sha256 == self.snapshot.file_sha256('source.mp4')
                 and request.run_config_sha256 == self.snapshot.file_sha256('run_config.yaml')
                 and request.window_half_size == 3 and request.crop_margin == .65,
                 'crop request differs from the original source and native window')
        if self.extractor is None:
            self._load(request)
        self.calls += 1
        result = self.extractor.extract_hit_window_features(frames=request.frames,
            player_boxes=request.player_boxes, hit_frame=request.hit_frame,
            player_id=request.player_id, handedness=request.handedness,
            court_orientation_sign=request.court_orientation_sign,
            window_half_size=request.window_half_size, crop_margin=request.crop_margin)
        if self.proxy.failure is not None:
            raise RuntimeError('Selected original player crop inference failed') from self.proxy.failure
        self._current()
        _resources(self.scratch, 'after_cuda', self.torch)
        return result

    def record(self):
        self._current()
        predictions = self.proxy.prediction_calls if self.proxy is not None else 0
        return {'schema_version': 1, 'new_vision_inference': predictions > 0,
            'pose_provider_calls': self.calls, 'pose_prediction_calls': predictions,
            'selected_player_checkpoint_sha256': self.checkpoint_sha,
            'selected_player_checkpoint_filename': self.checkpoint.name if self.checkpoint is not None else None,
            'source_sha256': self.snapshot.file_sha256('source.mp4'),
            'base_generation': self.snapshot.generation, 'analysis_revision': self.snapshot.analysis_revision+1,
            'run_config_sha256': self.snapshot.file_sha256('run_config.yaml'),
            'runtime_versions': self.runtime, 'input_spec': ({**self.options,
                'window_half_size': 3, 'crop_margin': .65,
                'tracking_policy': 'DISABLED_BOUND_SOURCE_CROPS'} if self.options is not None else None),
            'architecture': self.architecture, 'independent_accuracy_validated': False}

    def close(self):
        self.closed = True
        if self.extractor is not None:
            self.extractor.model = None
        if self.proxy is not None:
            self.proxy.model = None
        self.extractor = self.proxy = None
        gc.collect()
        if self.torch is not None:
            self.torch.cuda.empty_cache()
        self.torch = None


def committed_correction_generation(store, correction):
    """Reconcile lost acknowledgement from committed bytes, never intent."""
    from src.api.corrections import CorrectionStore
    return CorrectionStore(store).committed_generation(correction)


def _failed_correction(commands, store, ident, error):
    from src.api.worker import AnalysisError
    current = commands.get(ident, internal=True)
    try:
        committed = commands.committed_generation(current)
    except (ValueError, OSError):
        committed = None
    if committed is not None:
        commands.finish(ident, generation=committed)
    elif isinstance(error, JobCancelled) or current.get('cancel_requested'):
        commands.finish(ident, status='CANCELLED', error='Correction cancelled by its owner.')
    else:
        try:
            snapshot = resolve_generation(store.output_dir(current['analysis_id']))
        except (ValueError, OSError):
            snapshot = None
        conflict = snapshot is not None and snapshot.generation != current['base_generation']
        message = ('The analysis changed while this correction was processing. Reload its evidence before submitting a new review.'
            if conflict else str(error) if isinstance(error, (AnalysisError, CorrectionResourceError)) else
            'The correction could not be completed. The previous analysis is still available; review the source and retry.')
        commands.finish(ident, status='CONFLICT' if conflict else 'FAILED', error=message)


def process_correction(store, correction, settings):
    """Keep scratch/log failures local to this correction, like native failures."""
    from src.api.corrections import CorrectionStore
    try:
        _process_correction(store, correction, settings)
    except Exception as error:
        traceback.print_exc()
        _failed_correction(CorrectionStore(store), store, correction['id'], error)


def _process_correction(store, correction, settings):
    """Rebuild one queued owner command without changing original job status."""
    from src.api.corrections import CorrectionStore
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
    from src.scoring.corrected_exports import write_corrected_exports
    from src.scoring.corrected_rendering import render_corrected_video
    from src.scoring.corrected_preparation import prepare_corrected_result
    from src.scoring.corrected_publication import describe_prepared_correction, publish_corrected_result
    commands = CorrectionStore(store)
    from fastapi import HTTPException
    from src.api.resource_quotas import ResourceQuotas
    quotas = ResourceQuotas(store, settings)

    def check_storage():
        try:
            # The final callback runs inside publication_guard's writer lock.
            # A WAL reader sees the owner's committed admissions without
            # opening a second writer that would deadlock that guard.
            with store.connect() as conn:
                quotas.check_storage(conn, correction['owner_id'])
        except HTTPException as error:
            raise CorrectionResourceError(error.detail) from error
    ident, analysis_id = correction['id'], correction['analysis_id']
    parent = _safe_path(settings.root/'corrections'/ident)
    parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    with (scratch/'processing.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                original = store.get(analysis_id)
                command = _request(correction['command'])
                authorization = correction['authorization']

                def stage(label, step):
                    commands.stage(ident, label, step)
                    check_storage()
                    current = commands.get(ident, internal=True)
                    job = store.get(analysis_id)
                    _require(job is not None and original is not None
                             and job['owner_id'] == original['owner_id'] == current['owner_id']
                             and job['status'] == 'COMPLETED'
                             and _digest(current['command']) == _digest(command)
                             and _digest(current['authorization']) == _digest(authorization),
                             'owner or admitted command changed during correction')
                    snapshot = resolve_generation(store.output_dir(analysis_id))
                    _require(snapshot is not None and snapshot.generation == command['base_generation'],
                             'current generation changed from the reviewed base')

                stage('Checking correction source and resources', 1)
                _resources(scratch, 'cpu')
                with ExitStack() as owners:
                    rebuilt = owners.enter_context(rebuild_corrected_rule_timeline(
                        store.output_dir(analysis_id), scratch, request=command))
                    provider = SelectedCorrectionPose(rebuilt, scratch)
                    owners.callback(provider.close)
                    stage('Rebuilding shots and point statistics', 2)
                    shots = owners.enter_context(relink_corrected_shots(rebuilt, scratch, pose_provider=provider))
                    stage('Recomputing causal forecasts', 3)
                    forecasts = owners.enter_context(rebuild_corrected_forecasts(rebuilt, shots, scratch))
                    stage('Validating corrected event exports', 4)
                    exports = write_corrected_exports(rebuilt, shots, forecasts, scratch)
                    stage('Rendering corrected video', 5)
                    rendered = render_corrected_video(rebuilt, shots, forecasts, exports, scratch)
                    stage('Preparing matching reports', 6)
                    prepared = owners.enter_context(prepare_corrected_result(rebuilt, shots,
                        forecasts, exports, rendered, scratch, job=original))
                    descriptor = describe_prepared_correction(prepared)
                    execution = provider.record()
                # All borrowed owners have closed while the reviewed base was
                # current. Publication opens a separate result-storage owner.
                stage('Publishing corrected analysis', 7)
                _resources(scratch, 'cpu')
                with commands.publication_guard(ident) as admitted:
                    _require(_digest(admitted['command']) == _digest(command)
                             and _digest(admitted['authorization']) == _digest(authorization),
                             'publication command or owner authorization changed')
                    publish_corrected_result(descriptor, store.output_dir(analysis_id),
                        authorization=authorization, execution=execution, before_commit=check_storage)
                _require(commands.get(ident, internal=True)['status'] == 'COMPLETED',
                         'generation publication acknowledgement is incomplete')
            except Exception as error:
                traceback.print_exc()
                # A committed generation is authoritative even if an earlier
                # filesystem/SQLite acknowledgement was lost.
                _failed_correction(commands, store, ident, error)
