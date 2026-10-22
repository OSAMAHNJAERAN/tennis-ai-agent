"""Real worker/native pipeline process-loss probe with injected vision results.

FFmpeg normalization/playback, source mapping, native numerical/storage/render
stages, JobStore, seals, result builder and publication execute normally. Only
asset admission/provenance and the three selected model result boundaries are
synthetic. The loaded pipeline module contains its actual AST definitions.
"""
import builtins
from copy import deepcopy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'torchvision', 'ultralytics'}
SCOPE = 'NATIVE_WORKER_PROCESS_RECOVERY_WITH_SYNTHETIC_SELECTED_VISION'
IMPORT_ATTEMPTS = []
NATIVE_IMPORT = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        IMPORT_ATTEMPTS.append({'mechanism': 'builtins.__import__', 'name': name})
        raise AssertionError('Native worker child attempted forbidden inference import: ' + name)
    return NATIVE_IMPORT(name, *args, **kwargs)


class NoInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in FORBIDDEN:
            IMPORT_ATTEMPTS.append({'mechanism': 'meta_path', 'name': fullname})
            raise AssertionError('Native worker child attempted forbidden inference import: ' + fullname)
        return None


if any(name.split('.', 1)[0] in FORBIDDEN for name in sys.modules):
    raise AssertionError('Child already loaded an inference runtime before guard installation')
builtins.__import__ = guarded_import
sys.meta_path.insert(0, NoInferenceImports())

from filelock import FileLock
import numpy as np
import yaml
from src.api import worker
from src.api.jobs import JobStore, Settings
from src.api.result_document import ResultDocument
from src.detection import yolo26_bundle
from src.detection.yolo26_adapters import YOLO26BallDetector, YOLO26CourtPoseDetector, YOLO26PlayerPoseDetector
from src.utils.ball_inference_journal import BallInferenceJournal
from tests.helpers.native_pipeline_source import (
    Boxes, HEIGHT, WIDTH, InjectedModel, load_native_pipeline_source,
)


# This independent explicit list also checks the actual adapter's tile order.
# The selected model boundary supplies crop-local results; the real adapter
# performs origin translation, clipping, proposal collection and deduplication.
BALL_VIEW_REGIONS = ((0, 0, 320, 240), (0, 0, 160, 160),
                     (160, 0, 320, 160), (0, 80, 160, 240), (160, 80, 320, 240))
EMPTY_BALL_FRAME = 4


class TiledInjectedBallModel(InjectedModel):
    """Synthetic model results underneath unchanged native tile extraction."""

    def __init__(self, trace, after_result_prepared):
        super().__init__('ball', trace)
        self.view_calls, self.complete_view_frame_indices = [], []
        self._frame = self._image = None
        self._next_view = len(BALL_VIEW_REGIONS)
        self._after_result_prepared = after_result_prepared

    def predict(self, image, **options):
        if image.shape[:2] == (HEIGHT, WIDTH):
            assert self._next_view == len(BALL_VIEW_REGIONS), 'Previous tiled frame is incomplete'
            index = self.frame_index(image)
            assert 0 <= index < 8
            if self._frame is not None:
                assert index == self._frame + 1
            self._frame, self._image, self._next_view = index, image, 0
        assert self._frame is not None and 0 <= self._next_view < len(BALL_VIEW_REGIONS)
        index, view_index = self._frame, self._next_view
        left, top, right, bottom = BALL_VIEW_REGIONS[view_index]
        assert image.shape[:2] == (bottom - top, right - left)
        assert np.array_equal(image, self._image[top:bottom, left:right])
        assert options['device'] == 'cpu' and options['classes'] == [0]
        assert options['end2end'] is True and options['nms'] is False
        self.calls.append(deepcopy(options))
        self.frame_indices.append(index)
        self.trace.append(('ball', 'predict', index))
        result = self._result(index)
        coords = result.boxes.xyxy
        visible = ((coords[:, 0] >= left) & (coords[:, 1] >= top)
                   & (coords[:, 2] <= right) & (coords[:, 3] <= bottom))
        if index == EMPTY_BALL_FRAME:
            visible[:] = False
        result.boxes = Boxes(xyxy=coords[visible] - np.array([left, top, left, top]),
                             conf=result.boxes.conf[visible], id=None)
        record = {'frame_index': index, 'view_index': view_index,
                  'region': [left, top, right, bottom], 'candidates': len(result.boxes),
                  'result_prepared': True, 'returned_to_adapter': False}
        self.view_calls.append(record)
        self._next_view += 1
        # The crash callback is inside the synthetic model boundary. A native
        # extract_candidates call has already consumed frame 2's full view,
        # but cannot complete or journal this frame after the following exit.
        self._after_result_prepared(index, view_index)
        record['returned_to_adapter'] = True
        if self._next_view == len(BALL_VIEW_REGIONS):
            self.complete_view_frame_indices.append(index)
        return [result]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    mode, storage_arg, config_arg, analysis_id = sys.argv[1:]
    assert mode in {'crash', 'crash_mid_tile', 'resume', 'continuous'}
    storage, config_path = Path(storage_arg), Path(config_arg)
    store = JobStore(storage)
    settings = Settings(storage, config_path)
    output = store.output_dir(analysis_id)
    trace, selected_models, bundles = [], {}, []
    committed_raw_frontier = 0

    def audit(event, **details):
        value = {'event': event, 'scope': SCOPE, 'mode': mode, 'pid': os.getpid(),
                 'analysis_id': analysis_id, **details}
        with (storage / 'native_worker_observations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def runtime_observations():
        return {'import_attempts': list(IMPORT_ATTEMPTS),
                'forbidden_loaded': sorted(name for name in sys.modules if name.split('.', 1)[0] in FORBIDDEN),
                'source_loader_import_attempts': list(source_module.fixture_source_import_attempts)}

    def model_observations():
        return {'selected_models_constructed': len(selected_models), 'bundles_constructed': len(bundles),
                'players_track_calls': len(selected_models['players'].tracking_calls),
                'players_predict_calls': len(selected_models['players'].calls),
                'court_predict_calls': len(selected_models['court'].calls),
                'ball_predict_calls': len(selected_models['ball'].calls),
                'ball_frame_indices': list(selected_models['ball'].frame_indices),
                'ball_full_frame_indices': [row['frame_index'] for row in selected_models['ball'].view_calls
                                            if row['view_index'] == 0]
                    if isinstance(selected_models['ball'], TiledInjectedBallModel)
                    else list(selected_models['ball'].frame_indices),
                'ball_view_calls': list(getattr(selected_models['ball'], 'view_calls', [])),
                'ball_complete_view_frame_indices': list(getattr(selected_models['ball'], 'complete_view_frame_indices', [])),
                'trace': list(trace)}

    def after_ball_result_prepared(frame_index, view_index):
        if mode == 'crash_mid_tile' and (frame_index, view_index) == (2, 1):
            assert committed_raw_frontier == 2
            assert selected_models['ball'].complete_view_frame_indices == [0, 1]
            assert store.get(analysis_id)['status'] == 'PROCESSING'
            assert not (output / '.pipeline_checkpoint.json').exists()
            assert not (output / '.result/current.store').exists()
            audit('hard_exit_mid_tile', next_frame=committed_raw_frontier,
                  incomplete_frame_index=frame_index, incomplete_view_index=view_index,
                  full_frame_prediction_returned=True, tile_prediction_result_prepared=True,
                  **model_observations(), **runtime_observations())
            # Terminate without returning this tile result to the real native
            # adapter, and without Python finally/atexit or a raw-frame commit.
            os._exit(73)

    def check_models(path):
        assert Path(path) == config_path
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        assert config['scope'] == SCOPE
        assert config['model_system']['family'] == config['ball_detection']['backend'] == 'yolo26'
        assert config['event_detection']['authoritative_enabled'] is False
        return config

    def identity(job, effective_config, original):
        # Explicit synthetic substitute for real checkpoint/runtime admission.
        # It still binds job owner, all effective settings and exact input bytes.
        return {'scope': SCOPE, 'analysis_id': job['id'], 'owner_id': job['owner_id'],
                'original_sha256': digest(original), 'original_bytes': original.stat().st_size,
                'effective_config': effective_config, 'config_sha256': digest(config_path)}

    def build_bundle(config):
        assert not bundles, 'One worker attempt must construct one selected three-role bundle'
        player, ball, court = config['player_detection'], config['ball_detection'], config['court_detection']
        if ball['tiles']['enabled']:
            assert ball['tiles']['size_px'] == 160 and ball['tiles']['overlap'] == 0
        else:
            assert mode != 'crash_mid_tile'
        for role in ('players', 'ball', 'court'):
            selected_models[role] = (TiledInjectedBallModel(trace, after_ball_result_prepared)
                if role == 'ball' and ball['tiles']['enabled'] else InjectedModel(role, trace))
        bundle = SimpleNamespace(
            player_detector=YOLO26PlayerPoseDetector(player['model'], tracker=player['tracker'],
                confidence=player['confidence'], imgsz=player['imgsz'], device='cpu',
                _model=selected_models['players']),
            ball_detector=YOLO26BallDetector(ball['model'], imgsz=ball['imgsz'],
                high_conf=ball['high_conf'], low_conf=ball['low_conf'], device='cpu',
                tiles=ball['tiles'], _model=selected_models['ball']),
            court_detector=YOLO26CourtPoseDetector(court['model_path'], imgsz=court['imgsz'],
                device='cpu', _model=selected_models['court']),
            artifact_manifest={'status': 'SYNTHETIC_SELECTED_RESULTS_NOT_QUALIFIED'})
        bundles.append(bundle)
        return bundle

    # Bind the worker's normal import site to the actual complete source-loaded
    # module. No pipeline method or export/result implementation is stubbed.
    source_module = load_native_pipeline_source()
    pipeline_module = ModuleType('src.pipeline.phase6_pipeline')
    pipeline_module.__dict__.update(vars(source_module))
    sys.modules['src.pipeline.phase6_pipeline'] = pipeline_module
    yolo26_bundle.build_yolo26_bundle = build_bundle
    worker.check_models = check_models
    worker._checkpoint_identity = identity

    real_execute = worker.execute

    def execute(arguments, timeout=7200):
        destination = Path(arguments[-1])
        if destination == output / 'source.mp4' and '-stats_enc_pre' in arguments:
            audit('normalizing_source', real_ffmpeg=True)
        elif destination == output / 'annotated_web.mp4' and '-c:v' in arguments:
            audit('encoding_browser_video', real_ffmpeg=True)
        elif '-count_frames' in arguments:
            audit('counting_actual_decoded_frames', filename=destination.name, real_ffprobe=True)
        return real_execute(arguments, timeout=timeout)

    worker.execute = execute
    real_append = BallInferenceJournal.append

    def append(journal, frame_index, candidates):
        nonlocal committed_raw_frontier
        real_append(journal, frame_index, candidates)
        committed_raw_frontier = journal.next_frame
        audit('committed_raw_frame', frame_index=frame_index, next_frame=journal.next_frame)
        if mode == 'crash' and journal.next_frame == 2:
            assert frame_index == 1
            assert store.get(analysis_id)['status'] == 'PROCESSING'
            assert not (output / '.pipeline_checkpoint.json').exists()
            assert not (output / '.result/current.store').exists()
            audit('hard_exit_after_raw_frontier', next_frame=journal.next_frame,
                  **model_observations(), **runtime_observations())
            # No Python finally/atexit cleanup runs. The OS releases the worker
            # lock and SQLite handles after the real committed frame frontier.
            os._exit(73)

    BallInferenceJournal.append = append
    with FileLock(str(storage / 'worker.lock'), timeout=0):
        if mode == 'resume':
            previous = store.get(analysis_id)
            assert previous['status'] == 'PROCESSING' and previous['attempt'] == 1
            store.recover_interrupted(lambda job: worker.recoverable_completed_job(store, job))
            recovered = store.get(analysis_id)
            assert recovered['status'] == 'UPLOADED', recovered
            audit('recovered_same_analysis', previous_attempt=recovered['attempt'])
        job = store.claim()
        assert job['id'] == analysis_id
        assert job['attempt'] == (2 if mode == 'resume' else 1)
        audit('claimed', attempt=job['attempt'], owner_id=job['owner_id'])
        worker.process_job(store, job, settings)
        final = store.get(analysis_id)
        assert mode not in {'crash', 'crash_mid_tile'}, 'Hard process-exit frontier was never reached'
        assert final['status'] == 'COMPLETED', final
        assert final['attempt'] == (2 if mode == 'resume' else 1)
        assert store.claim() is None
        observations = runtime_observations()
        assert not any(observations.values()), observations
        with ResultDocument(output, analysis_id) as document:
            assert document.storage is not None
            assert document.value['summary']['frames'] == 8
            assert document.value['provenance']['productionQualified'] is False
            publication = {'generation': document.generation,
                           'frames': document.value['summary']['frames']}
        audit('completed', attempt=final['attempt'], publication=publication,
              **model_observations(), **observations)


if __name__ == '__main__':
    main()
