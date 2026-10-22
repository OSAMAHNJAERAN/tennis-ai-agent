"""Isolated process-crash probe; every media/model boundary is synthetic.

Invoked by test_checkpoint_process_restart.py, never a product worker entrypoint.
The durable queue, completed-stage checkpoint and result publication are real.
"""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import ModuleType


FORBIDDEN = {'torch', 'ultralytics'}
SCOPE = 'SYNTHETIC_PROCESS_CRASH_STORAGE_ONLY_NO_MEDIA_OR_INFERENCE'
blocked_imports = []


class NoInferenceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            blocked_imports.append(fullname)
            raise AssertionError('Inference import forbidden in process probe: ' + fullname)
        return None


if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):
    raise AssertionError('Process probe started with an inference runtime loaded')
sys.meta_path.insert(0, NoInferenceImports())
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Guard is already installed before importing the actual worker or queue.
from filelock import FileLock
from src.api import worker
from src.api.jobs import JobStore, Settings
from src.api.result_document import ResultDocument


def main():
    mode, root_arg, config_arg, analysis_id = sys.argv[1:]
    assert mode in {'crash', 'resume'}
    root, config_path = Path(root_arg), Path(config_arg)
    settings = Settings(root, config_path)
    store = JobStore(root)
    output = store.output_dir(analysis_id)
    audit_path = root / 'child_observations.jsonl'
    pipeline_log = root / 'pipeline_calls.log'  # Survives any analysis cleanup.

    def audit(event, **details):
        record = {'event': event, 'pid': os.getpid(), 'mode': mode,
                  'scope': SCOPE, 'analysis_id': analysis_id, **details}
        with audit_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def check_models(path):
        assert Path(path) == config_path
        value = json.loads(config_path.read_text(encoding='utf-8'))
        assert value['model_system']['family'] == 'yolo26'
        assert value['scope'] == SCOPE
        return value

    def identity(job, effective_config, original):
        return {'scope': SCOPE, 'analysis_id': job['id'], 'owner_id': job['owner_id'],
                'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                'original_bytes': original.stat().st_size,
                'effective_config': effective_config,
                'config_bytes_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()}

    def execute(args, timeout=7200):
        assert args[0] in {'SYNTHETIC_ffmpeg', 'SYNTHETIC_ffprobe'}
        if args[0] == 'SYNTHETIC_ffprobe':
            if '-count_frames' in args:
                audit('synthetic_decode_count', filename=Path(args[-1]).name)
                return '{"streams":[{"nb_read_frames":"2"}]}'
            assert timeout == 60
            audit('synthetic_original_probe')
            return json.dumps({'streams': [{'width': 100, 'height': 100,
                'avg_frame_rate': '30/1', 'r_frame_rate': '30/1',
                'duration': str(2 / 30), 'nb_frames': '2', 'start_time': '0',
                'time_base': '1/30000', 'codec_name': 'SYNTHETIC_NOT_MEDIA'}]})
        destination = Path(args[-1])
        assert destination.parent == output
        if destination.name == 'annotated_web.mp4':
            assert '-crf' in args and args[args.index('-crf') + 1] == '20'
            checkpoint = worker._load_completed_checkpoint(
                output, identity(store.get(analysis_id), effective_config,
                                 store.upload_dir(analysis_id) / 'original.mp4'))
            assert checkpoint is not None
            assert store.get(analysis_id)['status'] == 'PROCESSING'
            assert not (output / '.result/current.store').exists()
            audit('sealed_before_browser_transcode', attempt=store.get(analysis_id)['attempt'],
                  no_result_published=True, blocked_imports=blocked_imports,
                  forbidden_loaded=sorted(FORBIDDEN.intersection(sys.modules)))
            if mode == 'crash':
                # No Python finally/atexit cleanup runs; the OS releases the lock.
                os._exit(73)
        else:
            assert destination.name == 'source.mp4' and '-stats_enc_pre' in args
            stats = Path(args[args.index('-stats_enc_pre') + 1])
            stats.write_text('0 0 0 1/30000 0 1/30\n1 1 1000 1/30000 1 1/30\n',
                             encoding='utf-8')
            audit('synthetic_normalization')
        destination.write_bytes((SCOPE + '\n' + destination.name).encode('ascii'))
        return ''

    class PipelineStub:
        def __init__(self, path):
            assert Path(path) == output / 'inference_config.yaml'

        def run(self, source, directory):
            assert Path(source) == output / 'source.mp4' and Path(directory) == output
            with pipeline_log.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'analysis_id': analysis_id, 'pid': os.getpid(),
                                         'scope': SCOPE}) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            # Normalization/config were produced by the real worker. All other
            # required pipeline outputs are clearly marked synthetic fixtures.
            for name in worker.REQUIRED_PIPELINE_FILES:
                path = output / name
                if not path.exists():
                    path.write_text(json.dumps({'scope': SCOPE, 'artifact': name}),
                                    encoding='utf-8')
            audit('synthetic_pipeline_returned', required_files=len(worker.REQUIRED_PIPELINE_FILES))

    def build_result(directory, job, *, _storage):
        assert directory == output and job['id'] == analysis_id
        assert job['owner_id'] == 'process-restart-owner'
        assert store.get(analysis_id)['status'] == 'PROCESSING'
        rows = _storage.series({'id': f'{analysis_id}:frame:{index}', 'frame_index': index,
                                'analysisId': analysis_id, 'scope': SCOPE}
                               for index in range(2))
        audit('synthetic_result_builder', attempt=job['attempt'])
        return {'contractVersion': '2.0', 'scope': SCOPE, 'accuracyQualified': False,
                'summary': {'id': analysis_id, 'frames': 2, 'fps': 30.,
                            'durationSeconds': 2 / 30, 'status': 'COMPLETED'},
                'syntheticFrames': rows}

    pipeline_module = ModuleType('src.pipeline.phase6_pipeline')
    pipeline_module.Phase6Pipeline = PipelineStub
    sys.modules[pipeline_module.__name__] = pipeline_module
    worker.check_models = check_models
    worker._checkpoint_identity = identity
    worker.binary = lambda name: 'SYNTHETIC_' + name
    worker.execute = execute
    worker.build_result = build_result
    effective_config = check_models(config_path)
    effective_config.setdefault('match_configuration', {})['match_id'] = analysis_id

    # Match production's exclusive ownership contract during recovery/claim/run.
    with FileLock(str(root / 'worker.lock'), timeout=0):
        if mode == 'resume':
            previous = store.get(analysis_id)
            assert previous['status'] == 'PROCESSING' and previous['attempt'] == 1
            store.recover_interrupted(lambda job: worker.recoverable_completed_job(store, job))
            queued = store.get(analysis_id)
            assert queued['status'] == 'UPLOADED' and queued['attempt'] == 1
            audit('recovered_same_job', owner_id=queued['owner_id'], attempt=queued['attempt'])
        job = store.claim()
        assert job['id'] == analysis_id and job['owner_id'] == 'process-restart-owner'
        assert job['attempt'] == (1 if mode == 'crash' else 2)
        audit('claimed', attempt=job['attempt'], owner_id=job['owner_id'])
        real_finish = store.finish

        def finish(job_id, error=None):
            if error is None:
                assert store.get(job_id)['status'] == 'PROCESSING'
                with ResultDocument(output, job_id) as document:
                    assert document.storage is not None
                    assert document.value['summary']['frames'] == 2
                    assert len(document.value['syntheticFrames']) == 2
                audit('published_database_opened_before_completion', attempt=job['attempt'])
            real_finish(job_id, error)

        store.finish = finish
        worker.process_job(store, job, settings)
        assert mode == 'resume', 'Crash boundary was not reached'
        final = store.get(analysis_id)
        assert final['status'] == 'COMPLETED' and final['attempt'] == 2, final
        assert store.claim() is None
        assert not blocked_imports and not FORBIDDEN.intersection(sys.modules)
        audit('completed', attempt=final['attempt'], blocked_imports=blocked_imports,
              forbidden_loaded=sorted(FORBIDDEN.intersection(sys.modules)))


if __name__ == '__main__':
    main()
