"""Real local API + correction-only synthetic worker; never claim upload jobs.

The resource values are explicitly injected software-test constants. The
production guard, native rebuild, codecs, publication and authorization run
unchanged. No host admission probe or inference runtime is permitted here.
"""
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import socket
import sys
import threading
import traceback

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
os.chdir(ROOT)
manifest = json.loads((ART / 'fixture_manifest.json').read_text())
assert manifest['status'] == 'PREPARED_VERIFIED_SYNTHETIC_FIXTURE_SERVER_NOT_STARTED'
store_root = Path(manifest['storage_root']).resolve()
assert store_root == ART / 'service-v2'
os.environ.update(TENNIS_STORAGE_ROOT=str(store_root),
                  TENNIS_PIPELINE_CONFIG=manifest['config'],
                  TENNIS_REQUIRE_AUTH='1', TENNIS_SECURE_COOKIES='0',
                  PYTHONDONTWRITEBYTECODE='1', TEMP=str(ART / 'temp-v2'),
                  TMP=str(ART / 'temp-v2'), TMPDIR=str(ART / 'temp-v2'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'torchvision', 'ultralytics'}
attempts, resource_calls, claimed = [], [], []
native_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('No inference imports in this synthetic browser fixture: ' + name)
    return native_import(name, *args, **kwargs)


class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in FORBIDDEN:
            attempts.append(fullname)
            raise AssertionError('No inference module lookup: ' + fullname)


builtins.__import__ = guarded
sys.meta_path.insert(0, Tripwire())
from scripts.train import train_yolo26_author_ball_repair as resource_module


def synthetic_snapshot(directory, torch=None):
    assert torch is None, 'A model/GPU admission is outside this fixture'
    resource_calls.append('explicit_synthetic_cpu_snapshot')
    return {'ram': 8 * 1024**3, 'disk': 8 * 1024**3, 'rss': 0,
            'peak_rss': 0, 'vram': None}


resource_module.resource_snapshot = synthetic_snapshot
from src.api.app import create_app
from src.api.jobs import Settings
from src.api.corrections import CorrectionStore
from src.api.correction_worker import process_correction
import uvicorn

settings = Settings(store_root, Path(manifest['config']), require_auth=True,
                    secure_cookies=False)
app = create_app(settings, start_worker=False,
                 demos_manifest=ART / 'empty-demos.json',
                 frontend_root=ROOT / 'frontend/dist')
jobs = app.state.store
commands = CorrectionStore(jobs)
stop = threading.Event()


def corrections_only():
    try:
        while not stop.is_set():
            command = commands.claim()
            if command is not None:
                claimed.append({'id': command['id'], 'analysis_id': command['analysis_id']})
                process_correction(jobs, command, settings)
            stop.wait(.15)
    except BaseException:
        (ART / 'worker_failure.txt').write_text(traceback.format_exc())
        stop.set()


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('127.0.0.1', 0))
port = sock.getsockname()[1]
index = ROOT / 'frontend/dist/index.html'
runtime = {'server_pid': os.getpid(), 'host': '127.0.0.1', 'port': port,
           'base_url': f'http://127.0.0.1:{port}',
           'started_utc': datetime.now(timezone.utc).isoformat(),
           'storage_root': str(store_root), 'start_worker': False,
           'worker_scope': 'ONLY CorrectionStore.claim; original uploads remain queued',
           'synthetic_resources': True, 'actual_host_ram_gpu_probes': 0,
           'frontend_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
           'frontend_files': {p.relative_to(index.parent).as_posix():
                              hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(index.parent.rglob('*')) if p.is_file()}}
(ART / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
thread = threading.Thread(target=corrections_only, name='synthetic-corrections-only', daemon=True)
thread.start()
print(json.dumps({'host': '127.0.0.1', 'port': port, 'pid': os.getpid()}), flush=True)
try:
    uvicorn.Server(uvicorn.Config(app, log_level='warning')).run(sockets=[sock])
finally:
    stop.set()
    thread.join(timeout=30)
    sock.close()
    receipt = {'stopped_utc': datetime.now(timezone.utc).isoformat(),
               'server_pid': os.getpid(), 'correction_claims': claimed,
               'synthetic_resource_calls': len(resource_calls),
               'actual_host_ram_gpu_probes': 0, 'actual_model_inference': False,
               'inference_import_attempts': attempts,
               'forbidden_runtimes_loaded': sorted(FORBIDDEN & set(sys.modules)),
               'thread_joined': not thread.is_alive(),
               'original_job_dispatcher_started': False}
    (ART / 'server_stopped.json').write_text(json.dumps(receipt, indent=2) + '\n')
