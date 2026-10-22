"""Actual API process exit at the retry queue commit boundary; no models."""
import builtins
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'torchvision', 'ultralytics'}
attempts = []
native_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('Retry child attempted model import: '+name)
    return native_import(name, *args, **kwargs)


class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in FORBIDDEN:
            attempts.append(fullname)
            raise AssertionError('Retry child attempted model import: '+fullname)
        return None


builtins.__import__ = guarded
sys.meta_path.insert(0, Tripwire())
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.jobs import Settings

storage, source, failure, record = sys.argv[1:]
settings = Settings(Path(storage), Path(storage)/'unused.yaml', max_upload_bytes=64,
                    max_owner_pending_jobs=1, max_owner_storage_bytes=1024)
app = create_app(settings, start_worker=False)
jobs = app.state.store
create = jobs.create


def interrupted(*args, **kwargs):
    if failure == 'after':
        create(*args, **kwargs)
    with jobs.connect() as conn:
        selected = dict(conn.execute('SELECT * FROM analysis_retry_intents').fetchone())
    original = jobs.upload_dir(selected['analysis_id'])/'original.mp4'
    observation = {'pid': os.getpid(), 'case': failure, 'selected': selected,
        'queued_before_exit': jobs.get(selected['analysis_id']) is not None,
        'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'forbidden_import_attempts': attempts, 'forbidden_loaded': sorted(FORBIDDEN & set(sys.modules))}
    Path(record).write_text(json.dumps(observation, indent=2), encoding='utf-8')
    os._exit(85 if failure == 'before' else 86)


jobs.create = interrupted
with TestClient(app) as client:
    response = client.post('/api/analyses/'+source+'/retry',
                           headers={'Idempotency-Key': 'retry-intent-test-000001'})
    raise AssertionError('Expected child exit before HTTP acknowledgement: '+response.text)
