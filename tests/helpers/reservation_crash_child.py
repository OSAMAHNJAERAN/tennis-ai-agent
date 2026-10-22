"""Real legacy request process exits at admission/queue boundaries; no models."""
import builtins
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
FORBIDDEN = {'torch', 'torchvision', 'ultralytics'}
attempts = []
native_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('Legacy crash child attempted inference import: ' + name)
    return native_import(name, *args, **kwargs)


class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in FORBIDDEN:
            attempts.append(fullname)
            raise AssertionError('Legacy crash child attempted inference import: ' + fullname)
        return None


builtins.__import__ = guarded
sys.meta_path.insert(0, Tripwire())
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.jobs import Settings

storage, stage, record_path = sys.argv[1:]
settings = Settings(Path(storage), Path(storage) / 'unused.yaml', max_upload_bytes=64,
                    max_owner_pending_jobs=1, max_owner_storage_bytes=256)
app = create_app(settings, start_worker=False)
jobs, quotas = app.state.store, app.state.quotas


def exit_with_evidence():
    with jobs.connect() as conn:
        reservation = dict(conn.execute('SELECT * FROM resource_reservations').fetchone())
    ident = reservation['analysis_id']
    original = jobs.upload_dir(ident) / 'original.mp4'
    record = {'pid': os.getpid(), 'stage': stage, 'reservation': reservation,
              'queued_before_exit': jobs.get(ident) is not None,
              'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest() if original.is_file() else None,
              'forbidden_import_attempts': attempts,
              'forbidden_loaded': sorted(FORBIDDEN & set(sys.modules))}
    Path(record_path).write_text(json.dumps(record, indent=2), encoding='utf-8')
    os._exit({'reserved': 87, 'before': 88, 'after': 89}[stage])


if stage == 'reserved':
    reserve = quotas.reserve

    def interrupted_reserve(conn, *args, **kwargs):
        reserve(conn, *args, **kwargs)
        conn.commit()
        exit_with_evidence()

    quotas.reserve = interrupted_reserve
else:
    create = jobs.create

    def interrupted_create(*args, **kwargs):
        if stage == 'after':
            create(*args, **kwargs)
        exit_with_evidence()

    jobs.create = interrupted_create

with TestClient(app) as client:
    response = client.post('/api/analyses?filename=crash.mp4', content=b'\0\0\0\x18ftypisom' + b'x' * 20)
    raise AssertionError('Expected process exit before acknowledgement: ' + response.text)
