"""Crash the real creation route immediately before/after its SQL commit."""
from contextlib import contextmanager
import builtins
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
        raise AssertionError('Creation child attempted model import: '+name)
    return native_import(name, *args, **kwargs)


class Tripwire(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in FORBIDDEN:
            attempts.append(fullname)
            raise AssertionError('Creation child attempted model import: '+fullname)
        return None


builtins.__import__ = guarded
sys.meta_path.insert(0, Tripwire())
from fastapi.testclient import TestClient
from src.api.app import create_app
from src.api.creation_requests import CreationRequests
from src.api.jobs import Settings

storage, kind, failure, record_path, request_path = sys.argv[1:]
request = json.loads(Path(request_path).read_text(encoding='utf-8'))
app = create_app(Settings(Path(storage), Path(storage)/'config', max_upload_bytes=64,
    max_owner_pending_jobs=2, max_owner_storage_bytes=256), start_worker=False)
jobs = app.state.store
connect = jobs.connect
save = CreationRequests.save


def record_and_exit(conn, descriptor):
    record = {'pid': os.getpid(), 'kind': kind, 'case': failure, 'descriptor': descriptor,
        'visible_sessions_at_exit': conn.execute('SELECT COUNT(*) FROM upload_sessions').fetchone()[0],
        'visible_drafts_at_exit': conn.execute('SELECT COUNT(*) FROM matches').fetchone()[0],
        'forbidden_import_attempts': attempts, 'forbidden_loaded': sorted(FORBIDDEN & set(sys.modules))}
    Path(record_path).write_text(json.dumps(record, indent=2), encoding='utf-8')
    os._exit(88 if failure == 'before' else 89)


if failure == 'before':
    def interrupted_save(*args):
        save(*args)
        record_and_exit(args[0], args[-1])
    CreationRequests.save = staticmethod(interrupted_save)
else:
    @contextmanager
    def interrupted_connect():
        with connect() as conn:
            yield conn
        # The original context has actually committed and closed. Observe the
        # durable record on a new connection before the route presents its ack.
        with connect() as committed:
            row = committed.execute('SELECT descriptor_json FROM upload_creation_requests '
                'WHERE owner_id=? AND intent_key=?', ('local', request['key'])).fetchone()
            if row is not None:
                record_and_exit(committed, json.loads(row['descriptor_json']))
    jobs.connect = interrupted_connect

with TestClient(app) as client:
    route = '/api/uploads' if kind == 'upload' else '/api/batches'
    response = client.post(route, headers={'Idempotency-Key': request['key']}, json=request['payload'])
    raise AssertionError('Expected process exit before acknowledgement: '+response.text)
