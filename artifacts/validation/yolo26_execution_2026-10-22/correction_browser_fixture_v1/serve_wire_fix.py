"""Explicitly launched local synthetic fixture, using the real API/worker.

Run only after the separately running source-pinned verification is terminal.
The server is bound to loopback; nothing is published or transferred externally.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
manifest = json.loads((ART/'fixture_manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'PREPARED_VERIFIED_SYNTHETIC_FIXTURE_SERVER_NOT_STARTED'
assert manifest['host'] == '127.0.0.1'
root = Path(manifest['storage_root']).resolve()
assert root == ART/'service-v2' and root.is_relative_to(ART)
os.chdir(ROOT)
os.environ.update(TENNIS_STORAGE_ROOT=str(root), TENNIS_PIPELINE_CONFIG=manifest['config'],
    TENNIS_REQUIRE_AUTH='1', TENNIS_SECURE_COOKIES='0', PYTHONDONTWRITEBYTECODE='1',
    TEMP=str(ART/'temp-v2'), TMP=str(ART/'temp-v2'))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
    check.bind((manifest['host'], manifest['port_checked_available']))
from src.api.app import create_app
from src.api.jobs import Settings
import uvicorn
settings = Settings(root, Path(manifest['config']), require_auth=True, secure_cookies=False)
app = create_app(settings, start_worker=True, demos_manifest=ART/'empty-demos.json',
                 frontend_root=Path(manifest['frontend_root']))
index = Path(manifest['frontend_root'])/'index.html'
runtime = {'server_pid': os.getpid(), 'host': manifest['host'], 'port': manifest['port_checked_available'],
    'storage_root': str(root), 'frontend_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
    'scope': 'EXPLICIT LOCAL SYNTHETIC TEST; REAL API AND EXCLUSIVE WORKER; NO PUBLIC RELEASE'}
(ART/'runtime_wire_fix.json').write_text(json.dumps(runtime, indent=2)+'\n', encoding='utf-8')
uvicorn.run(app, host=manifest['host'], port=manifest['port_checked_available'], log_level='warning')
