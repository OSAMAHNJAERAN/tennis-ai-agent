"""Actual legacy process death releases its lease, never its retained media."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi.testclient import TestClient
import pytest

from src.api.app import create_app
from src.api.jobs import Settings

VIDEO = b'\0\0\0\x18ftypisom' + b'x' * 20


@pytest.mark.parametrize('stage', ['reserved', 'before', 'after'])
def test_actual_legacy_process_exit_then_owner_recovery_preserves_acceptance_and_bytes(tmp_path, stage):
    root, record_path = tmp_path / 'storage', tmp_path / 'child.json'
    helper = Path(__file__).parent / 'helpers/reservation_crash_child.py'
    result = subprocess.run([sys.executable, '-B', str(helper), str(root), stage, str(record_path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == {'reserved': 87, 'before': 88, 'after': 89}[stage], result.stderr
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if evidence_dir := os.environ.get('TENNIS_RECOVERY_EVIDENCE_DIR'):
        target = Path(evidence_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / ('legacy_' + stage + '.json')).open('x', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)
    assert record['pid'] != os.getpid()
    assert record['forbidden_import_attempts'] == record['forbidden_loaded'] == []
    assert record['reservation']['active'] == record['reservation']['recovery_version'] == 1
    assert record['queued_before_exit'] == (stage == 'after')
    assert record['original_sha256'] == (None if stage == 'reserved' else hashlib.sha256(VIDEO).hexdigest())
    settings = Settings(root, root / 'unused.yaml', max_upload_bytes=64,
                        max_owner_pending_jobs=1, max_owner_storage_bytes=256)
    app = create_app(settings, start_worker=False)
    ident = record['reservation']['analysis_id']
    before_job = app.state.store.get(ident)
    with TestClient(app) as client:
        before = client.get('/api/usage').json()
        assert before['pending_jobs'] == 1 and before['committed_bytes'] == 32
        response = client.post('/api/usage/reconcile')
        assert response.status_code == 200
        assert response.json()['reconciled_reservations'] == 1
        assert response.json()['busy_reservations'] == response.json()['unverified_reservations'] == 0
        usage = response.json()['usage']
        assert usage['pending_jobs'] == int(stage == 'after')
        assert usage['reserved_input_bytes'] == 0
        assert usage['stored_bytes'] == usage['committed_bytes'] == (0 if stage == 'reserved' else 32)
        again = client.post('/api/usage/reconcile')
        assert again.status_code == 200 and again.json()['reconciled_reservations'] == 0
        assert app.state.store.get(ident) == before_job
        if stage != 'reserved':
            assert (app.state.store.upload_dir(ident) / 'original.mp4').read_bytes() == VIDEO
        with app.state.store.connect() as conn:
            reservation = dict(conn.execute('SELECT * FROM resource_reservations WHERE analysis_id=?',
                                             (ident,)).fetchone())
        assert reservation['active'] == 0 and reservation['owner_id'] == 'local'
        assert reservation['size_bytes'] == 32
        next_upload = client.post('/api/uploads', headers={'Idempotency-Key': 'next-after-crash-recovery-0001'},
                                  json={'filename': 'next.mp4', 'size_bytes': 32})
        assert next_upload.status_code == (429 if stage == 'after' else 201)
