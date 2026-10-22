"""Stop only a verified duplicate server command with no listener or children."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import psutil

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).parent
RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()
inspection = json.loads((UNIT / 'inspect_runtime.json').read_text(encoding='utf-8'))
expected = next(record for record in inspection['runtimeCandidates'] if record['pid'] == 62896)
PYTHON = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe').resolve()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def snapshot():
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    assert len(rows) == 10 and not any(row['status'] in ('UPLOADED', 'PROCESSING') for row in rows)
    files = {path.relative_to(ROOT).as_posix(): digest(path)
             for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
    return {'jobs': rows, 'files': files}

process = psutil.Process(expected['pid'])
assert process.create_time() == expected['createTimeUnixSeconds']
assert Path(process.cwd()).resolve() == ROOT and Path(process.exe()).resolve() == PYTHON
assert process.cmdline() == [str(PYTHON), 'scripts/run/serve_product.py']
assert not process.children(recursive=True)
listeners = [row for row in psutil.net_connections(kind='tcp') if row.status == 'LISTEN' and row.laddr.port == 8000]
assert listeners and all(row.pid == 67460 for row in listeners)
assert not any(row.pid == process.pid for row in psutil.net_connections(kind='tcp') if row.status == 'LISTEN')
before = snapshot()
assert before == snapshot() and process.create_time() == expected['createTimeUnixSeconds']
process.terminate()
process.wait(timeout=10)
assert snapshot() == before
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'stoppedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'identity': expected, 'activeListeningServerPreserved': 67460,
               'savedJobsAndFilesUnchanged': True, 'snapshot': before,
               'inspectionSha256': digest(UNIT / 'inspect_runtime.json'),
               'scriptSha256': digest(Path(__file__)),
               'action': 'Stopped only the exact duplicate serve_product.py process with no children or listener. Its parent shell is untouched.',
               'acceptedWorkCancelled': False}, stream, indent=2)
print(json.dumps({'stoppedDuplicatePid': expected['pid'], 'savedJobsUnchanged': len(before['jobs']), 'activeServerPreserved': 67460}))
