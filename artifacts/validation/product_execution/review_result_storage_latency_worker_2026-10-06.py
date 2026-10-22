"""Check final tested source and the reviewed worker harness without importing it."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

inventory = json.loads((BASE / 'result_storage_latency_backend_2026-10-06/source_bindings.json').read_text(encoding='utf-8'))
assert all(digest(ROOT / name) == expected for name, expected in inventory.items())
worker = ROOT / 'scripts/audit/verify_result_storage_latency_worker_smoke.py'
assert digest(worker) == '279464db5328b87b0a524b9f669176f9498b609522718ddc4b2cab7873d7fefe'
ast.parse(worker.read_text(encoding='utf-8'))
unit = BASE / 'result_storage_latency_worker_smoke_2026-10-06'
preparation = json.loads((unit / 'preparation.json').read_text(encoding='utf-8'))
assert preparation['scriptSha256'] == digest(worker)
assert preparation['originReceiptSha256'] == digest(unit / 'origin.json')
assert not (unit / 'launch_marker.json').exists()
files = [worker, unit / 'preparation.json', unit / 'origin.json', Path(__file__),
         BASE / 'result_storage_latency_backend_2026-10-06/receipt.json',
         BASE / 'result_storage_latency_measurement_2026-10-06/audit/terminal_audit.json']
receipt = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
           'currentTestedSourceBindings': inventory,
           'inputs': {path.relative_to(ROOT).as_posix(): digest(path) for path in files},
           'helperImported': False, 'workloadsLaunched': False,
           'scope': 'Static source, preparation and full-suite binding check before one guarded actual-worker smoke.'}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'passed': True, 'testedInputsVerified': len(inventory), 'helperImported': False}))
