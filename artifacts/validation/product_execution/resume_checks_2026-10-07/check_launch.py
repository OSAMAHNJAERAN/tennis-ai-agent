"""Last read-only absence/source/job check immediately before documented startup."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
HELPER = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/verify_runtime.py'
spec = importlib.util.spec_from_file_location('_resume_last_launch_check', HELPER)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
before = json.loads((UNIT / 'runtime_preflight.json').read_text(encoding='utf-8'))
worker = json.loads((UNIT / 'worker_review.json').read_text(encoding='utf-8'))
assert worker['passed'] and worker['actualWorkerCompletionVerified']
assert checks.runtime_processes()['state'] == 'ABSENT'
assert checks.source_bindings() == before['sourceBindings']
assert checks.snapshot() == before['snapshot']
with (UNIT / 'launch_ready.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'runtimeState': 'ABSENT', 'sourceAndSavedJobsUnchanged': True,
               'workerAttemptCompleteAndRetired': True}, stream, indent=2)
print('Ready for documented startup')
