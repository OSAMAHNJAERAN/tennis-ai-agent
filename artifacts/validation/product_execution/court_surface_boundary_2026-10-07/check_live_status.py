"""Fresh read-only live website and available-memory check; no reload or inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import psutil

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06'
HELPER = OLD / 'verify_runtime.py'
OUTPUT = UNIT / 'live_status.json'
def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

assert not OUTPUT.exists(), 'Preserve the consumed read-only status check'
assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
spec = importlib.util.spec_from_file_location('_readonly_existing_runtime_checks', HELPER)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
record = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
          'runtimeStoppedOrStarted': False, 'inferenceStarted': False,
          'scope': 'Read-only current identities, API health, served frontend and retained source/product bindings. Prior exports verified by the preserved 2026-10-06 receipt; no new inference or accuracy verification.',
          'helperBinding': binding(HELPER), 'scriptBinding': binding(Path(__file__)),
          'availableRamBeforeBytes': psutil.virtual_memory().available}
try:
    original = json.loads((OLD / 'runtime_verification.json').read_text(encoding='utf-8'))
    before = checks.snapshot()
    sources = checks.source_bindings()
    assert sources == original['sourceBindings'], 'Production sources/frontend changed since retained verification'
    assert before == original['snapshotAfter'], 'Live saved jobs/files changed since retained verification'
    runtime = checks.runtime_processes()
    record['runtime'] = runtime
    assert runtime['state'] == 'OWNED_IDLE_PAIR'
    status, headers, data = checks.request('/api/health')
    health = json.loads(data)
    record['health'] = health
    assert status == 200 and health['status'] == 'ok' and health['worker_available']
    record['frontend'] = checks.frontend_checks()
    assert runtime == checks.runtime_processes(), 'Runtime changed during read-only checks'
    assert sources == checks.source_bindings()
    assert before == checks.snapshot()
    record.update(passed=True, sourceBindings=sources, savedJobsUnchanged=len(before['jobs']),
                  savedAnalysisFilesUnchanged=len(before['savedAnalysisFiles']),
                  sameRuntimeIdentitiesAsPriorReceipt=runtime == original['runtime'])
except BaseException as error:
    record.update(error=repr(error), traceback=traceback.format_exc())
finally:
    record['availableRamAfterBytes'] = psutil.virtual_memory().available
    record['realWorkerGuardRequiredAvailableBytes'] = 2500000000
    record['realWorkerMemoryPreflightCurrentlyPossible'] = record['availableRamAfterBytes'] >= 2500000000
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
print(json.dumps({key: record.get(key) for key in ('passed', 'error', 'savedJobsUnchanged', 'savedAnalysisFilesUnchanged', 'availableRamAfterBytes', 'realWorkerMemoryPreflightCurrentlyPossible')}))
raise SystemExit(0 if record['passed'] else 2)
