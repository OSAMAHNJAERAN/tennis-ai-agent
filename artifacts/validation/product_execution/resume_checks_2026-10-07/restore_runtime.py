"""Fresh preflight and verification around documented startup; never starts/stops."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
OLD = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06'
HELPER = OLD / 'verify_runtime.py'
PRIOR = ROOT / 'artifacts/validation/product_execution/court_surface_boundary_2026-10-07/runtime_restore_verification.json'

def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
spec = importlib.util.spec_from_file_location('_resume_runtime_restoration', HELPER)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
checks.PREFLIGHT = UNIT / 'runtime_preflight.json'
checks.RECEIPT = UNIT / 'runtime_verification.json'
preflight = '--preflight' in sys.argv
historical = json.loads(PRIOR.read_text(encoding='utf-8'))
assert checks.source_bindings() == historical['sourceBindings']
assert checks.snapshot() == historical['snapshotAfter']
if preflight:
    assert checks.runtime_processes()['state'] == 'ABSENT', 'Preserve any newly existing runtime'
checks.main()
assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
assert checks.source_bindings() == historical['sourceBindings']
assert checks.snapshot() == historical['snapshotAfter']
retired = checks.previous_identities_retired(historical['runtime'])
receipt = checks.PREFLIGHT if preflight else checks.RECEIPT
with (UNIT / ('runtime_preflight_wrapper.json' if preflight else 'runtime_verification_wrapper.json')).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'scope': 'Read-only checks around documented server restoration; no inference or saved-job modification.',
               'stage': 'ABSENT_PREFLIGHT' if preflight else 'POST_START_VERIFICATION',
               'previousRuntimeIdentitiesRetired': retired, 'helperBinding': binding(HELPER),
               'priorReceiptBinding': binding(PRIOR), 'wrapperBinding': binding(Path(__file__)),
               'receiptBinding': binding(receipt)}, stream, indent=2, allow_nan=False)
