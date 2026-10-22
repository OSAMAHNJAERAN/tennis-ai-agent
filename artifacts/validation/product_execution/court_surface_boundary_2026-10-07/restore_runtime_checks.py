"""Use the preserved read-only runtime verifier with fresh evidence paths."""
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
def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
before = binding(HELPER)
spec = importlib.util.spec_from_file_location('_fresh_runtime_restoration_checks', HELPER)
checks = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checks
spec.loader.exec_module(checks)
checks.PREFLIGHT = UNIT / 'runtime_restore_preflight.json'
checks.RECEIPT = UNIT / 'runtime_restore_verification.json'
preflight = '--preflight' in sys.argv
old_receipt = json.loads((OLD / 'runtime_verification.json').read_text(encoding='utf-8'))
assert checks.source_bindings() == old_receipt['sourceBindings']
assert checks.snapshot() == old_receipt['snapshotAfter']
if preflight:
    assert checks.runtime_processes()['state'] == 'ABSENT', 'Preserve any newly existing runtime'
checks.main()
assert binding(HELPER) == before
assert checks.source_bindings() == old_receipt['sourceBindings']
assert checks.snapshot() == old_receipt['snapshotAfter']
retired = checks.previous_identities_retired(old_receipt['runtime'])
receipt = checks.PREFLIGHT if preflight else checks.RECEIPT
with (UNIT / ('runtime_restore_preflight_wrapper.json' if preflight else 'runtime_restore_verification_wrapper.json')).open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'scope': 'Fresh read-only preflight/post-start compatibility checks for documented website restoration. No model inference or persistent analysis change.',
        'stage': 'ABSENT_RUNTIME_PREFLIGHT' if preflight else 'DOCUMENTED_SERVER_POST_START',
        'priorRuntimeIdentitiesRetired': retired, 'originalHelperUnchanged': True,
        'helperBinding': before, 'wrapperBinding': binding(Path(__file__)),
        'receiptBinding': binding(receipt)}, stream, indent=2, allow_nan=False)
