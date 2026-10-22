"""Fresh post-start verification with explicit fixed native console identities."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
HELPER = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/verify_runtime.py'
PRIOR = ROOT / 'artifacts/validation/product_execution/court_surface_boundary_2026-10-07/runtime_restore_verification.json'
REVIEW = UNIT / 'hidden_runtime_review.json'
ADAPTER = UNIT / 'hidden_console_runtime.py'

def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

assert binding(HELPER)['sha256'] == '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
bound_paths = (HELPER, REVIEW, ADAPTER, Path(__file__), UNIT / 'runtime_verification_failure.json')
before = {path.name: binding(path) for path in bound_paths}
review = json.loads(REVIEW.read_text(encoding='utf-8'))
assert review['passed'] is True and review['independentReviewPassed'] is True
checks = load_module('_resume_hidden_runtime_checks', HELPER)
adapter = load_module('_resume_native_console_adapter', ADAPTER)
checks.PREFLIGHT = UNIT / 'runtime_preflight.json'
checks.RECEIPT = UNIT / 'runtime_verification_v2.json'
checks.runtime_processes = lambda: adapter.runtime_processes(checks, review['runtime'])
historical = json.loads(PRIOR.read_text(encoding='utf-8'))
assert checks.source_bindings() == historical['sourceBindings']
assert checks.snapshot() == historical['snapshotAfter']
checks.main()
assert checks.source_bindings() == historical['sourceBindings']
assert checks.snapshot() == historical['snapshotAfter']
retired = checks.previous_identities_retired(historical['runtime'])
assert before == {path.name: binding(path) for path in bound_paths}
with (UNIT / 'runtime_verification_v2_wrapper.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'scope': 'Existing-result HTTP compatibility with exact independently reviewed hidden-launch native console identities. Original failed check retained; no inference or saved-state modification.',
               'previousRuntimeIdentitiesRetired': retired, 'evidenceBindings': before,
               'wrapperBinding': binding(Path(__file__)), 'receiptBinding': binding(checks.RECEIPT)},
              stream, indent=2, allow_nan=False)
