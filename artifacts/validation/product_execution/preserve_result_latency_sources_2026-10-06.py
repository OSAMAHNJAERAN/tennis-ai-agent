"""One-shot exact current source preservation before latency diagnostics/edits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).parent / 'result_storage_latency_baseline_2026-10-06'
assert not DIRECTORY.exists(), 'Preserve existing latency baseline'
FILES = ('src/api/result_storage.py', 'src/api/result_document.py', 'src/api/results.py',
         'src/api/reporting.py', 'src/api/app.py', 'src/api/worker.py', 'src/api/product.py', 'src/api/demos.py',
         'tests/test_result_storage.py', 'tests/test_result_document.py', 'tests/test_bounded_result_contract.py',
         'tests/test_bounded_reporting.py', 'tests/test_worker_result_publication.py',
         'docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
         'docs/deployment/PRODUCT_RESTORE_MANIFEST.json',
         'artifacts/validation/product_execution/finalize_result_storage_evidence_2026-10-06.json')
bindings = {}
for name in FILES:
    data = (ROOT / name).read_bytes()
    destination = DIRECTORY / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(data)
    assert destination.read_bytes() == data
    bindings[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
with (DIRECTORY / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'beforeLatencyChanges': True,
               'bindings': bindings}, stream, indent=2, allow_nan=False)
print(json.dumps({'preservedFiles': len(bindings), 'directory': DIRECTORY.relative_to(ROOT).as_posix()}))
