"""Preserve exact sources preceding bounded result construction and readers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent / 'result_storage_baseline_2026-10-06'
DEST.mkdir(exist_ok=False)
names = ('src/api/results.py', 'src/api/worker.py', 'src/api/app.py', 'src/api/reporting.py',
         'src/api/product.py', 'src/api/media_revision.py', 'src/api/demos.py',
         'tests/test_worker_result_publication.py', 'docs/plans/TENNIS_EXECUTION_PROGRESS.md',
         'docs/deployment/PRODUCT_INSTALLATION.md', 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
bindings = {}
for name in names:
    data = (ROOT / name).read_bytes()
    destination = DEST / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(data)
    assert destination.read_bytes() == data
    bindings[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
with (DEST / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'bindings': bindings,
               'scope': 'Exact pre-result-storage production, publication test and documentation; no inference.'}, stream, indent=2)
print(json.dumps({'preservedFiles': len(bindings), 'destination': DEST.relative_to(ROOT).as_posix()}))
