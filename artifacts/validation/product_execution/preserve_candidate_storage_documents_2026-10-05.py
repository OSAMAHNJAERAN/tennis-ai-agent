"""Preserve exact preceding documentation before recording this capacity unit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent / 'candidate_storage_document_baseline_2026-10-05'
DEST.mkdir(exist_ok=False)
bindings = {}
for name in ('docs/plans/TENNIS_EXECUTION_PROGRESS.md',
             'docs/deployment/PRODUCT_INSTALLATION.md',
             'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'):
    source, target = ROOT / name, DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    bindings[name] = expected
with (DEST / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'bindings': bindings,
               'scope': 'Exact documentation preceding candidate storage evidence; historical receipts remain unchanged.'},
              stream, indent=2, allow_nan=False)
print(json.dumps({'preserved': len(bindings), 'destination': str(DEST.relative_to(ROOT))}))
