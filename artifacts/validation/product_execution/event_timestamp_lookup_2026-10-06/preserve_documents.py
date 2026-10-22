"""Preserve current documents and the prior completed unit before event changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).parent / 'preserved_documents'
UNIT.mkdir()
names = ['docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
         'docs/deployment/PRODUCT_RESTORE_MANIFEST.json',
         'artifacts/validation/product_execution/finalize_result_storage_latency_evidence_2026-10-06.json',
         'artifacts/validation/product_execution/result_storage_latency_stale_audit_restore_2026-10-06/receipt.json']
bindings = {}
for name in names:
    source = ROOT / name
    data = source.read_bytes()
    destination = UNIT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(data)
    assert destination.read_bytes() == source.read_bytes()
    bindings[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
with (UNIT / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'beforeEventSourceChanges': True,
               'bindings': bindings, 'qualificationEvidence': False}, stream, indent=2)
print(json.dumps({'preservedFiles': len(bindings), 'documentsChanged': False}))
