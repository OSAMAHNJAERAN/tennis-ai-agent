"""Retain exact source baselines before the approved native analytics change."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
BASE = Path(__file__).resolve().parent
probe = json.loads((BASE.parent / 'analytics_capacity_probe_v1/receipt.json').read_text())
paths = list(probe['source_bindings_before']) + ['src/utils/mapped_sequence_view.py', 'src/utils/json_array_artifact.py']
records = []
for value in paths:
    source = ROOT / value
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if value in probe['source_bindings_before']:
        assert digest == probe['source_bindings_before'][value], value
    destination = BASE / 'baseline' / value
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(content)
    records.append({'source': str(source), 'baseline': str(destination), 'sha256': digest})
receipt = dict(status='EXACT_PRECHANGE_SOURCES_SNAPSHOTTED', created_utc=datetime.now(timezone.utc).isoformat(),
               files=records, production_source_changed=False, model_used=False, media_used=False)
with (BASE / 'baseline_manifest.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({'status': receipt['status'], 'files': len(records)}))
