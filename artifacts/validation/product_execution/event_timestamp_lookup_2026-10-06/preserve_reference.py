"""Preserve the exact pre-change event detector; no production imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

unit = Path(__file__).resolve().parent
root = unit.parents[3]
source = root / 'src/events/event_detector.py'
target = unit / 'src/events/event_detector.py'
target.parent.mkdir(parents=True, exist_ok=False)
with source.open('rb') as incoming, target.open('xb') as outgoing:
    for chunk in iter(lambda: incoming.read(1024 * 1024), b''):
        outgoing.write(chunk)
before = hashlib.sha256(source.read_bytes()).hexdigest()
preserved = hashlib.sha256(target.read_bytes()).hexdigest()
assert before == preserved
receipt = {'preservedAt': datetime.now(timezone.utc).isoformat(),
    'source': source.relative_to(root).as_posix(),
    'preserved': target.relative_to(root).as_posix(),
    'sha256': preserved, 'bytes': target.stat().st_size,
    'productionSourceChanged': False, 'inferenceRun': False,
    'scope': 'Pre-change reference for actual event feature timestamp lookup regression; no model or accuracy claim.'}
with (unit / 'baseline_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps(receipt), flush=True)

