"""Preserve the selected pipeline exactly before registration storage changes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent / 'registration_sequence_baseline_2026-10-05'
DEST.mkdir(exist_ok=False)
bindings = {}
for name in ('src/pipeline/phase6_pipeline.py', 'src/court/calibration.py',
             'src/court/camera_registration.py', 'src/tracking/court_player_tracker.py'):
    source, target = ROOT / name, DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(source.read_bytes())
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
    bindings[name] = expected
with (DEST / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'sources': bindings,
               'scope': 'Exact production source preceding owned frame-registration storage; no model or algorithm changes.'}, stream, indent=2)
print(json.dumps({'preserved': len(bindings), 'destination': str(DEST.relative_to(ROOT))}))
