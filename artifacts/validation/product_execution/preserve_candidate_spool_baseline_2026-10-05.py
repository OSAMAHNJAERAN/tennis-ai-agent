"""Preserve exact selected sources before the candidate-storage change."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent / 'candidate_spool_baseline_2026-10-05'
DEST.mkdir(exist_ok=False)
bindings = {}
for relative in ('src/pipeline/phase6_pipeline.py', 'src/tracking/temporal_ball_tracker.py', 'src/detection/image_people.py'):
    original = ROOT / relative
    copied = DEST / relative
    copied.parent.mkdir(parents=True, exist_ok=True)
    with copied.open('xb') as stream:
        stream.write(original.read_bytes())
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest(original) == digest(copied)
    bindings[relative] = digest(original)
with (DEST / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'preservedAt': datetime.now(timezone.utc).isoformat(), 'sources': bindings,
               'scope': 'Exact source snapshots before disk-backed candidate and early person exports; no inference/model changes.'}, stream, indent=2)
print(json.dumps({'directory': str(DEST.relative_to(ROOT)), 'sources': bindings}))
