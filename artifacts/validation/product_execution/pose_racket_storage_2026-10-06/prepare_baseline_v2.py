"""Preserve exact current sources before the pose/racket capacity intervention."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
SEAL = BASE.parent / 'frontend_entry_recovery_2026-10-06/final_verification.json'

def binding(path):
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': sha}

assert not (BASE / 'before.json').exists() and not (BASE / 'before').exists()
seal = json.loads(SEAL.read_text(encoding='utf-8'))
assert seal['passed'] and binding(SEAL)['sha256'] == 'e81b4e627af107b0aba7d706790b390f21c6e40d9ccea8335a951fd7d4a3fae5'
names = ['src/pipeline/phase6_pipeline.py', 'src/utils/frame_registration_sequence.py',
         'src/utils/json_array_artifact.py', 'src/tracking/player_motion_tracking.py',
         'src/tracking/racket_tracking.py', 'tests/test_phase6_pipeline_smoke.py',
         'docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
         'docs/deployment/PRODUCT_RESTORE_MANIFEST.json']
records = {}
for name in names:
    path = ROOT / name
    expected = seal['bindings'][name]
    assert binding(path) == expected
    target = BASE / 'before' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with path.open('rb') as source, target.open('xb') as output:
        shutil.copyfileobj(source, output)
    assert binding(target) == expected
    records[name] = {'before': expected, 'preservedPath': target.relative_to(ROOT).as_posix()}
with (BASE / 'before.json').open('x', encoding='utf-8') as output:
    json.dump({'preparedAt': datetime.now(timezone.utc).isoformat(), 'previousGoalTurn': 'PROGRESS',
               'previousFrontendSeal': {'path': SEAL.relative_to(ROOT).as_posix(), **binding(SEAL)},
               'files': records, 'productionChanged': False,
               'scope': 'Exact pre-intervention originals; no workload, inference or scientific qualification.'}, output, indent=2)
print(json.dumps({'preservedFiles': len(records), 'productionChanged': False}))
