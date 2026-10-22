"""Check unchanged worker/model/output and resource logic before a fresh attempt."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
old_path = ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v3.py'
new_path = ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v4.py'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert digest(old_path) == '967ddbd5ed2c28d4b64edb13508a6f5b7a81726b70f7d75bebcef398a3a35e11'
assert digest(new_path) == '6cad723439512a9feb309a42bf74d34f361e8d1749f3d7531df3fa0a9aab2015'
old, new = [ast.parse(path.read_text(encoding='utf-8')) for path in (old_path, new_path)]
functions = lambda tree: {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.FunctionDef)}
left, right = functions(old), functions(new)
allowed = {'binding_snapshot', 'origin_record', 'origin_history', 'prepare', 'run_parent'}
changed = {name for name in left if left[name] != right[name]}
assert changed <= allowed, changed
assert set(right) - set(left) == {'verify_v3_attempt'}
protected = set(left) - allowed
assert {'run_child', 'verify_saved_result', 'verify_raw_pose_racket', 'verify_videos', 'model_observer'} <= protected
constants = lambda tree: {node.targets[0].id: ast.dump(node.value, include_attributes=False)
                          for node in tree.body if isinstance(node, ast.Assign)
                          and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)}
before, after = constants(old), constants(new)
changed_constants = {key for key in before if before[key] != after[key]}
assert changed_constants == {'UNIT'}, changed_constants
new_constants = set(after) - set(before)
assert new_constants == {'V3_ATTEMPT', 'V3_UNIT', 'V3_SHA256', 'V3_REVIEW_SHA256'}
record = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'scope': 'Static AST identity for actual worker/model/output functions and every retained guard constant; fresh paths/history additions manually reviewed.',
          'v3Sha256': digest(old_path), 'v4Sha256': digest(new_path),
          'unchangedFunctions': sorted(protected), 'changedFunctions': sorted(changed),
          'changedConstants': sorted(changed_constants), 'newConstants': sorted(new_constants),
          'workloadStarted': False, 'qualificationEvidence': False}
with (UNIT / 'v4_source_review.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps({'passed': True, 'unchangedFunctions': len(protected), 'changedFunctions': sorted(changed)}))
