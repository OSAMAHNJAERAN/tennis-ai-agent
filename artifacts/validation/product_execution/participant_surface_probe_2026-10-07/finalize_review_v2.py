"""Fresh finalization after the first attempt stopped before writing outputs."""
from pathlib import Path
import hashlib
import json

unit = Path(__file__).resolve().parent
original = unit / 'finalize_review.py'
failure = unit / 'finalize_review_failure.json'
assert not any((unit / name).exists() for name in
               ('visual_review.json', 'TENNIS_EXECUTION_PROGRESS_before.md', 'final_verification.json'))
source = original.read_text(encoding='utf-8')
old = "resource_path = UNIT / 'resource_receipt.json'"
new = "resource_path = UNIT / 'run/resource_receipt.json'"
assert source.count(old) == 1
with failure.open('x', encoding='utf-8') as handle:
    json.dump({'status': 'FAILED_BEFORE_ANY_FINALIZATION_OUTPUT',
               'exitCode': 1, 'reason': 'Resource receipt was addressed at the unit root instead of run/resource_receipt.json.',
               'originalSourceSha256': hashlib.sha256(original.read_bytes()).hexdigest(),
               'originalSourcePreserved': True, 'experimentRerun': False}, handle, indent=2)
    handle.write('\n')
exec(compile(source.replace(old, new), str(original), 'exec'))
