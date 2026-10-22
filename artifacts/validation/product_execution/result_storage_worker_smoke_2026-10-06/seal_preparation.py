"""Seal the unexecuted observer revision; no product/model imports or work."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

unit = Path(__file__).resolve().parent
root = unit.parents[3]
script = root / 'scripts/audit/verify_result_storage_worker_smoke.py'
preparation = unit / 'preparation.json'
original_bytes = preparation.read_bytes()
original = json.loads(original_bytes)
assert original['scriptSha256'] == '6c5f0389d35333dcbd120d1ac305de2c2d74647bdb7fcf9e52719b14728a57cd'
assert original['preparedAt'] == '2026-10-06T04:41:16.738260+00:00'
assert original['inferenceRun'] is False and original['workerInvoked'] is False
assert original['modelEvidence'] == 'Passive interpreter profile observes real Phase6Pipeline construction/run and parameter/device counts. No model or worker functions replaced.'
# Initial --prepare used json.dump(indent=2, allow_nan=False), no final newline.
canonical = json.dumps(original, indent=2, allow_nan=False).encode('utf-8')
corrected_editor_newline = original_bytes == canonical + b'\n'
if corrected_editor_newline:
    # Restore the initial writer's exact EOF after the preparatory edit; all
    # original values/order/spacing are already asserted and unchanged.
    preparation.write_bytes(canonical)
    original_bytes = preparation.read_bytes()
assert original_bytes == canonical
source = script.read_bytes()
ast.parse(source, filename=str(script))
revision = {
    'recordedAt': datetime.now(timezone.utc).isoformat(),
    'status': 'PREPARED_NOT_EXECUTED',
    'originalScriptSha256': original['scriptSha256'],
    'originalPreparationSha256': hashlib.sha256(original_bytes).hexdigest(),
    'currentScriptSha256': hashlib.sha256(source).hexdigest(),
    'originalPreparationExactJsonDumpBytesPreserved': True,
    'restorationRemovedEditorAddedFinalNewline': corrected_editor_newline,
    'sourceAstParsed': True,
    'inferenceRun': False,
    'workerInvoked': False,
    'observerChange': 'Record actual model construction return, then disable profiling before inference. Worker elapsed time includes preparation/import/construction observer overhead; this is no performance comparison.',
    'dispatchChange': 'The parent verifies this revision current-source hash, original-source hash and exact original-preparation hash before any one-shot launch.',
    'rootObserverChangeReviewed': True,
    'executionStatus': 'Waiting for root dispatch after backend and synthetic measurement retire; no run launched.',
    'qualificationEvidence': False,
    'productionQualified': False,
}
with (unit / 'observer_revision.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(revision, stream, indent=2, allow_nan=False)
print(json.dumps(revision), flush=True)
