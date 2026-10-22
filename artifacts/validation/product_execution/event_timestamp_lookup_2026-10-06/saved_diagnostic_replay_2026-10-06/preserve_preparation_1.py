"""Preserve reviewed but unexecuted replay preparation before a fresh revision."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
unit=Path(__file__).resolve().parent
assert not (unit/'launch_marker.json').exists() and not (unit/'run').exists() and not (unit/'replay_receipt.json').exists()
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(unit/'verify_replay.py')=='d1b76a81a8f2bfe42329e837c37fde85c7f44a1065befb9663cf9788b5afe35f'
assert digest(unit/'preparation.json')=='0f90e18fc4f1639f680e0410b9ce81760d4414e4bea6d679cf1c2c4a72b26826'
destination=unit/'preserved_preparation_1'
destination.mkdir(exist_ok=False)
bindings={}
for name in ('verify_replay.py','preparation.json','seal_preparation.py'):
    source=unit/name
    preserved=destination/name
    with source.open('rb') as incoming,preserved.open('xb') as outgoing:
        outgoing.write(incoming.read())
    assert digest(source)==digest(preserved)
    bindings[name]={'sha256':digest(source),'bytes':source.stat().st_size,
                    'preservedPath':str(preserved)}
receipt={'preservedAt':datetime.now(timezone.utc).isoformat(),'status':'PREPARED_NEVER_EXECUTED',
         'bindings':bindings,'priorInputsAndReportsChanged':False,'replayRun':False,'helperImported':False,
         'reason':'A new exclusive revision will add module/interpreter continuity verification; prior preparation remains immutable.'}
with (unit/'preserved_preparation_1.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False)
print(json.dumps(receipt),flush=True)

