"""Bind every consumed v2 replay file and directory before a fresh v3."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
unit=Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def snapshot():
    paths=[path for path in unit.iterdir() if path.is_file() and
           ('_v2' in path.name or path.name=='prepared_revision_2.json')]
    paths += [path for path in (unit/'run_v2').rglob('*') if path.is_file()]
    return {'files':{path.relative_to(unit).as_posix():{'sha256':digest(path),'bytes':path.stat().st_size}
                     for path in sorted(set(paths))},
            'directories':['run_v2']+sorted(path.relative_to(unit).as_posix() for path in
                                         (unit/'run_v2').rglob('*') if path.is_dir())}
assert digest(unit/'helper_v2.py')=='99d6fc68508c423731b8aea6342189de89c70bd4e6d136f06c3fa1dd9979544e'
assert digest(unit/'prepared_revision_2.json')=='045b64a88cb57aa46b39e28bc61ceeb0e7ccbf982b81f5dce546609b0818bf1a'
failed=json.loads((unit/'replay_receipt_v2.json').read_text(encoding='utf-8'))
assert failed['status']=='FAILED' and failed['error']=="KeyError: 'pid'"
assert failed['samples']==2 and failed['allExactOwnedIdentitiesRetiredAtTerminal'] is True
assert (unit/'run_v2').is_dir() and list((unit/'run_v2').iterdir())==[]
assert not (unit/'actual_child_sampled_v2.json').exists() and not (unit/'child_receipt_v2.json').exists()
before=snapshot()
record={'preservedAt':datetime.now(timezone.utc).isoformat(),'consumedRevision':2,
    'status':'FAILED_BEFORE_ACKNOWLEDGEMENT_AND_NUMERIC_REPLAY','failure':failed['error'],
    'files':before['files'],'directories':before['directories'],
    'everyKnownOwnedIdentityRetired':True,'modelOrReplayStarted':False,
    'priorInputsAndReportsWritten':False,'noSelfHash':'This receipt binds consumed v2 files and directory names only; its hash is bound by fresh v3 preparation/history.'}
with (unit/'v2_history_bindings.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(record,stream,indent=2,allow_nan=False)
assert before==snapshot()
print(json.dumps({'files':len(before['files']),'directories':before['directories'],
                  'v2HistoryReceiptSha256':digest(unit/'v2_history_bindings.json')}),flush=True)

