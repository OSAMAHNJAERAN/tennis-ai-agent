"""Actual monitor method contract check, without constructing or launching processes."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
unit=Path(__file__).resolve().parent
root=unit.parents[4]
monitor_path=root/'scripts/audit/measure_candidate_storage.py'
helper=unit/'helper_v3.py'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(monitor_path)=='8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'
calls=[]
def reject_process(*args,**kwargs):
    calls.append('Popen')
    raise AssertionError('No subprocess allowed in this contract probe')
subprocess.Popen=reject_process
spec=importlib.util.spec_from_file_location('event_replay_actual_monitor_contract',monitor_path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
old=json.loads((unit/'actual_child_identity_v2.json').read_text(encoding='utf-8'))
canonical={'pid':old['actualChildPid'],'createTimeUnixSeconds':old['actualChildCreateTimeUnixSeconds']}
# Use the retained exact old identity as a method-level remembered fixture.
# Do not invoke __init__, psutil discovery, sample, terminate or any process.
tree=object.__new__(module.OwnedProcessTree)
tree._owned={(canonical['pid'],canonical['createTimeUnixSeconds']):{'sampleCount':2}}
try:
    tree.observed_workload(old)
except KeyError as error:
    old_outcome={'exception':type(error).__name__,'message':str(error)}
else:
    raise AssertionError('Consumed v2 schema must reproduce its actual monitor KeyError')
assert old_outcome=={'exception':'KeyError','message':"'pid'"}
assert tree.observed_workload(canonical) is True
assert tree.observed_workload({**canonical,'pid':canonical['pid']+1}) is False
assert tree.observed_workload({**canonical,'createTimeUnixSeconds':canonical['createTimeUnixSeconds']+1}) is False
tree._owned[(canonical['pid'],canonical['createTimeUnixSeconds'])]['sampleCount']=0
assert tree.observed_workload(canonical) is False
parsed=ast.parse(helper.read_text(encoding='utf-8'),filename=str(helper))
identity_nodes=[node for node in ast.walk(parsed) if isinstance(node,ast.Assign) and
                any(isinstance(target,ast.Name) and target.id=='identity' for target in node.targets)
                and isinstance(node.value,ast.Dict)]
assert len(identity_nodes)==1
keys=[ast.literal_eval(key) for key in identity_nodes[0].value.keys]
assert keys==['pid','createTimeUnixSeconds']
assert not calls and not any(name in sys.modules for name in ('torch','ultralytics','cv2','onnxruntime'))
record={'at':datetime.now(timezone.utc).isoformat(),'passed':True,
    'monitorSha256':digest(monitor_path),'helperSha256':digest(helper),
    'oldSchemaOutcome':old_outcome,'canonicalSchemaKeys':keys,
    'exactRememberedFixtureAccepted':True,'wrongPidRejected':True,'wrongCreateTimeRejected':True,
    'unsampledRecordRejected':True,'subprocessCalls':calls,'modelsLoaded':False,
    'actualProcessLaunched':False,'actualProcessSampled':False,
    'scope':'Method contract only using retained identity as a seeded remembered fixture; no live process/lifecycle verification and no replay.'}
with (unit/'monitor_identity_contract_receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(record,stream,indent=2,allow_nan=False)
print(json.dumps(record),flush=True)

