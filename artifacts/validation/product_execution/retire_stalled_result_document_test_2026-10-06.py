"""Retire only the exact isolated stalled pytest process; preserve evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()
process = psutil.Process(38436)
command = process.cmdline()
assert Path(process.cwd()).resolve() == ROOT
assert command[1:3] == ['-m', 'pytest']
assert 'tests/test_result_document.py' in command
assert '--basetemp=artifacts/validation/product_execution/result_document_tmp2_2026-10-06' in command
assert not process.children(recursive=True)
identity = {'pid': process.pid, 'createTime': process.create_time(), 'command': command,
            'cpuSeconds': process.cpu_times()._asdict(), 'memoryBytes': process.memory_info()._asdict()}
process.terminate()
process.wait(timeout=10)
try:
    retired = psutil.Process(identity['pid']).create_time() != identity['createTime']
except psutil.NoSuchProcess:
    retired = True
assert retired
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'exactOwnedTestRetired': retired,
               'identity': identity, 'scope': 'Stalled restricted-runtime test only; no product process touched.',
               'originalRunCompleted': False, 'originalRunPassClaim': False,
               'artifactsPreserved': True,
               'authorizedResponseProbe': 'result_document_thread_probe_2026-10-06.xml'}, stream, indent=2)
print(json.dumps({'retired': retired, 'productProcessesTouched': False}))
