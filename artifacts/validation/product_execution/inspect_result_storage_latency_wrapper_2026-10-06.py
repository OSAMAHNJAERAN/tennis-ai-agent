"""Inspect only the exact launch shell observed by the read-only host probe."""
import json
from pathlib import Path
import psutil

receipt = json.loads(Path(__file__).with_name('inspect_result_storage_latency_host_2026-10-06.json').read_text(encoding='utf-8'))
record = next(item for item in receipt['runtimeCandidates'] if Path(item['exe']).name.lower() == 'cmd.exe')
process = psutil.Process(record['pid'])
assert process.create_time() == record['create_time']
print(json.dumps({'process': process.cmdline(), 'children': [
    {'pid': child.pid, 'name': child.name(), 'cmdline': child.cmdline(), 'cwd': child.cwd(), 'exe': child.exe()}
    for child in process.children()]}))
