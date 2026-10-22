from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

rows = []
for pid in (58060, 59308):
    try:
        process = psutil.Process(pid)
        rows.append({'pid': pid, 'ppid': process.ppid(), 'exe': process.exe(), 'cwd': process.cwd(),
                     'command': process.cmdline(), 'createTime': process.create_time(),
                     'children': [{'pid': child.pid, 'name': child.name()} for child in process.children()]})
    except psutil.NoSuchProcess:
        rows.append({'pid': pid, 'missing': True})
print(json.dumps({'at': datetime.now(timezone.utc).isoformat(), 'processes': rows}))
