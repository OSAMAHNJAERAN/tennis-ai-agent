"""Read-only authoritative runtime topology after a refused idle preflight."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[4]
records = []
for process in psutil.process_iter():
    try:
        args = process.cmdline()
        if not ('src.api.worker' in args or any(Path(value).name.lower() == 'serve_product.py' for value in args[1:])):
            continue
        cwd = Path(process.cwd()).resolve()
        record = {'pid': process.pid, 'createTimeUnixSeconds': process.create_time(),
                  'parentPid': process.ppid(), 'name': process.name(),
                  'workspaceMatches': cwd == ROOT, 'rssBytes': process.memory_info().rss}
        if cwd == ROOT:
            record.update({'cwd': str(cwd), 'cmdline': args, 'exe': process.exe(),
                           'directChildren': [{'pid': child.pid, 'createTimeUnixSeconds': child.create_time(),
                                               'name': child.name(), 'exe': child.exe()}
                                              for child in process.children()]})
        records.append(record)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
report = {'observedAt': datetime.now(timezone.utc).isoformat(), 'readOnly': True,
          'availableRamBytes': psutil.virtual_memory().available, 'runtimeCandidates': records,
          'port8000Listeners': [{'pid': row.pid, 'host': row.laddr.ip} for row in psutil.net_connections(kind='tcp')
                               if row.status == 'LISTEN' and row.laddr.port == 8000]}
with Path(__file__).with_suffix('.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps(report))
