"""Read-only resource snapshot before configured-scale capacity measurements."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
if Path(sys.executable).resolve() != CLEAN.resolve():
    raise SystemExit(subprocess.run([str(CLEAN), str(Path(__file__).resolve())], cwd=ROOT).returncode)

import psutil

report = {
    'measuredAt': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only host readiness; no inference or capacity qualification.',
    'interpreter': sys.executable,
    'logicalCpuCount': psutil.cpu_count(),
    'physicalCpuCount': psutil.cpu_count(logical=False),
    'ram': psutil.virtual_memory()._asdict(),
    'swap': psutil.swap_memory()._asdict(),
    'workspaceDisk': shutil.disk_usage(ROOT)._asdict(),
    'productProcesses': [],
}
for process in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        args = process.info['cmdline'] or []
        if any('serve_product.py' in arg or arg == 'src.api.worker' for arg in args):
            if Path(process.cwd()).resolve() == ROOT:
                report['productProcesses'].append({
                    'pid': process.pid, 'name': process.name(),
                    'memory': process.memory_info()._asdict(),
                    'children': [{'pid': child.pid, 'name': child.name(),
                                  'memory': child.memory_info()._asdict()}
                                 for child in process.children(recursive=True)],
                })
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
gpu = shutil.which('nvidia-smi')
if gpu:
    result = subprocess.run([gpu, '--query-gpu=name,memory.total,memory.used',
                             '--format=csv,noheader,nounits'], capture_output=True,
                            text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    report['gpu'] = {'query': 'name,memory.total,memory.used (MiB)',
                     'returncode': result.returncode, 'output': result.stdout.strip()}
output = ROOT / 'artifacts/validation/product_execution/capacity_readiness_2026-10-05.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps(report, allow_nan=False))
