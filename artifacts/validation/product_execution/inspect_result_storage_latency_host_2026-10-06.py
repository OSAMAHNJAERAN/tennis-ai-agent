"""Read-only runtime candidate and available-memory inspection."""
from datetime import datetime, timezone
import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[3]
RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()
candidates = []
for process in psutil.process_iter(['pid', 'cmdline', 'exe', 'cwd', 'create_time', 'ppid']):
    args = process.info['cmdline'] or []
    if ('src.api.worker' in args or
            any(Path(argument).name.lower() == 'serve_product.py' for argument in args[1:])):
        candidates.append({**process.info, 'workingSetBytes': process.memory_info().rss})
listeners = [{'pid': row.pid, 'host': row.laddr.ip, 'port': row.laddr.port}
             for row in psutil.net_connections(kind='tcp') if row.status == 'LISTEN' and row.laddr.port == 8000]
receipt = {'observedAt': datetime.now(timezone.utc).isoformat(), 'runtimeCandidates': candidates,
           'listeners': listeners, 'availableRamBytes': psutil.virtual_memory().available,
           'runtimeStoppedOrStarted': False, 'inferenceRun': False}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps(receipt))
