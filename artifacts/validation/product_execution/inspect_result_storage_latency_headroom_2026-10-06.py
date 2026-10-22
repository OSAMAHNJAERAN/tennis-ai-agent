"""Observe memory headroom without launching or stopping any workload."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import psutil

RECEIPT = Path(__file__).with_suffix('.json')
assert not RECEIPT.exists()
samples = []
for index in range(5):
    samples.append({'at': datetime.now(timezone.utc).isoformat(), 'availableRamBytes': psutil.virtual_memory().available})
    if index < 4:
        time.sleep(3)
processes = sorted(psutil.process_iter(['pid', 'name', 'memory_info']),
                   key=lambda process: process.info['memory_info'].rss if process.info['memory_info'] else 0,
                   reverse=True)
largest = [{'pid': process.pid, 'name': process.info['name'], 'workingSetBytes': process.info['memory_info'].rss}
           for process in processes[:8] if process.info['memory_info']]
receipt = {'observedAt': datetime.now(timezone.utc).isoformat(), 'samples': samples,
           'preflightRequirementBytes': 2_500_000_000,
           'allSamplesMeetRequirement': all(item['availableRamBytes'] >= 2_500_000_000 for item in samples),
           'largestProcessNamesAndWorkingSets': largest, 'processCommandsRead': False,
           'processesStoppedOrStarted': False, 'inferenceRun': False,
           'scope': 'Read-only headroom observation; host memory includes unrelated apps and is not worker capacity evidence.'}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps(receipt))
