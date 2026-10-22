"""Run the bounded actual event regression once and retain its genuine RED."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

unit = Path(__file__).resolve().parent
root = unit.parents[3]
source = root / 'src/events/event_detector.py'
test = root / 'tests/test_event_timestamp_lookup.py'
baseline = unit / 'src/events/event_detector.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


assert sys.prefix == sys.base_prefix
assert digest(source) == digest(baseline) == '078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
assert not (unit / 'red_receipt.json').exists()
before = {'sourceSha256': digest(source), 'testSha256': digest(test),
          'baselineSha256': digest(baseline), 'runnerSha256': digest(Path(__file__))}
command = [sys.executable, '-m', 'pytest', 'tests/test_event_timestamp_lookup.py',
           '-q', '-s', '-o', 'junit_family=legacy', '--junitxml=' +
           (unit / 'red.xml').relative_to(root).as_posix()]
started = datetime.now(timezone.utc).isoformat()
began = time.perf_counter()
result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', EVENT_TIMESTAMP_LOOKUP_EVIDENCE=str(unit)))
output = result.stdout + result.stderr
with (unit / 'red_output.txt').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(output)
tree = ET.parse(unit / 'red.xml')
suite = tree.getroot().find('testsuite')
properties = {node.attrib['name']: node.attrib['value'] for node in tree.findall('.//property')}
measurements = json.loads(properties['nearestTimestampMeasurements'])
assert digest(source) == before['sourceSha256']
assert digest(test) == before['testSha256']
assert digest(baseline) == before['baselineSha256']
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
    'elapsedSeconds': time.perf_counter() - began, 'command': command,
    'returncode': result.returncode, 'status': 'RED' if result.returncode == 1 else 'UNEXPECTED',
    'tests': int(suite.attrib['tests']), 'failures': int(suite.attrib['failures']),
    'errors': int(suite.attrib['errors']), 'skipped': int(suite.attrib['skipped']),
    'bindings': before, 'nearestTimestampMeasurements': measurements,
    'xmlSha256': digest(unit / 'red.xml'), 'outputSha256': digest(unit / 'red_output.txt'),
    'productionSourceUnchanged': True, 'inferenceRun': False, 'modelsLoaded': False,
    'workloadBound': {'maximumFrames': 512, 'featureRuns': 6,
        'timeoutSeconds': 60, 'scope': 'Deterministic numeric regression, not a speed benchmark, memory-capacity or accuracy qualification.'}}
save(unit / 'red_receipt.json', receipt)
print(output, flush=True)
print(json.dumps(receipt, allow_nan=False), flush=True)
assert result.returncode == 1 and receipt['failures'] == 1 and receipt['errors'] == 0
raise SystemExit(result.returncode)

