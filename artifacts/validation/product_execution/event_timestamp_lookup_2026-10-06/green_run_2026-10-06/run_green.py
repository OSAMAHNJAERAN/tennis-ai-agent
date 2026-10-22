"""One focused GREEN run after the retained genuine RED; no production edits."""
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
evidence = unit.parent
root = evidence.parents[3]
source = root / 'src/events/event_detector.py'
test = root / 'tests/test_event_timestamp_lookup.py'
baseline = evidence / 'src/events/event_detector.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


assert sys.prefix == sys.base_prefix
assert digest(baseline) == '078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
assert digest(evidence / 'red_test_snapshot.py') == 'e3558972f8f86788ac7d2289af9d3a72e4dffa4bbc99f41e6e4d18581a648841'
red = json.loads((evidence / 'red_receipt.json').read_text(encoding='utf-8'))
assert red['status'] == 'RED' and red['failures'] == 1 and red['errors'] == 0
assert not (unit / 'green_receipt.json').exists()
before = {'sourceSha256': digest(source), 'testSha256': digest(test),
          'baselineSha256': digest(baseline), 'redTestSnapshotSha256': digest(evidence / 'red_test_snapshot.py'),
          'runnerSha256': digest(Path(__file__)), 'redReceiptSha256': digest(evidence / 'red_receipt.json')}
command = [sys.executable, '-m', 'pytest', 'tests/test_event_timestamp_lookup.py',
           '-q', '-s', '-o', 'junit_family=legacy', '--junitxml=' +
           (unit / 'green.xml').relative_to(root).as_posix()]
started = datetime.now(timezone.utc).isoformat()
began = time.perf_counter()
result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60,
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', EVENT_TIMESTAMP_LOOKUP_EVIDENCE=str(unit)))
output = result.stdout + result.stderr
with (unit / 'green_output.txt').open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(output)
tree = ET.parse(unit / 'green.xml')
suite = tree.getroot().find('testsuite')
properties = {node.attrib['name']: node.attrib['value'] for node in tree.findall('.//property')}
measurements = json.loads(properties['nearestTimestampMeasurements'])
unchanged = (digest(source) == before['sourceSha256'] and digest(test) == before['testSha256'] and
             digest(baseline) == before['baselineSha256'])
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
    'elapsedSeconds': time.perf_counter() - began, 'command': command,
    'returncode': result.returncode, 'status': 'GREEN' if result.returncode == 0 and unchanged else 'FAILED',
    'tests': int(suite.attrib['tests']), 'failures': int(suite.attrib['failures']),
    'errors': int(suite.attrib['errors']), 'skipped': int(suite.attrib['skipped']),
    'bindings': before, 'nearestTimestampMeasurements': measurements,
    'xmlSha256': digest(unit / 'green.xml'), 'outputSha256': digest(unit / 'green_output.txt'),
    'boundSourcesUnchangedDuringRun': unchanged, 'agentEditedProduction': False,
    'inferenceRun': False, 'modelsLoaded': False,
    'compatibility': 'Every serialized feature field plus complete analyze candidate/event/trace/settings/coordinate output against exact pre-change production source; finite sorted randomized nearest lookup and all old fallback/rejection cases.',
    'workloadBound': {'maximumDenseFrames': 512, 'featureRuns': 6,
        'randomSeed': 20261006, 'randomFiniteTimelines': 120, 'maximumRandomTimelineFrames': 96,
        'timeoutSeconds': 60, 'scope': 'Deterministic numeric regression, not a speed benchmark, memory-capacity or accuracy qualification.'}}
save(unit / 'green_receipt.json', receipt)
print(output, flush=True)
print(json.dumps(receipt, allow_nan=False), flush=True)
assert unchanged
raise SystemExit(result.returncode)

