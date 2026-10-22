"""Preserve an exclusive RED or GREEN snapshot-coherence regression receipt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
phase = sys.argv[1]
assert phase in {'red', 'green'}
unit = Path(__file__).parent / ('result_reader_snapshot_' + phase + '_2026-10-06')
unit.mkdir()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def snapshot():
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in (
        ROOT / 'src/api/result_storage.py', ROOT / 'tests/test_result_reader_snapshot.py', Path(__file__))}

before = snapshot()
started = datetime.now(timezone.utc).isoformat()
command = [sys.executable, '-m', 'pytest', 'tests/test_result_reader_snapshot.py', '-q', '--tb=short',
           '--basetemp=' + str(unit / 'tmp'), '--junitxml=' + str(unit / 'junit.xml')]
with (unit / 'log.txt').open('x', encoding='utf-8') as output:
    result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                            env={**os.environ, 'TENNIS_STORAGE_ROOT': str(unit / 'bootstrap')},
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), timeout=30)
suite = ET.parse(unit / 'junit.xml').getroot().find('testsuite')
counts = {key: int(suite.get(key, 0)) for key in ('tests', 'failures', 'errors', 'skipped')}
after = snapshot()
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
           'phase': phase, 'exitCode': result.returncode, 'counts': counts,
           'sourceBindingsBefore': before, 'sourceBindingsAfter': after,
           'sourcesUnchanged': before == after, 'command': command,
           'artifacts': {name: digest(unit / name) for name in ('log.txt', 'junit.xml')},
           'scope': 'Actual SQLite generation coherence and reader cleanup; no inference or accuracy evidence.'}
with (unit / 'receipt.json').open('x', encoding='utf-8') as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
print(json.dumps(receipt))
raise SystemExit(result.returncode)
