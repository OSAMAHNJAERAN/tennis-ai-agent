"""Bind and run the complete backend suite for the event lookup revision."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).parent / 'backend_final'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def snapshot():
    paths = [*ROOT.joinpath('src').rglob('*.py'), *ROOT.joinpath('tests').rglob('*.py'),
             ROOT / 'pytest.ini', Path(__file__)]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths if path.is_file()}

UNIT.mkdir()
before = snapshot()
save(UNIT / 'source_bindings.json', before)
started = datetime.now(timezone.utc).isoformat()
command = [sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=short',
           '-o', 'faulthandler_timeout=30', '-o', 'junit_family=legacy', '--basetemp=' + str(UNIT / 'tmp'),
           '--junitxml=' + str(UNIT / 'junit.xml')]
with (UNIT / 'log.txt').open('x', encoding='utf-8') as stream:
    result = subprocess.run(command, cwd=ROOT,
                            env={**os.environ, 'TENNIS_STORAGE_ROOT': str(UNIT / 'bootstrap')},
                            stdout=stream, stderr=subprocess.STDOUT, timeout=180,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
suite = ET.parse(UNIT / 'junit.xml').getroot().find('testsuite')
counts = {key: int(suite.get(key, 0)) for key in ('tests', 'failures', 'errors', 'skipped')}
unchanged = before == snapshot()
passed = result.returncode == 0 and not counts['failures'] and not counts['errors'] and unchanged
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
           'passed': passed, 'exitCode': result.returncode, 'counts': counts,
           'testElapsedSeconds': float(suite.get('time')), 'sourcesUnchanged': unchanged,
           'artifacts': {name: digest(UNIT / name) for name in ('source_bindings.json', 'junit.xml', 'log.txt')},
           'command': command, 'scope': 'Backend software regressions; no accuracy or full-match qualification.'}
save(UNIT / 'receipt.json', receipt)
print(json.dumps({'passed': passed, 'counts': counts, 'sourcesUnchanged': unchanged}))
raise SystemExit(0 if passed else 1)
