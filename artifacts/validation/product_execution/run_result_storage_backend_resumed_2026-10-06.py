"""Fresh backend verification after the prior run disappeared without a receipt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import psutil

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
XML = BASE / 'result_storage_backend_RESUMED_2026-10-06.xml'
LOG = BASE / 'result_storage_backend_RESUMED_2026-10-06.log'
BEFORE = BASE / 'result_storage_backend_resumed_sources_2026-10-06.json'
RECEIPT = Path(__file__).with_suffix('.json')
TEMP = BASE / 'result_storage_backend_resumed_tmp_2026-10-06'
INTERRUPTION = BASE / 'result_storage_backend_interrupted_2026-10-06.json'
assert not any(path.exists() for path in (XML, LOG, BEFORE, RECEIPT, TEMP, INTERRUPTION))

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

live = []
for process in psutil.process_iter(['pid', 'cmdline']):
    args = process.info['cmdline'] or []
    if any('result_storage_backend_tmp_2026-10-06' in arg or
           arg.endswith('run_result_storage_backend_2026-10-06.py') for arg in args):
        live.append(process.info)
assert not live, 'Prior exact verification is still alive; do not restart'
old_files = [BASE / name for name in ('run_result_storage_backend_2026-10-06.py',
             'result_storage_backend_sources_2026-10-06.json', 'result_storage_backend_FULL_2026-10-06.log')]
save(INTERRUPTION, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'priorRunCompleted': False,
     'priorRunPassed': False, 'matchingLiveProcesses': live, 'toolHandleMissing': 47017,
     'terminalReceiptPresent': (BASE / 'run_result_storage_backend_2026-10-06.json').exists(),
     'junitPresent': (BASE / 'result_storage_backend_FULL_2026-10-06.xml').exists(),
     'priorArtifactsPreserved': {path.name: digest(path) for path in old_files},
     'scope': 'Process and artifact readback; interruption cause is not independently established.'})
before = snapshot()
save(BEFORE, before)
started = datetime.now(timezone.utc).isoformat()
command = [sys.executable, '-m', 'pytest', 'tests', '-q', '--tb=short',
           '-o', 'faulthandler_timeout=30', '--basetemp=' + TEMP.relative_to(ROOT).as_posix(),
           '--junitxml=' + XML.relative_to(ROOT).as_posix()]
env = {**os.environ, 'TENNIS_STORAGE_ROOT': str(TEMP / 'bootstrap')}
with LOG.open('x', encoding='utf-8') as stream:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
suite = ET.parse(XML).getroot().find('testsuite') if XML.is_file() else None
counts = {key: int(suite.get(key, 0)) for key in ('tests', 'failures', 'errors', 'skipped')} if suite is not None else {}
unchanged = before == snapshot()
passed = bool(result.returncode == 0 and counts and not counts['failures'] and not counts['errors'] and unchanged)
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
           'passed': passed, 'exitCode': result.returncode, 'counts': counts,
           'testElapsedSeconds': float(suite.get('time')) if suite is not None else None,
           'sourceBindingsUnchanged': unchanged, 'sourceInventorySha256': digest(BEFORE),
           'junitSha256': digest(XML) if XML.is_file() else None, 'logSha256': digest(LOG),
           'command': command, 'scope': 'Backend software regressions only; no independent accuracy or full-match qualification.'}
save(RECEIPT, receipt)
print(json.dumps(receipt))
raise SystemExit(0 if passed else 1)
