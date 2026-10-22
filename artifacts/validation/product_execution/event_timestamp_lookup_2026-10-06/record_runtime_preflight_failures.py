"""Retain failed preflight facts without changing any consumed evidence."""
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[4]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
preflight = json.loads((UNIT / 'runtime_preflight.json').read_text(encoding='utf-8'))
assert preflight['runtime']['state'] == 'ABSENT'
data = {'scope': 'Refused/no-action runtime operations before successful read-only ABSENT preflight.',
        'failures': [
            {'action': 'verify_runtime.py --preflight', 'exitCode': 1,
             'error': 'AssertionError: Require exactly one owned server and one owned worker',
             'cause': 'Read-only inspection observed an additional serve_product.py command and its PowerShell launch wrapper; the strict verifier preserved all processes.'},
            {'action': 'stop_duplicate_unlistened_server.py', 'exitCode': 1,
             'error': 'psutil.NoSuchProcess: process PID not found (pid=62896)',
             'effect': 'The duplicate identity had exited before process construction. No terminate, product snapshot write or receipt publication occurred.'}],
        'subsequentSuccessfulPreflight': 'runtime_preflight.json',
        'subsequentRuntimeState': 'ABSENT', 'noRootStopOccurred': True,
        'savedJobs': 10, 'savedAnalysisFiles': 319, 'inferenceRun': False,
        'bindings': {name: digest(UNIT / name) for name in ('verify_runtime.py', 'inspect_runtime.py', 'inspect_runtime.json',
                    'stop_duplicate_unlistened_server.py', 'runtime_preflight.json')}}
with (UNIT / 'runtime_preflight_failures.json').open('x', encoding='utf-8') as stream:
    json.dump(data, stream, indent=2)
print(json.dumps({'failedOperationsPreserved': 2, 'rootStopOccurred': False, 'currentRuntimeState': 'ABSENT'}))
