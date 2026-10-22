"""Read back the concurrent frontend's isolated restore; never rebuild or publish."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RESTORE = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_event_dependencies_2026-10-06'
RECEIPT = UNIT / 'frontend_dependency_restore_verification.json'
assert not RECEIPT.exists()
bindings = {}

def bind(path, sha=None):
    value = {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}
    if sha:
        assert value['sha256'] == sha, str(path)
    name = path.relative_to(ROOT).as_posix()
    assert name not in bindings or bindings[name] == value
    bindings[name] = value
    return value

def read(path):
    bind(path)
    return json.loads(path.read_text(encoding='utf-8'))

report = read(RESTORE / 'receipt.json')
assert report['passed'] and report['boundSelectedInputsUnchanged'] and report['boundCopiedInputsUnchanged']
assert not report['inputDifferences'] and report['copiedFiles'] == 127
assert report['copiedFrontendFiles'] == 125 and report['packageStoreDirectories'] == 377
assert report['testSummary']['total'] == report['testSummary']['passed'] == 165
assert report['testSummary']['failed'] == 0 and report['testSummary']['allCopiedTestFilesExecuted']
assert report['testSummary']['expectedFiles'] == report['testSummary']['executedFiles'] == 23
bind(RESTORE / 'tests.json', report['testSummary']['receiptSha256'])
bind(ROOT / 'scripts/setup/verify_frontend_source_restore.py', report['verifierSha256'])
bind(RESTORE / 'verifier.py', report['verifierSha256'])
spec = importlib.util.spec_from_file_location('frontend_restore_inventory', ROOT / 'scripts/setup/verify_frontend_source_restore.py')
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)
for folder in (ROOT / 'frontend', RESTORE / 'frontend'):
    assert {path.relative_to(folder).as_posix() for path in inventory.frontend_inputs(folder)} == set(report['sourceBindings'])
    for name, sha in report['sourceBindings'].items():
        bind(folder / name, sha)
for name, sha in report['sharedBindings'].items():
    bind(ROOT / name, sha)
    bind(RESTORE / name, sha)
for folder in (ROOT / 'frontend/dist', RESTORE / 'frontend/dist'):
    assert {path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()} == set(report['buildBindings'])
    for name, sha in report['buildBindings'].items():
        bind(folder / name, sha)
assert len(report['buildBindings']) == 47
for stage in report['stages']:
    assert stage['passed'] and stage['returncode'] == 0
    bind(ROOT / stage['stdout'])
    bind(ROOT / stage['stderr'])
bind(Path(__file__))
for name, value in list(bindings.items()):
    assert bind(ROOT / name) == value
result = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'frontendInputsExact': 127, 'frontendTestsPassed': 165, 'frontendTestFiles': 23,
          'productionFilesExact': 47, 'selectedAndCopiedInputsUnchanged': True,
          'selectedAndCopiedBuildsExact': True, 'inferenceRun': False, 'productionQualified': False,
          'processMemoryOrLifecycleMeasured': False, 'browserBehaviorQualified': False,
          'scope': 'Current concurrent frontend frozen offline restore, software tests and exact served build bytes only.',
          'bindings': bindings}
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({key: result[key] for key in ('passed', 'frontendInputsExact', 'frontendTestsPassed', 'productionFilesExact')}))
