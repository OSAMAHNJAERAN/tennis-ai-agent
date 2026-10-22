"""Publish a fully checked local frontend, preserving the previous build and runtime."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
RESTORE = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_entry_recovery_2026-10-06'
OLD = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_event_dependencies_2026-10-06'
EVENT = BASE.parent / 'event_timestamp_lookup_2026-10-06'
LIVE = ROOT / 'frontend/dist'
STAGE = BASE / 'staged_dist'
BACKUP = BASE / 'previous_dist'
RECEIPT = BASE / 'publish_receipt.json'

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def inventory(folder):
    assert folder.resolve().is_relative_to(ROOT) and not folder.is_symlink()
    paths = sorted(path for path in folder.rglob('*') if path.is_file())
    assert all(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()) for path in paths)
    return {path.relative_to(folder).as_posix(): digest(path) for path in paths}

def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)

def load_runtime():
    source = EVENT / 'verify_runtime.py'
    expected = json.loads((EVENT / 'runtime_verification.json').read_text(encoding='utf-8'))
    assert digest(source) == expected['scriptSha256']
    spec = importlib.util.spec_from_file_location('entry_runtime_guard', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, expected

def main():
    assert not any(path.exists() for path in (BASE / 'publish_started.json', RECEIPT, STAGE, BACKUP))
    restore = json.loads((RESTORE / 'receipt.json').read_text(encoding='utf-8'))
    previous = json.loads((OLD / 'receipt.json').read_text(encoding='utf-8'))
    assert restore['passed'] and restore['boundSelectedInputsUnchanged'] and restore['boundCopiedInputsUnchanged']
    assert restore['offlineInstall'] and restore['frozenLockfile']
    assert all(stage['passed'] for stage in restore['stages'])
    tests = restore['testSummary']
    assert tests['success'] and tests['allCopiedTestFilesExecuted'] and tests['passed'] == tests['total'] and tests['failed'] == 0
    spec = importlib.util.spec_from_file_location('frontend_restore_contract', ROOT / 'scripts/setup/verify_frontend_source_restore.py')
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert digest(ROOT / 'scripts/setup/verify_frontend_source_restore.py') == restore['verifierSha256']
    for folder in (ROOT / 'frontend', RESTORE / 'frontend'):
        assert {path.relative_to(folder).as_posix(): digest(path) for path in contract.frontend_inputs(folder)} == restore['sourceBindings']
    frontend_inputs = {'frontend/' + name: sha for name, sha in restore['sourceBindings'].items()}
    frontend_inputs.update(restore['sharedBindings'])
    assert all(digest(ROOT / name) == sha and digest(RESTORE / name) == sha for name, sha in restore['sharedBindings'].items())
    build_before = inventory(LIVE)
    assert build_before == previous['buildBindings'] == inventory(OLD / 'frontend/dist')
    assert inventory(RESTORE / 'frontend/dist') == restore['buildBindings']
    runtime, event = load_runtime()
    backend = {name: row['sha256'] for name, row in event['sourceBindings']['files'].items() if name.startswith('src/') or name.startswith('scripts/run/') or name == 'data/demos/sources.json'}
    assert all(digest(ROOT / name) == sha for name, sha in backend.items())
    snapshot = runtime.snapshot()
    assert snapshot == event['snapshotAfter']
    process_before = runtime.runtime_processes()
    assert process_before == event['runtime'] and process_before['state'] == 'OWNED_IDLE_PAIR'
    assert Path(runtime.sys.executable).resolve() == runtime.BASE_PYTHON.resolve()
    started = datetime.now(timezone.utc).isoformat()
    save(BASE / 'publish_started.json', {'startedAt': started, 'helperSha256': digest(Path(__file__)), 'frontendRestoreSha256': digest(RESTORE / 'receipt.json')})
    shutil.copytree(RESTORE / 'frontend/dist', STAGE)
    assert inventory(STAGE) == restore['buildBindings']
    # Validate all final absolute rename targets before any directory move.
    assert LIVE.resolve() == ROOT / 'frontend/dist' and LIVE.resolve().is_relative_to(ROOT)
    assert BACKUP.resolve().parent == BASE and STAGE.resolve().parent == BASE
    assert all(path.resolve().is_relative_to(ROOT) and not path.is_symlink() for path in (LIVE, BACKUP, STAGE))
    assert inventory(LIVE) == build_before and runtime.snapshot() == snapshot and runtime.runtime_processes() == process_before
    assert all(digest(ROOT / name) == sha for name, sha in frontend_inputs.items())
    LIVE.rename(BACKUP)
    try:
        STAGE.rename(LIVE)
    except BaseException:
        BACKUP.rename(LIVE)
        raise
    after = inventory(LIVE)
    assert after == restore['buildBindings'] and inventory(BACKUP) == build_before
    assert runtime.snapshot() == snapshot and runtime.runtime_processes() == process_before
    assert all(digest(ROOT / name) == sha for name, sha in {**frontend_inputs, **backend}.items())
    save(RECEIPT, {'passed': True, 'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(),
         'frontendRestore': {'path': (RESTORE / 'receipt.json').relative_to(ROOT).as_posix(), 'sha256': digest(RESTORE / 'receipt.json')},
         'frontendInputs': frontend_inputs, 'buildBefore': build_before, 'buildAfter': after,
         'previousBuildPath': BACKUP.relative_to(ROOT).as_posix(), 'backendBindings': backend,
         'runtimeBefore': process_before, 'runtimeAfter': runtime.runtime_processes(),
         'snapshotBefore': snapshot, 'snapshotAfter': runtime.snapshot(), 'helperSha256': digest(Path(__file__)),
         'checks': {'qualifiedCopiedBuildExact': True, 'previousBuildPreservedExact': True, 'frontendInputsUnchanged': True,
                    'backendInputsUnchanged': True, 'runtimeRetained': True, 'savedJobsAndFilesUnchanged': True},
         'scope': 'Reversible local frontend publication only; no inference, runtime restart, or scientific qualification.'})
    print(json.dumps({'passed': True, 'tests': tests['passed'], 'buildFiles': len(after), 'previousBuildFilesPreserved': len(build_before)}))

if __name__ == '__main__':
    main()
