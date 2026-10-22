"""Stop only the exact idle pair bound by this unit's fresh preflight."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback

import psutil

BASE = Path(__file__).resolve().parent
HELPER = BASE / 'verify_runtime.py'
HELPER_SHA256 = '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def exact(record):
    process = psutil.Process(record['pid'])
    assert process.create_time() == record['createTimeUnixSeconds'], 'Runtime identity changed'
    return process


def main():
    assert digest(HELPER) == HELPER_SHA256, 'Reviewed preflight helper changed'
    spec = importlib.util.spec_from_file_location('pose_storage_runtime_gate', HELPER)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    preflight = json.loads(gate.PREFLIGHT.read_text(encoding='utf-8'))
    assert preflight['runtime']['state'] == 'OWNED_IDLE_PAIR', 'An exact owned idle pair is required'
    assert not (BASE / 'idle_stop_receipt.json').exists()
    save(BASE / 'idle_stop_marker.json', {'startedAt': datetime.now(timezone.utc).isoformat(),
        'oneShot': True, 'helperSha256': digest(Path(__file__)),
        'preflightSha256': digest(gate.PREFLIGHT)})
    result = {'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
        'action': 'Guarded idle reload; stop only exact preflight runtime identities.',
        'acceptedWorkCancelled': False, 'persistentFilesRemoved': False, 'terminationActions': []}
    try:
        assert gate.source_bindings() == preflight['sourceBindings'], 'Reviewed sources changed'
        assert gate.snapshot() == preflight['snapshot'], 'Saved jobs/files changed; preserve runtime'
        runtime = gate.runtime_processes()
        assert runtime == preflight['runtime'], 'Exact runtime or topology changed; preserve runtime'
        server_row, worker_row = runtime['processes']
        assert any(Path(arg).name == 'serve_product.py' for arg in server_row['cmdline'][1:])
        assert 'src.api.worker' in worker_row['cmdline']
        server, worker = exact(server_row), exact(worker_row)
        assert gate.snapshot() == preflight['snapshot'], 'Accepted work changed; preserve runtime'
        # Retire supervision before its exact worker so no replacement can be spawned.
        exact(server_row).terminate()
        result['terminationActions'].append({'identity': server_row, 'action': 'terminate idle supervisor'})
        server.wait(timeout=10)
        try:
            current_worker = exact(worker_row)
        except psutil.NoSuchProcess:
            current_worker = None
        if current_worker is not None:
            current_worker.terminate()
            result['terminationActions'].append({'identity': worker_row, 'action': 'terminate exact idle worker'})
            current_worker.wait(timeout=10)
        remembered = []
        for record in runtime['processes'] + runtime['ownedDescendants']:
            try:
                remembered.append(exact(record))
            except psutil.NoSuchProcess:
                pass
        if remembered:
            psutil.wait_procs(remembered, timeout=10)
        result['previousExactIdentitiesRetired'] = gate.previous_identities_retired(runtime)
        assert gate.runtime_processes()['state'] == 'ABSENT', 'Another runtime appeared; do not replace it'
        assert gate.snapshot() == preflight['snapshot'], 'Stored analyses changed during stop'
        assert gate.source_bindings() == preflight['sourceBindings'], 'Sources changed during stop'
        result['savedJobsAndAnalysisFilesUnchanged'] = True
        result['preflightSha256'] = digest(gate.PREFLIGHT)
        result['passed'] = True
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['completedAt'] = datetime.now(timezone.utc).isoformat()
        result['helperSha256'] = digest(Path(__file__))
        save(BASE / 'idle_stop_receipt.json', result)
        print(json.dumps({'passed': result['passed'], 'actionCount': len(result['terminationActions']),
                          'error': result.get('error')}), flush=True)


if __name__ == '__main__':
    main()
