"""Read-only proof of the exact hidden-start server and native console tree."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import urllib.request

sys.dont_write_bytecode = True
import psutil

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
HELPER = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/verify_runtime.py'
HELPER_SHA256 = '620285d1ea9446f92194fa6f4ba595573c21f724bc61e46497bd2a5c08b67743'
BASE_PYTHON = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe')
NATIVE_CONHOST = Path('C:/Windows/System32/conhost.exe')
EXPECTED = {
    'server': (32552, 1791348710.7337363),
    'worker': (42724, 1791348712.0018811),
    'serverConsole': (42148, 1791348710.7410307),
    'workerConsole': (31924, 1791348712.0111527),
}


def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size,
                'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def topology(checks):
    processes = {key: psutil.Process(identity[0]) for key, identity in EXPECTED.items()}
    for key, process in processes.items():
        assert process.create_time() == EXPECTED[key][1], 'Exact process identity differs: ' + key
    server, worker = processes['server'], processes['worker']
    server_console, worker_console = processes['serverConsole'], processes['workerConsole']
    assert Path(server.exe()).resolve() == BASE_PYTHON.resolve()
    assert Path(worker.exe()).resolve() == BASE_PYTHON.resolve()
    assert server.cmdline() == [str(BASE_PYTHON), 'scripts/run/serve_product.py']
    assert worker.cmdline() == [str(BASE_PYTHON), '-u', '-m', 'src.api.worker']
    assert Path(server.cwd()).resolve() == ROOT and Path(worker.cwd()).resolve() == ROOT
    assert worker.ppid() == server.pid
    assert server_console.ppid() == server.pid and worker_console.ppid() == worker.pid
    for console in (server_console, worker_console):
        assert console.name().lower() == 'conhost.exe'
        assert Path(console.exe()).resolve() == NATIVE_CONHOST.resolve()
        assert console.cmdline() == ['\\??\\C:\\Windows\\system32\\conhost.exe', '0x4']
        assert Path(console.cwd()).resolve() == Path('C:/Windows').resolve()
        assert not console.children(recursive=True), 'Native console has an unexpected descendant'
    assert {p.pid for p in server.children()} == {worker.pid, server_console.pid}
    assert {p.pid for p in worker.children()} == {worker_console.pid}
    assert {p.pid for p in server.children(recursive=True)} == {
        worker.pid, server_console.pid, worker_console.pid}
    recognized = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if ('src.api.worker' in args or
                any(Path(argument).name.lower() == 'serve_product.py' for argument in args[1:])):
            recognized.append(checks.process_record(process))
    recognized.sort(key=lambda row: row['pid'])
    assert {row['pid'] for row in recognized} == {server.pid, worker.pid}
    listeners = [{'pid': row.pid, 'host': row.laddr.ip, 'port': row.laddr.port}
                 for row in psutil.net_connections(kind='tcp')
                 if row.status == 'LISTEN' and row.laddr.port == 8000]
    listeners.sort(key=lambda row: (row['pid'] or -1, row['host']))
    assert listeners == [{'pid': server.pid, 'host': '127.0.0.1', 'port': 8000}]
    runtime = {'state': 'OWNED_IDLE_PAIR',
               'processes': [checks.process_record(server), checks.process_record(worker)],
               'ownedDescendants': [checks.process_record(server_console), checks.process_record(worker_console)],
               'port8000Listeners': listeners, 'launchWrappers': [],
               'serverDirectConsoleIds': [server_console.pid]}
    for row in runtime['processes'] + runtime['ownedDescendants']:
        assert psutil.Process(row['pid']).create_time() == row['createTimeUnixSeconds']
    return runtime, recognized


def main():
    assert Path(sys.executable).resolve() == BASE_PYTHON.resolve()
    receipt_path = UNIT / 'hidden_runtime_review.json'
    assert not receipt_path.exists(), 'Preserve the existing one-shot read-only review'
    report = {'schemaVersion': 1, 'reviewedAt': datetime.now(timezone.utc).isoformat(),
              'passed': False, 'independentReviewPassed': False,
              'scope': 'Read-only process, listener, loopback health and unchanged source/saved-job evidence for this exact hidden launch; no process or persistent-product mutation.',
              'runtimeStoppedOrStarted': False, 'productStateChanged': False,
              'qualificationEvidence': False, 'productionQualified': False}
    try:
        names = [HELPER, Path(__file__), UNIT / 'runtime_preflight.json',
                 UNIT / 'server_launch.json', UNIT / 'worker_review.json']
        before = {path.relative_to(ROOT).as_posix(): binding(path) for path in names}
        assert before[HELPER.relative_to(ROOT).as_posix()]['sha256'] == HELPER_SHA256
        spec = importlib.util.spec_from_file_location('_hidden_runtime_review_original', HELPER)
        checks = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = checks
        spec.loader.exec_module(checks)
        preflight = json.loads((UNIT / 'runtime_preflight.json').read_text(encoding='utf-8'))
        launch = json.loads((UNIT / 'server_launch.json').read_text(encoding='utf-8'))
        assert preflight['runtime']['state'] == 'ABSENT'
        assert launch['pid'] == EXPECTED['server'][0] and launch['windowStyle'] == 'Hidden'
        assert Path(launch['executable']).resolve() == BASE_PYTHON.resolve()
        assert launch['argument'] == 'scripts/run/serve_product.py'
        assert Path(launch['workingDirectory']).resolve() == ROOT
        assert launch['previousRuntimeStopped'] is False
        started = datetime.fromisoformat(launch['processStartedAt'].replace('Z', '+00:00')).timestamp()
        assert abs(started - EXPECTED['server'][1]) < .000001
        assert checks.source_bindings() == preflight['sourceBindings']
        assert checks.snapshot() == preflight['snapshot']
        runtime, recognized = topology(checks)
        parent_records = []
        all_records = runtime['processes'] + runtime['ownedDescendants']
        for pid in sorted({row['parentPid'] for row in all_records}):
            try:
                parent_records.append({'exists': True, **checks.process_record(psutil.Process(pid))})
            except psutil.NoSuchProcess:
                parent_records.append({'pid': pid, 'exists': False,
                                       'scope': 'Immediate launch parent had exited before review; child parent PID retained.'})
        status, headers, body = checks.request('/api/health')
        health = json.loads(body)
        assert status == 200 and health['status'] == 'ok' and health['worker_available'] is True
        assert checks.source_bindings() == preflight['sourceBindings']
        assert checks.snapshot() == preflight['snapshot']
        runtime_after, recognized_after = topology(checks)
        assert runtime_after == runtime and recognized_after == recognized
        after = {path.relative_to(ROOT).as_posix(): binding(path) for path in names}
        assert after == before
        report.update({'runtime': runtime, 'serverDirectConsoleIds': runtime['serverDirectConsoleIds'],
                       'recognizedRuntimeInventory': recognized, 'immediateParentRecords': parent_records,
                       'health': health, 'healthStatus': status,
                       'healthResponseSha256': hashlib.sha256(body).hexdigest(),
                       'bindingsBefore': before, 'bindingsAfter': after,
                       'allBoundReviewInputsUnchanged': True, 'originalHelperUnchanged': True,
                       'sourceBindingsMatchPreflight': True, 'savedSnapshotMatchesPreflight': True,
                       'savedJobs': len(preflight['snapshot']['jobs']),
                       'savedAnalysisFiles': len(preflight['snapshot']['savedAnalysisFiles']),
                       'runtimeStableAcrossChecks': True, 'noAdditionalRuntimeOrDescendant': True,
                       'nativeConsoleIdentitiesFixedForThisLaunch': True,
                       'passed': True, 'independentReviewPassed': True})
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    report['completedAt'] = datetime.now(timezone.utc).isoformat()
    with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report.get(key) for key in ('passed', 'savedJobs', 'savedAnalysisFiles', 'error')}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
