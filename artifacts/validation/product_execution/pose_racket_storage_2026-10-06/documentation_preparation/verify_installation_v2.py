"""Prepared fresh clean-runtime verification wrapper; root dispatch only."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
ROOT = UNIT.parents[3]
REPORT = ROOT / 'artifacts/validation/product_installation/pose_racket_storage_host_verification_2026-10-06.json'
LOG = UNIT / 'verify_installation.log'
RECEIPT = UNIT / 'verify_installation.json'
SAMPLES = UNIT / 'verify_installation_samples.jsonl'
MANIFEST = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
VERIFIER = ROOT / 'scripts/setup/verify_product_installation.py'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
MONITOR_SHA256 = '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    with Path(path).open('rb') as stream:
        return {'bytes': Path(path).stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(values):
    for name, value in values.items():
        assert binding(resolve(name)) == value, 'Verification input changed: ' + name


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    import psutil
    assert not any(path.exists() for path in (REPORT, LOG, RECEIPT, SAMPLES, UNIT / 'host_launch.json'))
    prepared = load(HERE / 'preparation_v2.json')
    verify(prepared['immutableInputs'])
    base_python = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe').resolve()
    assert Path(sys.executable).resolve() == base_python, 'Use the reviewed ordinary base interpreter'
    publication = load(UNIT / 'documentation_receipt.json')
    assert publication['passed'] and publication['status'] == 'COMPLETED'
    verify(publication['currentDocumentHashes'])
    assert binding(MONITOR)['sha256'] == MONITOR_SHA256
    names = [MANIFEST, VERIFIER, CLEAN, MONITOR, Path(__file__).resolve(), HERE / 'preparation_v2.json',
             UNIT / 'documentation_receipt.json'] + [resolve(path) for path in prepared['immutableInputs']]
    inputs = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): binding(path) for path in names}
    save(UNIT / 'host_launch.json', {'at': stamp(), 'oneShot': True, 'inputs': inputs,
        'guards': {'wallSeconds': 60, 'ownedTreeRssBytes': 1000000000,
                   'availableRamBytes': 300000000, 'lowRamSamples': 5, 'sampleSeconds': .25},
        'modelsOrInferenceAuthorized': False})
    result = {'startedAt': stamp(), 'passed': False, 'status': 'FAILED', 'inputs': inputs,
              'inferenceRun': False, 'productionQualified': False, 'sampleCount': 0,
              'peakSampledOwnedTreeRssBytes': 0, 'qualificationEvidence': False,
              'actualHelperInterpreter': {'path': str(base_python), 'binding': binding(base_python)},
              'observedExactOwnedExecutables': []}
    process = owned = None
    monitor = None
    began = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location('_pose_racket_host_owned_monitor', MONITOR)
        monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor)
        args = [str(CLEAN), '-B', str(VERIFIER), '--api-smoke', '--report', REPORT.relative_to(ROOT).as_posix()]
        result['command'] = args
        with LOG.open('x', encoding='utf-8') as log, SAMPLES.open('x', encoding='utf-8') as samples:
            process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            low = 0
            while process.poll() is None:
                rows = owned.sample()
                remembered = {(row['pid'], row['createTimeUnixSeconds'])
                              for row in result['observedExactOwnedExecutables']}
                for row in rows:
                    identity = (row['pid'], row['createTimeUnixSeconds'])
                    if identity in remembered:
                        continue
                    try:
                        actual = psutil.Process(row['pid'])
                        if actual.create_time() != row['createTimeUnixSeconds']:
                            continue
                        executable = Path(actual.exe())
                        result['observedExactOwnedExecutables'].append({
                            'pid': row['pid'], 'createTimeUnixSeconds': row['createTimeUnixSeconds'],
                            'path': str(executable), 'binding': binding(executable)})
                    except psutil.NoSuchProcess:
                        pass
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                available = psutil.virtual_memory().available
                elapsed = time.perf_counter() - began
                samples.write(json.dumps({'at': stamp(), 'elapsedSeconds': elapsed,
                    'processes': rows, 'ownedTreeRssBytes': rss, 'availableRamBytes': available}) + '\n')
                samples.flush()
                result['sampleCount'] += 1
                result['peakSampledOwnedTreeRssBytes'] = max(result['peakSampledOwnedTreeRssBytes'], rss)
                low = low + 1 if available < 300000000 else 0
                reason = ('owned_tree_rss_above_1GB' if rss > 1000000000 else
                          'host_available_ram_below_300MB_for_5_samples' if low >= 5 else
                          'owned_child_wall_above_60s' if elapsed > 60 else None)
                if reason:
                    result['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(.25)
            result['exitCode'] = process.returncode
        assert not owned.wait_for_retirement(timeout=2), 'Exact owned host-check descendants remain'
        verify(inputs)
        report = load(REPORT)
        checks = report['checks']
        assert process.returncode == 0 and report['checksPassed'] and checks and all(row['passed'] for row in checks)
        assert report['manifestSha256'] == binding(MANIFEST)['sha256']
        assert report['inferenceRun'] is False and report['productionQualified'] is False and report['modelsImported'] == []
        smoke = report['apiSmoke']
        assert smoke['healthStatus'] == smoke['analysesStatus'] == 200 and smoke['emptyIsolatedLibrary']
        assert smoke['invalidUploadStatus'] == 415 and smoke['workerDisabled'] and smoke['inferenceRun'] is False
        assert smoke['modelModulesImported'] == []
        result.update({'passed': True, 'status': 'COMPLETED', 'checkCount': len(checks),
                       'inputsUnchanged': True, 'report': REPORT.relative_to(ROOT).as_posix(),
                       'reportBinding': binding(REPORT), 'apiSmoke': smoke})
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        result['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                result['terminationActions'] = monitor.stop_owned(process, owned)
            except BaseException as cleanup:
                result['cleanupError'] = repr(cleanup)
    finally:
        if owned is not None:
            result['ownedProcessIdentities'] = owned.identities()
            try:
                remaining = owned.wait_for_retirement(timeout=2)
                result['remainingOwnedIdentitiesAtTerminal'] = remaining
                result['allOwnedIdentitiesRetiredAtTerminal'] = not remaining
                if remaining:
                    result['passed'], result['status'] = False, 'FAILED'
            except BaseException as retirement:
                result['retirementError'] = repr(retirement)
                result['passed'], result['status'] = False, 'FAILED'
        try:
            verify(inputs)
            result['inputsUnchangedAtTerminal'] = True
        except BaseException as changed:
            result['bindingError'] = repr(changed)
            result['passed'], result['status'] = False, 'FAILED'
        result['elapsedSeconds'] = time.perf_counter() - began
        result['completedAt'] = stamp()
        result['logBinding'] = binding(LOG) if LOG.exists() else None
        save(RECEIPT, result)
    print(json.dumps({'passed': result['passed'], 'status': result['status'], 'checkCount': result.get('checkCount')}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
