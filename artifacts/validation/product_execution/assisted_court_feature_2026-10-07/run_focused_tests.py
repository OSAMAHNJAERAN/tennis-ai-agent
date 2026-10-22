"""One guarded, source-bound assisted-anchor contract/geometry test attempt."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RUN = UNIT / 'focused_tests'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
SURFACE = ROOT / 'artifacts/validation/product_execution/participant_surface_probe_2026-10-07/experiment.py'
GUARDS = {'preflightAvailableBytes': 750_000_000, 'maximumOwnedRssBytes': 1_000_000_000,
          'minimumAvailableBytes': 400_000_000, 'lowSamples': 5, 'sampleSeconds': .25, 'wallSeconds': 180}
sys.path.insert(0, str(ROOT))

def stamp():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

def bindings():
    paths = [Path(__file__), UNIT / 'assisted_anchor.py', UNIT / 'test_assisted_anchor.py',
             UNIT / 'protocol.json', UNIT / 'prepared_manifest_v2.json', MONITOR, SURFACE,
             ROOT / 'src/court/calibration.py', ROOT / 'src/court/court_geometry.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): {
                'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths}

def verify(expected):
    for relative, value in expected.items():
        path = ROOT / relative
        assert path.stat().st_size == value['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value['sha256'], relative

def child():
    import psutil
    process = psutil.Process()
    identity = {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()}
    save(RUN / 'actual_identity.json', identity)
    began = time.monotonic()
    while not (RUN / 'actual_sampled.json').is_file():
        if time.monotonic() - began > 15:
            raise RuntimeError('Actual interpreter was not sampled before CV imports.')
        time.sleep(.02)
    expected = load(RUN / 'launch_marker.json')['sourceBindings']
    verify(expected)
    import pytest
    result = pytest.main([str(UNIT / 'test_assisted_anchor.py'), '-q', '-p', 'no:cacheprovider',
                          '--basetemp', str(RUN / 'pytest_tmp'), '--junitxml', str(RUN / 'junit.xml')])
    verify(expected)
    save(RUN / 'test_result.json', {'exitCode': int(result), 'passed': result == 0,
                                   'sourcesUnchanged': True, 'actualIdentity': identity,
                                   'modelInferenceRun': False, 'videoDecodeRun': False,
                                   'scientificQualificationEvidence': False})
    return int(result)

def run():
    import psutil
    source_bindings = bindings()
    surface = module('_assisted_test_product_snapshot', SURFACE)
    product_before = surface.product_snapshot()
    RUN.mkdir(exist_ok=False)
    save(RUN / 'launch_marker.json', {'at': stamp(), 'oneShot': True, 'guards': GUARDS,
                                     'sourceBindings': source_bindings, 'productBefore': product_before})
    resource = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False,
                'childStarted': False, 'samples': 0, 'peakOwnedTreeRssBytes': 0,
                'minimumAvailableRamBytes': psutil.virtual_memory().available,
                'qualificationEvidence': False, 'productionIntegrated': False}
    process = owned = monitor = None
    began = time.monotonic()
    try:
        assert resource['minimumAvailableRamBytes'] >= GUARDS['preflightAvailableBytes'], 'Preflight RAM below750MB'
        monitor = module('_assisted_focused_test_monitor', MONITOR)
        with (RUN / 'stdout.log').open('xb') as stdout, (RUN / 'samples.jsonl').open('x', encoding='utf-8') as samples:
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__)), '--child'], cwd=ROOT,
                                       stdout=stdout, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                                PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', OPENBLAS_NUM_THREADS='1',
                                                OMP_NUM_THREADS='1'), creationflags=subprocess.CREATE_NO_WINDOW)
            resource['childStarted'] = True
            owned = monitor.OwnedProcessTree(process)
            resource['launcherIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            low = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                elapsed = time.monotonic() - began
                samples.write(json.dumps({'at': stamp(), 'elapsedSeconds': elapsed, 'ownedProcesses': rows,
                                          'ownedTreeRssBytes': rss, 'availableRamBytes': available}) + '\n')
                samples.flush()
                resource['samples'] += 1
                resource['peakOwnedTreeRssBytes'] = max(resource['peakOwnedTreeRssBytes'], rss)
                resource['minimumAvailableRamBytes'] = min(resource['minimumAvailableRamBytes'], available)
                if (RUN / 'actual_identity.json').is_file() and not (RUN / 'actual_sampled.json').exists():
                    identity = load(RUN / 'actual_identity.json')
                    if owned.observed_workload(identity):
                        save(RUN / 'actual_sampled.json', {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                low = low + 1 if available < GUARDS['minimumAvailableBytes'] else 0
                reason = ('RSS above1GB' if rss > GUARDS['maximumOwnedRssBytes'] else
                          'Host RAM below400MB5samples' if low >= GUARDS['lowSamples'] else
                          'Wall above180seconds' if elapsed > GUARDS['wallSeconds'] else None)
                if reason:
                    resource['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(GUARDS['sampleSeconds'])
            resource['exitCode'] = process.returncode
        result = load(RUN / 'test_result.json')
        assert process.returncode == 0 and result['passed'], 'Focused tests failed; retain the consumed attempt.'
        assert owned.observed_workload(load(RUN / 'actual_identity.json'))
        resource['status'], resource['passed'] = 'COMPLETED', True
    except BaseException as error:
        resource['status'] = 'NOT_STARTED' if not resource['childStarted'] else 'STOPPED' if resource.get('safeguardStop') else 'FAILED'
        resource['error'] = repr(error)
        resource['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                resource['terminationActions'] = monitor.stop_owned(process, owned)
            except BaseException as cleanup:
                resource['cleanupError'] = repr(cleanup)
    finally:
        if process is not None:
            resource['exitCode'] = process.poll()
        if owned is not None:
            remaining = owned.wait_for_retirement()
            if remaining:
                resource['postExitTerminationActions'] = monitor.stop_owned(process, owned)
                remaining = owned.wait_for_retirement()
            resource['ownedIdentities'] = owned.identities()
            resource['remainingOwnedIdentities'] = remaining
            resource['allOwnedIdentitiesRetired'] = not remaining
            resource['passed'] = resource['passed'] and not remaining
        try:
            verify(source_bindings)
            resource['sourcesUnchanged'] = True
            resource['savedProductUnchanged'] = surface.product_snapshot() == product_before
            assert resource['savedProductUnchanged']
        except BaseException as error:
            resource['preservationError'] = repr(error)
            resource['passed'] = False
        if not resource['passed'] and resource['status'] == 'COMPLETED':
            resource['status'] = 'FAILED_FINALIZATION'
        resource['elapsedSeconds'] = time.monotonic() - began
        resource['completedAt'] = stamp()
        save(RUN / 'resource_receipt.json', resource)
    print(json.dumps({key: resource.get(key) for key in ('passed', 'status', 'error', 'samples',
                                                        'peakOwnedTreeRssBytes', 'minimumAvailableRamBytes', 'elapsedSeconds')}))
    return 0 if resource['passed'] else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    args = parser.parse_args()
    raise SystemExit(child() if args.child else run())
