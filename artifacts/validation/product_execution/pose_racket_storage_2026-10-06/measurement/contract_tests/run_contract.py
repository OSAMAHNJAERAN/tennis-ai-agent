"""One exclusive guarded run of the eight prepared helper contract checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MEASUREMENT = HERE.parent
ROOT = MEASUREMENT.parents[4]
BASE = Path(r'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe')
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
GUARDS = {'preflightAvailableBytes': 750_000_000, 'wallSeconds': 180,
          'ownedTreeRssBytes': 1_000_000_000, 'availableRamBytes': 400_000_000,
          'lowRamSamples': 5, 'sampleSeconds': .25}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot():
    expected = json.loads((MEASUREMENT / 'prepared_bindings.json').read_bytes())
    result = {}
    for pathname, value in expected.items():
        path = Path(pathname)
        path = path if path.is_absolute() else ROOT / path
        actual = binding(path)
        if actual != value:
            raise ValueError('Prepared source/evidence changed: ' + pathname)
        result[pathname] = actual
    for path in (MEASUREMENT / 'prepared_bindings.json', MEASUREMENT / 'preparation.json', Path(__file__).resolve(), BASE):
        key, value = name(path), binding(path)
        if key in result and result[key] != value:
            raise ValueError('Duplicate source binding conflict')
        result[key] = value
    return result


def main():
    import psutil
    sys.path.insert(0, str(ROOT))
    from scripts.audit.measure_candidate_storage import OwnedProcessTree, stop_owned

    if {path.name for path in HERE.iterdir()} != {'run_contract.py'} or (MEASUREMENT / 'run').exists():
        raise ValueError('Exclusive contract output directory or unlaunched benchmark condition changed')
    before = snapshot()
    save(HERE / 'source_bindings_before.json', before)
    command = [str(BASE), '-B', '-m', 'pytest', str(MEASUREMENT / 'test_helper_contract.py'),
               '-q', '--tb=short', '-o', 'faulthandler_timeout=30', '-o', 'junit_family=legacy',
               '--basetemp=' + str(HERE / 'tmp'), '--junitxml=' + str(HERE / 'junit.xml')]
    record = {'startedAt': stamp(), 'passed': False, 'status': 'RESERVED', 'command': command,
              'guards': GUARDS, 'sourceBindingsBefore': before, 'benchmarkLaunched': False,
              'inferenceRun': False, 'qualificationEvidence': False, 'productionEdited': False}
    save(HERE / 'launch.json', record)
    process = owned = None
    began, samples_count, peak_rss, low = time.perf_counter(), 0, 0, 0
    try:
        available = psutil.virtual_memory().available
        record['availableRamImmediatelyBeforeLaunchBytes'] = available
        if available < GUARDS['preflightAvailableBytes']:
            record['status'] = 'NOT_STARTED'
            raise RuntimeError('Unchanged 750 MB RAM preflight refused contract child')
        with (HERE / 'stdout.log').open('xb') as stdout, (HERE / 'stderr.log').open('xb') as stderr, \
                (HERE / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'YOLO_OFFLINE': '1'},
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = OwnedProcessTree(process)
            record['launchedIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                peak_rss = max(peak_rss, rss)
                elapsed = time.perf_counter() - began
                samples.write(json.dumps({'at': stamp(), 'elapsedSeconds': elapsed, 'ownedProcesses': rows,
                                         'ownedTreeRssBytes': rss, 'availableRamBytes': available}, allow_nan=False) + '\n')
                samples.flush()
                samples_count += 1
                low = low + 1 if available < GUARDS['availableRamBytes'] else 0
                if rss > GUARDS['ownedTreeRssBytes'] or low >= GUARDS['lowRamSamples'] or elapsed > GUARDS['wallSeconds']:
                    record['safeguardStop'] = 'unchanged_contract_resource_guard'
                    raise RuntimeError('Contract tests resource guard exceeded')
                time.sleep(GUARDS['sampleSeconds'])
            record['exitCode'] = process.returncode
        if owned.wait_for_retirement():
            raise RuntimeError('Contract tests retained exact owned identities')
        if not owned.observed_workload(record['launchedIdentity']):
            raise RuntimeError('Actual base interpreter contract child not sampled by exact identity')
        suites = list(ET.parse(HERE / 'junit.xml').getroot().iter('testsuite'))
        record['counts'] = {key: sum(int(suite.get(key, 0)) for suite in suites)
                            for key in ('tests', 'failures', 'errors', 'skipped')}
        record['testElapsedSeconds'] = sum(float(suite.get('time', 0)) for suite in suites)
        if process.returncode != 0 or record['counts']['tests'] != 8 or any(record['counts'][key] for key in ('failures', 'errors', 'skipped')):
            raise RuntimeError('The eight prepared contract checks did not entirely pass')
        record['passed'] = True
        record['status'] = 'PASSED'
    except BaseException as error:
        record['status'] = record['status'] if record['status'] == 'NOT_STARTED' else 'FAILED'
        record['error'] = type(error).__name__ + ': ' + str(error)
        record['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                record['terminationActions'] = stop_owned(process, owned)
            except BaseException as cleanup:
                record['cleanupError'] = repr(cleanup)
    finally:
        if owned is not None:
            record['ownedProcessIdentities'] = owned.identities()
            try:
                remaining = owned.wait_for_retirement()
                record['remainingOwnedIdentitiesAtTerminal'] = remaining
                record['allOwnedIdentitiesRetiredAtTerminal'] = not remaining
                if remaining:
                    record['passed'] = False
            except BaseException as retirement:
                record['retirementError'] = repr(retirement)
                record['passed'] = False
        try:
            after = snapshot()
            save(HERE / 'source_bindings_after.json', after)
            record['sourceBindingsAfter'] = after
            record['sourceBindingsUnchanged'] = before == after
            if before != after or (MEASUREMENT / 'run').exists():
                raise ValueError('Contract tests source/benchmark boundary changed')
        except BaseException as verification:
            record['bindingVerificationError'] = repr(verification)
            record['passed'] = False
        record['sampleCount'] = samples_count
        record['peakSampledOwnedTreeRssBytes'] = peak_rss
        record['elapsedSeconds'] = time.perf_counter() - began
        record['completedAt'] = stamp()
        if not record['passed'] and record['status'] == 'PASSED':
            record['status'] = 'FAILED'
        record['artifacts'] = {path.name: binding(path) for path in HERE.iterdir() if path.is_file()}
        save(HERE / 'receipt.json', record)
    print(json.dumps({key: record.get(key) for key in ('passed', 'status', 'counts', 'testElapsedSeconds', 'sourceBindingsUnchanged', 'allOwnedIdentitiesRetiredAtTerminal', 'error')}, allow_nan=False))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
