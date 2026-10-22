"""One guarded, source-bound complete backend suite; no real model inference."""
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

ROOT = Path(__file__).resolve().parents[5]
UNIT = Path(__file__).resolve().parent
BASE = Path(r'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe')
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
EXPECTED = {'src/pipeline/phase6_pipeline.py': 'ce88b62b44613509d5f4217d0d11fe8c13d9136c9f32c8bf97813d3bdc922811',
            'src/utils/identity_frame_sequence.py': 'f2d2118ccbe68493cd191de79c4b674ea2d6e36c7f81bb726bef6ef0deb17f04',
            'tests/test_pipeline_pose_storage.py': 'c26c9aaf9541c6aecbca01c01b598edf8edfc912d4b1ea27e84c5f9eb55367a0'}
GUARDS = {'wallSeconds': 180, 'ownedTreeRssBytes': 2_000_000_000,
          'availableRamBytes': 300_000_000, 'lowRamSamples': 5, 'sampleSeconds': .25}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def snapshot():
    paths = []
    for directory in ('src', 'tests', 'configs'):
        paths.extend(path for path in (ROOT / directory).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts
                     and path.suffix not in ('.pyc', '.pyo'))
    paths.extend(ROOT / name for name in ('pytest.ini', 'pyproject.toml', 'setup.cfg', 'tox.ini',
                                        '.coveragerc', 'requirements.txt', 'requirements-product.txt')
                 if (ROOT / name).is_file())
    paths.extend((MONITOR, Path(__file__).resolve()))
    source = {path.relative_to(ROOT).as_posix(): binding(path) for path in sorted(set(paths))}
    return {'files': source, 'baseInterpreter': {'path': str(BASE), **binding(BASE)},
            'actualHelperInterpreter': {'path': sys.executable, **binding(Path(sys.executable))},
            'inventoryExcludes': ['__pycache__', '*.pyc', '*.pyo']}


def main():
    import psutil
    sys.path.insert(0, str(ROOT))
    from scripts.audit.measure_candidate_storage import OwnedProcessTree, stop_owned

    if (UNIT / 'launch.json').exists():
        raise RuntimeError('This exclusive full backend verification is already consumed')
    before = snapshot()
    for path, digest in EXPECTED.items():
        if before['files'][path]['sha256'] != digest:
            raise RuntimeError(f'Reviewed source changed before full backend verification: {path}')
    save(UNIT / 'source_bindings_before.json', before)
    command = [str(BASE), '-B', '-m', 'pytest', 'tests', '-q', '--tb=short',
               '-o', 'faulthandler_timeout=30', '-o', 'junit_family=legacy',
               '--basetemp=' + str(UNIT / 'tmp'), '--junitxml=' + str(UNIT / 'junit.xml')]
    record = {'status': 'LAUNCHING', 'passed': False, 'startedAt': stamp(), 'command': command,
              'cwd': str(ROOT), 'guards': GUARDS, 'sourceInventoryCount': len(before['files']),
              'sourceBeforePath': (UNIT / 'source_bindings_before.json').relative_to(ROOT).as_posix(),
              'scope': 'Complete backend software regressions; no real models/media jobs, accuracy or full-match qualification.',
              'storageRoot': str(UNIT / 'bootstrap'), 'qualificationEvidence': False}
    save(UNIT / 'launch.json', record)
    env = {**os.environ, 'TENNIS_STORAGE_ROOT': str(UNIT / 'bootstrap'),
           'YOLO_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    process = owned = None
    failure = None
    began, low_ram, peak_rss, samples_count = time.monotonic(), 0, 0, 0
    actual_executables = {}
    try:
        with (UNIT / 'stdout.log').open('xb') as stdout, (UNIT / 'stderr.log').open('xb') as stderr, (UNIT / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            owned = OwnedProcessTree(process)
            record['launcherIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            while process.poll() is None:
                rows = owned.sample()
                for row in rows:
                    identity = (row['pid'], row['createTimeUnixSeconds'])
                    if identity not in actual_executables:
                        try:
                            live = psutil.Process(row['pid'])
                            if live.create_time() == identity[1]:
                                executable = Path(live.exe())
                                actual_executables[identity] = {'identity': {'pid': identity[0], 'createTimeUnixSeconds': identity[1]},
                                                                'path': str(executable), **binding(executable)}
                        except psutil.NoSuchProcess:
                            pass
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                peak_rss = max(peak_rss, rss)
                available = psutil.virtual_memory().available
                sample = {'elapsedSeconds': time.monotonic() - began, 'ownedProcesses': rows,
                          'ownedTreeRssBytes': rss, 'availableRamBytes': available}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                samples_count += 1
                low_ram = low_ram + 1 if available < GUARDS['availableRamBytes'] else 0
                if rss > GUARDS['ownedTreeRssBytes'] or low_ram >= GUARDS['lowRamSamples'] or sample['elapsedSeconds'] > GUARDS['wallSeconds']:
                    raise RuntimeError('Full backend resource guard exceeded')
                time.sleep(GUARDS['sampleSeconds'])
            record['exitCode'] = process.wait()
            if owned.wait_for_retirement():
                raise RuntimeError('Full backend verification retained exact owned descendant identities')
            record['allOwnedIdentitiesRetired'] = True
        suites = list(ET.parse(UNIT / 'junit.xml').getroot().iter('testsuite'))
        record['counts'] = {key: sum(int(suite.get(key, 0)) for suite in suites)
                            for key in ('tests', 'failures', 'errors', 'skipped')}
        record['counts']['passed'] = record['counts']['tests'] - record['counts']['failures'] - record['counts']['errors'] - record['counts']['skipped']
        record['testElapsedSeconds'] = sum(float(suite.get('time', 0)) for suite in suites)
        record['status'] = 'TESTS_PASSED' if record['exitCode'] == 0 and record['counts']['tests'] > 0 and not record['counts']['failures'] and not record['counts']['errors'] else 'TESTS_FAILED'
    except BaseException as error:
        failure = error
        record['status'] = 'RUNNER_FAILED'
        record['error'] = repr(error)
        record['traceback'] = traceback.format_exc()
    finally:
        if process is not None and (process.poll() is None or (owned is not None and owned.live_identities())):
            try:
                record['stopActions'] = stop_owned(process, owned)
            except BaseException as error:
                record['cleanupError'] = repr(error)
                if failure is not None:
                    failure.add_note(f'Owned cleanup also failed: {error}')
            record['allOwnedIdentitiesRetired'] = not owned.live_identities() if owned is not None else process.poll() is not None
        if owned is not None:
            record['ownedIdentities'] = owned.identities()
        record['actualProcessExecutables'] = list(actual_executables.values())
        after = snapshot()
        save(UNIT / 'source_bindings_after.json', after)
        record['sourcesUnchanged'] = before == after
        record['sourceAfterPath'] = (UNIT / 'source_bindings_after.json').relative_to(ROOT).as_posix()
        record['baseInterpreterAfterMatchesBefore'] = before['baseInterpreter'] == after['baseInterpreter']
        record['completedAt'] = stamp()
        record['elapsedSeconds'] = time.monotonic() - began
        record['sampleCount'] = samples_count
        record['peakSampledOwnedTreeRssBytes'] = peak_rss
        record['passed'] = (record['status'] == 'TESTS_PASSED' and record['sourcesUnchanged']
                            and record.get('allOwnedIdentitiesRetired') is True and not record.get('cleanupError'))
        if record['passed']:
            record['status'] = 'PASSED'
        record['artifacts'] = {name: binding(UNIT / name) for name in
                               ('launch.json', 'source_bindings_before.json', 'source_bindings_after.json',
                                'stdout.log', 'stderr.log', 'samples.jsonl', 'junit.xml')
                               if (UNIT / name).is_file()}
        save(UNIT / 'receipt.json', record)
    print(json.dumps({key: record.get(key) for key in ('status', 'passed', 'exitCode', 'counts', 'testElapsedSeconds', 'sourcesUnchanged', 'sourceInventoryCount', 'allOwnedIdentitiesRetired', 'peakSampledOwnedTreeRssBytes')}, indent=2))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
