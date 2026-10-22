"""One exclusive bounded pytest run; deterministic substitutes, no inference."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[6]
UNIT = Path(__file__).resolve().parent
BASE = Path(r'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe')
EXPECTED_PIPELINE = '93b1bcf44a56ed70d76264ae7ab8cd2e8423616821632da09904d71ddc3c75ec'
SOURCE_PATHS = ('src/pipeline/phase6_pipeline.py', 'tests/test_pipeline_pose_storage.py',
                'scripts/audit/measure_candidate_storage.py', 'configs/phase6_analytics/pipeline.yaml')


def binding(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    import psutil
    sys.path.insert(0, str(ROOT))
    from scripts.audit.measure_candidate_storage import OwnedProcessTree, stop_owned

    launch = UNIT / 'launch.json'
    if launch.exists():
        raise RuntimeError('This exclusive RED run has already been consumed')
    sources = {name: binding(ROOT / name) for name in SOURCE_PATHS}
    if sources['src/pipeline/phase6_pipeline.py']['sha256'] != EXPECTED_PIPELINE:
        raise RuntimeError('RED requires untouched frozen original pipeline')
    snapshot = UNIT / 'sources'
    snapshot.mkdir()
    for name in SOURCE_PATHS:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    command = [str(BASE), '-B', '-m', 'pytest', 'tests/test_pipeline_pose_storage.py', '-q',
               '--junitxml=' + str(UNIT / 'junit.xml'), '--basetemp=' + str(UNIT / 'tmp')]
    guards = {'wallSeconds': 180, 'ownedTreeRssBytes': 1_500_000_000,
              'availableRamBytes': 300_000_000, 'lowRamSamples': 5, 'sampleSeconds': .25}
    record = {'status': 'PREPARED', 'scope': 'Actual pipeline orchestration with deterministic inference/media substitutes; no real models/video jobs',
              'command': command, 'cwd': str(ROOT), 'sourceBefore': sources,
              'baseInterpreter': {'path': str(BASE), **binding(BASE)},
              'helper': {'path': str(Path(__file__)), **binding(Path(__file__))},
              'guards': guards, 'startedAtUtc': datetime.now(timezone.utc).isoformat()}
    save(launch, record)
    env = os.environ.copy()
    env['YOLO_OFFLINE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    process = owned = None
    began, low_ram = time.monotonic(), 0
    failure = None
    try:
        with (UNIT / 'stdout.log').open('xb') as stdout, (UNIT / 'stderr.log').open('xb') as stderr, (UNIT / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            owned = OwnedProcessTree(process)
            record['launchedIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                sample = {'elapsedSeconds': time.monotonic() - began, 'ownedProcesses': rows,
                          'ownedTreeRssBytes': rss, 'availableRamBytes': available}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                low_ram = low_ram + 1 if available < guards['availableRamBytes'] else 0
                if rss > guards['ownedTreeRssBytes'] or low_ram >= guards['lowRamSamples'] or sample['elapsedSeconds'] > guards['wallSeconds']:
                    raise RuntimeError('Focused RED resource guard exceeded')
                time.sleep(guards['sampleSeconds'])
            record['exitCode'] = process.wait()
            remaining = owned.wait_for_retirement()
            if remaining:
                raise RuntimeError('Focused pytest retained owned descendant identities')
            record['allOwnedIdentitiesRetired'] = True
        junit = ET.parse(UNIT / 'junit.xml').getroot()
        record['suites'] = [dict(suite.attrib) for suite in junit.iter('testsuite')]
        record['failureCases'] = [{'name': case.attrib['name'], 'message': child.attrib.get('message', '')}
                                  for case in junit.iter('testcase') for child in case if child.tag in ('failure', 'error')]
        known_failures = all(case['name'].startswith(('test_pose_and_racket_original_records_are_bounded_before_shot_linking', 'test_render_handles_release_on_failure_before_traceback_or_gc_cleanup', 'test_render_cleanup_failure_preserves_primary_and_releases_other_handle')) for case in record['failureCases'])
        record['status'] = 'GENUINE_RED' if record['exitCode'] == 1 and record['failureCases'] and known_failures else 'UNEXPECTED_TEST_OUTCOME'
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
        record['sourceAfter'] = {name: binding(ROOT / name) for name in SOURCE_PATHS}
        record['sourceBindingsUnchanged'] = record['sourceBefore'] == record['sourceAfter']
        record['elapsedSeconds'] = time.monotonic() - began
        record['completedAtUtc'] = datetime.now(timezone.utc).isoformat()
        save(UNIT / 'receipt.json', record)
    print(json.dumps({key: record.get(key) for key in ('status', 'exitCode', 'suites', 'failureCases', 'elapsedSeconds', 'sourceBindingsUnchanged', 'allOwnedIdentitiesRetired')}, indent=2))
    return 0 if record['status'] == 'GENUINE_RED' and record['sourceBindingsUnchanged'] and record['allOwnedIdentitiesRetired'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
