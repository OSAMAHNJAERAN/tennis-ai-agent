"""Prespecified isolated registration-record measurements; no models or footage."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'artifacts/validation/product_execution/registration_storage_measurement_2026-10-05'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
CASES = [('dynamic', count) for count in (600, 18000, 432000, 864000)] + [('static', 864000)]
SOURCES = ('scripts/audit/measure_registration_storage.py', 'scripts/audit/measure_candidate_storage.py',
           'src/utils/frame_registration_sequence.py', 'src/court/calibration.py',
           'src/pipeline/phase6_pipeline.py', 'src/court/camera_registration.py', 'src/tracking/court_player_tracker.py',
           'requirements-product.txt', 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml',
           'artifacts/validation/product_installation/clean_env/Scripts/python.exe')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def publish_handshake(path, value):
    # Neither process observes a partially written handshake file. These exact
    # names live only in the fresh, exclusively owned case directory.
    temporary = path.with_name('.' + path.name + '.tmp')
    save(temporary, value)
    assert not path.exists(), 'Preserve an existing handshake'
    temporary.rename(path)


def source_bindings():
    return {name: digest(ROOT / name) for name in SOURCES}


def load_monitor():
    spec = importlib.util.spec_from_file_location('registration_owned_monitor', ROOT / 'scripts/audit/measure_candidate_storage.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def child(profile, mode, count, directory):
    # Imports belong to the isolated actual interpreter, never the monitoring parent.
    import psutil
    process = psutil.Process()
    publish_handshake(directory / 'actual_identity.json', {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()})
    acknowledged = directory / 'actual_identity_sampled.json'
    deadline = time.monotonic() + 10
    while not acknowledged.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Parent did not acknowledge exact actual interpreter sampling')
        time.sleep(.01)
    import cv2
    import numpy as np
    from src.court.calibration import CourtCalibration
    from src.utils.frame_registration_sequence import FrameRegistration, FrameRegistrationSequence

    started = time.perf_counter()
    anchor = CourtCalibration(np.array([[.125, -.0, .5], [0., .25, -.125], [0., 0., 1.]]),
                              (True, False, True), .125, .25, .0625)
    points = np.arange(28, dtype=np.float32).reshape(1, 14, 2)

    def generated(index):
        if profile == 'static':
            return FrameRegistration(anchor, np.eye(3))
        # Alternating precision and rejection intervals stress exact private fields,
        # including a rejected calibration carrying its diagnostic matrix.
        dtype = np.float32 if index % 2 else np.float64
        matrix = np.array([[.125, -.0, (index % 127) / 64], [0., .25, -.125], [0., 0., 1.]], dtype=dtype)
        reason = 'CAMERA_CUT' if index % 19 == 0 else None
        calibration = CourtCalibration(None if index % 17 == 0 else matrix, (True, False, True),
                                       (index % 127) / 64, -.0, .0625, reason)
        transform = None if reason or calibration.image_to_court is None else np.array(
            [[1., -.0, (index % 31) / 8], [0., 1., .25], [0., 0., 1.]], dtype=dtype)
        return FrameRegistration(calibration, transform)

    def encoded(record):
        calibration = record.calibration
        def array(value):
            return None if value is None else [value.dtype.str, list(value.shape), value.tobytes(order='C').hex()]
        def scalar(value):
            return None if value is None else float(value).hex()
        projected = calibration.project_ground_point((11.5, 40.25))
        transformed = (None if record.anchor_to_frame_px is None else
                       cv2.perspectiveTransform(points, record.anchor_to_frame_px)[0])
        value = [array(calibration.image_to_court), list(calibration.inlier_mask),
                 scalar(calibration.reprojection_error_px), scalar(calibration.reprojection_error_m),
                 scalar(calibration.inlier_reprojection_error_px), calibration.rejection_reason,
                 array(record.anchor_to_frame_px), calibration.is_valid, calibration.inlier_count,
                 None if projected is None else [scalar(item) for item in projected], array(transformed)]
        return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('ascii')

    report = {'profile': profile, 'mode': mode, 'frames': count, 'passed': False,
              'actualWorkloadIdentity': {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()},
              'elapsedScope': 'Synthetic construction/checks/owned cleanup after imports and actual-identity sampling handshake; monitor includes startup.',
              'pythonVersion': sys.version, 'libraries': {'numpy': np.__version__, 'opencv': cv2.__version__, 'psutil': psutil.__version__},
              'qualificationEvidence': False, 'modelsImported': [], 'inferenceRun': False}
    owned_path = None
    try:
        context = (FrameRegistrationSequence(directory) if mode == 'new' and profile == 'dynamic' else
                   FrameRegistrationSequence.constant(anchor, count) if mode == 'new' else nullcontext())
        with context as storage:
            append_started = time.perf_counter()
            if mode == 'old':
                calibrations, transforms = [], []
                for index in range(count):
                    record = generated(index)
                    calibrations.append(record.calibration)
                    transforms.append(record.anchor_to_frame_px)
                del record
            elif profile == 'dynamic':
                for index in range(count):
                    storage.append(generated(index))
                # Instrumentation flushes the final bounded transaction, which
                # production also commits on success, to measure complete DB size.
                storage._connection.commit()
                owned_path = storage.path
                report['logicalDatabaseBytesBeforeCleanup'] = owned_path.stat().st_size
                report['finalTransactionFlushedForDiskMeasurement'] = True
            report['appendSeconds'] = time.perf_counter() - append_started
            def actual(index):
                return FrameRegistration(calibrations[index], transforms[index]) if mode == 'old' else storage[index]
            check_started = time.perf_counter()
            forward = hashlib.sha256()
            for index in range(count):
                expected = encoded(generated(index))
                observed = encoded(actual(index))
                if observed != expected:
                    raise AssertionError('Exact fields/matrix/projection differ at frame ' + str(index))
                forward.update(len(observed).to_bytes(4, 'big'))
                forward.update(observed)
            report['allFrameFieldsMatricesAndProjectionsExactlyMatchGeneratedInput'] = True
            report['forwardSha256'] = forward.hexdigest()
            report['forwardCheckSeconds'] = time.perf_counter() - check_started
            # Reverse and boundary/random access digest checks avoid retaining
            # a full record index or full serialized output during validation.
            indices = sorted(set([0, 1, count // 2, count - 2, count - 1] +
                                 [(index * 104729 + 37) % count for index in range(256)]), reverse=True)
            reverse = hashlib.sha256()
            for index in indices:
                observed = encoded(actual(index))
                assert observed == encoded(generated(index))
                reverse.update(index.to_bytes(8, 'big'))
                reverse.update(observed)
            assert encoded(actual(-1)) == encoded(generated(count - 1))
            if mode == 'new':
                assert len(storage.calibrations) == len(storage.transforms) == count
                for index in (0, count // 2, count - 1):
                    assert encoded(FrameRegistration(storage.calibrations[index], storage.transforms[index])) == encoded(generated(index))
                for record, index in zip(storage[-8::2], range(count - 8, count, 2), strict=True):
                    assert encoded(record) == encoded(generated(index))
            report['reverseAndRandomSampleSha256'] = reverse.hexdigest()
            report['reverseAndRandomSamples'] = len(indices)
        if owned_path is not None:
            report['databaseCleanup'] = {'path': owned_path.relative_to(ROOT).as_posix(),
                'allExactOwnedFilesAbsent': all(not Path(str(owned_path) + suffix).exists() for suffix in ('', '-journal', '-wal', '-shm'))}
            assert report['databaseCleanup']['allExactOwnedFilesAbsent']
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        memory = process.memory_info()
        report.update({'totalChildElapsedSeconds': time.perf_counter() - started,
                       'windowsPeakWorkingSetBytes': getattr(memory, 'peak_wset', None),
                       'windowsPeakPagefileBytes': getattr(memory, 'peak_pagefile', None),
                       'modelsImported': [name for name in ('torch', 'torchvision', 'ultralytics') if name in sys.modules],
                       'completedAt': stamp()})
        save(directory / 'child_receipt.json', report)
        print(json.dumps({'passed': report['passed'], 'profile': profile, 'mode': mode, 'frames': count}), flush=True)


def parent():
    import psutil
    monitor = load_monitor()
    UNIT.mkdir(parents=True, exist_ok=False)
    before = source_bindings()
    with (UNIT / 'executed_measure_registration_storage.py').open('xb') as stream:
        stream.write(Path(__file__).read_bytes())
    assert digest(UNIT / 'executed_measure_registration_storage.py') == before['scripts/audit/measure_registration_storage.py']
    protocol = {'declaredAt': stamp(), 'sourceBindingsBefore': before, 'cases': CASES,
                'modes': ['old', 'new'], 'scope': 'Synthetic registration records only; no video, models, training or accuracy qualification.',
                'sampleSeconds': .25, 'preflightAvailableRamBytes': 2500000000, 'maximumOwnedTreeRssBytes': 1400000000,
                'minimumAvailableRamBytes': 400000000, 'lowRamConsecutiveSamples': 5, 'maximumChildWallSeconds': 300,
                'minimumFreeDiskBytes': 4000000000, 'retention': 'Preserve all logs and partials; no retry or broad cleanup.'}
    protocol['actualIdentitySamplingHandshake'] = 'Child waits up to10seconds before heavy imports/work until parent acknowledges its exact sampled identity.'
    save(UNIT / 'protocol.json', protocol)
    report = {'startedAt': stamp(), 'passed': False, 'runs': [], 'comparisons': [], 'qualificationEvidence': False,
              'inferenceRun': False, 'scope': protocol['scope']}
    try:
        assert psutil.virtual_memory().available >= protocol['preflightAvailableRamBytes'], 'Insufficient preflight RAM; no launch'
        assert shutil.disk_usage(ROOT).free >= protocol['minimumFreeDiskBytes'], 'Insufficient preflight disk; no launch'
        for profile, count in CASES:
            for mode in ('old', 'new'):
                directory = UNIT / (profile + '_' + str(count) + '_' + mode)
                directory.mkdir()
                assert psutil.virtual_memory().available >= protocol['preflightAvailableRamBytes'], 'Insufficient RAM before next child; no retry'
                process = None
                owned = None
                receipt = {'profile': profile, 'mode': mode, 'frames': count, 'samples': 0,
                           'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
                           'minimumSampledAvailableRamBytes': psutil.virtual_memory().available, 'passed': False}
                begin = time.perf_counter()
                try:
                    with (directory / 'stdout.txt').open('x', encoding='utf-8') as log, \
                            (directory / 'samples.jsonl').open('x', encoding='utf-8') as samples:
                        process = subprocess.Popen([str(CLEAN), str(Path(__file__)), '--child', profile, mode, str(count), str(directory)],
                                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                        owned = monitor.OwnedProcessTree(process)
                        low_ram = 0
                        while process.poll() is None:
                            rows = owned.sample()
                            identity_path = directory / 'actual_identity.json'
                            acknowledged = directory / 'actual_identity_sampled.json'
                            if identity_path.exists() and not acknowledged.exists():
                                identity = json.loads(identity_path.read_text(encoding='utf-8'))
                                if owned.observed_workload(identity):
                                    publish_handshake(acknowledged, {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                            available = psutil.virtual_memory().available
                            private_values = [row['memoryBytes'].get('private') for row in rows]
                            sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - begin, 'processes': rows,
                                      'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                                      'ownedTreePrivateBytes': sum(value for value in private_values if value is not None),
                                      'allPrivateValuesAvailable': all(value is not None for value in private_values),
                                      'availableRamBytes': available}
                            samples.write(json.dumps(sample) + '\n')
                            samples.flush()
                            receipt['samples'] += 1
                            receipt['peakSampledOwnedTreeRssBytes'] = max(receipt['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                            receipt['peakSampledOwnedTreePrivateBytes'] = max(receipt['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                            receipt['minimumSampledAvailableRamBytes'] = min(receipt['minimumSampledAvailableRamBytes'], available)
                            low_ram = low_ram + 1 if available < protocol['minimumAvailableRamBytes'] else 0
                            reason = ('Owned RSS safeguard' if sample['ownedTreeRssBytes'] > protocol['maximumOwnedTreeRssBytes'] else
                                      'Available RAM safeguard' if low_ram >= 5 else
                                      'Owned wall-time safeguard' if sample['elapsedSeconds'] > 300 else None)
                            if reason:
                                receipt['safeguardStop'] = reason
                                raise RuntimeError(reason)
                            time.sleep(.25)
                    receipt['exitCode'] = process.returncode
                    assert process.returncode == 0, 'Child failed; preserve partial evidence'
                    child_report = json.loads((directory / 'child_receipt.json').read_text(encoding='utf-8'))
                    assert child_report['passed'] and not child_report['modelsImported']
                    assert owned.observed_workload(child_report['actualWorkloadIdentity']), 'Actual workload was not sampled'
                    receipt['actualWorkloadWasSampled'] = True
                    receipt['remainingOwnedIdentities'] = owned.wait_for_retirement()
                    assert not receipt['remainingOwnedIdentities'], 'Owned descendant remained before next child'
                    receipt['passed'] = True
                    receipt['child'] = child_report
                except BaseException as error:
                    receipt['error'] = type(error).__name__ + ': ' + str(error)
                    if process is not None:
                        try:
                            receipt['terminationActions'] = monitor.stop_owned(process, owned)
                        except BaseException as cleanup_error:
                            error.add_note('Owned cleanup also failed: ' + str(cleanup_error))
                            receipt['cleanupError'] = str(cleanup_error)
                    raise
                finally:
                    receipt['completedAt'] = stamp()
                    receipt['monitoredChildWallSeconds'] = time.perf_counter() - begin
                    if owned is not None:
                        receipt['ownedIdentities'] = owned.identities()
                    save(directory / 'resource_receipt.json', receipt)
                    report['runs'].append(receipt)
                print(json.dumps({'completed': mode, 'profile': profile, 'frames': count,
                                  'peakWorkingSetBytes': child_report['windowsPeakWorkingSetBytes'],
                                  'childSeconds': child_report['totalChildElapsedSeconds']}), flush=True)
            old, new = report['runs'][-2:]
            equal = all(old['child'][name] == new['child'][name] for name in ('forwardSha256', 'reverseAndRandomSampleSha256'))
            assert equal
            report['comparisons'].append({'profile': profile, 'frames': count, 'passed': True,
                                         'allFramesDirectlyComparedToFrozenSyntheticInput': True,
                                         'forwardDigestEqual': True, 'reverseRandomDigestEqual': True})
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        report['completedAt'] = stamp()
        report['sourceBindingsAfter'] = source_bindings()
        report['sourcesUnchanged'] = before == report['sourceBindingsAfter']
        if not report['sourcesUnchanged']:
            report['passed'] = False
        save(UNIT / 'report.json', report)
        print(json.dumps({'passed': report['passed'], 'completedChildren': len(report['runs']),
                          'comparisons': len(report['comparisons']), 'error': report.get('error')}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', nargs=4, metavar=('PROFILE', 'MODE', 'COUNT', 'DIRECTORY'))
    args = parser.parse_args()
    if args.child:
        profile, mode, count, directory = args.child
        sys.path.insert(0, str(ROOT))
        child(profile, mode, int(count), Path(directory))
        return 0
    assert sys.prefix == sys.base_prefix, 'Launch monitor from the base interpreter'
    return parent()


if __name__ == '__main__':
    raise SystemExit(main())
