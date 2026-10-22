"""One reviewed saved-artifact event replay; never run the historical evaluator main.

All seven original B-H configurations and all three already-consumed diagnostic
trajectory/player timelines use the actual frozen and current analyze methods.
Full outputs must match before coverage/metric filtering. Existing reports and
inputs are read only. This neither runs models/media nor qualifies accuracy.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
EVIDENCE = UNIT.parent
ROOT = EVIDENCE.parents[3]
RUN = UNIT / 'run'
BASELINE = EVIDENCE / 'src/events/event_detector.py'
CURRENT = ROOT / 'src/events/event_detector.py'
REPORT = ROOT / 'artifacts/validation/phase6_4_event_recovery_results.json'
EVALUATOR = ROOT / 'scripts/evaluate_phase6_4_event_recovery.py'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
EXPECTED_BASELINE_SHA = '078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
EXPECTED_CURRENT_SHA = 'ccc00b450aa144d59a5fff13ede8a675749dc62fdc500143d3c114af715eff63'
SAMPLE_SECONDS = .25
MAXIMUM_WALL_SECONDS = 180
MAXIMUM_OWNED_RSS_BYTES = 1_000_000_000
PREFLIGHT_AVAILABLE_BYTES = 800_000_000
MINIMUM_AVAILABLE_BYTES = 400_000_000
LOW_RAM_SAMPLES = 5
SCOPE = ('Exact current-vs-frozen event-analysis replay on saved consumed diagnostic artifacts. '
         'No model/media inference, historical report rewrite, accuracy or full-match qualification.')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(paths):
    return {path.relative_to(ROOT).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in sorted(set(paths))}


def historical_paths():
    report = read(REPORT)
    provenance = report['input_provenance']
    files = [REPORT, UNIT / 'preserved_integrity_test.json',
             UNIT / 'preserved/tests/test_phase6_4_recovery_artifacts.py']
    for label, record in provenance.items():
        if label == 'preserved_video_inputs':
            for video in record.values():
                files += [ROOT / item['path'] for item in video.values()]
        else:
            files.append(BASELINE if label == 'event_detector_implementation' else ROOT / record['path'])
    files += [EVIDENCE / 'backend' / name for name in ('receipt.json', 'log.txt', 'junit.xml', 'source_bindings.json')]
    return files


def source_paths():
    return list((ROOT / 'src').rglob('*.py')) + [
        Path(__file__).resolve(), BASELINE, EVALUATOR, MONITOR,
        ROOT / 'scripts/evaluate_phase6_4_cross_match.py',
        ROOT / 'configs/phase6_analytics/pipeline.yaml',
        UNIT / 'seal_preparation.py', UNIT / 'preserved_integrity_test.json']


def check_historical_provenance():
    report = read(REPORT)
    if report['qualification_evidence'] is not False or report['scientific_split'] != 'CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED':
        raise ValueError('Expected consumed, unqualified historical diagnostic report')
    provenance = report['input_provenance']
    if set(provenance['preserved_video_inputs']) != {'video_08', 'video_09', 'video_10'}:
        raise ValueError('Unexpected diagnostic video set')
    for label, record in provenance.items():
        records = list(record.values()) if label == 'preserved_video_inputs' else [record]
        if label == 'preserved_video_inputs':
            records = [item for video in records for item in video.values()]
        for item in records:
            path = BASELINE if label == 'event_detector_implementation' else ROOT / item['path']
            if not path.resolve().is_relative_to(ROOT) or digest(path) != item['sha256'] or path.stat().st_size != item['size_bytes']:
                raise ValueError('Historical input provenance mismatch: ' + item['path'])
    if provenance['event_detector_implementation']['path'] != 'src/events/event_detector.py':
        raise ValueError('Historical original detector path changed')
    if digest(BASELINE) != EXPECTED_BASELINE_SHA or digest(CURRENT) != EXPECTED_CURRENT_SHA:
        raise ValueError('Frozen/current detector binding changed')
    return report


def analysis_bytes(analysis):
    return json.dumps(asdict(analysis), sort_keys=True, allow_nan=True,
                      default=lambda value: value.value).encode('utf-8')


def save_bytes(path, value):
    with path.open('xb') as stream:
        stream.write(value)


def run_child(token):
    import psutil
    protocol = read(UNIT / 'protocol.json')
    if token != protocol['launchToken'] or (RUN / 'started.json').exists():
        raise ValueError('Fresh owned child token required')
    process = psutil.Process()
    identity = {'actualChildPid': os.getpid(), 'actualChildCreateTimeUnixSeconds': process.create_time()}
    save(UNIT / 'actual_child_identity.json', identity)
    deadline = time.perf_counter() + 15
    while not (UNIT / 'actual_child_sampled.json').is_file():
        if time.perf_counter() > deadline:
            raise RuntimeError('Exact child identity was not sampled before imports')
        time.sleep(.05)
    save(RUN / 'started.json', {'at': stamp(), 'identity': identity})
    report = {'startedAt': stamp(), 'passed': False, 'scope': SCOPE, 'actualWorkloadIdentity': identity,
              'comparisons': [], 'inferenceRun': False, 'modelsLoaded': False, 'historicalReportsWritten': False,
              'qualificationEvidence': False}
    attempts = []
    def blocked(*args, **kwargs):
        attempts.append(repr(args[:1]))
        raise RuntimeError('Network disabled during saved diagnostic replay')
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    try:
        if snapshot(source_paths()) != protocol['sourceBindingsBefore']:
            raise ValueError('Source bindings changed before actual replay')
        historical = check_historical_provenance()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        evaluator = load_module('saved_event_recovery_evaluator_20261006', EVALUATOR)
        frozen = load_module('frozen_saved_event_timestamp_detector_20261006', BASELINE)
        import yaml
        provenance = historical['input_provenance']
        videos = read(ROOT / provenance['video_manifest']['path'])['videos']
        coverage = read(ROOT / provenance['annotation_coverage']['path'])
        with (ROOT / provenance['active_event_config']['path']).open('r', encoding='utf-8') as stream:
            base_config = yaml.safe_load(stream)['event_detection']
        configs = evaluator._variant_configs(base_config)
        if set(configs) != {row['variant'] for row in historical['ablations'] if row['variant'] != 'A_INTEGRATED_PRE_RECOVERY'}:
            raise ValueError('Expected all seven original rerunnable B-H variant settings')
        if set(videos) != set(provenance['preserved_video_inputs']):
            raise ValueError('Manifest and preserved input video sets differ')
        save(RUN / 'variant_settings.json', configs)
        inputs = {}
        frame_counts = {}
        for video_id, records in provenance['preserved_video_inputs'].items():
            trajectory_path = ROOT / records['trajectories']['path']
            detection_path = ROOT / records['detections']['path']
            if trajectory_path.parent != detection_path.parent:
                raise ValueError('Saved timeline pair path mismatch')
            points, p1, p2, detections = evaluator._load_video_inputs(str(trajectory_path.parent.parent), video_id)
            if len(points) != int(videos[video_id]['frame_count']) or len(p1) != len(points) or len(p2) != len(points):
                raise ValueError('Saved trajectory/player/manifest counts differ')
            inputs[video_id] = (points, p1, p2)
            frame_counts[video_id] = len(points)
        report['frameCounts'] = frame_counts
        for variant, settings in configs.items():
            for video_id, metadata in videos.items():
                points, p1, p2 = inputs[video_id]
                kwargs = {'fps': float(metadata['fps']),
                          'frame_size': (int(metadata['width']), int(metadata['height'])),
                          'camera_offsets_px': None}
                # Separate actual detector instances. The evaluator main and
                # metric writer are never called, and no GT enters analyze.
                old = frozen.TennisEventDetector(config=dict(settings)).analyze(points, p1, p2, **kwargs)
                new = evaluator.TennisEventDetector(config=dict(settings)).analyze(points, p1, p2, **kwargs)
                old_bytes, new_bytes = analysis_bytes(old), analysis_bytes(new)
                save_bytes(RUN / f'{variant}_{video_id}_frozen.json', old_bytes)
                save_bytes(RUN / f'{variant}_{video_id}_current.json', new_bytes)
                comparison = {'variant': variant, 'video': video_id, 'frames': len(points),
                    'fps': kwargs['fps'], 'dimensions': list(kwargs['frame_size']),
                    'cameraOffsets': None, 'poseSupport': None,
                    'candidates': len(new.candidates), 'events': len(new.events),
                    'verificationTraces': len(new.verification_traces), 'bytes': len(new_bytes),
                    'frozenSha256': hashlib.sha256(old_bytes).hexdigest(),
                    'currentSha256': hashlib.sha256(new_bytes).hexdigest(),
                    'exactAnalysisBytes': new_bytes == old_bytes}
                if variant == 'H_FINAL_INTEGRATED':
                    old_events = evaluator._covered_records(
                        [evaluator._event_record(event, video_id) for event in old.events], video_id, coverage)
                    comparison['historicalHFilteredEventsEqual'] = (
                        json.dumps(old_events, sort_keys=True, allow_nan=True) ==
                        json.dumps(historical['final_events_by_video'][video_id], sort_keys=True, allow_nan=True))
                    if not comparison['historicalHFilteredEventsEqual']:
                        raise AssertionError('Frozen H replay differs from the preserved historical events: ' + video_id)
                report['comparisons'].append(comparison)
                if not comparison['exactAnalysisBytes']:
                    raise AssertionError('Current/frozen full analyze bytes differ: ' + variant + '/' + video_id)
        if len(report['comparisons']) != 21:
            raise AssertionError('Exactly seven variants by three videos required')
        forbidden = [name for name in ('torch', 'ultralytics', 'cv2', 'onnxruntime') if name in sys.modules]
        report['forbiddenInferenceImports'] = forbidden
        if forbidden or attempts:
            raise AssertionError('Unexpected model runtime import or network attempt')
        imported_paths = sorted({Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                                 if getattr(module, '__file__', None) and Path(module.__file__).is_file()})
        report['ordinaryImportedModuleBindings'] = {
            str(path): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in imported_paths}
        report['passed'] = True
        return 0
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        return 1
    finally:
        memory = process.memory_info()._asdict()
        report['actualProcessMemoryBytes'] = memory
        report['windowsPeakWorkingSetBytes'] = memory.get('peak_wset')
        report['windowsPeakPagefileBytes'] = memory.get('peak_pagefile')
        report['networkAttempts'] = attempts
        report['completedAt'] = stamp()
        save(UNIT / 'child_receipt.json', report)


def run_parent():
    import psutil
    preparation = read(UNIT / 'preparation.json')
    if preparation['status'] != 'PREPARED_NOT_EXECUTED' or preparation['scriptSha256'] != digest(__file__):
        raise ValueError('Reviewed preparation/script binding required')
    if os.name != 'nt' or sys.prefix != sys.base_prefix:
        raise ValueError('Use base Windows Python for this model-free replay')
    token = uuid.uuid4().hex
    save(UNIT / 'launch_marker.json', {'at': stamp(), 'launchToken': token, 'oneShot': True})
    monitor = load_module('saved_event_replay_owned_monitor_20261006', MONITOR)
    report = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False, 'scope': SCOPE,
              'replayChildStarted': False, 'inferenceRun': False, 'modelsLoaded': False,
              'samples': 0, 'peakSampledOwnedTreeRssBytes': 0, 'qualificationEvidence': False}
    process = owned = None
    before_sources = before_history = None
    began = None
    protocol = {'at': stamp(), 'launchToken': token, 'scope': SCOPE,
        'sourceBindingsBefore': snapshot(source_paths()), 'historicalArtifactsBefore': snapshot(historical_paths()),
        'preparationSha256': digest(UNIT / 'preparation.json'),
        'monitorSha256': digest(MONITOR), 'interpreter': sys.executable,
        'host': {'platform': platform.platform(), 'ramBefore': psutil.virtual_memory()._asdict()},
        'sampleIntervalSeconds': SAMPLE_SECONDS, 'wallLimitSeconds': MAXIMUM_WALL_SECONDS,
        'rssLimitBytes': MAXIMUM_OWNED_RSS_BYTES, 'preflightAvailableRamBytes': PREFLIGHT_AVAILABLE_BYTES,
        'minimumAvailableRamBytes': MINIMUM_AVAILABLE_BYTES, 'lowRamSamples': LOW_RAM_SAMPLES}
    save(UNIT / 'protocol.json', protocol)
    try:
        if protocol['sourceBindingsBefore'] != preparation['sourceBindings']:
            raise ValueError('Reviewed source bindings changed before replay launch')
        check_historical_provenance()
        before_sources, before_history = protocol['sourceBindingsBefore'], protocol['historicalArtifactsBefore']
        if psutil.virtual_memory().available < PREFLIGHT_AVAILABLE_BYTES:
            report['notStartedReason'] = 'Available host RAM below 800 MB; no child or retry'
            return 2
        RUN.mkdir(exist_ok=False)
        argv = [sys.executable, str(Path(__file__).resolve()), '--child', '--launch-token', token]
        began = time.perf_counter()
        with (UNIT / 'child_stdout.txt').open('x', encoding='utf-8') as log, (UNIT / 'resource_samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            report['replayChildStarted'] = True
            low = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - began,
                    'processes': rows, 'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                    'availableRamBytes': available}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                report['samples'] += 1
                report['peakSampledOwnedTreeRssBytes'] = max(report['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                if (UNIT / 'actual_child_identity.json').exists() and not (UNIT / 'actual_child_sampled.json').exists():
                    identity = read(UNIT / 'actual_child_identity.json')
                    if owned.observed_workload(identity):
                        save(UNIT / 'actual_child_sampled.json', {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                low = low + 1 if available < MINIMUM_AVAILABLE_BYTES else 0
                if sample['ownedTreeRssBytes'] > MAXIMUM_OWNED_RSS_BYTES or low >= LOW_RAM_SAMPLES or sample['elapsedSeconds'] > MAXIMUM_WALL_SECONDS:
                    report['safeguardStop'] = sample
                    raise RuntimeError('Owned RSS, host low-RAM or wall guard exceeded; retain partial replay')
                time.sleep(SAMPLE_SECONDS)
        report['returncode'] = process.returncode
        child = read(UNIT / 'child_receipt.json')
        report['comparisons'] = child['comparisons']
        report['actualWorkloadIdentity'] = child['actualWorkloadIdentity']
        report['actualWorkloadSampledByExactIdentity'] = owned.observed_workload(child['actualWorkloadIdentity'])
        if process.returncode != 0 or not child['passed'] or not report['actualWorkloadSampledByExactIdentity']:
            raise AssertionError('Owned actual replay child/comparisons failed')
        report['remainingOwnedIdentitiesAfterNormalExit'] = owned.wait_for_retirement()
        if report['remainingOwnedIdentitiesAfterNormalExit']:
            raise RuntimeError('Owned child identities remained after normal exit')
        report['allOwnedIdentitiesRetiredAfterNormalExit'] = True
        report['passed'] = True
        report['status'] = 'COMPLETED'
        return 0
    except BaseException as error:
        report['passed'] = False
        report['status'] = 'FAILED'
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                report['terminationActions'] = monitor.stop_owned(process, owned)
            except BaseException as cleanup_error:
                report['cleanupError'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
        return 1
    finally:
        if owned is not None:
            report['ownedProcessIdentitiesAndPeaks'] = owned.identities()
            try:
                leftovers = owned.wait_for_retirement()
                report['remainingExactOwnedIdentitiesAtTerminal'] = leftovers
                report['allExactOwnedIdentitiesRetiredAtTerminal'] = not leftovers
                if leftovers:
                    report['passed'] = False
                    report['status'] = 'FAILED'
            except BaseException as error:
                report['retirementVerificationError'] = type(error).__name__ + ': ' + str(error)
                report['passed'] = False
                report['status'] = 'FAILED'
        for label, before, reader in (('sourceBindings', before_sources, lambda: snapshot(source_paths())),
                                      ('historicalArtifacts', before_history, lambda: snapshot(historical_paths()))):
            if before is not None:
                try:
                    after = reader()
                    report[label + 'After'] = after
                    report[label + 'Unchanged'] = before == after
                    if before != after:
                        report['passed'] = False
                        report['status'] = 'FAILED'
                except BaseException as error:
                    report[label + 'VerificationError'] = type(error).__name__ + ': ' + str(error)
                    report['passed'] = False
                    report['status'] = 'FAILED'
        report['preparationUnchanged'] = digest(UNIT / 'preparation.json') == protocol['preparationSha256']
        if not report['preparationUnchanged']:
            report['passed'] = False
            report['status'] = 'FAILED'
        report['completedAt'] = stamp()
        report['monitoredWallSeconds'] = time.perf_counter() - began if began else None
        save(UNIT / 'replay_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('status', 'passed', 'samples', 'error')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--launch-token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return run_child(args.launch_token)
    code = run_parent()
    return 1 if code == 0 and not read(UNIT / 'replay_receipt.json')['passed'] else code


if __name__ == '__main__':
    raise SystemExit(main())

