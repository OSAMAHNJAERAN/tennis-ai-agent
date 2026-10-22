"""Guarded fresh 30-frame compatibility smoke; never publishes product state.

The base interpreter monitors exactly one clean-environment child. Existing
licensed media and historical smoke evidence are read only. This engineering
smoke cannot qualify accuracy, full-match capacity or deployment permissions.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'artifacts/validation/product_execution/candidate_storage_real_smoke_2026-10-05'
RUN = UNIT / 'run'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
HISTORY = ROOT / 'artifacts/validation/product_installation/pipeline_smoke_30_frames'
OLD_ANALYSIS = HISTORY / 'analysis_offline_retry'
EXCERPT = HISTORY / 'original_frames_0000_0029.mkv'
EXCERPT_SHA256 = '12d8eaf3f668738d3082849db92f731a3d1c3bb8124c16ab8e11e80c0c182552'
HELPER = ROOT / 'scripts/setup/verify_clean_pipeline_smoke.py'
CONFIG = ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'
PRODUCT_DB = Path(os.getenv('TENNIS_STORAGE_ROOT', str(ROOT / 'outputs/product'))).resolve() / 'analyses.sqlite3'
SAMPLE_SECONDS = .25
PREFLIGHT_AVAILABLE_BYTES = 2_500_000_000
MAXIMUM_OWNED_RSS_BYTES = 3_000_000_000
MINIMUM_AVAILABLE_BYTES = 400_000_000
LOW_RAM_SAMPLES = 5
MAXIMUM_WALL_SECONDS = 300
EXPECTED_JSON = (
    'ball_metrics.json', 'camera_registration.json', 'court_geometry.json', 'detections.json',
    'event_candidates.json', 'event_verification_trace.json', 'image_people.json', 'line_calls.json',
    'match_analytics.json', 'match_events.json', 'match_state.json', 'metrics.json',
    'patch_persistence_audit.json', 'pixel_motion_audit.json', 'player_metrics.json',
    'player_motion.json', 'point_analytics.json', 'racket_tracking.json', 'rallies.json',
    'score_history.json', 'scoring_events.json', 'shot_events.json',
    'stationary_candidate_audit.json', 'temporal_detour_audit.json',
    'temporal_spacing_audit.json', 'trajectories.json',
)
VOLATILE_PATHS = {
    'metrics.json': ['$.processing_fps', '$.total_time_s'],
}
SOURCE_BINDINGS = (
    'src/pipeline/phase6_pipeline.py', 'src/utils/ball_candidate_sequence.py',
    'src/utils/json_array_artifact.py', 'src/utils/video_frame_sequence.py',
    'src/tracking/temporal_ball_tracker.py', 'src/detection/image_people.py',
    'src/detection/wasb_ball_detector.py', 'src/detection/tiled_wasb_candidates.py',
    'src/detection/spaced_ball_stream.py', 'src/tracking/stationary_candidate_filter.py',
    'src/tracking/candidate_pixel_motion.py', 'src/tracking/selected_patch_persistence.py',
    'src/tracking/selected_temporal_detours.py', 'src/tracking/court_player_tracker.py',
    'scripts/setup/verify_clean_pipeline_smoke.py',
    'scripts/audit/measure_candidate_storage.py',
    'scripts/audit/verify_candidate_storage_pipeline_smoke.py',
    'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml',
    'data/demos/sources.json', 'models/keypoints_model.pth', 'yolo11m.pt',
    'artifacts/models/ball/wasb_tennis_best.pth.tar', 'yolo11s-pose.pt',
)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def strict_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def source_record():
    return next(row for row in strict_json(ROOT / 'data/demos/sources.json')['sources']
                if row['id'] == 'dimitrov_tiafoe_2024')


def snapshot(paths):
    result = {}
    for path in sorted(set(Path(value).resolve() for value in paths)):
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or out-of-workspace read-only binding: ' + str(path))
        result[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    return result


def bindings():
    source = source_record()
    files = [ROOT / relative for relative in SOURCE_BINDINGS]
    files.extend([ROOT / source['sourcePath'], ROOT / source['inputPath'], EXCERPT, CLEAN])
    # Bind the external model implementation as well as its checkpoint.
    files.extend((ROOT / 'artifacts/research/WASB-SBDT').rglob('*.py'))
    value = snapshot(files)
    for field, sha_field in (('sourcePath', 'sourceSha256'), ('inputPath', 'inputSha256')):
        if value[source[field]]['sha256'] != source[sha_field]:
            raise ValueError('Licensed source binding changed: ' + field)
    if digest(EXCERPT) != EXCERPT_SHA256:
        raise ValueError('Historical 30-frame excerpt hash changed')
    return value


def history_snapshot():
    paths = [path for path in HISTORY.rglob('*') if path.is_file()]
    paths.append(HISTORY.parent / 'real_pipeline_smoke.json')
    return snapshot(paths)


def product_jobs():
    # Do not construct JobStore: its constructor creates directories/migrates DBs.
    if not PRODUCT_DB.is_file():
        return {'path': str(PRODUCT_DB), 'exists': False, 'jobs': []}
    connection = sqlite3.connect(PRODUCT_DB.as_uri() + '?mode=ro', uri=True, timeout=5)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA query_only=ON')
        connection.execute('BEGIN')
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    finally:
        connection.close()
    return {'path': str(PRODUCT_DB), 'exists': True, 'jobs': rows}


def load_helper():
    spec = importlib.util.spec_from_file_location('candidate_storage_smoke_helper', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = RUN
    return module


def load_monitor():
    # Reuse the benchmark's verified redirector/descendant ownership logic.
    path = ROOT / 'scripts/audit/measure_candidate_storage.py'
    spec = importlib.util.spec_from_file_location('candidate_storage_owned_monitor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verified_input(helper):
    import cv2
    import numpy as np
    source = source_record()
    retained = ROOT / source['inputPath']
    original = ROOT / source['sourcePath']
    if (digest(EXCERPT) != EXCERPT_SHA256 or digest(retained) != source['inputSha256']
            or digest(original) != source['sourceSha256']):
        raise ValueError('Licensed input/excerpt hash changed before decode')
    probes = {name: helper.probe(path) for name, path in (
        ('original', original), ('retained', retained), ('excerpt', EXCERPT))}
    for name, probe in probes.items():
        stream = probe['streams'][0]
        if Fraction(stream['avg_frame_rate']) != 60 or (stream['width'], stream['height']) != (1920, 1080):
            raise ValueError('Native FPS/dimensions changed: ' + name)
    if len(probes['excerpt']['frames']) != 30 or len(probes['retained']['frames']) != 600:
        raise ValueError('Licensed excerpt/retained decoded frame count changed')
    original_start = int(source['excerpt']['startFrame'])
    if original_start != 390 or len(probes['original']['frames']) != source['sourceVideo']['frames']:
        raise ValueError('Original decoded frame count or retained-source range changed')
    selected = {
        'original': probes['original']['frames'][original_start:original_start + 30],
        'retained': probes['retained']['frames'][:30],
        'excerpt': probes['excerpt']['frames'],
    }
    pts = {name: [Fraction(frame['best_effort_timestamp_time']) for frame in frames]
           for name, frames in selected.items()}
    relative = {name: [point - values[0] for point in values] for name, values in pts.items()}
    if not all(len(values) == 30 for values in pts.values()) or not (
            relative['original'] == relative['retained'] == relative['excerpt']):
        raise ValueError('Original and excerpt PTS cadence differs')
    if any(b <= a for values in pts.values() for a, b in zip(values, values[1:])):
        raise ValueError('Non-increasing original/excerpt PTS')
    captures = {name: cv2.VideoCapture(str(path)) for name, path in (
        ('original', original), ('retained', retained), ('excerpt', EXCERPT))}
    pixel_hashes = []
    try:
        if not all(capture.isOpened() for capture in captures.values()):
            raise ValueError('Cannot open read-only source/excerpt decoder')
        for _ in range(original_start):
            if not captures['original'].read()[0]:
                raise ValueError('Original decoder failed before the retained range')
        for index in range(30):
            decoded = {name: capture.read() for name, capture in captures.items()}
            if not all(ok for ok, _ in decoded.values()):
                raise ValueError('Source/excerpt decoder failed at frame ' + str(index))
            frame = decoded['original'][1]
            if not (np.array_equal(frame, decoded['retained'][1])
                    and np.array_equal(frame, decoded['excerpt'][1])):
                raise ValueError('Original decoded pixels differ at frame ' + str(index))
            pixel_hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
        if captures['excerpt'].read()[0]:
            raise ValueError('30-frame excerpt has additional decoded frames')
    finally:
        for capture in captures.values():
            capture.release()
    report = {
        'verifiedAt': stamp(), 'passed': True, 'source': source,
        'excerptPath': EXCERPT.relative_to(ROOT).as_posix(), 'excerptSha256': EXCERPT_SHA256,
        'sourceFrames': [0, 30], 'originalSourceFrames': [390, 420],
        'frames': 30, 'width': 1920, 'height': 1080, 'averageFrameRate': '60/1',
        'decodedFramesIdentical': True, 'originalPixelsCompared': 30,
        'decodedPixelSha256PerFrame': pixel_hashes,
        'originalPtsSeconds': [str(value) for value in pts['original']],
        'retainedPtsSeconds': [str(value) for value in pts['retained']],
        'excerptPtsSeconds': [str(value) for value in pts['excerpt']],
        'maximumRelativeTimestampDifferenceSeconds': 0,
        'timestampComparison': 'Exact rational values parsed from original best-effort PTS strings.',
        'modifications': 'Existing retained lossless 30-frame excerpt reused read only; no media generated or changed.',
        'qualificationEvidence': False, 'inferenceRun': False,
    }
    save(RUN / 'input_excerpt.json', report)
    return report


def first_difference(left, right, path='$'):
    if type(left) is not type(right):
        return path + ' (JSON value types differ)'
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ' (object keys differ)'
        for key in left:
            difference = first_difference(left[key], right[key], path + '.' + key)
            if difference is not None:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ' (array lengths differ)'
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, path + '[' + str(index) + ']')
            if difference is not None:
                return difference
    elif isinstance(left, float):
        if left.hex() != right.hex():
            return path + ' (exact floating-point values differ)'
    elif left != right:
        return path + ' (values differ)'
    return None


def normalized_json(path):
    value = strict_json(path)
    if path.name == 'metrics.json':
        for key in ('processing_fps', 'total_time_s'):
            timing = value.pop(key)
            if isinstance(timing, bool) or not isinstance(timing, (int, float)) or not math.isfinite(timing) or timing <= 0:
                raise ValueError('Invalid volatile runtime measurement: ' + key)
    return value


def compare_artifacts(helper):
    import cv2
    import numpy as np
    analysis = RUN / 'analysis_offline_retry'
    expected = set(EXPECTED_JSON)
    if {path.name for path in OLD_ANALYSIS.glob('*.json')} != expected:
        raise ValueError('Historical JSON artifact set differs from the frozen 26-file contract')
    if {path.name for path in analysis.glob('*.json')} != expected:
        raise ValueError('New JSON artifact set differs from the frozen 26-file contract')
    comparison = {'passed': False, 'volatilePaths': VOLATILE_PATHS,
                  'volatileProof': 'phase6_pipeline.py assigns total_time_s from time.time() minus start and processing_fps from frames / elapsed.',
                  'jsonFiles': {}, 'expectedJsonFiles': 26}
    for name in EXPECTED_JSON:
        old, new = OLD_ANALYSIS / name, analysis / name
        difference = first_difference(normalized_json(old), normalized_json(new))
        comparison['jsonFiles'][name] = {
            'exactParsedValuesEqual': difference is None, 'firstDifference': difference,
            'oldSha256': digest(old), 'newSha256': digest(new),
            'ignoredPaths': VOLATILE_PATHS.get(name, []),
        }
    comparison['runConfigBytesEqual'] = (OLD_ANALYSIS / 'run_config.yaml').read_bytes() == (analysis / 'run_config.yaml').read_bytes()
    comparison['runConfigSha256'] = digest(analysis / 'run_config.yaml')
    video_paths = {'old': OLD_ANALYSIS / 'annotated.mp4', 'new': analysis / 'annotated.mp4'}
    comparison['annotatedVideos'] = {}
    for name, path in video_paths.items():
        probe = helper.probe(path)
        stream = probe['streams'][0]
        ticks = [int(frame['best_effort_timestamp']) for frame in probe['frames']]
        exact_pts = [Fraction(tick) * Fraction(stream['time_base']) for tick in ticks]
        if (len(probe['frames']) != 30 or Fraction(stream['avg_frame_rate']) != 60
                or (stream['width'], stream['height']) != (1920, 1080)
                or exact_pts != [Fraction(index, 60) for index in range(30)]):
            raise ValueError('Annotated video frame count/dimensions/FPS/exact PTS differs: ' + name)
        comparison['annotatedVideos'][name] = {
            'frames': 30, 'averageFrameRate': stream['avg_frame_rate'], 'sha256': digest(path),
            'bytes': path.stat().st_size, 'width': stream['width'], 'height': stream['height'],
            'timeBase': stream['time_base'], 'framePtsTicks': ticks,
            'exactNormalizedFramePtsEqualIndexOver60': True,
        }
    captures = {name: cv2.VideoCapture(str(path)) for name, path in video_paths.items()}
    identical = []
    try:
        for _ in range(30):
            old_ok, old_frame = captures['old'].read()
            new_ok, new_frame = captures['new'].read()
            if not old_ok or not new_ok or old_frame.shape != (1080, 1920, 3) or new_frame.shape != (1080, 1920, 3):
                raise ValueError('Annotated full decoder failed or dimensions changed')
            identical.append(bool(np.array_equal(old_frame, new_frame)))
        if captures['old'].read()[0] or captures['new'].read()[0]:
            raise ValueError('Annotated video decoder found extra frames')
    finally:
        for capture in captures.values():
            capture.release()
    comparison['annotatedDecodedPixelsEqual'] = all(identical)
    comparison['annotatedPixelComparisonScope'] = 'Additional descriptive presentation evidence only; pixel equality is not required for the measurement JSON compatibility gate.'
    comparison['annotatedFramesCompared'] = len(identical)
    comparison['annotatedVideosFullyDecoded'] = True
    scratch = [path.relative_to(RUN).as_posix() for path in RUN.rglob('*')
               if path.is_file() and (path.name.startswith('ball_candidates_')
                                     or path.name.endswith('.tmp'))]
    comparison['unpublishedScratchFiles'] = scratch
    comparison['passed'] = (all(row['exactParsedValuesEqual'] for row in comparison['jsonFiles'].values())
                            and comparison['runConfigBytesEqual'] and not scratch)
    save(UNIT / 'artifact_comparison.json', comparison)
    if not comparison['passed']:
        raise ValueError('Actual 30-frame artifact compatibility failed; exact comparison receipt retained')
    return comparison


def run_child():
    import psutil
    actual = psutil.Process()
    report = {'startedAt': stamp(), 'passed': False, 'inferenceRun': False,
              'scope': 'Licensed 30-frame compatibility smoke only.', 'productionQualified': False,
              'qualificationEvidence': False, 'productStatePublished': False,
              'interpreter': sys.executable,
              'actualWorkloadIdentity': {'pid': actual.pid, 'createTimeUnixSeconds': actual.create_time()},
              'modelInferenceHelperInvoked': False}
    try:
        if Path(sys.executable).resolve() != CLEAN.resolve() or sys.prefix == sys.base_prefix:
            raise ValueError('Inference child must use the existing clean isolated interpreter')
        protocol = strict_json(UNIT / 'protocol.json')
        if bindings() != protocol['sourceBindingsBefore'] or history_snapshot() != protocol['historicalArtifactsBefore']:
            raise ValueError('Read-only sources/historical evidence changed before the child started')
        if RUN.exists():
            raise FileExistsError('Fresh smoke run directory already exists; no reuse or automatic retry')
        RUN.mkdir(exist_ok=False)
        helper = load_helper()
        input_report = verified_input(helper)
        if (RUN / 'analysis_offline_retry').exists():
            raise FileExistsError('Fresh analysis output must not exist before inference')
        report['modelInferenceHelperInvoked'] = True
        result = helper.run_smoke(EXCERPT, input_report)
        smoke = strict_json(UNIT / 'real_pipeline_smoke.json')
        report['inferenceRun'] = smoke['inferenceRun']
        if (result != 0 or not smoke['passed'] or smoke['networkAttempts']
                or smoke['productionQualified'] or smoke['productStatePublished']):
            raise ValueError('Offline clean helper did not complete the unqualified isolated smoke')
        report['comparison'] = compare_artifacts(helper)
        report['helperReceiptSha256'] = digest(UNIT / 'real_pipeline_smoke.json')
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        if (UNIT / 'real_pipeline_smoke.json').is_file():
            report['inferenceRun'] = strict_json(UNIT / 'real_pipeline_smoke.json')['inferenceRun']
        memory = actual.memory_info()
        report['actualProcessMemoryBytes'] = memory._asdict()
        report['windowsPeakWorkingSetBytes'] = getattr(memory, 'peak_wset', None)
        report['windowsPeakPagefileBytes'] = getattr(memory, 'peak_pagefile', None)
        report['peakMemoryScope'] = 'OS peaks for the actual workload interpreter, separate from parent sampled aggregate RSS/private; pagefile peak is process commit.'
        report['completedAt'] = stamp()
        save(UNIT / 'child_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'inferenceRun', 'error')}), flush=True)


def run_parent():
    import psutil
    monitor = load_monitor()
    if not UNIT.resolve().is_relative_to((ROOT / 'artifacts/validation/product_execution').resolve()):
        raise ValueError('Owned smoke directory escaped the validation workspace')
    UNIT.mkdir(parents=True, exist_ok=False)
    report = {'startedAt': stamp(), 'passed': False, 'status': 'NOT_STARTED',
              'parentInterpreter': sys.executable, 'parentIsBaseInterpreter': sys.prefix == sys.base_prefix,
              'inferenceChildStarted': False, 'scope': 'Sampled owned-child 30-frame compatibility smoke only.',
              'qualificationEvidence': False, 'productionQualified': False, 'productStatePublished': False,
              'sampleIntervalSeconds': SAMPLE_SECONDS, 'samples': 0,
              'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
              'minimumSampledAvailableRamBytes': psutil.virtual_memory().available}
    protocol = {
        'declaredAt': stamp(), 'scriptSha256': digest(Path(__file__).resolve()),
        'cleanInterpreter': str(CLEAN), 'freshRunDirectory': str(RUN),
        'readOnlyProductionDatabase': str(PRODUCT_DB), 'frames': 30, 'nativeFps': 60,
        'sampleIntervalSeconds': SAMPLE_SECONDS, 'preflightAvailableRamBytes': PREFLIGHT_AVAILABLE_BYTES,
        'maximumSampledOwnedTreeRssBytes': MAXIMUM_OWNED_RSS_BYTES,
        'minimumAvailableRamBytes': MINIMUM_AVAILABLE_BYTES, 'lowRamConsecutiveSamplesToStop': LOW_RAM_SAMPLES,
        'maximumChildWallSeconds': MAXIMUM_WALL_SECONDS,
        'volatilePaths': VOLATILE_PATHS, 'expectedJsonFiles': list(EXPECTED_JSON),
        'ownedTreeMonitor': 'scripts/audit/measure_candidate_storage.py::OwnedProcessTree and stop_owned',
        'retention': 'Preserve every owned log, partial artifact, sample and receipt; no deletion or retry.',
        'qualificationEvidence': False, 'productionQualified': False, 'productStatePublished': False,
        'hostRamBeforeBytes': psutil.virtual_memory()._asdict(), 'diskBeforeBytes': shutil.disk_usage(ROOT)._asdict(),
    }
    process = None
    owned = None
    try:
        initial_available = psutil.virtual_memory().available
        report['preflightAvailableRamBytes'] = initial_available
        if initial_available < PREFLIGHT_AVAILABLE_BYTES:
            report['notStartedReason'] = 'Available RAM below the predeclared 2.5 GB launch threshold; no retry.'
            save(UNIT / 'protocol.json', protocol)
            return 2
        protocol['sourceBindingsBefore'] = bindings()
        protocol['historicalArtifactsBefore'] = history_snapshot()
        protocol['productionJobStatesBefore'] = product_jobs()
        before_launch = psutil.virtual_memory().available
        report['availableRamImmediatelyBeforeLaunchBytes'] = before_launch
        save(UNIT / 'protocol.json', protocol)
        if before_launch < PREFLIGHT_AVAILABLE_BYTES:
            report['notStartedReason'] = 'Available RAM fell below the launch threshold during read-only binding; no retry.'
            return 2
        begin = time.perf_counter()
        with (UNIT / 'child_stdout.txt').open('x', encoding='utf-8') as log, \
                (UNIT / 'resource_samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen([str(CLEAN), str(Path(__file__).resolve()), '--child'],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            owned = monitor.OwnedProcessTree(process)
            report['inferenceChildStarted'] = True
            report['status'] = 'RUNNING'
            save(UNIT / 'running_process.json', {'pid': process.pid, 'createTimeUnixSeconds': owned.root_identity[1],
                                                'startedAt': stamp(), 'command': [str(CLEAN), str(Path(__file__).resolve()), '--child']})
            low_ram = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                private_values = [row['memoryBytes'].get('private') for row in rows]
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - begin, 'processes': rows,
                          'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                          'ownedTreePrivateBytes': sum(value for value in private_values if value is not None),
                          'allPrivateValuesAvailable': all(value is not None for value in private_values),
                          'availableRamBytes': available}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                report['samples'] += 1
                report['peakSampledOwnedTreeRssBytes'] = max(report['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                report['peakSampledOwnedTreePrivateBytes'] = max(report['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                report['minimumSampledAvailableRamBytes'] = min(report['minimumSampledAvailableRamBytes'], available)
                low_ram = low_ram + 1 if available < MINIMUM_AVAILABLE_BYTES else 0
                reason = None
                if sample['ownedTreeRssBytes'] > MAXIMUM_OWNED_RSS_BYTES:
                    reason = 'Owned process-tree RSS exceeded 3 GB'
                elif low_ram >= LOW_RAM_SAMPLES:
                    reason = 'Available RAM below 400 MB for five consecutive samples'
                elif sample['elapsedSeconds'] > MAXIMUM_WALL_SECONDS:
                    reason = 'Owned child wall time exceeded 300 seconds'
                if reason is not None:
                    report['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(SAMPLE_SECONDS)
            report['returncode'] = process.returncode
            report['monitoredChildWallSeconds'] = time.perf_counter() - begin
        if process.returncode != 0:
            raise RuntimeError('Owned smoke child failed; partial outputs and log retained')
        child = strict_json(UNIT / 'child_receipt.json')
        if not child['passed']:
            raise ValueError('Owned smoke child did not pass exact artifact comparison')
        report['actualWorkloadIdentity'] = child['actualWorkloadIdentity']
        report['actualWorkloadWasSampled'] = owned.observed_workload(child['actualWorkloadIdentity'])
        report['actualWorkloadWindowsPeakWorkingSetBytes'] = child['windowsPeakWorkingSetBytes']
        report['actualWorkloadWindowsPeakPagefileBytes'] = child['windowsPeakPagefileBytes']
        report['actualWorkloadProcessMemoryBytes'] = child['actualProcessMemoryBytes']
        if not report['actualWorkloadWasSampled']:
            raise ValueError('Actual workload interpreter identity was not sampled; wrapper-only evidence is invalid')
        report['remainingOwnedIdentitiesAfterNormalExit'] = owned.wait_for_retirement()
        if report['remainingOwnedIdentitiesAfterNormalExit']:
            raise RuntimeError('An exact owned descendant remained after normal smoke exit')
        report['allOwnedIdentitiesRetiredAfterNormalExit'] = True
        report['passed'] = True
        report['status'] = 'COMPLETED'
        return 0
    except BaseException as error:
        report['status'] = 'STOPPED' if report.get('safeguardStop') else 'FAILED'
        report['passed'] = False
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                report['terminationActions'] = monitor.stop_owned(process, owned)
                report['returncode'] = process.returncode
            except BaseException as cleanup_error:
                report['ownedProcessCleanupError'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
        return 1
    finally:
        if owned is not None:
            report['ownedProcessIdentitiesAndPeaks'] = owned.identities()
        if process is not None and (UNIT / 'child_receipt.json').is_file():
            try:
                child = strict_json(UNIT / 'child_receipt.json')
                report['actualWorkloadIdentity'] = child['actualWorkloadIdentity']
                report['actualWorkloadWasSampled'] = owned is not None and owned.observed_workload(child['actualWorkloadIdentity'])
                report['actualWorkloadWindowsPeakWorkingSetBytes'] = child['windowsPeakWorkingSetBytes']
                report['actualWorkloadWindowsPeakPagefileBytes'] = child['windowsPeakPagefileBytes']
                report['actualWorkloadProcessMemoryBytes'] = child['actualProcessMemoryBytes']
            except BaseException as receipt_error:
                report['childReceiptVerificationError'] = type(receipt_error).__name__ + ': ' + str(receipt_error)
                report['passed'] = False
                report['status'] = 'FAILED'
        if 'sourceBindingsBefore' in protocol:
            for label, before_key, reader in (
                    ('sourceBindings', 'sourceBindingsBefore', bindings),
                    ('historicalArtifacts', 'historicalArtifactsBefore', history_snapshot),
                    ('productionJobStates', 'productionJobStatesBefore', product_jobs)):
                try:
                    after = reader()
                    report[label + 'After'] = after
                    unchanged = after == protocol[before_key]
                    report[label + 'Unchanged'] = unchanged
                    if not unchanged:
                        report['passed'] = False
                        report['status'] = 'FAILED'
                except BaseException as verification_error:
                    report[label + 'VerificationError'] = type(verification_error).__name__ + ': ' + str(verification_error)
                    report['passed'] = False
                    report['status'] = 'FAILED'
        report['completedAt'] = stamp()
        report['diskAfterBytes'] = shutil.disk_usage(ROOT)._asdict()
        report['retainedOwnedFilesLogicalBytes'] = sum(path.stat().st_size for path in UNIT.rglob('*') if path.is_file())
        report['samplingScope'] = 'Owned process-tree RSS may count shared pages twice and miss sub-0.25-second peaks; this is not total-pipeline capacity qualification.'
        save(UNIT / 'resource_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'status', 'inferenceChildStarted', 'samples',
                                                         'peakSampledOwnedTreeRssBytes', 'safeguardStop', 'error')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child()
        return 0
    if sys.prefix != sys.base_prefix or Path(sys.executable).resolve() == CLEAN.resolve():
        raise ValueError('Launch the monitor with base Python; only its owned child may use the clean environment')
    code = run_parent()
    # Post-run invariants are finalized in run_parent's finally block. A
    # successful child cannot override a changed source/history/product state.
    if code == 0 and not strict_json(UNIT / 'resource_receipt.json')['passed']:
        return 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())
