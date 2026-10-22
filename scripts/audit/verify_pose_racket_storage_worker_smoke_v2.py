"""One fresh guarded real worker run after a declined RAM preflight.

The retained licensed input is read only. A fresh private JobStore executes the
actual worker and models once. Frozen normalizers/reporting use the exact new
raw artifact bytes, before URL presentation. This 30-frame engineering smoke
does not qualify accuracy, full-match capacity, or public deployment.
"""

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/worker_smoke_v2'
RUN = UNIT / 'run'
ISOLATED = RUN / 'isolated_product'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
RETAINED_GUARD = ROOT / 'scripts/audit/verify_candidate_storage_pipeline_smoke.py'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_baseline_2026-10-06'
ORIGIN = ROOT / 'scripts/audit/verify_result_storage_worker_smoke.py'
ORIGIN_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06'
ORIGIN_SHA256 = '8874064cac5a27ae70a7b0784d5f8eca17b1388f1e80665ab304cd1cab112b82'
FIRST_ATTEMPT = ROOT / 'scripts/audit/verify_result_storage_latency_worker_smoke.py'
FIRST_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_worker_smoke_2026-10-06'
FIRST_SHA256 = '279464db5328b87b0a524b9f669176f9498b609522718ddc4b2cab7873d7fefe'
SECOND_ATTEMPT = ROOT / 'scripts/audit/verify_result_storage_latency_headroom_worker_smoke.py'
SECOND_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_headroom_worker_smoke_2026-10-06'
SECOND_SHA256 = '99285464e48ed647a8886ee49a860c91ecc72d23f343381dbf5c3e67f288fee5'
CONFIG = ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'
SAMPLE_SECONDS = .25
PREFLIGHT_AVAILABLE_BYTES = 2_500_000_000
MAXIMUM_OWNED_RSS_BYTES = 3_000_000_000
MINIMUM_AVAILABLE_BYTES = 400_000_000
LOW_RAM_SAMPLES = 5
MAXIMUM_WALL_SECONDS = 300
EXCERPT_SHA256 = '12d8eaf3f668738d3082849db92f731a3d1c3bb8124c16ab8e11e80c0c182552'
SCOPE = ('One actual 30-frame, native-60-FPS worker/model and saved-result compatibility smoke. '
         'No independent labels, accuracy, full-match capacity or deployment qualification.')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def strict_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)
    check_finite(value)
    return value


def check_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON numeric overflow')
    if isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_finite(child)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def retained_guard():
    module = load_module('result_worker_retained_input_guard', RETAINED_GUARD)
    module.RUN = RUN
    return module


def binding_snapshot(guard):
    # Bind every in-repository production Python module, not just the older
    # smoke's selected list (which did not bind calibration/camera code).
    files = list((ROOT / 'src').rglob('*.py'))
    files += [Path(__file__).resolve(), RETAINED_GUARD, MONITOR, guard.HELPER,
              CONFIG, ROOT / 'data/demos/sources.json', CLEAN, BASELINE / 'receipt.json',
              BASELINE / 'src/api/results.py', BASELINE / 'src/api/reporting.py',
              UNIT / 'preparation.json', UNIT / 'origin.json', ORIGIN,
              ORIGIN_UNIT / 'preparation.json', ORIGIN_UNIT / 'observer_revision.json', FIRST_ATTEMPT]
    files += [path for path in FIRST_UNIT.rglob('*') if path.is_file()]
    files += [ROOT / name for name in ('models/keypoints_model.pth', 'yolo11m.pt',
                                     'artifacts/models/ball/wasb_tennis_best.pth.tar', 'yolo11s-pose.pt')]
    files += list((ROOT / 'artifacts/research/WASB-SBDT').rglob('*.py'))
    source = guard.source_record()
    files += [ROOT / source['sourcePath'], ROOT / source['inputPath'], guard.EXCERPT]
    result = guard.snapshot(files)
    for field, hash_field in (('sourcePath', 'sourceSha256'), ('inputPath', 'inputSha256')):
        if result[source[field]]['sha256'] != source[hash_field]:
            raise ValueError('Licensed source binding differs: ' + field)
    if digest(guard.EXCERPT) != EXCERPT_SHA256:
        raise ValueError('Retained exact 30-frame input differs')
    preserved = strict_json(BASELINE / 'receipt.json')['bindings']
    for name in ('src/api/results.py', 'src/api/reporting.py'):
        path = BASELINE / name
        if (digest(path) != preserved[name]['sha256'] or path.stat().st_size != preserved[name]['bytes']):
            raise ValueError('Frozen legacy module differs from its preservation receipt: ' + name)
    return result


def first_attempt_history():
    return {path.relative_to(ROOT).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in [FIRST_ATTEMPT] + sorted(path for path in FIRST_UNIT.rglob('*') if path.is_file())}


def verify_first_attempt():
    if digest(FIRST_ATTEMPT) != FIRST_SHA256:
        raise ValueError('The consumed first helper binding changed')
    receipt = strict_json(FIRST_UNIT / 'resource_receipt.json')
    protocol = strict_json(FIRST_UNIT / 'protocol.json')
    if (receipt['status'] != 'NOT_STARTED' or receipt['passed'] is not False or
            receipt['inferenceChildStarted'] is not False or receipt['samples'] != 0 or
            receipt['minimumSampledAvailableRamBytes'] != 2_355_306_496 or
            receipt['minimumSampledAvailableRamBytes'] >= PREFLIGHT_AVAILABLE_BYTES or
            protocol['scriptSha256'] != FIRST_SHA256 or 'sourceBindingsBefore' in protocol or
            (FIRST_UNIT / 'run').exists() or (FIRST_UNIT / 'actual_child_identity.json').exists() or
            (FIRST_UNIT / 'child_receipt.json').exists() or (FIRST_UNIT / 'resource_samples.jsonl').exists()):
        raise ValueError('The first attempt must be terminal NOT_STARTED before any child or model launch')
    return first_attempt_history()


def verify_second_attempt():
    if digest(SECOND_ATTEMPT) != SECOND_SHA256:
        raise ValueError('The consumed second helper changed')
    receipt = strict_json(SECOND_UNIT / 'resource_receipt.json')
    protocol = strict_json(SECOND_UNIT / 'protocol.json')
    if (receipt['status'] != 'NOT_STARTED' or receipt['passed'] is not False or
            receipt['inferenceChildStarted'] is not False or receipt['samples'] != 0 or
            receipt['minimumSampledAvailableRamBytes'] != 2_227_232_768 or
            protocol['scriptSha256'] != SECOND_SHA256 or 'sourceBindingsBefore' in protocol or
            (SECOND_UNIT / 'run').exists() or (SECOND_UNIT / 'actual_child_identity.json').exists()):
        raise ValueError('The second attempt must remain declined before model launch')
    return {path.relative_to(ROOT).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in [SECOND_ATTEMPT] + sorted(path for path in SECOND_UNIT.rglob('*') if path.is_file())}


def origin_record():
    if digest(ORIGIN) != ORIGIN_SHA256:
        raise ValueError('Original reviewed helper source changed')
    return {'recordedAt': stamp(), 'helper': ORIGIN.relative_to(ROOT).as_posix(),
            'helperSha256': ORIGIN_SHA256,
            'historicalDirectory': ORIGIN_UNIT.relative_to(ROOT).as_posix(),
            'historicalPreparationSha256': digest(ORIGIN_UNIT / 'preparation.json'),
            'historicalObserverRevisionSha256': digest(ORIGIN_UNIT / 'observer_revision.json'),
            'newHelper': Path(__file__).relative_to(ROOT).as_posix(),
            'reuse': 'Cloned genuine worker/input/normalization/reporting/owned-monitor checks. Fresh paths and initial source-hash gate; no fabricated observer revision.',
            'status': 'PREPARED_NOT_EXECUTED', 'inferenceRun': False,
            'firstLatencyAttemptBindings': verify_first_attempt(),
            'firstLatencyAttemptStatus': 'NOT_STARTED', 'firstLatencyAttemptInferenceChildStarted': False,
            'firstLatencyAttemptAvailableRamBytes': 2_355_306_496,
            'secondLatencyAttemptBindings': verify_second_attempt(),
            'secondLatencyAttemptStatus': 'NOT_STARTED',
            'secondLatencyAttemptAvailableRamBytes': 2_227_232_768,
            'retryPolicy': 'Explicit root-reviewed fresh launch only after other workloads finish and independently observed headroom recovers. Both unchanged 2.5 GB preflight checks run again; no automatic retry.'}


def origin_history(guard):
    return guard.snapshot([ORIGIN, FIRST_ATTEMPT] + [path for path in ORIGIN_UNIT.rglob('*') if path.is_file()] +
                          [path for path in FIRST_UNIT.rglob('*') if path.is_file()])


def live_product_tree(guard):
    directory = guard.PRODUCT_DB.parent.resolve()
    if not directory.is_relative_to(ROOT) or not directory.is_dir():
        raise ValueError('Expected live product tree within read-only workspace')
    paths = list(directory.rglob('*'))
    return {'directory': str(directory),
            'directories': sorted(path.relative_to(directory).as_posix() for path in paths if path.is_dir()),
            'files': guard.snapshot(path for path in paths if path.is_file()),
            'scope': 'Every live product file byte/hash, including SQLite sidecars, and directory names. Read-only; any change prevents a passing receipt.'}


def prepare():
    ast.parse(Path(__file__).read_text(encoding='utf-8'), filename=str(Path(__file__)))
    verify_first_attempt()
    UNIT.mkdir(parents=True, exist_ok=False)
    save(UNIT / 'origin.json', origin_record())
    save(UNIT / 'preparation.json', {
        'preparedAt': stamp(), 'status': 'PREPARED_NOT_EXECUTED', 'scope': SCOPE,
        'script': Path(__file__).relative_to(ROOT).as_posix(), 'scriptSha256': digest(__file__),
        'originReceiptSha256': digest(UNIT / 'origin.json'),
        'inferenceRun': False, 'workerInvoked': False, 'rootReviewRequiredBeforeRun': True,
        'launchCommand': 'C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe scripts\\audit\\verify_pose_racket_storage_worker_smoke_v2.py --run',
        'frames': 30, 'nativeFps': 60, 'inputSha256': EXCERPT_SHA256,
        'preflightAvailableRamBytes': PREFLIGHT_AVAILABLE_BYTES,
        'maximumSampledOwnedTreeRssBytes': MAXIMUM_OWNED_RSS_BYTES,
        'minimumAvailableRamBytes': MINIMUM_AVAILABLE_BYTES,
        'lowRamConsecutiveSamplesToStop': LOW_RAM_SAMPLES,
        'sampleIntervalSeconds': SAMPLE_SECONDS, 'maximumChildWallSeconds': MAXIMUM_WALL_SECONDS,
        'productionWrites': False, 'isolatedValidationJobPublication': True,
        'comparison': 'Frozen results.py and reporting.py use byte-identical new raw artifacts and the original claimed job; no volatile values removed; exact strict JSON/CSV/HTML bytes before presentation.',
        'normalizationDimensions': 'Record actual worker source/annotation dimensions, scale and padding. The bound worker currently normalizes fps, PTS and padding without a resize filter.',
        'network': 'YOLO_OFFLINE=true, YOLO_AUTOINSTALL=false; socket connect/connect_ex fail and are recorded.',
        'modelEvidence': 'Passive return observer records real Phase6Pipeline construction and parameter/device counts, then disables profiling before inference. Actual completed worker and raw artifacts establish run completion. No model or worker functions replaced.',
        'retention': 'One exclusive launch marker. Retain every partial artifact, log and receipt on failure; no retry or unrelated cleanup.',
        'qualificationEvidence': False, 'productionQualified': False,
    })
    print(json.dumps({'prepared': True, 'inferenceRun': False, 'directory': str(UNIT),
                      'scriptSha256': digest(__file__)}), flush=True)
    return 0


def equal_files(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            first, second = a.read(1024 * 1024), b.read(1024 * 1024)
            if first != second:
                return False
            if not first:
                return True


def write_chunks(path, chunks):
    with path.open('x', encoding='utf-8', newline='') as stream:
        for chunk in chunks:
            stream.write(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest(path)}


def model_observer(report, previous_profile):
    pipeline_file = str(ROOT / 'src/pipeline/phase6_pipeline.py')
    def observe(frame, event, _argument):
        if event != 'return' or frame.f_code.co_filename != pipeline_file:
            return
        if frame.f_code.co_name != '__init__':
            return
        pipeline = frame.f_locals.get('self')
        if pipeline is None or type(pipeline).__name__ != 'Phase6Pipeline':
            return
        try:
            # Retain only primitive audit records, never the pipeline/tensors.
            assets = {'court': pipeline.court_detector.model,
                      'personAndRacket': pipeline.player_detector.model.model,
                      'ball': pipeline.ball_detector.model,
                      'pose': pipeline.motion_tracker.model.model}
            counts = {name: {'parameters': sum(parameter.numel() for parameter in model.parameters()),
                             'devices': sorted({parameter.device.type for parameter in model.parameters()})}
                      for name, model in assets.items()}
            report.setdefault('actualPipelineObservations', []).append({'stage': 'construction', 'at': stamp(), 'loadedAssets': counts})
        except BaseException as error:
            # An exception-return observation must not replace the worker's
            # original model/producer error. Successful validation rejects it.
            report.setdefault('modelObservationErrors', []).append(type(error).__name__ + ': ' + str(error))
        finally:
            # Profiling covers only worker preparation/import/construction.
            # The actual model inference and artifact stages run unprofiled.
            sys.setprofile(previous_profile)
            report['constructionObserverStoppedBeforeInference'] = True
    return observe


def verify_videos(output, helper, input_report):
    import cv2
    result = {}
    for name in ('source.mp4', 'annotated.mp4', 'annotated_web.mp4'):
        path = output / name
        probe = helper.probe(path)
        stream, frames = probe['streams'][0], probe['frames']
        if len(frames) != 30 or Fraction(stream['avg_frame_rate']) != 60:
            raise ValueError('Worker playback frame count/FPS differs: ' + name)
        time_base = Fraction(stream['time_base'])
        pts = [int(frame['best_effort_timestamp']) * time_base for frame in frames]
        if pts != [Fraction(index, 60) for index in range(30)]:
            raise ValueError('Worker playback PTS clock differs: ' + name)
        capture = cv2.VideoCapture(str(path))
        hashes, dimensions = [], []
        try:
            if not capture.isOpened():
                raise ValueError('Cannot independently decode worker playback: ' + name)
            for index in range(30):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Worker playback decode failed: ' + name + ':' + str(index))
                dimensions.append([frame.shape[1], frame.shape[0]])
                hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
            if capture.read()[0]:
                raise ValueError('Worker playback contains more than 30 decoded frames: ' + name)
        finally:
            capture.release()
        expected = [int(stream['width']), int(stream['height'])]
        if any(shape != expected for shape in dimensions):
            raise ValueError('Decoded/probed dimensions differ: ' + name)
        result[name] = {'frames': 30, 'width': expected[0], 'height': expected[1],
                        'averageFrameRate': stream['avg_frame_rate'], 'timeBase': stream['time_base'],
                        'ptsSecondsExactRational': [str(point) for point in pts],
                        'decodedPixelSha256PerFrame': hashes, 'sha256': digest(path)}
    source_dimensions = [result['source.mp4']['width'], result['source.mp4']['height']]
    if any([row['width'], row['height']] != source_dimensions for row in result.values()):
        raise ValueError('Source/annotation dimensions do not align')
    media = strict_json(output / 'media_metadata.json')
    if ([media['original']['width'], media['original']['height']] != [input_report['width'], input_report['height']]
            or media['original']['fps'] != 60):
        raise ValueError('Worker original media metadata differs from verified input')
    normalization = media['normalization']
    for axis, key, padding in ((0, 'scaleX', 'paddingRightPx'), (1, 'scaleY', 'paddingBottomPx')):
        dimension = input_report['width'] if axis == 0 else input_report['height']
        if dimension * normalization[key] + normalization[padding] != source_dimensions[axis]:
            raise ValueError('Worker normalization scale/padding does not explain actual dimensions')
    mapping = media['timeMapping']
    records = []
    with (output / 'source_frame_mapping.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            records.append(json.loads(line))
    if (len(records) != 30 or mapping.get('frames') != 30 or
            mapping.get('status') != 'CAPTURED_FFMPEG_FRAME_MAP' or
            mapping.get('uniqueMappedSourceFrames') != 30 or mapping.get('duplicatedOutputFrames') != 0 or
            mapping.get('droppedSourceFrames') not in (0, None)):
        raise ValueError('Worker normalized/source mapping counts differ')
    source_probe = helper.probe(RUN / 'isolated_original.mkv')
    source_time_base = Fraction(source_probe['streams'][0]['time_base'])
    input_pts = [int(frame['best_effort_timestamp']) * source_time_base for frame in source_probe['frames']]
    for index, record in enumerate(records):
        normalized = int(record['normalized_pts']) * Fraction(record['normalized_time_base'])
        source = int(record['source_pts']) * Fraction(record['source_time_base'])
        if (record['frame_index'] != index or record['source_frame_index'] != index or
                normalized != Fraction(index, 60) or source != input_pts[index] or
                record['timestamp_seconds'] != float(normalized) or
                record['source_timestamp_seconds'] != float(source) or
                record['validMeasurementTransitionFromPrevious'] is not (index > 0)):
            raise ValueError('Worker source/normalized frame mapping differs at ' + str(index))
    if mapping.get('validMeasurementTransitionIntervals') != [{'startFrame': 1, 'endFrameExclusive': 30}]:
        raise ValueError('Worker source measurement transition coverage differs')
    save(UNIT / 'media_alignment.json', {'passed': True, 'inputOriginalPixelsPreserved': True,
         'workerNormalizedPixelsExpectedIdenticalToLosslessInput': False,
         'normalization': normalization, 'playback': result, 'timeMapping': mapping,
         'sourceFrameMappingFrames': len(records), 'originalSourceFrames': [390, 420],
         'semantics': 'Input pixels/quantized native PTS are proved against original media; normalized lossy pixels are independently decoded, not equated with historical inference.'})
    return result


def validate_semantics(value):
    capabilities = value['summary']['capabilities']
    if value['summary']['frames'] != 30 or value['summary']['fps'] != 60:
        raise ValueError('Published result frame clock differs')
    if (value['world3d'] is not None or value['ballMetrics'] is not None or
            capabilities['world3d'] or capabilities['ballMetrics'] or capabilities['events'] or
            capabilities['lineCalls'] or capabilities['scoring'] or value['events'] or value['lineCalls']):
        raise ValueError('Unqualified physical/event capabilities were exposed')
    if (value['court'] is not None or capabilities['court'] or
            any(frame['court_registration_valid'] for frame in value['detections']['frames']) or
            'court' not in value['unavailableSections']):
        raise ValueError('This predeclared smoke input did not retain unavailable court semantics')
    if (value['pipelineMetrics']['raw']['production_qualified'] is not False or
            value['pipelineMetrics']['raw']['authoritative_events_enabled'] is not False or
            value['provenance']['productionQualified'] is not False):
        raise ValueError('Production/event authority changed')
    for index, point in enumerate(value['trajectories']['ballTrajectory']):
        if (point['frame_index'] != index or point['speed_kmh'] is not None or
                point['world_position_m'] is not None or point['court_x_m'] is not None or point['court_y_m'] is not None):
            raise ValueError('Physical ball/missing court values were not withheld')
    counts = {'detections': len(value['detections']['frames']),
              'ballTrajectory': len(value['trajectories']['ballTrajectory']),
              'cameraFrames': len(value['cameraTracking']['frames'])}
    for key in ('player_1', 'player_2'):
        counts[key + 'Motion'] = len(value['playerTracking'][key]['samples'])
    for key in ('player1CourtPositions', 'player2CourtPositions', 'player1Speeds', 'player2Speeds'):
        counts[key] = len(value['trajectories'][key])
        if any(point is not None for point in value['trajectories'][key]):
            raise ValueError('Unavailable court player position/speed was not withheld')
    for name in ('poseTracking', 'racketTracking'):
        if value.get(name) is None or len(value[name]['frames']) != 30:
            raise ValueError('Enabled pose/racket section is missing or incomplete: ' + name)
    for name in ('imagePeople', 'poseTracking', 'racketTracking'):
        if value[name] is not None:
            counts[name] = len(value[name]['frames'])
    if any(count != 30 for count in counts.values()):
        raise ValueError('Published normalized observation sections do not cover 30 frames')
    return counts


def verify_raw_pose_racket(output):
    pose = strict_json(output / 'player_motion.json')
    racket = strict_json(output / 'racket_tracking.json')
    if pose.get('pose_enabled') is not True or racket.get('enabled') is not True:
        raise ValueError('This selected recipe must export both enabled observation sections')
    checked = {}
    for name, values in (('pose_frames', pose.get('pose_frames')), ('racket_frames', racket.get('frames'))):
        if not isinstance(values, list) or len(values) != 30 or not all(isinstance(frame, dict) for frame in values):
            raise ValueError('Enabled raw pose/racket frame dictionaries missing or incomplete: ' + name)
        checked[name] = len(values)
    return {'enabled': True, 'rawFrameCounts': checked,
            'positiveObservationCoverageRequired': False,
            'scope': 'Complete enabled raw series; missing selected roles and null measurements remain legitimate.'}


def verify_saved_result(output, claimed_job, source_credit):
    from src.api.result_document import ResultDocument
    from src.api.result_storage import DiskSeries
    from src.api import reporting
    old_results = load_module('result_worker_frozen_results', BASELINE / 'src/api/results.py')
    old_reporting = load_module('result_worker_frozen_reporting', BASELINE / 'src/api/reporting.py')
    raw_pose_racket = verify_raw_pose_racket(output)
    raw = RUN / 'raw_artifact_view'
    raw.mkdir(exist_ok=False)
    copied = {}
    # Legacy build_result inventories top-level files before publication. Only
    # those two generated exports are excluded; every raw byte is untouched.
    for source in sorted(output.iterdir()):
        if not source.is_file() or source.name in ('result.json', 'summary.json'):
            continue
        destination = raw / source.name
        shutil.copyfile(source, destination)
        if not equal_files(source, destination):
            raise ValueError('Raw artifact copy differs: ' + source.name)
        copied[source.name] = {'bytes': source.stat().st_size, 'sha256': digest(source)}
    save(UNIT / 'golden_raw_artifact_bindings.json', {'copiedBytesExactly': True, 'artifacts': copied,
         'excludedPublicationFiles': ['result.json', 'summary.json'],
         'claimedJobUsedUnchanged': claimed_job,
         'volatileRuntimeFieldsRemoved': [], 'presentationApplied': False})
    golden = old_results.build_result(raw, claimed_job)
    check_finite(golden)
    golden_json = RUN / 'golden_result.json'
    write_chunks(golden_json, [json.dumps(golden, allow_nan=False)])
    marker = strict_json(output / '.result/current.store')
    database = output / '.result' / marker['database']
    if database.stat().st_size != marker['databaseBytes'] or digest(database) != marker['databaseSha256']:
        raise ValueError('Committed generation does not match manifest')
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        if connection.execute('PRAGMA integrity_check').fetchone() != ('ok',):
            raise ValueError('Committed result database integrity check failed')
    finally:
        connection.close()
    comparisons = {}
    with ResultDocument(output, claimed_job['id']) as document:
        counts = validate_semantics(document.value)
        if not isinstance(document.value['detections']['frames'], DiskSeries):
            raise ValueError('Worker normalized detections are not reopened lazily')
        if document.value['mediaSource'] != source_credit:
            raise ValueError('Licensed source attribution was lost in saved result')
        lazy_json = RUN / 'reopened_result.json'
        write_chunks(lazy_json, document.json_chunks())
        if not equal_files(golden_json, lazy_json) or not equal_files(golden_json, output / 'result.json'):
            difference = retained_guard().first_difference(strict_json(golden_json), strict_json(lazy_json))
            raise ValueError('Exact golden/reopened/published JSON bytes differ: ' + str(difference))
        comparisons['json'] = {'exactBytes': True, 'parsedValuesExact': strict_json(golden_json) == strict_json(lazy_json),
                               'bytes': lazy_json.stat().st_size, 'sha256': digest(lazy_json)}
        for name, suffix, current, legacy in (
                ('csv', '.csv', reporting.telemetry_csv_chunks(document.value),
                 old_reporting.telemetry_csv_chunks(golden)),
                ('html', '.html', reporting.readable_report_chunks(document.value, output),
                 [old_reporting.readable_report(golden, output)])):
            expected, actual = RUN / ('golden_report' + suffix), RUN / ('reopened_report' + suffix)
            left, right = write_chunks(expected, legacy), write_chunks(actual, current)
            if not equal_files(expected, actual):
                raise ValueError('Exact frozen/current ' + name.upper() + ' report bytes differ before presentation')
            comparisons[name] = {'exactBytes': True, 'bytes': right['bytes'], 'sha256': right['sha256']}
        root_summary = document.value['summary']
        if strict_json(output / 'summary.json') != root_summary:
            raise ValueError('Compatibility summary differs from committed result')
    if not document.closed or document.storage._connection is not None:
        raise ValueError('Owned lazy result reader did not close')
    # A complete publication has one closed immutable generation, no scratch
    # database, staged exports or SQLite journal/WAL/shared-memory sidecars.
    generation_files = sorted(path.name for path in (output / '.result').iterdir() if path.is_file())
    if generation_files != sorted([database.name, 'current.store']):
        raise ValueError('Successful first publication left result scratch/staging files: ' + repr(generation_files))
    if any((output / name).exists() for name in ('result.tmp', 'summary.tmp')):
        raise ValueError('Successful worker retained staged result exports')
    spool = [path.relative_to(output).as_posix() for path in output.rglob('*')
             if path.is_file() and path.name.startswith(('ball_candidates_', 'frame_registration_', 'identity_frames_'))]
    if spool:
        raise ValueError('Successful worker retained candidate/registration scratch: ' + repr(spool))
    save(UNIT / 'saved_result_comparison.json', {'passed': True, 'scope': SCOPE,
         'frozenModuleBindings': {name: {'bytes': (BASELINE / name).stat().st_size, 'sha256': digest(BASELINE / name)}
                                  for name in ('src/api/results.py', 'src/api/reporting.py')},
         'comparisons': comparisons, 'normalizedSectionCounts': counts,
         'enabledRawPoseRacket': raw_pose_racket,
         'manifest': marker, 'databaseIntegrity': 'ok', 'lazyDiskSeriesReader': True,
         'closedOwnedReaderConnection': True, 'generationFiles': generation_files,
         'unpublishedScratchFiles': spool, 'presentationApplied': False,
         'comparisonUsesSameNewRawBytes': True, 'volatileRuntimeFieldsRemoved': [],
         'qualificationEvidence': False, 'productionQualified': False})
    return comparisons


def run_child(token):
    import psutil
    actual = psutil.Process()
    identity = {'pid': actual.pid, 'createTimeUnixSeconds': actual.create_time()}
    protocol = strict_json(UNIT / 'protocol.json')
    if not token or token != protocol['launchToken'] or not RUN.is_dir() or ISOLATED.exists():
        raise ValueError('Child requires one fresh authorized parent launch')
    save(UNIT / 'actual_child_identity.json', identity)
    # The real clean interpreter (not only its redirector) must be sampled
    # before models or video work begin.
    deadline = time.perf_counter() + 15
    while not (UNIT / 'actual_child_sampled.json').exists():
        if time.perf_counter() >= deadline:
            raise RuntimeError('Parent did not acknowledge actual child identity')
        time.sleep(.02)
    if strict_json(UNIT / 'actual_child_sampled.json')['identity'] != identity:
        raise ValueError('Parent acknowledged a different workload identity')
    report = {'startedAt': stamp(), 'passed': False, 'scope': SCOPE,
              'actualWorkloadIdentity': identity, 'interpreter': sys.executable,
              'isolatedEnvironment': sys.prefix != sys.base_prefix,
              'workerInvoked': False, 'inferenceRun': False, 'networkAttempts': [],
              'timingScope': 'Worker timing includes preparation/import/model-construction passive observer overhead; profiling is disabled before model inference. This is no performance comparison.',
              'productionStatePublished': False, 'isolatedValidationJobPublished': False,
              'qualificationEvidence': False, 'productionQualified': False}
    began = time.perf_counter()
    previous_profile = sys.getprofile()
    previous_connect, previous_connect_ex = socket.socket.connect, socket.socket.connect_ex
    def no_network(_socket, address):
        report['networkAttempts'].append(repr(address))
        raise RuntimeError('Offline real worker smoke attempted a network connection')
    socket.socket.connect = socket.socket.connect_ex = no_network
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_CONFIG_DIR'] = str(RUN / 'ultralytics_settings')
    os.environ['MPLCONFIGDIR'] = str(RUN / 'matplotlib_settings')
    os.environ['TENNIS_STORAGE_ROOT'] = str(ISOLATED)
    os.environ['TENNIS_PIPELINE_CONFIG'] = str(CONFIG)
    (RUN / 'ultralytics_settings').mkdir(exist_ok=False)
    (RUN / 'matplotlib_settings').mkdir(exist_ok=False)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    try:
        if not report['isolatedEnvironment']:
            raise RuntimeError('Actual worker child is not the clean virtual environment')
        guard = retained_guard()
        if binding_snapshot(guard) != protocol['sourceBindingsBefore']:
            raise ValueError('Source/model/media bindings changed before child work')
        helper = guard.load_helper()
        input_report = guard.verified_input(helper)
        shutil.copyfile(guard.EXCERPT, RUN / 'isolated_original.mkv')
        if digest(RUN / 'isolated_original.mkv') != EXCERPT_SHA256:
            raise ValueError('Owned original copy differs from retained input')
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        from src.api.jobs import JobStore, Settings
        from src.api.media_source import validate_media_source
        from src.api.worker import process_job
        store = JobStore(ISOLATED)
        settings = Settings(ISOLATED, CONFIG)
        analysis_id = store.new_id()
        upload = store.upload_dir(analysis_id)
        upload.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(RUN / 'isolated_original.mkv', upload / 'original.mkv')
        output = store.output_dir(analysis_id)
        output.mkdir(parents=True, exist_ok=False)
        credit = dict(guard.source_record())
        credit.update({'id': 'dimitrov_tiafoe_2024_worker_smoke', 'inputSha256': EXCERPT_SHA256,
                       'modifications': 'Original source frames 390-419 retained as a lossless half-second native-60-FPS excerpt; audio omitted. The isolated worker normalizes playback and adds analysis overlays. Engineering smoke only.'})
        credit['excerpt'] = dict(credit['excerpt'], endSeconds=7.0, endFrameExclusive=420,
                                 lastSourcePts=str(int(Fraction(input_report['originalPtsSeconds'][-1]) / Fraction(credit['excerpt']['sourceTimeBase']))))
        credit = validate_media_source(credit)
        if credit is None:
            raise ValueError('Licensed half-second attribution is invalid')
        save(output / 'media_source.json', credit)
        store.create(analysis_id, 'Licensed original frames 390-419.mkv', '.mkv',
                     (upload / 'original.mkv').stat().st_size, CONFIG)
        claimed = store.claim()
        if claimed is None or claimed['id'] != analysis_id or claimed['status'] != 'PROCESSING':
            raise ValueError('Isolated worker claim failed')
        save(UNIT / 'isolated_claimed_job.json', claimed)
        report['analysisId'] = analysis_id
        report['outputDirectory'] = output.relative_to(ROOT).as_posix()
        report['workerInvoked'] = report['inferenceRun'] = True
        worker_start = time.perf_counter()
        sys.setprofile(model_observer(report, previous_profile))
        try:
            process_job(store, claimed, settings)
        finally:
            sys.setprofile(previous_profile)
        report['actualWorkerElapsedSeconds'] = time.perf_counter() - worker_start
        completed = store.get(analysis_id)
        save(UNIT / 'isolated_completed_job.json', completed)
        report['completedJob'] = completed
        if (completed['status'] != 'COMPLETED' or completed['step'] != 11 or
                completed['error'] is not None or not completed['completed_at']):
            raise ValueError('Actual isolated worker did not complete; inspect processing.log')
        observations = report.get('actualPipelineObservations', [])
        if (report.get('modelObservationErrors') or not report.get('constructionObserverStoppedBeforeInference') or
                [row['stage'] for row in observations] != ['construction'] or
                any(asset['parameters'] <= 0 for row in observations for asset in row['loadedAssets'].values())):
            raise ValueError('Whole actual model construction was not observed before unprofiled inference')
        report['actualPipelineRunCompletedByWorker'] = True
        report['isolatedValidationJobPublished'] = True
        report['playback'] = verify_videos(output, helper, input_report)
        report['comparison'] = verify_saved_result(output, claimed, credit)
        report['retainedOriginalUploadUnchanged'] = digest(upload / 'original.mkv') == EXCERPT_SHA256
        if report['networkAttempts'] or not report['retainedOriginalUploadUnchanged']:
            raise ValueError('Offline worker/input-preservation invariant failed')
        if binding_snapshot(guard) != protocol['sourceBindingsBefore']:
            raise ValueError('Source/model/media binding changed during actual worker')
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        sys.setprofile(previous_profile)
        socket.socket.connect, socket.socket.connect_ex = previous_connect, previous_connect_ex
        report['elapsedSeconds'] = time.perf_counter() - began
        memory = actual.memory_info()
        report['actualProcessMemoryBytes'] = memory._asdict()
        report['windowsPeakWorkingSetBytes'] = getattr(memory, 'peak_wset', None)
        report['windowsPeakPagefileBytes'] = getattr(memory, 'peak_pagefile', None)
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_available():
            report['cudaAllocatorPeakBytes'] = {'allocated': torch.cuda.max_memory_allocated(),
                'reserved': torch.cuda.max_memory_reserved(),
                'scope': 'PyTorch process allocator; excludes driver/other-process memory.'}
        report['completedAt'] = stamp()
        save(UNIT / 'child_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'workerInvoked', 'analysisId', 'error')}), flush=True)


def run_parent():
    import psutil
    preparation = strict_json(UNIT / 'preparation.json')
    if (preparation['status'] != 'PREPARED_NOT_EXECUTED' or
            preparation['scriptSha256'] != digest(__file__) or
            preparation['originReceiptSha256'] != digest(UNIT / 'origin.json') or
            strict_json(UNIT / 'origin.json')['helperSha256'] != ORIGIN_SHA256 or
            digest(ORIGIN) != ORIGIN_SHA256):
        raise ValueError('Prepared source or preserved origin binding changed before launch')
    if verify_first_attempt() != strict_json(UNIT / 'origin.json')['firstLatencyAttemptBindings']:
        raise ValueError('The first declined attempt history changed before this conditional launch')
    if verify_second_attempt() != strict_json(UNIT / 'origin.json')['secondLatencyAttemptBindings']:
        raise ValueError('The second declined attempt history changed before launch')
    if os.name != 'nt' or not CLEAN.is_file() or sys.prefix != sys.base_prefix:
        raise ValueError('Launch with base Windows Python and the installed clean interpreter')
    if not UNIT.resolve().is_relative_to((ROOT / 'artifacts/validation/product_execution').resolve()):
        raise ValueError('Owned output directory escaped product_execution')
    # The marker is exclusively created before preflight. Even a stopped or
    # failed launch consumes this one-shot directory and cannot be retried.
    token = uuid.uuid4().hex
    save(UNIT / 'launch_marker.json', {'declaredAt': stamp(), 'launchToken': token, 'oneShot': True})
    guard = retained_guard()
    monitor = load_module('result_worker_owned_process_monitor', MONITOR)
    protocol = {'declaredAt': stamp(), 'scope': SCOPE, 'launchToken': token,
        'scriptSha256': digest(__file__), 'originReceipt': strict_json(UNIT / 'origin.json'),
        'conditionalFreshLaunch': 'Manually reviewed after other workloads complete and headroom recovers; unchanged preflight rechecks. First consumed attempt is retained. No automatic retry.',
        'cleanInterpreter': str(CLEAN),
        'readOnlyLiveProductDatabase': str(guard.PRODUCT_DB), 'isolatedProductRoot': str(ISOLATED),
        'frames': 30, 'nativeFps': 60, 'sourceOriginalFrameRange': [390, 420],
        'sampleIntervalSeconds': SAMPLE_SECONDS, 'preflightAvailableRamBytes': PREFLIGHT_AVAILABLE_BYTES,
        'maximumSampledOwnedTreeRssBytes': MAXIMUM_OWNED_RSS_BYTES,
        'minimumAvailableRamBytes': MINIMUM_AVAILABLE_BYTES, 'lowRamConsecutiveSamplesToStop': LOW_RAM_SAMPLES,
        'maximumChildWallSeconds': MAXIMUM_WALL_SECONDS,
        'ownedTreeMonitor': str(MONITOR.relative_to(ROOT)) + '::OwnedProcessTree and stop_owned',
        'ownedTreeMonitorSha256': digest(MONITOR), 'productionWrites': False,
        'qualificationEvidence': False, 'productionQualified': False,
        'retention': 'Retain all owned logs, partial outputs and receipts. No retry or unrelated cleanup.',
        'host': {'hostname': platform.node(), 'platform': platform.platform(), 'processor': platform.processor(),
                 'baseInterpreter': sys.executable, 'python': sys.version, 'psutilVersion': psutil.__version__,
                 'ramBeforeBytes': psutil.virtual_memory()._asdict(), 'diskBeforeBytes': shutil.disk_usage(ROOT)._asdict()}}
    report = {'startedAt': stamp(), 'passed': False, 'status': 'NOT_STARTED', 'scope': SCOPE,
        'inferenceChildStarted': False, 'productionStatePublished': False, 'samples': 0,
        'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
        'minimumSampledAvailableRamBytes': psutil.virtual_memory().available,
        'qualificationEvidence': False, 'productionQualified': False}
    process = owned = None
    began = None
    try:
        if psutil.virtual_memory().available < PREFLIGHT_AVAILABLE_BYTES:
            report['notStartedReason'] = 'Available RAM below 2.5 GB before bindings; no launch or retry.'
            save(UNIT / 'protocol.json', protocol)
            return 2
        protocol['sourceBindingsBefore'] = binding_snapshot(guard)
        protocol['historicalArtifactsBefore'] = guard.history_snapshot()
        protocol['productionJobStatesBefore'] = guard.product_jobs()
        protocol['liveProductFileTreeBefore'] = live_product_tree(guard)
        protocol['originHistoricalArtifactsBefore'] = origin_history(guard)
        protocol['firstLatencyAttemptHistoryBefore'] = verify_first_attempt()
        protocol['secondLatencyAttemptHistoryBefore'] = verify_second_attempt()
        if len(protocol['productionJobStatesBefore']['jobs']) != 10:
            raise ValueError('Expected exactly ten live read-only job states before this isolated smoke')
        report['availableRamImmediatelyBeforeLaunchBytes'] = psutil.virtual_memory().available
        save(UNIT / 'protocol.json', protocol)
        if report['availableRamImmediatelyBeforeLaunchBytes'] < PREFLIGHT_AVAILABLE_BYTES:
            report['notStartedReason'] = 'Available RAM fell below 2.5 GB during bindings; no launch or retry.'
            return 2
        RUN.mkdir(exist_ok=False)
        argv = [str(CLEAN), str(Path(__file__).resolve()), '--child', '--launch-token', token]
        began = time.perf_counter()
        with (UNIT / 'child_stdout.txt').open('x', encoding='utf-8') as log, \
                (UNIT / 'resource_samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            report['inferenceChildStarted'] = True
            report['status'] = 'RUNNING'
            save(UNIT / 'running_process.json', {'pid': process.pid,
                'createTimeUnixSeconds': owned.root_identity[1], 'startedAt': stamp(), 'command': argv})
            low_ram = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                private = [row['memoryBytes'].get('private') for row in rows]
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - began, 'processes': rows,
                    'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                    'ownedTreePrivateBytes': sum(value for value in private if value is not None),
                    'allPrivateValuesAvailable': all(value is not None for value in private),
                    'availableRamBytes': available, 'freeDiskBytes': shutil.disk_usage(ROOT).free}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                report['samples'] += 1
                report['peakSampledOwnedTreeRssBytes'] = max(report['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                report['peakSampledOwnedTreePrivateBytes'] = max(report['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                report['minimumSampledAvailableRamBytes'] = min(report['minimumSampledAvailableRamBytes'], available)
                identity_path = UNIT / 'actual_child_identity.json'
                if identity_path.is_file() and not (UNIT / 'actual_child_sampled.json').exists():
                    identity = strict_json(identity_path)
                    if owned.observed_workload(identity):
                        save(UNIT / 'actual_child_sampled.json', {'at': stamp(), 'identity': identity,
                                                               'sampledByExactIdentity': True})
                low_ram = low_ram + 1 if available < MINIMUM_AVAILABLE_BYTES else 0
                reason = None
                if sample['ownedTreeRssBytes'] > MAXIMUM_OWNED_RSS_BYTES:
                    reason = 'Aggregate exact-owned-tree RSS exceeded 3 GB'
                elif low_ram >= LOW_RAM_SAMPLES:
                    reason = 'Host available RAM below 400 MB for five consecutive samples'
                elif sample['elapsedSeconds'] > MAXIMUM_WALL_SECONDS:
                    reason = 'Owned child wall time exceeded 300 seconds'
                if reason:
                    report['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(SAMPLE_SECONDS)
            report['returncode'] = process.returncode
        if process.returncode != 0:
            raise RuntimeError('Actual owned worker smoke child failed; partial artifacts retained')
        child = strict_json(UNIT / 'child_receipt.json')
        if not child['passed'] or not child['isolatedValidationJobPublished']:
            raise ValueError('Actual worker/result compatibility verification did not pass')
        report['actualWorkloadIdentity'] = child['actualWorkloadIdentity']
        report['actualWorkloadSampledByExactIdentity'] = owned.observed_workload(child['actualWorkloadIdentity'])
        if not report['actualWorkloadSampledByExactIdentity']:
            raise ValueError('Actual clean interpreter was not sampled; redirector-only evidence is invalid')
        report['remainingOwnedIdentitiesAfterNormalExit'] = owned.wait_for_retirement()
        if report['remainingOwnedIdentitiesAfterNormalExit']:
            raise RuntimeError('Exact owned descendants remained after normal worker smoke exit')
        report['allOwnedIdentitiesRetiredAfterNormalExit'] = True
        report['status'] = 'COMPLETED'
        report['passed'] = True
        return 0
    except BaseException as error:
        report['passed'] = False
        report['status'] = 'STOPPED' if report.get('safeguardStop') else 'FAILED'
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        if not (UNIT / 'protocol.json').exists():
            save(UNIT / 'protocol.json', protocol)
        if process is not None:
            try:
                report['terminationActions'] = monitor.stop_owned(process, owned)
            except BaseException as cleanup_error:
                report['ownedProcessCleanupError'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
            report['returncode'] = process.returncode
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
            except BaseException as cleanup_error:
                report['retirementVerificationError'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
                report['passed'] = False
                report['status'] = 'FAILED'
        if (UNIT / 'child_receipt.json').is_file():
            try:
                child = strict_json(UNIT / 'child_receipt.json')
                report['actualWorkloadIdentity'] = child['actualWorkloadIdentity']
                report['actualWorkloadSampledByExactIdentity'] = owned is not None and owned.observed_workload(child['actualWorkloadIdentity'])
                for field in ('windowsPeakWorkingSetBytes', 'windowsPeakPagefileBytes', 'actualProcessMemoryBytes', 'cudaAllocatorPeakBytes'):
                    report['actualWorkload_' + field] = child.get(field)
                report['isolatedValidationJobPublished'] = child['isolatedValidationJobPublished']
            except BaseException as receipt_error:
                report['childReceiptVerificationError'] = type(receipt_error).__name__ + ': ' + str(receipt_error)
                report['passed'] = False
                report['status'] = 'FAILED'
        for label, before_key, reader in (
                ('sourceBindings', 'sourceBindingsBefore', lambda: binding_snapshot(guard)),
                ('historicalArtifacts', 'historicalArtifactsBefore', guard.history_snapshot),
                ('productionJobStates', 'productionJobStatesBefore', guard.product_jobs),
                ('liveProductFileTree', 'liveProductFileTreeBefore', lambda: live_product_tree(guard)),
                ('originHistoricalArtifacts', 'originHistoricalArtifactsBefore', lambda: origin_history(guard)),
                ('firstLatencyAttemptHistory', 'firstLatencyAttemptHistoryBefore', verify_first_attempt),
                ('secondLatencyAttemptHistory', 'secondLatencyAttemptHistoryBefore', verify_second_attempt)):
            if before_key not in protocol:
                continue
            try:
                after = reader()
                report[label + 'After'] = after
                report[label + 'Unchanged'] = after == protocol[before_key]
                if not report[label + 'Unchanged']:
                    report['passed'] = False
                    report['status'] = 'FAILED'
            except BaseException as binding_error:
                report[label + 'VerificationError'] = type(binding_error).__name__ + ': ' + str(binding_error)
                report['passed'] = False
                report['status'] = 'FAILED'
        report['completedAt'] = stamp()
        report['monitoredChildWallSeconds'] = time.perf_counter() - began if began is not None else None
        report['diskAfterBytes'] = shutil.disk_usage(ROOT)._asdict()
        report['retainedOwnedFilesLogicalBytes'] = sum(path.stat().st_size for path in UNIT.rglob('*') if path.is_file())
        report['memoryScope'] = ('Sampled aggregate owned RSS can double-count shared pages and miss short peaks. '
            'Actual interpreter OS peak working set and process-commit/pagefile peaks are separately reported. '
            'Host low RAM includes unrelated processes and does not prove a worker leak. CUDA allocator is separate.')
        save(UNIT / 'resource_receipt.json', report)
        print(json.dumps({key: report.get(key) for key in ('passed', 'status', 'inferenceChildStarted', 'samples', 'error')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--launch-token', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare:
        return prepare()
    if args.child:
        run_child(args.launch_token)
        return 0
    code = run_parent()
    if code == 0 and not strict_json(UNIT / 'resource_receipt.json')['passed']:
        return 1
    return code


if __name__ == '__main__':
    raise SystemExit(main())

