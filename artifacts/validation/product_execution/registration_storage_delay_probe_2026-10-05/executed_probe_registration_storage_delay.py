"""Bounded read-only timing probe against the retained interrupted database."""
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UNIT = ROOT / 'artifacts/validation/product_execution/registration_storage_delay_probe_2026-10-05'
DATA = ROOT / 'artifacts/validation/product_execution/registration_storage_measurement_2026-10-05/dynamic_432000_new'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def load_monitor():
    spec = importlib.util.spec_from_file_location('delay_probe_monitor', ROOT / 'scripts/audit/measure_candidate_storage.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def child():
    import cv2
    import numpy as np
    import psutil
    from src.utils.frame_registration_sequence import _decode_record
    database, = DATA.glob('frame_registration_*.sqlite3')
    points = np.arange(28, dtype=np.float32).reshape(1, 14, 2)

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

    stages = []
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA cache_size=-2048')
        count, minimum, maximum = connection.execute('SELECT COUNT(*),MIN(frame_index),MAX(frame_index) FROM registration_frames').fetchone()
        query = 'SELECT metadata,image_matrix,anchor_transform FROM registration_frames WHERE frame_index=?'
        plan = connection.execute('EXPLAIN QUERY PLAN ' + query, (0,)).fetchall()
        for start in (0, count - 1000):
            begin = time.perf_counter()
            records = []
            for index in range(start, start + 1000):
                records.append(_decode_record(connection.execute(query, (index,)).fetchone()))
            read_seconds = time.perf_counter() - begin
            profiler = cProfile.Profile()
            begin = time.perf_counter()
            profiler.enable()
            checksum = hashlib.sha256()
            for record in records:
                checksum.update(encoded(record))
            profiler.disable()
            encode_seconds = time.perf_counter() - begin
            stats = pstats.Stats(profiler).stats
            top = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:12]
            stages.append({'startFrame': start, 'frames': 1000, 'readDecodeSeconds': read_seconds,
                           'profiledGeometryEncodingSeconds': encode_seconds, 'sha256': checksum.hexdigest(),
                           'topCumulative': [{'function': key[2], 'file': Path(key[0]).name, 'line': key[1],
                                              'calls': value[1], 'totalSeconds': value[2], 'cumulativeSeconds': value[3]}
                                             for key, value in top]})
            del records
    finally:
        connection.close()
    actual = psutil.Process()
    result = {'recordedAt': datetime.now(timezone.utc).isoformat(), 'actualWorkloadIdentity': {'pid': actual.pid, 'createTimeUnixSeconds': actual.create_time()},
              'database': database.relative_to(ROOT).as_posix(), 'rows': count, 'minimumFrame': minimum, 'maximumFrame': maximum,
              'queryPlan': plan, 'stages': stages, 'threads': actual.num_threads(),
              'pythonVersion': sys.version, 'libraries': {'numpy': np.__version__, 'opencv': cv2.__version__},
              'peakWorkingSetBytes': actual.memory_info().peak_wset,
              'fastLoop': all(row['readDecodeSeconds'] + row['profiledGeometryEncodingSeconds'] < .3 for row in stages),
              'diagnosticThresholdSecondsPer1000Records': .3, 'inferenceRun': False, 'readOnlyDatabase': True,
              'scope': '1000-record first/last windows only; diagnostic threshold, not product throughput or qualification.'}
    save(UNIT / 'child_receipt.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['fastLoop'] else 1


def parent():
    UNIT.mkdir(parents=True, exist_ok=False)
    import psutil
    monitor = load_monitor()
    process = owned = None
    receipt = {'startedAt': datetime.now(timezone.utc).isoformat(), 'inferenceRun': False}
    try:
        with (UNIT / 'stdout.txt').open('x', encoding='utf-8') as log:
            process = subprocess.Popen([str(CLEAN), str(Path(__file__)), '--child'], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            begin = time.monotonic()
            while process.poll() is None:
                rows = owned.sample()
                if sum(row['memoryBytes']['rss'] for row in rows) > 1000000000 or time.monotonic() - begin > 30:
                    raise RuntimeError('Probe exceeded predeclared RAM/wall guard')
                time.sleep(.05)
        result = json.loads((UNIT / 'child_receipt.json').read_text(encoding='utf-8'))
        receipt['exitCode'] = process.returncode
        receipt['actualWorkloadWasSampled'] = owned.observed_workload(result['actualWorkloadIdentity'])
        receipt['remainingOwnedIdentities'] = owned.wait_for_retirement()
        assert receipt['actualWorkloadWasSampled'] and not receipt['remainingOwnedIdentities']
        receipt['probeExecuted'] = True
        receipt['fastLoop'] = result['fastLoop']
        print(json.dumps({'probeExecuted': True, 'fastLoop': result['fastLoop'], 'rows': result['rows'], 'stages': result['stages'], 'threads': result['threads']}))
        return process.returncode
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        if process is not None:
            receipt['terminationActions'] = monitor.stop_owned(process, owned)
        raise
    finally:
        receipt['completedAt'] = datetime.now(timezone.utc).isoformat()
        if owned is not None:
            receipt['ownedIdentities'] = owned.identities()
        save(UNIT / 'resource_receipt.json', receipt)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    raise SystemExit(child() if '--child' in sys.argv else parent())
