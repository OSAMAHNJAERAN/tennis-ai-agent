"""One-shot synthetic preoptimization-disk/current-disk latency comparison.

Preparation does not authorize a run. Launch only after the root reviewer has
frozen sources and completed other workloads. No footage or model is used.
"""
import argparse
from datetime import datetime, timezone
import gc
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06'
MONITOR = 'scripts/audit/measure_candidate_storage.py'
MONITOR_SHA256 = '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'
FROZEN = {'src/api/result_storage.py': {'bytes': 22807, 'sha256': '6c553e36e55bbf54888819162b6207297686fd050cd62e085821d04fedee5fc2'}, 'src/api/result_document.py': {'bytes': 8158, 'sha256': 'd30e7abe74d5228bae6bfea14ec9ae07bca317d5032cd3aeda6bf92f811d1cdc'}, 'src/api/results.py': {'bytes': 35337, 'sha256': '30a64418e141d2ab5d847c33aa6fcc0386807ff7a04a5891b5172a904f9c1a55'}, 'src/api/reporting.py': {'bytes': 17376, 'sha256': '7854e4952e840d43e495b36cfced57b69a3f38107b1a2849e53dd99705880f17'}, 'src/api/app.py': {'bytes': 11742, 'sha256': '0639b614f8f1ed3b00ee9abd3aee6fa6eab7498f5b7e1c27232d2348fb747253'}, 'src/api/worker.py': {'bytes': 11967, 'sha256': '02e9e14ce2a8857bd65fdfdfa8f8b3d43baaf2612521d0659bec1258b490aedb'}, 'src/api/product.py': {'bytes': 16609, 'sha256': '0e3c03dc07987b3d0f40f4dd0c1743bf8b168276be7573846a4eaef09bcadacb'}, 'src/api/demos.py': {'bytes': 3370, 'sha256': '88fa345e184e2b80c179691d16aa53a4a4a0ec042c6ce18a4d4e0d1076d39b04'}, 'tests/test_result_storage.py': {'bytes': 19198, 'sha256': '1817f7cda77b8ef7a5c829c4bff5fad62aa0308c196b33be13bcb1c3757ac6c9'}, 'tests/test_result_document.py': {'bytes': 16001, 'sha256': '6e1bcc9db483d5e05bbf5f85db070d8f1d9a5a9b11e8a85f91f8bba5aa929b2e'}, 'tests/test_bounded_result_contract.py': {'bytes': 21805, 'sha256': '28268bdd9aaf3a9fc4f8c3015f5c5ffadcb89699406629b89305fff92179dba9'}, 'tests/test_bounded_reporting.py': {'bytes': 15308, 'sha256': 'b9f00f841bc72dc7dbcbfa32e45ee553050a01b3e6ff4c16ddc3319926247447'}, 'tests/test_worker_result_publication.py': {'bytes': 12747, 'sha256': '581845e11ecfadbd6e415d09bf0055b153e8eddb321baea8978e2eab3b991f2a'}, 'docs/plans/TENNIS_EXECUTION_PROGRESS.md': {'bytes': 84083, 'sha256': '0215a5dedf9777001aa8323b0ceeae50a7916430a1213d0c5b0e05667e09b4fc'}, 'docs/deployment/PRODUCT_INSTALLATION.md': {'bytes': 23585, 'sha256': 'b0fda5dc0ed56a24ec57376a173bb470b6e79e35b0bc66c542d1d786098ce7b2'}, 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json': {'bytes': 76355, 'sha256': 'e05c7933689fd305dd910f20ed6cbc54337cab45331bfeb3ecabbb078073781c'}, 'artifacts/validation/product_execution/finalize_result_storage_evidence_2026-10-06.json': {'bytes': 88827, 'sha256': '05f2cdce18ece5066e4507a2c846e894eeaf7fd800f83f2450c42431bde9e19f'}}
FROZEN_RECEIPT = {'bytes': 2780, 'sha256': 'ee8ebdb6e4723bf608a8eb0daf0684efd49de9c3af32d31aab64cac1e2607a85'}
CURRENT = (
    'src/__init__.py', 'src/api/__init__.py',
    'src/api/results.py', 'src/api/result_storage.py', 'src/api/result_document.py',
    'src/api/reporting.py', 'src/api/worker.py', 'src/api/app.py', 'src/api/product.py',
    'src/api/media_source.py', 'src/api/media_revision.py', 'src/api/jobs.py',
    'src/detection/image_people.py', 'src/tracking/player_motion_tracking.py',
    'src/utils/bbox_utils.py', 'requirements-product.txt', MONITOR,
    'artifacts/validation/product_installation/clean_env/Scripts/python.exe',
    'artifacts/validation/product_installation/clean_env/pyvenv.cfg',
)
HISTORICAL = (
    'artifacts/validation/product_execution/finalize_result_storage_evidence_2026-10-06.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/helper.py',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/protocol.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/run/report.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit_correction.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/protocol.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/resource_receipt.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/child_receipt.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/saved_result_comparison.json',
)
COUNTS = (600, 18000)
MODES = ('old', 'new')
FPS = 60
MATRIX = [[.1, -.0, 0.], [0., .1, 0.], [0., 0., 1.]]
TRANSFORM = [[1., -.0, 0.], [0., 1., 0.], [0., 0., 1.]]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def publish_handshake(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    save(temporary, value)
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def source_bindings():
    names = list(CURRENT) + list(HISTORICAL) + [str((BASELINE / name).relative_to(ROOT).as_posix()) for name in FROZEN]
    names += [str((BASELINE / 'receipt.json').relative_to(ROOT).as_posix()),
              str(Path(__file__).resolve().relative_to(ROOT).as_posix()),
              str((UNIT / 'protocol.json').relative_to(ROOT).as_posix()),
              str((UNIT / 'PROTOCOL.md').relative_to(ROOT).as_posix())]
    return {name: {'bytes': (ROOT / name).stat().st_size, 'sha256': digest(ROOT / name)} for name in names}


def validate_baseline():
    receipt = BASELINE / 'receipt.json'
    if {'bytes': receipt.stat().st_size, 'sha256': digest(receipt)} != FROZEN_RECEIPT:
        raise AssertionError('Frozen baseline preservation receipt changed')
    for name, expected in FROZEN.items():
        path = BASELINE / name
        if {'bytes': path.stat().st_size, 'sha256': digest(path)} != expected:
            raise AssertionError('Frozen baseline changed: ' + name)
    if digest(ROOT / MONITOR) != MONITOR_SHA256:
        raise AssertionError('Verified owned-process helper changed; review before a new protocol')


def load_file(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_disk_implementation(mode):
    # Importing this package runs only its empty initializer, before production
    # dependencies. Frozen modules occupy their actual aliases so every imported
    # DiskSeries/ResultStorage/file_hash belongs to the selected implementation.
    import src.api as api_package
    names = ('result_storage', 'results', 'reporting', 'result_document')
    aliases = tuple('src.api.' + name for name in names)
    if any(name in sys.modules for name in aliases):
        raise AssertionError('Selected production aliases were imported before implementation selection')
    modules, bindings = {}, {}
    base = BASELINE if mode == 'old' else ROOT
    for name in names:
        path = base / 'src/api' / (name + '.py')
        module = load_file('src.api.' + name, path)
        setattr(api_package, name, module)
        modules[name] = module
        bindings[name] = {'moduleAlias': module.__name__,
                          'relativeSource': path.relative_to(ROOT).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)}
    storage, results = modules['result_storage'], modules['results']
    reporting, document = modules['reporting'], modules['result_document']
    assert results.DiskSeries is storage.DiskSeries
    assert reporting.DiskSeries is storage.DiskSeries
    assert document.ResultStorage is storage.ResultStorage
    assert document.file_hash is results.file_hash
    assert sys.modules['src.api.media_revision'].file_hash is results.file_hash
    return results, reporting, storage.ResultStorage, document.ResultDocument, document.publish_result, bindings


class Rows:
    """One iterator factory; a growing input is never retained as a list."""
    def __init__(self, factory):
        self.factory = factory


def write_generated(value, stream):
    if isinstance(value, Rows):
        stream.write('[')
        first = True
        for item in value.factory():
            if not first:
                stream.write(', ')
            first = False
            write_generated(item, stream)
        stream.write(']')
    elif isinstance(value, dict):
        stream.write('{')
        for index, (key, item) in enumerate(value.items()):
            if index:
                stream.write(', ')
            json.dump(key, stream, allow_nan=False)
            stream.write(': ')
            write_generated(item, stream)
        stream.write('}')
    else:
        # Only fixed-size frame/scalar containers reach stdlib serialization.
        json.dump(value, stream, allow_nan=False)


def camera_valid(index):
    return not 65 <= index % 257 <= 69


def selected(index, player):
    absent = index % (191 if player == 1 else 223) == (17 if player == 1 else 21)
    return None if absent else {
        'bbox': [1., 1., 10., 20.] if player == 1 else [40., 3., 50., 30.],
        'confidence': .8 if player == 1 else .7,
        'source_track_id': 7 if player == 1 else 8,
    }


def ball_record(index):
    state = ('MISSING' if index % 97 == 10 else 'OCCLUDED' if index % 97 == 11 else
             ('DETECTED', 'TRACKED', 'PREDICTED', 'INTERPOLATED')[index % 4])
    return {'frame_index': index, 'timestamp_seconds': index / FPS,
            'x_px': 30. + (index % 23) / 16, 'y_px': 50. + (index % 17) / 32,
            'court_x_m': 4. + (index % 23) / 160, 'court_y_m': 8.,
            'confidence': .2 if state == 'OCCLUDED' else .9,
            'speed_kmh': 180, 'state': state, 'source': 'SYNTHETIC_IMAGE_RECORD',
            'world_position_m': [4., 8., 3.],
            'unknown_payload': {'ints': [0, 1, -5], 'floats': [0., -.0, 1e-12],
                                'text': 'Synthetic é 🎾 <value> \\ "quoted"', 'null': None}}


def motion_record(index, player):
    if player == 1 and index % 181 == 37:
        return None
    return {'frame_index': index, 'timestamp_s': index / FPS,
            'ground_position_m': [4. + (index % 101) / 100, 8. if player == 1 else 12.],
            'speed_kmh': 0. if index % 101 == 0 else (10. if player == 1 else 12.),
            'velocity_window_seconds': .1 if player == 1 else .05,
            'position_fit_rmse_m': -.0}


def people_record(index):
    if index % 13 == 0:
        people = None
    elif index % 13 == 1:
        people = []
    else:
        people = []
        for player in (1, 2):
            evidence = selected(index, player) or selected(0, player)
            people.append({'bbox': evidence['bbox'], 'confidence': evidence['confidence'],
                           'sourceTrackId': evidence['source_track_id'], 'courtRole': player,
                           'participantStatus': 'COURT_ROLE_CANDIDATE'})
        # A fixed 67-person burst also crosses the private parser's nested
        # array boundary. Same-role/value distractors remain unverified people.
        for offset in range(65 if index % 97 == 3 else 1):
            people.append({'bbox': [60., 2., 70., 22.], 'confidence': .625,
                           'sourceTrackId': 100 + offset, 'courtRole': None,
                           'participantStatus': 'UNKNOWN'})
    return {'frame_index': index, 'timestamp_seconds': index / FPS, 'people': people}


def timing_valid(index):
    return index >= 1 and not 90 <= index % 257 <= 91


def timing_intervals(count):
    start = None
    for index in range(1, count + 1):
        valid = index < count and timing_valid(index)
        if valid and start is None:
            start = index
        elif not valid and start is not None:
            # Pin the existing boolean/int input behavior as well as later
            # ordinary interval edges; JSON retains this exact value.
            yield {'startFrame': True if start == 1 else start, 'endFrameExclusive': index}
            start = None
    yield {'startFrame': -1, 'endFrameExclusive': 3}
    yield {'startFrame': count, 'endFrameExclusive': count + 1}


def generate_inputs(directory, count):
    directory.mkdir()
    values = {
        'detections.json': {
            'metadata': {'frames': count, 'fps': FPS, 'width': 100, 'height': 100,
                         'player_identity_scope': 'SYNTHETIC_NEAR_FAR_ROLE'},
            'frames': Rows(lambda: ({'frame_index': index, 'court_registration_valid': True,
                                    'player_1': selected(index, 1), 'player_2': selected(index, 2)}
                                   for index in range(count)))},
        'trajectories.json': {'ball_trajectory': Rows(lambda: (ball_record(i) for i in range(count)))},
        'court_geometry.json': {'is_valid': True, 'homography_matrix': MATRIX,
                                'reprojection_error_px': .125,
                                'court_keypoints_px': [[1., 2.], [3., 4.]],
                                'canonical_keypoints_m': [[0., 0.], [1., 1.]]},
        'camera_registration.json': {'enabled': True, 'frames': Rows(lambda: (
            {'frame_index': i, 'timestamp_seconds': i / FPS, 'is_valid': camera_valid(i),
             'image_to_court': MATRIX, 'anchor_to_frame_px': TRANSFORM if camera_valid(i) else None,
             'rejection_reason': None if camera_valid(i) else 'SYNTHETIC_CAMERA_CUT',
             'inlier_mean_residual_reference_px': .125, 'reference_width': 960}
            for i in range(count)))},
        'player_motion.json': {
            'ground_position_source': 'SYNTHETIC_BBOX_GROUND_PROXY',
            'player_1': {'observed_distance_m': 5.75, 'mean_speed_kmh': 10.,
                         'samples': Rows(lambda: (motion_record(i, 1) for i in range(count)))},
            'player_2': {'observed_distance_m': 7.25, 'mean_speed_kmh': 12.,
                         'samples': Rows(lambda: (motion_record(i, 2) for i in range(count)))}},
        # A valid 65-frame prefix gives pose observations followed by explicit
        # missing frames; this is deliberately not a dense-pose memory claim.
        'pose_tracking.json': {'frames': Rows(lambda: (
            {'1': {'left_wrist': {'position_px': [4., 9.], 'confidence': .8},
                   'right_wrist': {'position_px': [100., 2.], 'confidence': .9}},
             '2': {'left_wrist': {'position_px': [44., 9.], 'confidence': .7}}}
            for _ in range(65)))},
        'racket_tracking.json': {'enabled': True, 'frames': Rows(lambda: (
            {'1': {'state': 'DETECTED', 'bbox_xyxy': [2., 3., 10., 12.],
                   'position_px': [6., 7.] if i % 113 != 23 else [20., 7.], 'confidence': .7},
             '2': {'state': 'DETECTED', 'bbox_xyxy': [41., 3., 49., 12.],
                   'position_px': [45., 7.], 'confidence': .7}}
            for i in range(count)))},
        'image_people.json': {'coordinateSystem': 'normalized_video_pixels',
                              'identityScope': 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
                              'frames': Rows(lambda: (people_record(i) for i in range(count)))},
        'metrics.json': {'processing_fps': 5., 'authoritative_events_enabled': False,
                         'production_qualified': False, 'ball_backend': 'SYNTHETIC_NO_MODEL',
                         'tracking_strategy': 'SYNTHETIC_RECORDS_ONLY'},
        # Deliberately present but wholly abstained while authoritative=false.
        'match_events.json': {'events': Rows(lambda: ({'event_type': 'UNQUALIFIED', 'timestamp_s': i / FPS}
                                                    for i in range(count)))},
        'line_calls.json': {'line_calls': Rows(lambda: ({'frame_index': i, 'call': 'UNQUALIFIED'}
                                                      for i in range(count)))},
        'media_metadata.json': {
            'original': {'width': 100, 'height': 100, 'frameRate': '60/1', 'timeBase': '1/60',
                         'firstSourcePts': '0', 'lastSourcePts': str(count - 1),
                         'evidenceKind': 'SYNTHETIC_INDEX_CLOCK_NO_DECODED_FOOTAGE'},
            'normalization': {'synthetic': True, 'clock': 'index / 60'},
            'timeMapping': {'status': 'CAPTURED_FFMPEG_FRAME_MAP',
                            'synthetic': True, 'evidenceKind': 'GENERATED_RECORDS_NOT_FFMPEG_EXECUTION',
                            'sourceTimeBase': '1/60', 'firstSourcePts': '0', 'lastSourcePts': str(count - 1),
                            'validMeasurementTransitionIntervals': Rows(lambda: timing_intervals(count))}},
    }
    for name, value in values.items():
        with (directory / name).open('x', encoding='utf-8', newline='') as stream:
            write_generated(value, stream)
    mapping = directory / 'source_frame_mapping.jsonl'
    with mapping.open('x', encoding='utf-8', newline='') as stream:
        for index in range(count):
            json.dump({'normalizedFrameIndex': index, 'normalizedTimestampSeconds': index / FPS,
                       'sourcePts': str(index), 'sourceTimeBase': '1/60',
                       'synthetic': True}, stream, allow_nan=False)
            stream.write('\n')
    # Attribution fields test escaping/formula neutralization only. They are
    # generated records, and certify neither an asset nor an actual license.
    credit = {'id': 'synthetic_records_no_footage', 'title': 'Synthetic, "records" <fixture>',
              'author': '=Synthetic é author', 'sourcePage': 'https://commons.wikimedia.org/wiki/Commons:Sandbox',
              'license': 'CC BY-SA 4.0', 'licenseUrl': 'https://creativecommons.org/licenses/by-sa/4.0/',
              'derivativeLicense': 'CC BY-SA 4.0',
              'modifications': 'Generated record fixture; no footage or rights assertion.',
              'sourceSha256': digest(mapping), 'inputSha256': digest(mapping),
              'excerpt': {'startSeconds': 0., 'endSeconds': count / FPS,
                          'startFrame': 0, 'endFrameExclusive': count,
                          'sourceTimeBase': '1/60', 'firstSourcePts': '0', 'lastSourcePts': str(count - 1)}}
    with (directory / 'media_source.json').open('x', encoding='utf-8', newline='') as stream:
        json.dump(credit, stream, allow_nan=False)
    names = tuple(values) + ('source_frame_mapping.jsonl', 'media_source.json')
    job = {'id': 'match_' + ('0' * 28) + f'{count:04x}',
           'filename': 'Synthetic é 🎾 <records>.mp4', 'started_at': '2026-10-06T00:00:00+00:00'}
    return names, job


def input_bindings(directory, names):
    return {name: {'bytes': (directory / name).stat().st_size, 'sha256': digest(directory / name)} for name in names}


def disk_snapshot(directory):
    files, size, databases, journals = 0, 0, 0, 0
    for current, _directories, names in os.walk(directory):
        for name in names:
            path = Path(current) / name
            if path.is_file():
                logical = path.stat().st_size
                files += 1
                size += logical
                databases += logical if name.endswith('.sqlite3') else 0
                journals += logical if name.endswith(('-journal', '-wal', '-shm')) else 0
    return {'files': files, 'logicalFileBytes': size, 'sqliteDatabaseBytes': databases,
            'sqliteSidecarBytes': journals, 'allocatedDiskBlocksMeasured': False}


def write_chunks(path, chunks):
    maximum, count = 0, 0
    begin = time.perf_counter()
    first_chunk_seconds = None
    with path.open('x', encoding='utf-8', newline='') as stream:
        for chunk in chunks:
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - begin
            maximum = max(maximum, len(chunk))
            count += 1
            stream.write(chunk)
    closed_seconds = time.perf_counter() - begin
    hash_begin = time.perf_counter()
    checksum = digest(path)
    return {'chunks': count, 'maximumChunkCharacters': maximum, 'bytes': path.stat().st_size,
            'sha256': checksum, 'firstChunkAvailableSeconds': first_chunk_seconds,
            'allChunksWrittenAndClosedSeconds': closed_seconds,
            'verificationHashFullFileReadSeconds': time.perf_counter() - hash_begin}


def expected_location(index, player):
    if not camera_valid(index) or selected(index, player) is None or motion_record(index, player) is None:
        return None
    return motion_record(index, player)['ground_position_m']


def expected_speed(index, player):
    start = max(0, index - (6 if player == 1 else 3))
    if start >= index or any(expected_location(i, player) is None for i in range(start, index + 1)):
        return None
    if any(not timing_valid(i) for i in range(start + 1, index + 1)):
        return None
    return motion_record(index, player)['speed_kmh']


def check_contract(value, count):
    """Independent generated-frame expectations, not only old/new equality."""
    assert value['contractVersion'] == '2.0'
    assert value['summary']['frames'] == count and value['summary']['fps'] == FPS
    assert value['summary']['durationSeconds'] == count / FPS
    assert value['world3d'] is None and value['ballMetrics'] is None and value['segmentationMasks'] is None
    assert value['pipelineMetrics']['raw']['authoritative_events_enabled'] is False
    assert value['events'] == [] and value['lineCalls'] == [] and value['scoreHistory'] == []
    assert value['playerMetrics'] is None
    assert value['summary']['capabilities']['events'] is False
    original = value['mediaMetadata']['original']
    assert original['firstSourcePts'] == '0' and original['lastSourcePts'] == str(count - 1)
    assert original['timeBase'] == '1/60' and original['evidenceKind'].startswith('SYNTHETIC_')
    trajectory = value['trajectories']
    assert len(trajectory['ballTrajectory']) == len(value['detections']['frames']) == count
    assert len(value['cameraTracking']['frames']) == len(value['imagePeople']['frames']) == count
    zero_speeds, null_speeds, unknown_people, null_people, empty_people = 0, 0, 0, 0, 0
    for index, ball in enumerate(trajectory['ballTrajectory']):
        source = ball_record(index)
        assert ball['frame_index'] == index and ball['timestamp_seconds'] == index / FPS
        assert ball['state'] == source['state']
        assert ball['speed_kmh'] is None and ball['world_position_m'] is None
        assert ball['confidence'] == (None if source['state'] == 'MISSING' else source['confidence'])
        assert ball['unknown_payload']['floats'][1].hex() == '-0x0.0p+0'
        assert type(ball['unknown_payload']['ints'][0]) is int
        assert type(ball['unknown_payload']['floats'][0]) is float
        missing_image = source['state'] in ('MISSING', 'OCCLUDED')
        assert ball['x_px'] == (None if missing_image else source['x_px'])
        assert ball['y_px'] == (None if missing_image else source['y_px'])
        assert ball['court_x_m'] == (None if missing_image or not camera_valid(index) else source['court_x_m'])
        assert ball['court_y_m'] == (None if missing_image or not camera_valid(index) else source['court_y_m'])
        detection = value['detections']['frames'][index]
        assert detection['court_registration_valid'] is camera_valid(index)
        assert value['cameraTracking']['frames'][index]['isValid'] is camera_valid(index)
        for player in (1, 2):
            positions = trajectory[f'player{player}CourtPositions']
            speeds = trajectory[f'player{player}Speeds']
            assert len(positions) == len(speeds) == count
            assert positions[index] == expected_location(index, player)
            assert speeds[index] == expected_speed(index, player)
            zero_speeds += speeds[index] == 0.
            null_speeds += speeds[index] is None
        people = value['imagePeople']['frames'][index]['people']
        if index % 13 == 0:
            assert people is None
            null_people += 1
        elif index % 13 == 1:
            assert len(people) == 0
            empty_people += 1
        else:
            assert len(people) == (67 if index % 97 == 3 else 3)
            for proposal_index, proposal in enumerate(people):
                role = proposal_index + 1 if proposal_index < 2 else None
                accepted = role is not None and camera_valid(index) and selected(index, role) is not None
                assert proposal['courtRole'] == (role if accepted else None)
                assert proposal['participantStatus'] == ('COURT_ROLE_CANDIDATE' if accepted else 'UNKNOWN')
                unknown_people += not accepted
    assert null_speeds and unknown_people and null_people and empty_people
    # Raw zero speed is retained once its short window and timing are valid.
    assert zero_speeds > 0
    assert len(value['courtSegments']) > 1
    assert value['poseTracking']['frames'][65]['player_1'] is None
    assert value['racketTracking']['frames'][23]['player_1']['state'] == 'MISSING'
    return {'allFrameTimestampsAndGapSemanticsChecked': True, 'physicalBallSpeedAndXyzAlwaysNull': True,
            'authoritativeFalsePreserved': True, 'sourcePtsStringEndpointsPreserved': True,
            'unknownPeople': unknown_people, 'unavailablePeopleFrames': null_people,
            'emptyPeopleFrames': empty_people, 'withheldSpeedSamples': null_speeds,
            'measuredZeroSpeedSamples': zero_speeds}


def child(mode, count, directory):
    import psutil
    process = psutil.Process()
    identity = {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()}
    report = {'startedAt': stamp(), 'passed': False, 'mode': mode, 'frames': count,
              'actualWorkloadIdentity': identity, 'pythonVersion': sys.version,
              'qualificationEvidence': False, 'inferenceRun': False,
              'scope': 'Generated artifacts only; no video, models or measured PTS.',
              'stages': {}, 'processSnapshots': {}, 'diskSnapshots': {}}
    started = time.perf_counter()
    try:
        publish_handshake(directory / 'actual_identity.json', identity)
        acknowledged = directory / 'actual_identity_sampled.json'
        deadline = time.monotonic() + 10
        while not acknowledged.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Exact actual interpreter sampling was not acknowledged')
            time.sleep(.01)
        ack = json.loads(acknowledged.read_text(encoding='utf-8'))
        assert ack['identity'] == identity and ack['sampledByExactIdentity'] is True
        expected = json.loads((RUN / 'source_bindings.json').read_text(encoding='utf-8'))
        validate_baseline()
        assert source_bindings() == expected, 'Sources changed before child imports'
        report['sourceBindingsBefore'] = expected
        # These are the production-used functions, without importing/running a
        # worker, FFmpeg, API server, video decoder, detector or model.
        results, reporting, ResultStorage, ResultDocument, publish_result, loaded = load_disk_implementation(mode)
        report['implementation'] = 'FROZEN_PREOPTIMIZATION_DISK' if mode == 'old' else 'CURRENT_DISK_LATENCY_CANDIDATE'
        report['loadedProductionModules'] = loaded
        report['allSelectedStorageClassesConsistent'] = True
        report['psutilVersion'] = psutil.__version__
        analysis = directory / 'analysis'

        def boundary(name):
            report['processSnapshots'][name] = process.memory_info()._asdict()
            report['diskSnapshots'][name] = disk_snapshot(directory)

        work_started = time.perf_counter()
        begin = time.perf_counter()
        input_names, job = generate_inputs(analysis, count)
        report['stages']['inputGenerationSeconds'] = time.perf_counter() - begin
        report['inputBindingsBefore'] = input_bindings(analysis, input_names)
        boundary('afterInputGeneration')
        begin = time.perf_counter()
        context = ResultStorage(analysis)
        with context as storage:
            value = results.build_result(analysis, job, _storage=storage)
            report['stages']['constructionSeconds'] = time.perf_counter() - begin
            boundary('afterConstruction')
            begin = time.perf_counter()
            publish_result(storage, value, analysis)
            report['stages']['publicationSeconds'] = time.perf_counter() - begin
        del value
        gc.collect()
        boundary('afterPublicationAndConstructionRelease')
        report['publishedJson'] = {'bytes': (analysis / 'result.json').stat().st_size,
                                   'sha256': digest(analysis / 'result.json')}
        begin = time.perf_counter()
        document_context = ResultDocument(analysis, job['id'])
        report['stages']['readerOpenSeconds'] = time.perf_counter() - begin
        report['readerOpenScope'] = 'Selected production ResultDocument construction, including database SHA256 verification and root validation; no presentation decoration yet. Shared file_hash cache follows the bound selected results.py implementation.' 
        with document_context as document:
            begin = time.perf_counter()
            value = document.presented(job['id'])
            report['stages']['presentationSeconds'] = time.perf_counter() - begin
            boundary('afterReaderOpen')
            begin = time.perf_counter()
            chunks = document.json_chunks(value)
            report['readJson'] = write_chunks(directory / 'read_result.json', chunks)
            report['stages']['strictReadJsonSeconds'] = time.perf_counter() - begin
            begin = time.perf_counter()
            # Both selected disk implementations stream the production CSV rows.
            report['csv'] = write_chunks(directory / 'telemetry.csv', reporting.telemetry_csv_chunks(value))
            report['stages']['csvExportSeconds'] = time.perf_counter() - begin
            begin = time.perf_counter()
            html_chunks = reporting.readable_report_chunks(value, None)
            report['html'] = write_chunks(directory / 'report.html', html_chunks)
            report['stages']['htmlExportSeconds'] = time.perf_counter() - begin
            begin = time.perf_counter()
            report['semanticChecks'] = check_contract(value, count)
            report['stages']['independentSemanticCheckSeconds'] = time.perf_counter() - begin
            boundary('afterReadAndExport')
        del value, document, document_context
        gc.collect()
        boundary('afterReaderCloseAndRelease')
        marker = json.loads((analysis / '.result/current.store').read_text(encoding='utf-8'))
        database = analysis / '.result' / marker['database']
        assert marker['databaseSha256'] == digest(database) and marker['databaseBytes'] == database.stat().st_size
        assert marker['analysisId'] == job['id'] and marker['frames'] == count
        assert not (analysis / 'result.tmp').exists() and not (analysis / 'summary.tmp').exists()
        assert not tuple(analysis.glob('.result_storage_*.sqlite3*'))
        assert not tuple((analysis / '.result').glob('*.store.tmp'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-journal'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-wal'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-shm'))
        report['publicationManifest'] = marker
        report['publishedDatabaseBytes'] = database.stat().st_size
        report['ownedScratchAndPendingExportsAbsent'] = True
        report['inputBindingsAfter'] = input_bindings(analysis, input_names)
        assert report['inputBindingsBefore'] == report['inputBindingsAfter'], 'Generated original inputs changed'
        assert report['publishedJson']['sha256'] == report['readJson']['sha256'], 'Presented read changed saved bytes'
        report['sourceBindingsAfter'] = source_bindings()
        assert report['sourceBindingsAfter'] == expected, 'Sources changed during child'
        report['workAfterImportsSeconds'] = time.perf_counter() - work_started
        report['measuredProductionStagesSeconds'] = sum(report['stages'][name] for name in (
            'constructionSeconds', 'publicationSeconds', 'readerOpenSeconds', 'presentationSeconds', 'strictReadJsonSeconds',
            'csvExportSeconds', 'htmlExportSeconds'))
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        primary = sys.exception()
        try:
            memory = process.memory_info()
            report.update({'completedAt': stamp(), 'totalChildElapsedSeconds': time.perf_counter() - started,
                           'windowsPeakWorkingSetBytes': getattr(memory, 'peak_wset', None),
                           'windowsPeakPrivateCommitBytes': getattr(memory, 'peak_pagefile', None),
                           'modelsImported': [name for name in ('torch', 'torchvision', 'ultralytics', 'onnxruntime') if name in sys.modules]})
            if report['modelsImported']:
                report['passed'] = False
                report['unexpectedModelRuntimeImport'] = True
            save(directory / 'child_receipt.json', report)
            print(json.dumps({'passed': report['passed'], 'mode': mode, 'frames': count,
                              'error': report.get('error')}), flush=True)
        except BaseException as receipt_error:
            if primary is None:
                raise
            primary.add_note('Child receipt save also failed: ' + str(receipt_error))
    return 0 if report['passed'] else 1


def files_equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            first, second = a.read(65536), b.read(65536)
            if first != second:
                return False
            if not first:
                return True


def compare_pair(count):
    old, new = RUN / f'{count}_old', RUN / f'{count}_new'
    left = json.loads((old / 'child_receipt.json').read_text(encoding='utf-8'))
    right = json.loads((new / 'child_receipt.json').read_text(encoding='utf-8'))
    comparisons = {}
    for name in ('analysis/result.json', 'analysis/summary.json', 'read_result.json', 'telemetry.csv', 'report.html'):
        comparisons[name] = {'exactBytesEqual': files_equal(old / name, new / name),
                             'oldBytes': (old / name).stat().st_size, 'newBytes': (new / name).stat().st_size,
                             'oldSha256': digest(old / name), 'newSha256': digest(new / name)}
    inputs_equal = left['inputBindingsBefore'] == right['inputBindingsBefore']
    semantics_equal = left['semanticChecks'] == right['semanticChecks']
    passed = inputs_equal and semantics_equal and all(row['exactBytesEqual'] for row in comparisons.values())
    result = {'frames': count, 'passed': passed, 'outputs': comparisons,
              'streamedInputsIdentical': inputs_equal, 'independentSemanticChecksEqual': semantics_equal,
              'volatileFieldsExcluded': [], 'baselineStagesSeconds': left['stages'],
              'currentStagesSeconds': right['stages'],
              'baselineDatabaseBytes': left.get('publishedDatabaseBytes'),
              'currentDatabaseBytes': right.get('publishedDatabaseBytes'),
              'baselineImplementation': left['implementation'], 'currentImplementation': right['implementation'],
              'noPerformanceQualificationGate': True,
              'qualificationEvidence': False}
    save(RUN / f'comparison_{count}.json', result)
    if not passed:
        raise AssertionError('Exact result/read/CSV/HTML comparison failed at ' + str(count) + '; halt later cases')
    return result


def parent():
    import psutil
    RUN.mkdir(exist_ok=False)
    report = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False,
              'runs': [], 'comparisons': [], 'qualificationEvidence': False, 'inferenceRun': False,
              'syntheticAttributionIsNotRightsEvidence': True}
    before = None
    try:
        protocol = json.loads((UNIT / 'protocol.json').read_text(encoding='utf-8'))
        assert protocol['counts'] == list(COUNTS) and protocol['modes'] == list(MODES)
        assert protocol['minimumAvailableRamBytes'] == 400 * 1024 * 1024
        assert protocol['maximumChildWallSeconds'] == 180 and protocol['lowRamConsecutiveSamples'] == 5
        validate_baseline()
        before = source_bindings()
        prepared = UNIT / 'prepared_source_bindings.json'
        report['preparedBindingsSha256'] = digest(prepared)
        assert before == json.loads(prepared.read_text(encoding='utf-8')), 'Reviewed prepared sources changed; refuse launch'
        save(RUN / 'source_bindings.json', before)
        with (RUN / 'executed_helper.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        assert digest(RUN / 'executed_helper.py') == digest(Path(__file__))
        report['sourceBindingsBefore'] = before
        report['monitorInterpreter'] = {'path': sys.executable, 'sha256': digest(Path(sys.executable)), 'pythonVersion': sys.version}
        monitor = load_file('result_owned_monitor_20261006', ROOT / MONITOR)
        for count in COUNTS:
            for mode in MODES:
                assert source_bindings() == before, 'Bound sources changed before next child; no launch'
                assert digest(prepared) == report['preparedBindingsSha256'], 'Prepared binding record changed; no launch'
                assert psutil.virtual_memory().available >= protocol['preflightAvailableRamBytes'], 'Insufficient preflight RAM; no retry'
                assert shutil.disk_usage(UNIT).free >= protocol['minimumFreeDiskBytes'], 'Insufficient preflight disk; no retry'
                directory = RUN / f'{count}_{mode}'
                directory.mkdir()
                process, owned = None, None
                receipt = {'mode': mode, 'frames': count, 'passed': False, 'samples': 0,
                           'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
                           'minimumSampledAvailableRamBytes': psutil.virtual_memory().available,
                           'allPrivateSamplesAvailable': True}
                begin = time.perf_counter()
                try:
                    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
                    with (directory / 'stdout.txt').open('x', encoding='utf-8') as log, \
                            (directory / 'samples.jsonl').open('x', encoding='utf-8', newline='') as samples:
                        process = subprocess.Popen([str(CLEAN), str(Path(__file__)), '--child', mode, str(count), str(directory)],
                                                   cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                                   creationflags=subprocess.CREATE_NO_WINDOW)
                        report['status'] = 'RUNNING'
                        owned = monitor.OwnedProcessTree(process)
                        low_ram = 0
                        while process.poll() is None:
                            rows = owned.sample()
                            identity_path = directory / 'actual_identity.json'
                            ack_path = directory / 'actual_identity_sampled.json'
                            if identity_path.exists() and not ack_path.exists():
                                identity = json.loads(identity_path.read_text(encoding='utf-8'))
                                if owned.observed_workload(identity):
                                    publish_handshake(ack_path, {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                            private = [row['memoryBytes'].get('private') for row in rows]
                            available = psutil.virtual_memory().available
                            sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - begin,
                                      'processes': rows, 'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                                      'ownedTreePrivateBytes': sum(value for value in private if value is not None),
                                      'allPrivateValuesAvailable': all(value is not None for value in private),
                                      'availableRamBytes': available}
                            samples.write(json.dumps(sample, allow_nan=False) + '\n')
                            samples.flush()
                            receipt['samples'] += 1
                            receipt['peakSampledOwnedTreeRssBytes'] = max(receipt['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                            receipt['peakSampledOwnedTreePrivateBytes'] = max(receipt['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                            receipt['minimumSampledAvailableRamBytes'] = min(receipt['minimumSampledAvailableRamBytes'], available)
                            receipt['allPrivateSamplesAvailable'] &= sample['allPrivateValuesAvailable']
                            low_ram = low_ram + 1 if available < protocol['minimumAvailableRamBytes'] else 0
                            reason = ('Owned tree RSS safeguard' if sample['ownedTreeRssBytes'] > protocol['maximumOwnedTreeRssBytes'] else
                                      'Available RAM safeguard' if low_ram >= protocol['lowRamConsecutiveSamples'] else
                                      'Child wall-time safeguard' if sample['elapsedSeconds'] >= protocol['maximumChildWallSeconds'] else None)
                            if reason:
                                receipt['safeguardStop'] = reason
                                raise RuntimeError(reason + '; preserve partials and halt later cases')
                            time.sleep(protocol['sampleSeconds'])
                    receipt['exitCode'] = process.returncode
                    assert process.returncode == 0, 'Child failed; retain receipts and halt later cases'
                    child_report = json.loads((directory / 'child_receipt.json').read_text(encoding='utf-8'))
                    assert child_report['passed'] and not child_report['modelsImported']
                    assert owned.observed_workload(child_report['actualWorkloadIdentity']), 'Actual workload identity was not sampled'
                    receipt['actualWorkloadWasSampled'] = True
                    receipt['remainingOwnedIdentities'] = owned.wait_for_retirement()
                    assert not receipt['remainingOwnedIdentities'], 'Remembered owned identity remained after normal exit'
                    receipt['allRememberedOwnedIdentitiesRetired'] = True
                    assert source_bindings() == before, 'Bound sources changed during child'
                    receipt['passed'] = True
                    receipt['childReceiptSha256'] = digest(directory / 'child_receipt.json')
                except BaseException as error:
                    receipt['error'] = type(error).__name__ + ': ' + str(error)
                    if process is not None:
                        try:
                            receipt['terminationActions'] = monitor.stop_owned(process, owned)
                        except BaseException as cleanup_error:
                            error.add_note('Exact owned cleanup also failed: ' + str(cleanup_error))
                            receipt['cleanupError'] = str(cleanup_error)
                        if owned is not None:
                            try:
                                receipt['remainingOwnedIdentitiesAfterCleanup'] = owned.wait_for_retirement()
                                receipt['allRememberedOwnedIdentitiesRetired'] = not receipt['remainingOwnedIdentitiesAfterCleanup']
                            except BaseException as retirement_error:
                                error.add_note('Owned retirement verification also failed: ' + str(retirement_error))
                                receipt['retirementVerificationError'] = str(retirement_error)
                                receipt['allRememberedOwnedIdentitiesRetired'] = False
                    raise
                finally:
                    receipt['completedAt'] = stamp()
                    receipt['monitoredChildWallSeconds'] = time.perf_counter() - begin
                    receipt['retainedDisk'] = disk_snapshot(directory)
                    if owned is not None:
                        receipt['ownedIdentities'] = owned.identities()
                    save(directory / 'resource_receipt.json', receipt)
                    report['runs'].append(receipt)
                print(json.dumps({'completed': mode, 'frames': count, 'wallSeconds': receipt['monitoredChildWallSeconds'],
                                  'peakSampledOwnedTreeRssBytes': receipt['peakSampledOwnedTreeRssBytes']}), flush=True)
            report['comparisons'].append(compare_pair(count))
        report['passed'] = True
        report['status'] = 'COMPLETED'
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        report['status'] = 'FAILED' if report['runs'] or report['status'] == 'RUNNING' else 'NOT_STARTED'
    finally:
        report['completedAt'] = stamp()
        report['laterCasesHaltedAfterFirstFailure'] = not report['passed']
        if before is not None:
            try:
                report['sourceBindingsAfter'] = source_bindings()
                report['sourcesUnchanged'] = before == report['sourceBindingsAfter']
                report['preparedBindingsUnchanged'] = digest(prepared) == report['preparedBindingsSha256']
                if not report['sourcesUnchanged'] or not report['preparedBindingsUnchanged']:
                    report['passed'] = False
                    report['status'] = 'FAILED'
            except BaseException as source_error:
                report['sourceBindingError'] = str(source_error)
                report['passed'] = False
                report['status'] = 'FAILED'
        save(RUN / 'report.json', report)
        print(json.dumps({'passed': report['passed'], 'status': report['status'],
                          'completedChildren': len(report['runs']), 'comparisons': len(report['comparisons']),
                          'error': report.get('error')}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', nargs=3, metavar=('MODE', 'COUNT', 'DIRECTORY'))
    arguments = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if arguments.child:
        mode, raw_count, raw_directory = arguments.child
        count, directory = int(raw_count), Path(raw_directory).resolve()
        assert mode in MODES and count in COUNTS
        assert directory == RUN / f'{count}_{mode}' and directory.is_dir()
        assert not (directory / 'child_receipt.json').exists()
        return child(mode, count, directory)
    assert sys.prefix == sys.base_prefix, 'Launch monitoring parent with base Python313'
    return parent()


if __name__ == '__main__':
    raise SystemExit(main())
