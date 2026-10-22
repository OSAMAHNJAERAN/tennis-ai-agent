"""Independent stdlib evidence audit; never runs the selector or decodes video."""
import ast
from collections import defaultdict
import csv
import ctypes
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import median
import struct
import sys
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
OUT = UNIT / 'audit'
DATASET = ROOT / 'data/external/uvy_tennis_videos'
CACHE = ROOT / 'outputs/vision_upgrade_audit/uvy_person_tracked640'
SEQUENCES = ('tennis_V01', 'tennis_V02', 'tennis_V03')
COUNTS = ('annotatedPlayers', 'proposalObservations', 'proposalMatched', 'proposalMissed',
          'proposalPublisherUnmatched', 'selectedMatched', 'selectedPublisherUnmatched',
          'selectedMissed', 'selectedRefereeOverlap', 'selectorLostMatched',
          'selectedObservations', 'framesWithoutSelection', 'ambiguityFrames')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def resolve(path):
    return Path(path) if Path(path).is_absolute() else ROOT / path


def verify(bindings):
    for path, expected in bindings.items():
        assert binding(resolve(path)) == expected, 'Changed bound file: ' + path


def verify_record(record):
    assert binding(resolve(record['path'])) == {key: record[key] for key in ('bytes', 'sha256')}


def close(a, b, tolerance=1e-12):
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)


def finite_box(box):
    assert len(box) == 4 and all(type(value) in (int, float) and math.isfinite(value) for value in box)
    assert box[0] < box[2] and box[1] < box[3]


def adapt(cached):
    observations = []
    for frame in cached:
        seen, result = set(), []
        for item in frame:
            assert set(item) == {'id', 'box', 'confidence'}
            assert type(item['id']) is int and item['id'] >= 0 and item['id'] not in seen
            seen.add(item['id'])
            finite_box(item['box'])
            assert type(item['confidence']) in (int, float) and math.isfinite(item['confidence']) and 0 < item['confidence'] <= 1
            result.append({'sourceTrackId': item['id'], 'bbox': list(item['box']),
                           'confidence': item['confidence'], 'courtRole': None, 'participantStatus': 'UNKNOWN'})
        observations.append(result)
    return observations


def iou(a, b):
    finite_box(a)
    finite_box(b)
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def match_counts(predicted, targets):
    # Independent maximum-cardinality bipartite augmenting paths. No SciPy or
    # production matcher, and no use of persisted matches as the answer.
    edges = [[j for j, target in enumerate(targets) if iou(box, target) >= .5] for box in predicted]
    owners = {}

    def augment(index, visited):
        for target in edges[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in owners or augment(owners[target], visited):
                owners[target] = index
                return True
        return False

    matched = sum(augment(index, set()) for index in range(len(predicted)))
    return matched, len(predicted) - matched, len(targets) - matched


def publisher_gt(path, count):
    frames, seen = [[] for _ in range(count)], set()
    with Path(path).open(newline='') as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            assert len(row) == 9
            values = [float(value) for value in row]
            assert all(math.isfinite(value) for value in values)
            frame, identity, category = [int(values[index]) for index in (0, 1, 7)]
            assert all(values[index] == int(values[index]) for index in (0, 1, 7))
            assert 1 <= frame <= count and identity >= 0 and 1 <= category <= 7
            assert values[4] > 0 and values[5] > 0 and (frame, identity) not in seen
            assert values[6] == 1 and 0 <= values[8] <= 1
            seen.add((frame, identity))
            x, y, width, height = values[2:6]
            frames[frame - 1].append({'id': identity, 'class': category,
                                     'box': [x, y, x + width, y + height], 'visibility': values[8]})
    return frames


def project(matrix, point):
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert all(type(value) in (int, float) and math.isfinite(value) for row in matrix for value in row)
    x, y = point
    divisor = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(divisor) <= 1e-8:
        return None
    result = [(matrix[axis][0] * x + matrix[axis][1] * y + matrix[axis][2]) / divisor for axis in (0, 1)]
    return result if all(math.isfinite(value) for value in result) else None


def reconstruct_selection(observations, camera, fps, protocol, saved_audit):
    observed, support = defaultdict(list), defaultdict(list)
    for index, people in enumerate(observations):
        for person in people:
            identity, box = person['sourceTrackId'], person['bbox']
            observed[identity].append(index)
            record = camera[index]
            if not record['isValid']:
                continue
            assert record['frame_index'] == index and close(record['timestamp_seconds'], index / fps)
            foot = project(record['frameToAnchor'], [(box[0] + box[2]) / 2, box[3]])
            top = project(record['frameToAnchor'], [(box[0] + box[2]) / 2, box[1]])
            if foot is None or top is None or math.dist(foot, top) <= 1:
                continue
            support[identity].append((index, foot, math.dist(foot, top)))
    expected_statistics = {}
    for identity, indices in observed.items():
        samples = support[identity]
        if samples:
            horizontal = max(row[1][0] for row in samples) - min(row[1][0] for row in samples)
            vertical = max(row[1][1] for row in samples) - min(row[1][1] for row in samples)
            extent = math.hypot(horizontal, vertical) / median(row[2] for row in samples)
            span = (samples[-1][0] - samples[0][0]) / fps
        else:
            extent, span = None, 0
        eligible = (len(indices) / fps >= protocol['minimum_observed_seconds']
                    and len(samples) >= protocol['minimum_camera_samples']
                    and span >= protocol['minimum_motion_span_seconds']
                    and extent is not None and extent >= protocol['minimum_motion_extent_heights'])
        expected_statistics[str(identity)] = {'observedFrames': len(indices), 'cameraSupportedSamples': len(samples),
            'motionSpanSeconds': span, 'motionExtentBodyHeights': extent,
            'eligibleMovingPersonCandidate': eligible, 'identityScope': 'SOURCE_TRACK_FRAGMENT_NOT_VERIFIED_PERSON_ID'}
    actual_statistics = saved_audit['statistics']
    assert set(actual_statistics) == set(expected_statistics)
    for identity, expected in expected_statistics.items():
        assert set(actual_statistics[identity]) == set(expected)
        assert all(close(actual_statistics[identity][key], value) for key, value in expected.items()), identity
    expected_selected, ambiguity = [], []
    for index, people in enumerate(observations):
        current = [person for person in people if expected_statistics[str(person['sourceTrackId'])]['eligibleMovingPersonCandidate']]
        if len(current) > 2:
            expected_selected.append([])
            ambiguity.append(index)
        else:
            expected_selected.append(current)
    assert saved_audit['ambiguousFrames'] == ambiguity
    assert saved_audit['qualificationEvidence'] is False and saved_audit['identityMerging'] is False and saved_audit['gapFilling'] is False
    return expected_selected, ambiguity, expected_statistics


def verify_camera(camera, count, fps, size, protocol):
    expected_indices = set(range(0, count, protocol['sampleFrameStride'])) | {count - 1}
    accepted = 0
    assert len(camera) == count
    for index, record in enumerate(camera):
        assert record['frame_index'] == index and close(record['timestamp_seconds'], index / fps)
        assert record['sampled'] is (index in expected_indices)
        if index not in expected_indices:
            assert record['isValid'] is False and record['frameToAnchor'] is None
            assert record['rejectionReason'] == 'NOT_SAMPLED_NO_INTERPOLATION'
            continue
        width, height = size
        reference_width = protocol['referenceWidth']
        reference_height = round(height * reference_width / width)
        assert record['nativeSize'] == size and record['referenceSize'] == [reference_width, reference_height]
        assert len(record['nativeBgrPixelSha256']) == 64
        if index == 0:
            assert record['anchorIdentityException'] is True and record['isValid'] is True
            assert record['frameToAnchor'] == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            accepted += 1
            continue
        if 'reducedTransform' in record:
            transform = record['reducedTransform']
            assert len(transform) == 3 and all(len(row) == 3 for row in transform)
            assert all(math.isfinite(value) for row in transform for value in row)
            assert close(transform[0][0], transform[1][1]) and close(transform[0][1], -transform[1][0])
            assert transform[2] == [0., 0., 1.]
            assert 0 < record['inliers'] <= record['matches']
            assert close(record['inlierFraction'], record['inliers'] / record['matches'])
            assert record['meanResidualReferencePx'] >= 0
            assert len(record['matchedSpanFraction']) == 2 and all(value >= 0 for value in record['matchedSpanFraction'])
            gates = (record['inliers'] >= protocol['minimumInliers']
                     and record['inlierFraction'] >= protocol['minimumInlierFraction']
                     and record['meanResidualReferencePx'] <= protocol['maximumMeanInlierResidualReferencePx']
                     and all(value >= minimum for value, minimum in zip(record['matchedSpanFraction'], protocol['minimumCurrentInlierSpanFraction'])))
            assert record['isValid'] is bool(gates)
        else:
            assert record['isValid'] is False and record['frameToAnchor'] is None
        if record['isValid']:
            scale = [width / reference_width, height / reference_height, 1.]
            expected = [[scale[i] * record['reducedTransform'][i][j] / scale[j] for j in range(3)] for i in range(3)]
            assert all(close(record['frameToAnchor'][i][j], expected[i][j]) for i in range(3) for j in range(3))
            assert record['nativeConjugationScale'] == [[scale[0], 0., 0.], [0., scale[1], 0.], [0., 0., 1.]]
            assert record['rejectionReason'] is None
            accepted += 1
        else:
            assert record['frameToAnchor'] is None and record['rejectionReason']
    return {'sampled': len(expected_indices), 'acceptedIncludingIdentityAnchor': accepted}


def png(path, decode_pixels):
    data = Path(path).read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    offset, compressed, width, height, bit_depth, color = 8, bytearray(), None, None, None, None
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        checksum = struct.unpack('>I', data[offset + 8 + length:offset + 12 + length])[0]
        assert len(payload) == length and (zlib.crc32(kind + payload) & 0xffffffff) == checksum
        offset += length + 12
        if kind == b'IHDR':
            width, height, bit_depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            assert bit_depth == 8 and color == 2 and compression == 0 and filtering == 0 and interlace == 0
        elif kind == b'IDAT':
            compressed.extend(payload)
        elif kind == b'IEND':
            assert offset == len(data)
            break
    assert width and height and compressed
    if not decode_pixels:
        return [width, height], None
    unpacked = zlib.decompress(compressed)
    stride = width * 3
    assert len(unpacked) == height * (stride + 1)
    prior = bytearray(stride)
    pixel_hash = hashlib.sha256()
    for index in range(height):
        start = index * (stride + 1)
        filter_type = unpacked[start]
        current = bytearray(unpacked[start + 1:start + 1 + stride])
        assert filter_type in range(5)
        for position in range(stride):
            left = current[position - 3] if position >= 3 else 0
            up = prior[position]
            upper_left = prior[position - 3] if position >= 3 else 0
            if filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = up
            elif filter_type == 3:
                prediction = (left + up) // 2
            elif filter_type == 4:
                value = left + up - upper_left
                distances = (abs(value - left), abs(value - up), abs(value - upper_left))
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            else:
                prediction = 0
            current[position] = (current[position] + prediction) & 255
        bgr = bytearray(stride)
        bgr[0::3], bgr[1::3], bgr[2::3] = current[2::3], current[1::3], current[0::3]
        pixel_hash.update(bgr)
        prior = current
    return [width, height], pixel_hash.hexdigest()


def product_snapshot():
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        jobs = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    files = {path.relative_to(ROOT).as_posix(): binding(path)
             for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
    return {'jobs': jobs, 'files': files}


def exact_identity_retired(identity):
    assert os.name == 'nt', 'Exact terminal identity check requires Windows'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    from ctypes import wintypes
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, identity['pid'])
    if not handle:
        error = ctypes.get_last_error()
        assert error == 87, 'Cannot establish retirement: OpenProcess error ' + str(error)
        return {'pid': identity['pid'], 'createTimeUnixSeconds': identity['createTimeUnixSeconds'], 'retired': True,
                'method': 'Exact PID absent; ERROR_INVALID_PARAMETER'}
    try:
        creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time))
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        observed_creation = ticks / 10000000 - 11644473600
        if abs(observed_creation - identity['createTimeUnixSeconds']) > 1e-6:
            return {'pid': identity['pid'], 'createTimeUnixSeconds': identity['createTimeUnixSeconds'], 'retired': True,
                    'method': 'PID reused by different creation time', 'currentCreateTimeUnixSeconds': observed_creation}
        exit_code = wintypes.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        assert exit_code.value != 259, 'Exact owned identity remains active'
        return {'pid': identity['pid'], 'createTimeUnixSeconds': identity['createTimeUnixSeconds'], 'retired': True,
                'method': 'Exact creation-time handle reports process exited', 'exitCode': exit_code.value}
    finally:
        assert kernel.CloseHandle(handle)


def resource_checks(resource, report, protocol):
    sample_rows = rows(RUN / 'samples.jsonl')
    identity = load(RUN / 'actual_identity.json')
    acknowledgement = load(RUN / 'actual_sampled.json')
    assert acknowledgement['identity'] == identity == report['actualIdentity']
    assert acknowledgement['sampledByExactIdentity'] is True
    assert resource['samples'] == len(sample_rows) and len(sample_rows) > 0
    assert resource['exitCode'] == 0 and resource['allOwnedIdentitiesRetired'] is True
    assert resource['remainingOwnedIdentities'] == [] and not resource.get('safeguardStop')
    owned = {(row['pid'], row['createTimeUnixSeconds']): row for row in resource['ownedIdentities']}
    assert (identity['pid'], identity['createTimeUnixSeconds']) in owned
    launcher = resource['launcherIdentity']
    assert (launcher['pid'], launcher['createTimeUnixSeconds']) in owned
    seen = defaultdict(list)
    peak, minimum, low = 0, math.inf, 0
    acknowledged_at = datetime.fromisoformat(acknowledgement['at'])
    exact_samples_before_ack = 0
    for index, sample in enumerate(sample_rows):
        if index:
            assert sample['elapsedSeconds'] >= sample_rows[index - 1]['elapsedSeconds']
        assert sample['elapsedSeconds'] <= protocol['wallSeconds']
        process_rows = sample['ownedProcesses']
        keys = [(row['pid'], row['createTimeUnixSeconds']) for row in process_rows]
        assert len(keys) == len(set(keys)) and all(key in owned for key in keys)
        total = sum(row['memoryBytes']['rss'] for row in process_rows)
        assert total == sample['ownedTreeRssBytes'] and total <= protocol['maximumOwnedRssBytes']
        available = sample['availableRamBytes']
        peak, minimum = max(peak, total), min(minimum, available)
        low = low + 1 if available < protocol['minimumAvailableBytes'] else 0
        assert low < protocol['lowSamples']
        for row, key in zip(process_rows, keys):
            assert row['depth'] == owned[key]['depth'] and row['parentPid'] == owned[key]['parentPid']
            seen[key].append(row)
        if (identity['pid'], identity['createTimeUnixSeconds']) in keys and datetime.fromisoformat(sample['at']) <= acknowledged_at:
            exact_samples_before_ack += 1
    assert exact_samples_before_ack >= 1
    assert resource['peakOwnedTreeRssBytes'] == peak
    # Preflight RAM is not saved separately; its possible contribution to the
    # terminal minimum cannot be independently reconstructed from samples.
    assert resource['minimumAvailableRamBytes'] <= minimum
    assert resource['minimumAvailableRamBytes'] >= protocol['preflightAvailableBytes']
    for key, record in owned.items():
        assert len(seen[key]) == record['sampleCount']
        assert max(row['memoryBytes']['rss'] for row in seen[key]) == record['peakSampledRssBytes']
        private = [row['memoryBytes'].get('private') for row in seen[key]]
        if all(value is not None for value in private):
            assert max(private) == record['peakSampledPrivateBytes']
        if record['depth'] == 0:
            assert key == (launcher['pid'], launcher['createTimeUnixSeconds']) and record['parentPid'] is None
        else:
            assert any(parent['pid'] == record['parentPid'] and parent['depth'] + 1 == record['depth'] for parent in owned.values())
    assert resource['elapsedSeconds'] < protocol['wallSeconds']
    retired = [exact_identity_retired(record) for record in owned.values()]
    return {'sampleCount': len(sample_rows), 'exactActualSamplesBeforeAcknowledgement': exact_samples_before_ack,
            'peakAggregateOwnedRssBytes': peak, 'minimumSampledAvailableRamBytes': minimum,
            'recordedMinimumAvailableRamBytes': resource['minimumAvailableRamBytes'],
            'preflightMinimumLimitation': 'Initial preflight sample not separately persisted; terminal minimum and sampled bounds verified',
            'exactRetirementChecks': retired}


def source_order_check():
    source = ast.parse((UNIT / 'experiment.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    child_body = functions['child'].body
    protected_body = next(node.body for node in child_body if isinstance(node, ast.Try))
    select_indices, score_indices = [], []
    for index, node in enumerate(protected_body):
        for descendant in ast.walk(node):
            if not isinstance(descendant, ast.Call) or not isinstance(descendant.func, ast.Name):
                continue
            if descendant.func.id == 'select_sequence':
                select_indices.append(index)
            if descendant.func.id == 'score_sequence':
                score_indices.append(index)
    assert len(select_indices) == len(score_indices) == 1 and select_indices[0] < score_indices[0]
    read_gt_functions = []
    for function_name, function in functions.items():
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'read_gt' for node in ast.walk(function)):
            read_gt_functions.append(function_name)
    assert read_gt_functions == ['score_sequence']
    return {'staticSemanticGroundTruthReadOnlyInScoring': True, 'allSequenceSelectionLoopPrecedesScoringLoop': True,
            'limitation': 'Frozen source ordering and staged receipts verify intended execution order; no independent instrumented read trace'}


def sequence_check(sequence, report, stage, old_record, info, timing, protocol):
    count = info['publisher_frames']
    result = report['sequences'][sequence]
    assert result['decodedFrames'] == count and result['nativeSize'] == info['size']
    fps = float(Fraction(timing['stream']['avg_frame_rate']))
    assert result['fps'] == old_record['fps'] == fps
    assert timing['timestamp_count'] == count and timing['strictly_increasing'] is True
    assert len(timing['frame_timestamps_seconds']) == count
    assert all(abs(value - index / fps) < .001 for index, value in enumerate(timing['frame_timestamps_seconds']))
    assert result['fpsFraction'] == timing['stream']['avg_frame_rate']
    assert result['qualificationEvidence'] is False and result['sourceObservationIntegrity'] is True
    assert result['semanticGroundTruthReadBeforeSelection'] is False
    assert result['source'] == (DATASET / info['video']).relative_to(ROOT).as_posix()
    assert binding(resolve(result['source'])) == result['sourceBinding']
    assert result['cachedProposals'] == (CACHE / old_record['predictions_file']).relative_to(ROOT).as_posix()
    assert binding(resolve(result['cachedProposals'])) == result['cachedProposalsBinding']
    assert result['cachedProposalsBinding']['sha256'] == old_record['predictions_sha256']
    for key in ('camera', 'selected', 'selectionAudit'):
        verify_record(result[key])
        assert stage['sequences'][sequence][key] == result[key]
    assert stage['sequences'][sequence]['semanticGroundTruthReadBeforeSelection'] is False
    observations = adapt(load(resolve(result['cachedProposals'])))
    assert len(observations) == count
    directory = RUN / sequence
    camera = rows(directory / 'camera.jsonl')
    camera_check = verify_camera(camera, count, fps, info['size'], protocol['camera'])
    assert camera_check['sampled'] == result['cameraSamples']
    assert camera_check['acceptedIncludingIdentityAnchor'] == result['acceptedCameraSamplesIncludingIdentityAnchor']
    saved_selection_audit = load(directory / 'selection_audit.json')
    selected, ambiguous, statistics = reconstruct_selection(observations, camera, fps, protocol['selector'], saved_selection_audit)
    saved_selected = rows(directory / 'selected.jsonl')
    assert len(saved_selected) == count
    for index, row in enumerate(saved_selected):
        assert row['frame_index'] == index and close(row['timestamp_seconds'], index / fps)
        assert row['people'] == selected[index]
    assert sum(len(frame) for frame in selected) == result['selectedActualObservations']
    labels = (DATASET / 'UVY' / sequence / 'gt/labels.txt').read_text(encoding='utf-8').splitlines()
    assert labels == ['player', 'sports ball', 'goal', 'referee', 'goalkeeper', 'pcamera', 'person playing']
    gt_path = DATASET / 'UVY' / sequence / 'gt/gt.txt'
    gt = publisher_gt(gt_path, count)
    scoring = rows(directory / 'scoring.jsonl')
    assert len(scoring) == len(old_record['per_frame_counts']) == count
    totals = dict.fromkeys(COUNTS, 0)
    reasons = {0: ['FIRST_FRAME'], count // 2: ['MIDDLE_FRAME'], count - 1: ['LAST_FRAME']}
    for defect in protocol['scoring']['knownLabelDefectsRetained']:
        if defect['sequence'] == sequence:
            reasons.setdefault(defect['frameIndexZeroBased'], []).append('KNOWN_ORIGINAL_LABEL_DEFECT_RETAINED')
    first_extra = first_lost = first_ambiguity = None
    for index, row in enumerate(scoring):
        assert row['frame_index'] == index and close(row['timestamp_seconds'], index / fps)
        players = [person for person in gt[index] if person['class'] == 1]
        referees = [person for person in gt[index] if person['class'] == 4]
        assert row['observations'] == observations[index] and row['selected'] == selected[index]
        assert row['originalPublisherPlayers'] == players and row['originalPublisherReferees'] == referees
        p_tp, p_fp, p_fn = match_counts([person['bbox'] for person in observations[index]], [person['box'] for person in players])
        s_tp, s_fp, s_fn = match_counts([person['bbox'] for person in selected[index]], [person['box'] for person in players])
        r_tp, _, _ = match_counts([person['bbox'] for person in selected[index]], [person['box'] for person in referees])
        historical = {'matched': p_tp, 'missed': p_fn, 'unmatched_proposals': p_fp}
        assert row['historicalProposalCountsReproduced'] == historical == old_record['per_frame_counts'][index]
        lost = p_tp - s_tp
        assert lost >= 0
        current = {'annotatedPlayers': len(players), 'proposalObservations': len(observations[index]),
                   'proposalMatched': p_tp, 'proposalMissed': p_fn, 'proposalPublisherUnmatched': p_fp,
                   'selectedMatched': s_tp, 'selectedPublisherUnmatched': s_fp, 'selectedMissed': s_fn,
                   'selectedRefereeOverlap': r_tp, 'selectorLostMatched': lost,
                   'selectedObservations': len(selected[index]), 'framesWithoutSelection': int(not selected[index]),
                   'ambiguityFrames': int(index in ambiguous)}
        assert row['counts'] == current
        assert row['scope'] == 'REUSED_FLAWED_PUBLISHER_LABEL_AGREEMENT_NOT_VALIDATED_PARTICIPANT_PRECISION'
        for key in COUNTS:
            totals[key] += current[key]
        if s_fp and first_extra is None:
            first_extra = index
            reasons.setdefault(index, []).append('FIRST_PUBLISHER_UNMATCHED_SELECTION_NOT_VALIDATED_FALSE_POSITIVE')
        if lost and first_lost is None:
            first_lost = index
            reasons.setdefault(index, []).append('FIRST_CACHED_PLAYER_MATCH_LOST_BY_SELECTOR')
        if index in ambiguous and first_ambiguity is None:
            first_ambiguity = index
            reasons.setdefault(index, []).append('FIRST_MORE_THAN_TWO_CANDIDATE_ABSTENTION')
    summary = load(directory / 'scoring_summary.json')
    assert summary == result['summary'] and summary['counts'] == totals
    verify_record(summary['groundTruth'])
    verify_record(summary['scoring'])
    assert summary['groundTruth']['path'] == gt_path.relative_to(ROOT).as_posix()
    assert summary['historicalProposalReplayPassedAllFrames'] is True
    expected = {'annotatedPlayers': old_record['annotated_player_boxes'], 'proposalMatched': old_record['matched_player_boxes'],
                'proposalMissed': old_record['missed_player_boxes'], 'proposalObservations': old_record['person_proposals'],
                'proposalPublisherUnmatched': old_record['unmatched_proposals_not_false_positives']}
    assert all(totals[key] == value for key, value in expected.items())
    expected_defects = [{**defect, 'retainedOriginalPlayerIds': [person['id'] for person in gt[defect['frameIndexZeroBased']] if person['class'] == 1]}
                       for defect in protocol['scoring']['knownLabelDefectsRetained'] if defect['sequence'] == sequence]
    assert summary['knownLabelDefectsRetained'] == expected_defects
    assert all(summary[key] is None for key in ('participantPrecision', 'participantIdentityAccuracy', 'independentGeneralizationAccuracy'))
    reviews = load(directory / 'review_receipt.json')
    assert reviews['humanVisualReviewComplete'] is False and reviews['qualificationEvidence'] is False
    assert reviews['originalPngsVerifiedLossless'] is True
    assert reviews['reviews'] == result['reviews']
    assert {row['frame_index'] for row in reviews['reviews']} == set(reasons)
    pixel_bindings = []
    for review in reviews['reviews']:
        index = review['frame_index']
        assert review['reasons'] == reasons[index] and review['nativeSize'] == info['size']
        assert review['sourceFrameMethod'] == 'Exact sequential decode from frame zero; no seek'
        for kind in ('original', 'overlay'):
            verify_record(review[kind])
            size, bgr_hash = png(resolve(review[kind]['path']), kind == 'original')
            assert size == info['size']
            if kind == 'original':
                assert bgr_hash == review['nativeBgrPixelSha256']
                if camera[index]['sampled']:
                    assert bgr_hash == camera[index]['nativeBgrPixelSha256']
        pixel_bindings.append({'frame_index': index, 'nativeSize': info['size'],
                               'originalPngDecodedBgrSha256': review['nativeBgrPixelSha256'], 'reasons': review['reasons']})
    return {'framesIndependentlyScored': count, 'counts': totals, 'camera': camera_check,
            'sourceFragmentsReconstructed': len(statistics),
            'eligibleMovingFragments': [identity for identity, value in statistics.items() if value['eligibleMovingPersonCandidate']],
            'allSelectedObservationsExactlyReconstructed': True, 'knownPublisherDefectsRetained': expected_defects,
            'nativePngReviewBindings': pixel_bindings, 'participantPrecision': None, 'qualificationEvidence': False}


def main():
    assert not OUT.exists(), 'Exclusive independent audit already consumed'
    OUT.mkdir()
    receipt = {'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
               'scope': 'Independent persisted-evidence arithmetic audit; not source video, camera or participant accuracy validation',
               'modelInferenceRun': False, 'videoDecodeRun': False, 'selectorInvoked': False,
               'qualificationEvidence': False, 'productionPromoted': False, 'fullExecutionObjectiveComplete': False,
               'auditSource': binding(Path(__file__)), 'originalTrialStatus': None}
    frozen_run = {path.relative_to(ROOT).as_posix(): binding(path) for path in RUN.rglob('*') if path.is_file()}
    receipt['terminalEvidenceBefore'] = frozen_run
    try:
        assert not any(module == 'torch' or module.startswith(('torch.', 'ultralytics', 'numpy', 'cv2', 'scipy')) for module in sys.modules)
        ast.parse(Path(__file__).read_text(encoding='utf-8'))
        preparation = load(UNIT / 'preparation.json')
        protocol = load(UNIT / 'protocol.json')
        report = load(RUN / 'report.json')
        resource = load(RUN / 'resource_receipt.json')
        receipt['originalTrialStatus'] = {'resourceStatus': resource['status'], 'resourcePassed': resource['passed'],
                                         'childComplete': report['complete'], 'childError': report.get('error'),
                                         'finalizationErrors': report.get('finalizationErrors')}
        assert resource['status'] == 'COMPLETED' and resource['passed'] is True and report['complete'] is True
        assert report['finalizationErrors'] == [] and 'error' not in report
        verify(preparation['bindings'])
        assert preparation['guards'] == {key: protocol['guards'][key] for key in preparation['guards']}
        assert preparation['semanticGroundTruthRead'] is False and preparation['modelInferenceRun'] is False and preparation['videoDecodeRun'] is False
        assert product_snapshot() == preparation['productBefore']
        assert len(preparation['productBefore']['jobs']) == 10 and len(preparation['productBefore']['files']) == 319
        verify(report['ordinarySourcesBefore'])
        verify(report['ordinarySourcesAfter'])
        assert report['ordinarySourcesBefore'] == report['ordinarySourcesAfter']
        assert report['modelInferenceRun'] is False and report['qualificationEvidence'] is False and report['productionPromoted'] is False
        assert report['sourcesUnchanged'] is True and report['savedProductUnchanged'] is True
        assert resource['sourcesUnchanged'] is True and resource['savedProductUnchanged'] is True
        assert report['allThreeSelectionsPersistedBeforeSemanticGroundTruth'] is True and report['semanticGroundTruthRead'] is True
        assert all(report[key] is None for key in ('participantPrecision', 'participantIdentityAccuracy', 'independentGeneralizationAccuracy'))
        receipt['semanticReadOrderChecks'] = source_order_check()
        receipt['resources'] = resource_checks(resource, report, protocol['guards'])
        stage = load(RUN / 'selection_stage_receipt.json')
        assert stage['semanticGroundTruthRead'] is False and stage['qualificationEvidence'] is False
        assert set(stage['sequences']) == set(report['sequences']) == set(SEQUENCES)
        manifest = load(DATASET / 'manifest.json')
        old = load(CACHE / 'report.json')
        timing = load(ROOT / 'artifacts/validation/vision_upgrade/uvy_tennis_timing_audit.json')
        assert old['mode'] == 'tracked640' and old['complete'] is True and old['qualification_evidence'] is False
        assert binding(DATASET / 'manifest.json')['sha256'] == old['dataset_manifest_sha256']
        receipt['sequences'] = {sequence: sequence_check(sequence, report, stage, old['sequences'][sequence],
                                                       manifest['sequences'][sequence], timing[sequence], protocol)
                                for sequence in SEQUENCES}
        assert sum(row['framesIndependentlyScored'] for row in receipt['sequences'].values()) == 2230
        pooled = {key: sum(row['counts'][key] for row in receipt['sequences'].values()) for key in COUNTS}
        assert report['pooledCounts'] == pooled
        receipt['pooledCounts'] = pooled
        receipt['reviewFrames'] = sum(len(row['nativePngReviewBindings']) for row in receipt['sequences'].values())
        receipt['boundInputFiles'] = len(preparation['bindings'])
        receipt['boundLoadedModuleFiles'] = len(report['ordinarySourcesBefore'])
        receipt['preservedProductJobs'] = 10
        receipt['preservedProductFiles'] = 319
        verify(preparation['bindings'])
        verify(frozen_run)
        assert product_snapshot() == preparation['productBefore']
        assert binding(Path(__file__)) == receipt['auditSource']
        receipt['passed'] = True
        receipt['limitations'] = [
            'Persisted camera consensus statistics and transform arithmetic are verified; background feature matches were not rerun or independently validated.',
            'PNG bytes, CRCs, native dimensions and decoded original pixel hashes are verified; source-video decoding is not independently repeated.',
            'Publisher labels remain exposed, incomplete and flawed; agreement and publisher-unmatched counts are not participant precision or independent accuracy.',
            'Source read ordering is verified statically and by staged receipts; no independent instrumented semantic file-read trace exists.',
            'Sampled RSS guards do not measure every transient allocation or full-match inference; private bytes may exceed RSS and were not a declared stop guard.'
        ]
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        receipt['traceback'] = traceback.format_exc()
    finally:
        receipt['completedAt'] = datetime.now(timezone.utc).isoformat()
        try:
            verify(frozen_run)
            receipt['originalTerminalEvidenceUnchanged'] = True
        except BaseException as error:
            receipt['passed'] = False
            receipt['terminalEvidencePreservationError'] = repr(error)
        with (OUT / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'passed': receipt['passed'], 'error': receipt.get('error'),
                      'framesScored': sum(row['framesIndependentlyScored'] for row in receipt.get('sequences', {}).values()),
                      'reviewFrames': receipt.get('reviewFrames'), 'originalTrialStatus': receipt['originalTrialStatus']}), flush=True)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
