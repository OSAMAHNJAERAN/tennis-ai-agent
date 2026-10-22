"""Independent persisted surface-probe arithmetic and grayscale-mask audit."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
OUT = UNIT / 'audit'
OLD = ROOT / 'artifacts/validation/product_execution/participant_motion_uvy_2026-10-07'
DATASET = ROOT / 'data/external/uvy_tennis_videos'
CACHE = ROOT / 'outputs/vision_upgrade_audit/uvy_person_tracked640'
HELPER = OLD / 'audit.py'
HELPER_SHA = 'a49cb19c96a25adb249cf279f3a53ae43d0e759bfe8a1772f66d2a96740e38bd'
COMPARISONS = ('cachedProposals', 'cachedMotion', 'surfaceSupported', 'motionSurfaceDiagnosticIntersection')


def helper():
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    specification = importlib.util.spec_from_file_location('_independent_evidence_arithmetic', HELPER)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    module.RUN = RUN
    return module


def gray_png(path):
    # Independent byte-width-one PNG filtering, retaining values 0 and 1.
    data = Path(path).read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    offset, compressed, size, ended = 8, bytearray(), None, False
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        checksum = struct.unpack('>I', data[offset + 8 + length:offset + 12 + length])[0]
        assert len(payload) == length and zlib.crc32(kind + payload) & 0xffffffff == checksum
        offset += length + 12
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', payload)
            assert size is None and depth == 8 and color == 0 and compression == filtering == interlace == 0
            size = [width, height]
        elif kind == b'IDAT':
            compressed.extend(payload)
        elif kind == b'IEND':
            assert not payload and offset == len(data)
            ended = True
            break
    assert size and compressed and ended
    width, height = size
    raw = zlib.decompress(compressed)
    assert len(raw) == height * (width + 1)
    prior, pixels = bytearray(width), bytearray()
    for index in range(height):
        start = index * (width + 1)
        method, current = raw[start], bytearray(raw[start + 1:start + 1 + width])
        assert method in range(5)
        for column in range(width):
            left = current[column - 1] if column else 0
            up = prior[column]
            upper_left = prior[column - 1] if column else 0
            if method == 1:
                prediction = left
            elif method == 2:
                prediction = up
            elif method == 3:
                prediction = (left + up) // 2
            elif method == 4:
                p = left + up - upper_left
                distances = (abs(p - left), abs(p - up), abs(p - upper_left))
                prediction = (left, up, upper_left)[distances.index(min(distances))]
            else:
                prediction = 0
            current[column] = (current[column] + prediction) & 255
        pixels.extend(current)
        prior = current
    assert set(pixels) <= {0, 1}
    return size, pixels


def context(h, box, mask, native_size, reduced_size, has_surface):
    h.finite_box(box)
    width, height = box[2] - box[0], box[3] - box[1]
    center = [(box[0] + box[2]) / 2, box[3]]
    native = [[center[0] + x * width, center[1] + y * height]
              for y in (-.03, 0., .03) for x in (-.2, -.1, 0., .1, .2)]
    # Python round and numpy.rint both implement round-to-nearest/ties-to-even;
    # multiply by the frozen ratio before rounding, matching the protocol.
    reduced = [[point[axis] * (reduced_size[axis] / native_size[axis]) for axis in (0, 1)] for point in native]
    integer = [[int(round(value)) for value in point] for point in reduced]
    bounds = (all(0 <= point[axis] < native_size[axis] for point in native for axis in (0, 1))
              and all(0 <= point[axis] < reduced_size[axis] for point in integer for axis in (0, 1)))
    result = {'nativeLocations': native, 'reducedLocations': reduced,
              'reducedNearestIntegerLocations': integer, 'all15InBounds': bounds,
              'sampleCount': 15, 'componentValues': None, 'insideCount': None, 'insideFraction': None,
              'courtRole': None, 'participantStatus': 'UNKNOWN'}
    if not bounds:
        result['status'] = 'UNKNOWN_OUT_OF_BOUNDS'
    elif not has_surface:
        result['status'] = 'UNKNOWN_NO_ELIGIBLE_SURFACE'
    else:
        values = [mask[point[1] * reduced_size[0] + point[0]] for point in integer]
        result.update(componentValues=values, insideCount=sum(values), insideFraction=sum(values) / 15)
        result['status'] = ('SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE' if sum(values) / 15 >= .5 else 'REJECTED_SURFACE_CONTEXT')
    return result


def context_equal(h, actual, expected):
    assert set(actual) == set(expected)
    for key in ('nativeLocations', 'reducedLocations'):
        assert len(actual[key]) == len(expected[key]) == 15
        assert all(len(point) == 2 for point in actual[key])
        assert all(h.close(actual[key][i][j], expected[key][i][j]) for i in range(15) for j in (0, 1))
    for key in set(expected) - {'nativeLocations', 'reducedLocations', 'insideFraction'}:
        assert actual[key] == expected[key], key
    assert h.close(actual['insideFraction'], expected['insideFraction'])


def optimal_agreement(h, boxes, targets, stored):
    tp, fp, fn = h.match_counts(boxes, targets)
    # A second independent bitmask dynamic program verifies the maximum sum
    # of IoU among matchings of maximum cardinality, without SciPy.
    states = {0: 0.}
    valid_ious = []
    for box in boxes:
        edges = [(index, h.iou(box, target)) for index, target in enumerate(targets)]
        edges = [(index, value) for index, value in edges if value >= .5]
        valid_ious.extend(value for _, value in edges)
        updated = dict(states)
        for mask, reward in states.items():
            for index, value in edges:
                if not mask & (1 << index):
                    new_mask = mask | (1 << index)
                    updated[new_mask] = max(updated.get(new_mask, -math.inf), reward + value)
        states = updated
    best_mask, best_sum = max(states.items(), key=lambda pair: (pair[0].bit_count(), pair[1]))
    assert best_mask.bit_count() == tp
    expected = {'matchedOriginalClass1': tp, 'publisherUnmatched': fp, 'missedOriginalClass1': fn,
                'proposalCount': len(boxes), 'participantPrecision': None}
    assert {key: stored[key] for key in expected} == expected
    assert len(stored['matchedIous']) == tp and all(.5 <= value <= 1 for value in stored['matchedIous'])
    assert h.close(sum(stored['matchedIous']), best_sum)
    assert all(any(h.close(value, candidate) for candidate in valid_ious) for value in stored['matchedIous'])
    return {key: expected[key] for key in ('matchedOriginalClass1', 'publisherUnmatched', 'missedOriginalClass1', 'proposalCount')}


def verify_overlap(h, item, players, referees):
    for category, group in ((1, players), (4, referees)):
        key = f'originalPublisherClass{category}Overlaps'
        expected = [{'id': row['id'], 'iou': h.iou(item['person']['bbox'], row['box'])} for row in group]
        assert len(item[key]) == len(expected)
        assert all(a['id'] == b['id'] and h.close(a['iou'], b['iou']) for a, b in zip(item[key], expected))
        assert item[f'anyOriginalClass{category}IouAtLeastHalf'] is any(row['iou'] >= .5 for row in expected)


def ordering_check():
    source = ast.parse((UNIT / 'experiment.py').read_text(encoding='utf-8'))
    functions = {node.name: node for node in source.body if isinstance(node, ast.FunctionDef)}
    protected = next(node.body for node in functions['child'].body if isinstance(node, ast.Try))
    pixel, score = [], []
    for index, node in enumerate(protected):
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                if call.func.id == 'pixel_phase':
                    pixel.append(index)
                if call.func.id == 'score_phase':
                    score.append(index)
    assert len(pixel) == len(score) == 1 and pixel[0] < score[0]
    reads = [name for name, function in functions.items() if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
             and call.func.attr == 'read_gt' for call in ast.walk(function))]
    assert reads == ['score_phase']
    assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'VideoCapture' for call in ast.walk(source))
    pixel_source = ast.get_source_segment((UNIT / 'experiment.py').read_text(encoding='utf-8'), functions['pixel_phase'])
    assert "candidates[0]['mask']" in pixel_source
    assert all(token not in pixel_source for token in ('fillConvexPoly', 'fillPoly', 'convexHull', 'read_gt'))
    return {'frozenSourceAllPixelDecisionsPrecedePublisherCsvParsing': True,
            'maskObtainedDirectlyFromFirstActualExtractorComponent': True,
            'noVideoDecodeInHarness': True, 'instrumentedFileReadTraceAvailable': False}


def verify_components(h, record, pixels):
    width, height = record['reducedSize']
    candidates = record['candidateComponents']
    area = sum(pixels)
    assert record['mask']['insidePixels'] == area
    assert record['mask']['shape'] == [height, width]
    assert record['mask']['rawArraySha256'] == hashlib.sha256(pixels).hexdigest()
    assert record['mask']['isNoCandidatePlaceholder'] is (not bool(candidates))
    assert record['selectedComponentIndex'] == (0 if candidates else None)
    assert [row['area'] for row in candidates] == sorted((row['area'] for row in candidates), reverse=True)
    for candidate in candidates:
        assert type(candidate['area']) is int and candidate['area'] >= width * height * .025
        assert candidate['centroid'][1] >= height * .3 and candidate['solidity'] >= .6
        assert candidate['hue_center'] in range(0, 180, 15)
        hull = candidate['hull']
        assert len(hull) >= 3 and all(len(point) == 2 for point in hull)
        assert all(type(value) is int for point in hull for value in point)
        assert all(0 <= point[0] < width and 0 <= point[1] < height for point in hull)
        twice_area = abs(sum(point[0] * hull[(index + 1) % len(hull)][1] - point[1] * hull[(index + 1) % len(hull)][0]
                             for index, point in enumerate(hull)))
        assert h.close(candidate['solidity'], candidate['area'] / max(twice_area / 2, 1))
    if candidates:
        assert area == candidates[0]['area'] and area > 0
        x_total = y_total = 0
        for position, value in enumerate(pixels):
            if value:
                x_total += position % width
                y_total += position // width
        assert h.close(candidates[0]['centroid'][0], x_total / area)
        assert h.close(candidates[0]['centroid'][1], y_total / area)
    else:
        assert area == 0


def frame_check(h, pixel, scored, frozen, cached, prior_motion, prior_scoring, gt):
    sequence, index = pixel['sequence'], pixel['frame_index']
    assert (sequence, index) == (frozen['sequence'], frozen['frame_index'])
    for key, value in frozen.items():
        assert pixel[key] == value
    assert scored['semanticGroundTruthRead'] is True and pixel['semanticGroundTruthRead'] is False
    assert scored['qualificationEvidence'] is False and pixel['qualificationEvidence'] is False
    directory = ROOT / pixel['directory']
    assert directory.resolve().is_relative_to(RUN) and directory.is_dir()
    assert h.load(directory / 'pixel_decisions.json') == pixel
    assert h.load(directory / 'scored_context.json') == scored
    for key, value in pixel.items():
        if key == 'proposals':
            assert len(scored[key]) == len(value)
            for left, right in zip(value, scored[key]):
                assert all(right[subkey] == item for subkey, item in left.items())
        elif key != 'semanticGroundTruthRead':
            assert scored[key] == value
    original = ROOT / pixel['nativeOriginal']
    assert original.resolve().is_relative_to(OLD) and original.is_file()
    h.verify_record(pixel['originalCopy'])
    assert h.binding(original) == {key: pixel['originalCopy'][key] for key in ('bytes', 'sha256')}
    native_size, bgr_hash = h.png(h.resolve(pixel['originalCopy']['path']), True)
    assert native_size == pixel['nativeSize'] and bgr_hash == pixel['nativeBgrPixelSha256']
    assert pixel['reducedSize'] == [960, round(native_size[1] * 960 / native_size[0])]
    h.verify_record(pixel['mask'])
    mask_size, mask_pixels = gray_png(h.resolve(pixel['mask']['path']))
    assert mask_size == pixel['reducedSize']
    verify_components(h, pixel, mask_pixels)
    h.verify_record(scored['overlay'])
    assert h.png(h.resolve(scored['overlay']['path']), False)[0] == native_size
    assert h.png(directory / 'reduced_original.png', False)[0] == pixel['reducedSize']
    raw = cached[sequence][index]
    adapted = h.adapt([raw])[0]
    motion = prior_motion[sequence][index]['people']
    previous = prior_scoring[sequence][index]
    assert adapted == previous['observations'] and motion == previous['selected']
    assert len(pixel['proposals']) == len(raw)
    assert [row['originalCachedProposal'] for row in pixel['proposals']] == raw
    assert [row['person'] for row in pixel['proposals']] == adapted
    statuses, repeated = Counter(), []
    for item in pixel['proposals']:
        expected_context = context(h, item['person']['bbox'], mask_pixels, native_size, mask_size, bool(pixel['candidateComponents']))
        context_equal(h, item['footContext'], expected_context)
        assert item['cachedMotionSelected'] is (item['person'] in motion)
        statuses[expected_context['status']] += 1
        grid = Counter(tuple(point) for point in expected_context['reducedNearestIntegerLocations'])
        if len(grid) < 15:
            repeated.append({'sourceTrackId': item['person']['sourceTrackId'], 'uniqueReducedGridPixels': len(grid),
                             'maximumPixelMultiplicity': max(grid.values()), 'samplesRetained': 15})
    assert [item['person'] for item in pixel['proposals'] if item['cachedMotionSelected']] == motion
    players = [row for row in gt[sequence][index] if row['class'] == 1]
    referees = [row for row in gt[sequence][index] if row['class'] == 4]
    assert scored['originalPublisherPlayers'] == previous['originalPublisherPlayers'] == players
    assert scored['originalPublisherReferees'] == previous['originalPublisherReferees'] == referees
    for item in scored['proposals']:
        verify_overlap(h, item, players, referees)
    assert len(scored['originalPublisherBoxContexts']) == len(players) + len(referees)
    for publisher, stored in zip(players + referees, scored['originalPublisherBoxContexts']):
        assert {key: stored[key] for key in publisher} == publisher
        context_equal(h, stored['footContext'], context(h, publisher['box'], mask_pixels, native_size, mask_size, bool(pixel['candidateComponents'])))
    supported = [item['person'] for item in pixel['proposals'] if item['footContext']['status'] == 'SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE']
    intersection = [item['person'] for item in pixel['proposals'] if item['cachedMotionSelected']
                    and item['footContext']['status'] == 'SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE']
    target_boxes = [row['box'] for row in players]
    groups = dict(zip(COMPARISONS, (adapted, motion, supported, intersection)))
    counts = {group: optimal_agreement(h, [row['bbox'] for row in people], target_boxes, scored['counts'][group])
              for group, people in groups.items()}
    assert counts['cachedProposals']['matchedOriginalClass1'] == previous['counts']['proposalMatched']
    assert counts['cachedMotion']['matchedOriginalClass1'] == previous['counts']['selectedMatched']
    if sequence == 'tennis_V01' and index == 604:
        assert players == [] and scored['knownLabelDefect'] == 'Original player annotation omission retained'
    elif sequence == 'tennis_V03' and index == 140:
        assert any(row['id'] == 9 for row in players)
        assert scored['knownLabelDefect'] == 'Original class1 identity9 chair-umpire annotation retained'
    else:
        assert scored['knownLabelDefect'] is None
    return {'sequence': sequence, 'frameIndexZeroBased': index, 'nativeSize': native_size,
            'decodedNativeBgrPixelSha256': bgr_hash, 'maskPixels': len(mask_pixels), 'insideMaskPixels': sum(mask_pixels),
            'rawMaskPixelSha256': hashlib.sha256(mask_pixels).hexdigest(), 'proposalStatuses': dict(statuses),
            'proposalsWithRepeatedReducedGridPixels': repeated, 'counts': counts,
            'knownLabelDefectRetained': scored['knownLabelDefect'], 'qualificationEvidence': False}


def main():
    assert not OUT.exists(), 'One-shot audit already consumed'
    OUT.mkdir()
    receipt = {'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
               'scope': 'Independent persisted pixel-context evidence audit; no extractor, models or video decoding',
               'qualificationEvidence': False, 'productionPromoted': False, 'fullExecutionObjectiveComplete': False,
               'extractorRerun': False, 'modelInferenceRun': False, 'videoDecodeRun': False, 'originalTrialStatus': None}
    h = helper()
    receipt['auditSource'] = h.binding(Path(__file__))
    receipt['independentArithmeticHelper'] = {'path': HELPER.relative_to(ROOT).as_posix(), **h.binding(HELPER)}
    frozen_run = {path.relative_to(ROOT).as_posix(): h.binding(path) for path in RUN.rglob('*') if path.is_file()}
    receipt['originalTerminalEvidenceBefore'] = frozen_run
    try:
        ast.parse(Path(__file__).read_text(encoding='utf-8'))
        assert not any(module == 'torch' or module.startswith(('torch.', 'ultralytics', 'numpy', 'cv2', 'scipy')) for module in sys.modules)
        prepared, protocol = h.load(UNIT / 'preparation.json'), h.load(UNIT / 'protocol.json')
        report, resource = h.load(RUN / 'report.json'), h.load(RUN / 'resource_receipt.json')
        receipt['originalTrialStatus'] = {'resourceStatus': resource['status'], 'resourcePassed': resource['passed'],
                                         'childComplete': report['complete'], 'childError': report.get('error'),
                                         'finalizationErrors': report.get('finalizationErrors')}
        assert resource['status'] == 'COMPLETED' and resource['passed'] is True and report['complete'] is True
        assert report['finalizationErrors'] == [] and 'error' not in report
        h.verify(prepared['bindings'])
        h.verify(report['ordinarySourcesBefore'])
        h.verify(report['ordinarySourcesAfter'])
        assert report['ordinarySourcesBefore'] == report['ordinarySourcesAfter']
        assert prepared['guards'] == {key: protocol['guards'][key] for key in prepared['guards']}
        for record in (prepared, report, resource):
            assert record['qualificationEvidence'] is False and record['modelInferenceRun'] is False and record['videoDecodeRun'] is False
        assert prepared['semanticGroundTruthRead'] is False and report['semanticGroundTruthRead'] is True
        assert report['allThirteenPixelDecisionsPersistedBeforeSemanticGroundTruth'] is True
        assert report['productionPromoted'] is False
        assert report['sourcesUnchanged'] is True and report['savedProductUnchanged'] is True
        assert resource['sourcesUnchanged'] is True and resource['savedProductUnchanged'] is True
        assert h.product_snapshot() == prepared['productBefore']
        assert len(prepared['productBefore']['jobs']) == 10 and len(prepared['productBefore']['files']) == 319
        assert all(report[key] is None for key in ('participantPrecision', 'participantIdentityAccuracy', 'independentGeneralizationAccuracy'))
        assert protocol['referenceWidth'] == 960 and protocol['candidateCount'] == 1
        assert protocol['footPatch']['xOffsetsBoxWidths'] == [-.2, -.1, 0, .1, .2]
        assert protocol['footPatch']['yOffsetsBoxHeights'] == [-.03, 0, .03] and protocol['footPatch']['minimumInsideFraction'] == .5
        receipt['sourceOrdering'] = ordering_check()
        receipt['resources'] = h.resource_checks(resource, report, protocol['guards'])
        stage = h.load(RUN / 'pixel_stage_receipt.json')
        assert h.binding(RUN / 'pixel_stage_receipt.json') == report['pixelStageReceipt']
        assert stage['semanticGroundTruthRead'] is False and stage['qualificationEvidence'] is False
        assert stage['frames'] == 13 and stage['candidateCount'] == 1
        assert h.binding(RUN / 'pixel_decisions.jsonl') == stage['decisions']
        pixels = h.rows(RUN / 'pixel_decisions.jsonl')
        assert len(pixels) == len(prepared['frames']) == len(report['frames']) == 13
        visual = h.load(OLD / 'visual_review.json')
        assert [(row['sequence'], row['frame_index']) for row in prepared['frames']] == [(row['sequence'], row['frameIndexZeroBased']) for row in visual['frames']]
        cached = {sequence: h.load(CACHE / (sequence + '_proposals.json')) for sequence in ('tennis_V01', 'tennis_V02', 'tennis_V03')}
        prior_motion = {sequence: {row['frame_index']: row for row in h.rows(OLD / 'run' / sequence / 'selected.jsonl')} for sequence in cached}
        prior_scoring = {sequence: {row['frame_index']: row for row in h.rows(OLD / 'run' / sequence / 'scoring.jsonl')} for sequence in cached}
        manifest = h.load(DATASET / 'manifest.json')
        gt = {sequence: h.publisher_gt(DATASET / 'UVY' / sequence / 'gt/gt.txt', manifest['sequences'][sequence]['publisher_frames']) for sequence in cached}
        receipt['frames'] = [frame_check(h, pixel, scored, frozen, cached, prior_motion, prior_scoring, gt)
                             for pixel, scored, frozen in zip(pixels, report['frames'], prepared['frames'])]
        pooled = {group: {key: sum(row['counts'][group][key] for row in receipt['frames'])
                          for key in ('matchedOriginalClass1', 'publisherUnmatched', 'missedOriginalClass1', 'proposalCount')}
                  for group in COMPARISONS}
        assert pooled == report['pooledCounts']
        receipt['pooledCounts'] = pooled
        statuses = Counter()
        for frame in receipt['frames']:
            statuses.update(frame['proposalStatuses'])
        receipt['proposalStatuses'] = dict(statuses)
        receipt['proposalsWithRepeatedReducedGridPixels'] = sum(len(frame['proposalsWithRepeatedReducedGridPixels']) for frame in receipt['frames'])
        receipt['boundInputFiles'] = len(prepared['bindings'])
        receipt['boundLoadedModuleFiles'] = len(report['ordinarySourcesBefore'])
        receipt['preservedProductJobs'] = 10
        receipt['preservedProductFiles'] = 319
        h.verify(prepared['bindings'])
        h.verify(frozen_run)
        assert h.product_snapshot() == prepared['productBefore']
        assert h.binding(Path(__file__)) == receipt['auditSource']
        assert h.binding(HELPER) == {key: receipt['independentArithmeticHelper'][key] for key in ('bytes', 'sha256')}
        receipt['passed'] = True
        receipt['limitations'] = [
            'Saved masks, 0/1 bytes, component geometry and protocol sampling are verified; HSV/smoothness extraction and actual semantic court membership are not rerun or validated.',
            'Repeated nearest-grid pixels retain their original multiplicity among fifteen samples; these samples are not fifteen independent measurements.',
            'Native PNG pixels are independently decoded and bound to previous saved source hashes; source video decoding is not repeated.',
            'Previously exposed visual comments and retrospective image selection remain known; staged receipts and frozen code establish publisher CSV parsing order without an instrumented read trace.',
            'Flawed publisher label agreement and unmatched counts do not establish participant precision, identity, independent generalization or physical court position.',
            'Sampled RSS and owned retirement are verified; transient allocations and full-match memory or inference are not qualified.'
        ]
    except BaseException as error:
        receipt['error'] = type(error).__name__ + ': ' + str(error)
        receipt['traceback'] = traceback.format_exc()
    finally:
        receipt['completedAt'] = datetime.now(timezone.utc).isoformat()
        try:
            h.verify(frozen_run)
            receipt['originalTerminalEvidenceUnchanged'] = True
        except BaseException as error:
            receipt['passed'] = False
            receipt['terminalEvidencePreservationError'] = repr(error)
        with (OUT / 'receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'passed': receipt['passed'], 'error': receipt.get('error'),
                      'frames': len(receipt.get('frames', [])), 'pooledCounts': receipt.get('pooledCounts'),
                      'originalTrialStatus': receipt['originalTrialStatus']}), flush=True)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
