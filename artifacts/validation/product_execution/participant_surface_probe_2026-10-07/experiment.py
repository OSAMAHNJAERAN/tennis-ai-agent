"""One guarded pixel-context probe of thirteen reused original PNGs; no models."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
OLD = ROOT / 'artifacts/validation/product_execution/participant_motion_uvy_2026-10-07'
DATASET = ROOT / 'data/external/uvy_tennis_videos'
CACHE = ROOT / 'outputs/vision_upgrade_audit/uvy_person_tracked640'
BASE = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe')
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
GUARDS = {'preflightAvailableBytes': 750000000, 'maximumOwnedRssBytes': 1000000000,
          'minimumAvailableBytes': 400000000, 'lowSamples': 5, 'sampleSeconds': .25, 'wallSeconds': 180}
SEQUENCES = ('tennis_V01', 'tennis_V02', 'tennis_V03')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.prepared')
    save(temporary, value)
    assert not path.exists(), 'Exclusive receipt already exists'
    temporary.rename(path)


def write_row(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()


def verify(bindings):
    for path, expected in bindings.items():
        target = Path(path) if Path(path).is_absolute() else ROOT / path
        assert binding(target) == expected, 'Frozen input changed: ' + path


def module_at(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def ordinary_sources():
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            result[name(path)] = binding(path)
    return result


def product_snapshot():
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    assert len(rows) == 10 and not any(row['status'] in ('UPLOADED', 'PROCESSING') for row in rows)
    files = {name(path): binding(path) for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
    assert len(files) == 319, 'Expected preserved ten-job 319-file product snapshot'
    return {'jobs': rows, 'files': files}


def frame_inputs():
    visual = load(OLD / 'visual_review.json')
    report = load(OLD / 'run/report.json')
    assert visual['allSavedNativeOriginalsAndOverlaysViewed'] and visual['reviewedFrames'] == 13
    assert len(visual['frames']) == 13 and report['complete']
    assert binding(OLD / 'run/report.json') == visual['reportBinding']
    assert visual['decision'] == 'REJECT_MOTION_ONLY_PARTICIPANT_PROMOTION'
    frames = []
    for row in visual['frames']:
        sequence, index = row['sequence'], row['frameIndexZeroBased']
        assert sequence in SEQUENCES and type(index) is int and index >= 0
        matches = [r for r in report['sequences'][sequence]['reviews'] if r['frame_index'] == index]
        assert len(matches) == 1
        original = ROOT / row['nativeOriginal']
        assert original.resolve().is_relative_to(OLD) and original.is_file()
        assert binding(original) == visual['imageBindings'][row['nativeOriginal']]
        assert binding(original) == {k: matches[0]['original'][k] for k in ('bytes', 'sha256')}
        frames.append({'sequence': sequence, 'frame_index': index, 'nativeOriginal': row['nativeOriginal'],
                       'nativeSize': matches[0]['nativeSize'],
                       'nativeBgrPixelSha256': matches[0]['nativeBgrPixelSha256'],
                       'previousVisualObservation': row['observation']})
    assert len({(r['sequence'], r['frame_index']) for r in frames}) == 13
    return frames


def input_bindings():
    protocol = load(UNIT / 'protocol.json')
    assert {key: protocol['guards'][key] for key in GUARDS} == GUARDS
    assert protocol['referenceWidth'] == 960 and protocol['candidateCount'] == 1
    assert protocol['footPatch']['xOffsetsBoxWidths'] == [-.2, -.1, 0, .1, .2]
    assert protocol['footPatch']['yOffsetsBoxHeights'] == [-.03, 0, .03]
    assert protocol['footPatch']['minimumInsideFraction'] == .5
    expected = [(protocol['surface']['source'], protocol['surface']['sha256']),
                (protocol['groundTruth']['publisherParser'], protocol['groundTruth']['publisherParserSha256']),
                (protocol['groundTruth']['matcher'], protocol['groundTruth']['matcherSha256']),
                (protocol['guards']['monitor'], protocol['guards']['monitorSha256'])]
    for path, digest in expected:
        assert binding(ROOT / path)['sha256'] == digest
    inputs = [Path(__file__), UNIT / 'protocol.json', BASE, MONITOR, DATASET / 'manifest.json',
              CACHE / 'report.json', OLD / 'visual_review.json', OLD / 'protocol.json',
              OLD / 'experiment.py', OLD / 'run/report.json', OLD / 'run/resource_receipt.json',
              OLD / 'audit/receipt.json',
              ROOT / 'scripts/evaluate/probe_court_surface_support.py',
              ROOT / 'scripts/evaluate/audit_uvy_videos.py', ROOT / 'src/evaluation/detection_metrics.py',
              ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml']
    manifest, cache_report = load(DATASET / 'manifest.json'), load(CACHE / 'report.json')
    assert manifest['complete'] and cache_report['complete'] and cache_report['mode'] == 'tracked640'
    assert binding(DATASET / 'manifest.json')['sha256'] == cache_report['dataset_manifest_sha256']
    for item in manifest['files']:
        path = DATASET / item['path']
        assert binding(path) == {k: item[k] for k in ('bytes', 'sha256')}
        inputs.append(path)
    for sequence in SEQUENCES:
        old_record = cache_report['sequences'][sequence]
        proposals = CACHE / old_record['predictions_file']
        assert proposals.parent == CACHE and proposals.name == sequence + '_proposals.json'
        assert binding(proposals)['sha256'] == old_record['predictions_sha256']
        inputs.extend([proposals, OLD / 'run' / sequence / 'selected.jsonl',
                       OLD / 'run' / sequence / 'scoring.jsonl', OLD / 'run' / sequence / 'camera.jsonl',
                       OLD / 'run' / sequence / 'selection_audit.json'])
    inputs.extend(ROOT / row['nativeOriginal'] for row in frame_inputs())
    inputs.extend(path for path in (ROOT / 'src').rglob('*.py'))
    return {name(path): binding(path) for path in inputs}


def prepare():
    assert not RUN.exists(), 'This one-shot unit has already been consumed'
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    frozen = input_bindings()
    save(UNIT / 'preparation.json', {'preparedAt': stamp(), 'bindings': frozen,
         'productBefore': product_snapshot(), 'guards': GUARDS, 'frames': frame_inputs(),
         'semanticGroundTruthRead': False, 'modelInferenceRun': False, 'videoDecodeRun': False,
         'qualificationEvidence': False, 'sourcePreparedOnly': True})
    print(json.dumps({'prepared': True, 'boundFiles': len(frozen), 'executionStarted': False}), flush=True)


def finite_box(box):
    assert isinstance(box, list) and len(box) == 4
    assert all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
    assert box[0] < box[2] and box[1] < box[3]
    return box


def foot_context(np, box, mask, native_size, reduced_size, has_surface):
    box = finite_box(box)
    width, height = box[2] - box[0], box[3] - box[1]
    center = [(box[0] + box[2]) / 2, box[3]]
    native = [[center[0] + x * width, center[1] + y * height]
              for y in [-.03, 0, .03] for x in [-.2, -.1, 0, .1, .2]]
    scales = np.asarray(reduced_size, dtype=float) / np.asarray(native_size, dtype=float)
    reduced = np.asarray(native, dtype=float) * scales
    indices = np.rint(reduced).astype(int)
    native_array = np.asarray(native)
    inside_bounds = bool((native_array >= 0).all() and (native_array < np.asarray(native_size)).all()
                         and (indices >= 0).all() and (indices < np.asarray(reduced_size)).all())
    result = {'nativeLocations': native, 'reducedLocations': reduced.tolist(),
              'reducedNearestIntegerLocations': indices.tolist(), 'all15InBounds': inside_bounds,
              'sampleCount': 15, 'componentValues': None, 'insideCount': None, 'insideFraction': None,
              'courtRole': None, 'participantStatus': 'UNKNOWN'}
    if not inside_bounds:
        result['status'] = 'UNKNOWN_OUT_OF_BOUNDS'
    elif not has_surface:
        result['status'] = 'UNKNOWN_NO_ELIGIBLE_SURFACE'
    else:
        values = mask[indices[:, 1], indices[:, 0]].astype(int).tolist()
        assert set(values) <= {0, 1}
        result.update(componentValues=values, insideCount=sum(values), insideFraction=sum(values) / 15)
        result['status'] = ('SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE' if sum(values) / 15 >= .5
                            else 'REJECTED_SURFACE_CONTEXT')
    return result


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def cached_frame_rows(path, indices):
    wanted = set(indices)
    result = {row['frame_index']: row for row in jsonl(path) if row['frame_index'] in wanted}
    assert set(result) == wanted
    return result


def pixel_phase(cv2, np, surface, protocol, frames):
    cached = {s: load(CACHE / (s + '_proposals.json')) for s in SEQUENCES}
    wanted = {s: [r['frame_index'] for r in frames if r['sequence'] == s] for s in SEQUENCES}
    motion = {s: cached_frame_rows(OLD / 'run' / s / 'selected.jsonl', wanted[s]) for s in SEQUENCES}
    records = []
    with (RUN / 'pixel_decisions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for frame in frames:
            sequence, index = frame['sequence'], frame['frame_index']
            source = ROOT / frame['nativeOriginal']
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            assert image is not None and image.dtype == np.uint8
            height, width = image.shape[:2]
            assert [width, height] == frame['nativeSize']
            assert hashlib.sha256(image.tobytes()).hexdigest() == frame['nativeBgrPixelSha256']
            reduced_size = [protocol['referenceWidth'], round(height * protocol['referenceWidth'] / width)]
            reduced = cv2.resize(image, tuple(reduced_size), interpolation=cv2.INTER_LINEAR)
            candidates = surface.surface_candidates(reduced)
            mask = candidates[0]['mask'] if candidates else np.zeros(reduced.shape[:2], dtype=np.uint8)
            assert mask.shape == reduced.shape[:2] and mask.dtype == np.uint8 and np.isin(mask, [0, 1]).all()
            if candidates:
                assert int(mask.sum()) == candidates[0]['area']
            directory = RUN / sequence / ('frame_' + str(index).zfill(6))
            directory.mkdir(parents=True, exist_ok=False)
            import shutil
            shutil.copyfile(source, directory / 'native_original.png')
            assert binding(directory / 'native_original.png') == binding(source)
            assert cv2.imwrite(str(directory / 'component_mask_01.png'), mask)
            boxes, seen = [], set()
            old_people = motion[sequence][index]['people']
            selected_by_id = {p['sourceTrackId']: p for p in old_people}
            assert len(selected_by_id) == len(old_people)
            for original in cached[sequence][index]:
                assert set(original) == {'id', 'box', 'confidence'}
                identity = original['id']
                assert type(identity) is int and identity >= 0 and identity not in seen
                seen.add(identity)
                finite_box(original['box'])
                assert isinstance(original['confidence'], (int, float)) and not isinstance(original['confidence'], bool)
                assert math.isfinite(original['confidence']) and 0 < original['confidence'] <= 1
                person = {'sourceTrackId': identity, 'bbox': list(original['box']),
                          'confidence': original['confidence'], 'courtRole': None, 'participantStatus': 'UNKNOWN'}
                if identity in selected_by_id:
                    assert person == selected_by_id[identity]
                boxes.append({'originalCachedProposal': original, 'person': person,
                              'cachedMotionSelected': identity in selected_by_id,
                              'footContext': foot_context(np, person['bbox'], mask, [width, height], reduced_size, bool(candidates))})
            assert set(selected_by_id) <= seen
            component_metadata = [{k: v for k, v in row.items() if k != 'mask'} for row in candidates]
            record = {**frame, 'reducedSize': reduced_size, 'directory': name(directory),
                      'originalCopy': {'path': name(directory / 'native_original.png'), **binding(directory / 'native_original.png')},
                      'mask': {'path': name(directory / 'component_mask_01.png'), **binding(directory / 'component_mask_01.png'),
                               'shape': list(mask.shape), 'insidePixels': int(mask.sum()),
                               'rawArraySha256': hashlib.sha256(mask.tobytes()).hexdigest(),
                               'isNoCandidatePlaceholder': not bool(candidates)},
                      'candidateComponents': component_metadata, 'selectedComponentIndex': 0 if candidates else None,
                      'proposals': boxes, 'semanticGroundTruthRead': False, 'qualificationEvidence': False}
            records.append(record)
            save(directory / 'pixel_decisions.json', record)
            write_row(stream, record)
            print(json.dumps({'pixelFrame': [sequence, index], 'components': len(candidates),
                              'proposalCount': len(boxes), 'semanticGroundTruthRead': False}), flush=True)
    assert len(records) == 13
    atomic_save(RUN / 'pixel_stage_receipt.json', {'completedAt': stamp(), 'frames': 13,
                'semanticGroundTruthRead': False, 'candidateCount': 1, 'decisions': binding(RUN / 'pixel_decisions.jsonl'),
                'qualificationEvidence': False})
    return records


def iou(box, target):
    finite_box(box)
    finite_box(target)
    intersection = max(0, min(box[2], target[2]) - max(box[0], target[0])) * max(0, min(box[3], target[3]) - max(box[1], target[1]))
    union = (box[2]-box[0])*(box[3]-box[1]) + (target[2]-target[0])*(target[3]-target[1]) - intersection
    return intersection / union


def agreement(metrics, boxes, gt):
    counts = metrics.match_detections(boxes, [row['box'] for row in gt], iou_threshold=.5)
    return {'matchedOriginalClass1': counts.true_positives,
            'publisherUnmatched': counts.false_positives, 'missedOriginalClass1': counts.false_negatives,
            'matchedIous': list(counts.matched_ious), 'proposalCount': len(boxes),
            'participantPrecision': None}


def annotate_overlap(record, players, referees):
    for item in record['proposals']:
        box = item['person']['bbox']
        item['originalPublisherClass1Overlaps'] = [{'id': row['id'], 'iou': iou(box, row['box'])} for row in players]
        item['originalPublisherClass4Overlaps'] = [{'id': row['id'], 'iou': iou(box, row['box'])} for row in referees]
        item['anyOriginalClass1IouAtLeastHalf'] = any(row['iou'] >= .5 for row in item['originalPublisherClass1Overlaps'])
        item['anyOriginalClass4IouAtLeastHalf'] = any(row['iou'] >= .5 for row in item['originalPublisherClass4Overlaps'])


def draw_overlay(cv2, np, record):
    directory = ROOT / record['directory']
    image = cv2.imread(str(directory / 'native_original.png'), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(directory / 'component_mask_01.png'), cv2.IMREAD_UNCHANGED)
    assert image is not None and mask is not None and np.isin(mask, [0, 1]).all()
    native_mask = cv2.resize(mask, tuple(record['nativeSize']), interpolation=cv2.INTER_NEAREST)
    canvas = image.copy()
    canvas[native_mask > 0] = (canvas[native_mask > 0].astype(float) * .75 + np.asarray([50, 200, 50]) * .25).astype(np.uint8)
    for item in record['proposals']:
        person, context = item['person'], item['footContext']
        accepted = context['status'] == 'SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE'
        color = (0, 220, 220) if accepted else (140, 140, 140) if context['status'].startswith('UNKNOWN') else (60, 60, 230)
        x1, y1, x2, y2 = np.rint(person['bbox']).astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        for x, y in context['nativeLocations']:
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, color, -1)
        short_status = 'SUPPORTED' if accepted else 'UNKNOWN' if context['status'].startswith('UNKNOWN') else 'REJECTED'
        label = f"fragment {person['sourceTrackId']} {short_status}" + (' motion' if item['cachedMotionSelected'] else '')
        cv2.putText(canvas, label, (max(0, x1), max(12, min(image.shape[0] - 4, y1 - 3))),
                    cv2.FONT_HERSHEY_SIMPLEX, .32 if image.shape[1] == 640 else .5, color, 1, cv2.LINE_AA)
    assert cv2.imwrite(str(directory / 'native_context_overlay.png'), canvas)
    assert cv2.imwrite(str(directory / 'reduced_original.png'), cv2.resize(image, tuple(record['reducedSize']), interpolation=cv2.INTER_LINEAR))
    return {'path': name(directory / 'native_context_overlay.png'), **binding(directory / 'native_context_overlay.png')}


def score_phase(cv2, np, metrics, parser, records):
    assert (RUN / 'pixel_stage_receipt.json').is_file()
    manifest = load(DATASET / 'manifest.json')
    gt = {s: parser.read_gt(DATASET / 'UVY' / s / 'gt/gt.txt', manifest['sequences'][s]['publisher_frames']) for s in SEQUENCES}
    wanted = {s: [r['frame_index'] for r in records if r['sequence'] == s] for s in SEQUENCES}
    previous = {s: cached_frame_rows(OLD / 'run' / s / 'scoring.jsonl', wanted[s]) for s in SEQUENCES}
    for record in records:
        sequence, index = record['sequence'], record['frame_index']
        players = [row for row in gt[sequence][index] if row['class'] == 1]
        referees = [row for row in gt[sequence][index] if row['class'] == 4]
        assert players == previous[sequence][index]['originalPublisherPlayers']
        assert referees == previous[sequence][index]['originalPublisherReferees']
        assert [r['person'] for r in record['proposals']] == previous[sequence][index]['observations']
        annotate_overlap(record, players, referees)
        raw = [r['person']['bbox'] for r in record['proposals']]
        motion = [r['person']['bbox'] for r in record['proposals'] if r['cachedMotionSelected']]
        supported = [r['person']['bbox'] for r in record['proposals'] if r['footContext']['status'] == 'SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE']
        intersection = [r['person']['bbox'] for r in record['proposals'] if r['cachedMotionSelected'] and r['footContext']['status'] == 'SURFACE_SUPPORTED_IMAGE_PERSON_CANDIDATE']
        counts = {label: agreement(metrics, boxes, players) for label, boxes in
                  [('cachedProposals', raw), ('cachedMotion', motion), ('surfaceSupported', supported), ('motionSurfaceDiagnosticIntersection', intersection)]}
        assert counts['cachedProposals']['matchedOriginalClass1'] == previous[sequence][index]['counts']['proposalMatched']
        assert counts['cachedMotion']['matchedOriginalClass1'] == previous[sequence][index]['counts']['selectedMatched']
        directory = ROOT / record['directory']
        mask = cv2.imread(str(directory / 'component_mask_01.png'), cv2.IMREAD_UNCHANGED)
        publisher_context = [{**row, 'footContext': foot_context(np, row['box'], mask, record['nativeSize'], record['reducedSize'], bool(record['candidateComponents']))}
                             for row in players + referees]
        record.update(originalPublisherPlayers=players, originalPublisherReferees=referees,
                      originalPublisherBoxContexts=publisher_context, counts=counts,
                      semanticGroundTruthRead=True, knownLabelDefect=None)
        if sequence == 'tennis_V01' and index == 604:
            assert players == []
            record['knownLabelDefect'] = 'Original player annotation omission retained'
        if sequence == 'tennis_V03' and index == 140:
            assert any(row['id'] == 9 for row in players)
            record['knownLabelDefect'] = 'Original class1 identity9 chair-umpire annotation retained'
        record['overlay'] = draw_overlay(cv2, np, record)
        save(directory / 'scored_context.json', record)
        print(json.dumps({'scoredFrame': [sequence, index], 'counts': counts, 'qualificationEvidence': False}), flush=True)
    return records


def child():
    began = time.perf_counter()
    report = {'startedAt': stamp(), 'complete': False, 'qualificationEvidence': False,
              'productionPromoted': False, 'modelInferenceRun': False, 'videoDecodeRun': False,
              'semanticGroundTruthRead': False, 'sourcesUnchanged': False, 'savedProductUnchanged': False,
              'scope': 'THIRTEEN_REUSED_DEVELOPMENT_PIXEL_CONTEXT_DIAGNOSTIC'}
    before = workload_process = None
    try:
        import psutil
        assert RUN.is_dir() and (RUN / 'launch_marker.json').is_file()
        before = load(UNIT / 'preparation.json')
        verify(before['bindings'])
        workload_process = psutil.Process()
        identity = {'pid': workload_process.pid, 'createTimeUnixSeconds': workload_process.create_time()}
        report['actualIdentity'] = identity
        atomic_save(RUN / 'actual_identity.json', identity)
        deadline = time.monotonic() + 10
        while not (RUN / 'actual_sampled.json').exists():
            assert time.monotonic() < deadline, 'Actual interpreter sampling acknowledgement timed out'
            time.sleep(.01)
        assert load(RUN / 'actual_sampled.json')['identity'] == identity
        assert Path(workload_process.exe()).resolve() == BASE.resolve()
        import cv2
        import numpy as np
        import scipy
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        surface = module_at('_participant_surface_extractor', ROOT / 'scripts/evaluate/probe_court_surface_support.py')
        metrics = module_at('_participant_surface_metrics', ROOT / 'src/evaluation/detection_metrics.py')
        parser = module_at('_participant_surface_publisher_parser', ROOT / 'scripts/evaluate/audit_uvy_videos.py')
        assert not any(key == 'torch' or key.startswith(('torch.', 'ultralytics')) for key in sys.modules)
        report['environment'] = {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
                                 'scipy': scipy.__version__, 'psutil': psutil.__version__}
        report['ordinarySourcesBefore'] = ordinary_sources()
        protocol = load(UNIT / 'protocol.json')
        records = pixel_phase(cv2, np, surface, protocol, before['frames'])
        report['allThirteenPixelDecisionsPersistedBeforeSemanticGroundTruth'] = True
        report['pixelStageReceipt'] = binding(RUN / 'pixel_stage_receipt.json')
        report['semanticGroundTruthRead'] = True
        report['frames'] = score_phase(cv2, np, metrics, parser, records)
        assert len(report['frames']) == 13
        report['pooledCounts'] = {comparison: {key: sum(r['counts'][comparison][key] for r in records)
                                              for key in ('matchedOriginalClass1', 'publisherUnmatched', 'missedOriginalClass1', 'proposalCount')}
                                 for comparison in ('cachedProposals', 'cachedMotion', 'surfaceSupported', 'motionSurfaceDiagnosticIntersection')}
        report['pooledCountsScope'] = 'Only thirteen already reviewed images from at most two source match groups; flawed publisher-label agreement'
        report['participantPrecision'] = report['participantIdentityAccuracy'] = report['independentGeneralizationAccuracy'] = None
        report['ordinarySourcesAfter'] = ordinary_sources()
        verify(report['ordinarySourcesBefore'])
        verify(report['ordinarySourcesAfter'])
        assert all(report['ordinarySourcesAfter'].get(k) == v for k, v in report['ordinarySourcesBefore'].items())
        assert not any(key == 'torch' or key.startswith(('torch.', 'ultralytics')) for key in sys.modules)
        verify(before['bindings'])
        assert product_snapshot() == before['productBefore']
        report['sourcesUnchanged'] = report['savedProductUnchanged'] = True
        report['complete'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        errors = []
        try:
            if workload_process is not None:
                report['actualProcessMemoryBytes'] = workload_process.memory_info()._asdict()
        except BaseException as error:
            errors.append('process memory: ' + repr(error))
        try:
            if before is not None:
                verify(before['bindings'])
                report['sourcesUnchanged'] = True
                report['savedProductUnchanged'] = product_snapshot() == before['productBefore']
                assert report['savedProductUnchanged']
        except BaseException as error:
            errors.append('input/product preservation: ' + repr(error))
        report['finalizationErrors'] = errors
        if errors:
            report['complete'] = False
        report['childElapsedSeconds'] = time.perf_counter() - began
        report['completedAt'] = stamp()
        save(RUN / 'report.json', report)
    return 0 if report['complete'] else 1


def run():
    import psutil
    prepared = load(UNIT / 'preparation.json')
    verify(prepared['bindings'])
    assert product_snapshot() == prepared['productBefore']
    RUN.mkdir(exist_ok=False)
    save(RUN / 'launch_marker.json', {'at': stamp(), 'oneShot': True, 'guards': GUARDS})
    resource = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False, 'childStarted': False,
                'samples': 0, 'peakOwnedTreeRssBytes': 0,
                'minimumAvailableRamBytes': psutil.virtual_memory().available,
                'modelInferenceRun': False, 'videoDecodeRun': False, 'qualificationEvidence': False}
    process = owned = monitor = None
    began = time.monotonic()
    try:
        assert resource['minimumAvailableRamBytes'] >= GUARDS['preflightAvailableBytes'], 'RAM preflight below750MB'
        monitor = module_at('_participant_surface_owned_monitor', MONITOR)
        arguments = [str(BASE), '-B', str(Path(__file__)), '--child']
        with (RUN / 'stdout.log').open('xb') as log, (RUN / 'samples.jsonl').open('x', encoding='utf-8', newline='\n') as samples:
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            resource['childStarted'] = True
            owned = monitor.OwnedProcessTree(process)
            resource['launcherIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            low = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                elapsed = time.monotonic() - began
                write_row(samples, {'at': stamp(), 'elapsedSeconds': elapsed, 'ownedProcesses': rows,
                                    'ownedTreeRssBytes': rss, 'availableRamBytes': available})
                resource['samples'] += 1
                resource['peakOwnedTreeRssBytes'] = max(resource['peakOwnedTreeRssBytes'], rss)
                resource['minimumAvailableRamBytes'] = min(resource['minimumAvailableRamBytes'], available)
                if (RUN / 'actual_identity.json').exists() and not (RUN / 'actual_sampled.json').exists():
                    identity = load(RUN / 'actual_identity.json')
                    if owned.observed_workload(identity):
                        atomic_save(RUN / 'actual_sampled.json', {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                low = low + 1 if available < GUARDS['minimumAvailableBytes'] else 0
                reason = ('RSS above1GB' if rss > GUARDS['maximumOwnedRssBytes'] else
                          'HostRAM below400MB5samples' if low >= GUARDS['lowSamples'] else
                          'Wall above180seconds' if elapsed > GUARDS['wallSeconds'] else None)
                if reason:
                    resource['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(GUARDS['sampleSeconds'])
            resource['exitCode'] = process.returncode
        completed = load(RUN / 'report.json')
        assert process.returncode == 0 and completed['complete'], 'Incomplete surface measurement; retain partial evidence'
        assert completed['sourcesUnchanged'] and completed['savedProductUnchanged']
        assert completed['allThirteenPixelDecisionsPersistedBeforeSemanticGroundTruth']
        verify(completed['ordinarySourcesBefore'])
        verify(completed['ordinarySourcesAfter'])
        assert owned.observed_workload(load(RUN / 'actual_identity.json'))
        resource['status'], resource['passed'] = 'COMPLETED', True
    except BaseException as error:
        resource['status'] = ('NOT_STARTED' if not resource['childStarted'] else
                              'STOPPED' if resource.get('safeguardStop') else 'FAILED')
        resource['error'] = type(error).__name__ + ': ' + str(error)
        resource['traceback'] = traceback.format_exc()
        if process is not None:
            try:
                resource['terminationActions'] = monitor.stop_owned(process, owned)
            except BaseException as cleanup_error:
                resource['cleanupError'] = repr(cleanup_error)
    finally:
        if process is not None:
            resource['exitCode'] = process.poll()
        if owned is not None:
            try:
                remaining = owned.wait_for_retirement()
                if remaining:
                    resource['passed'] = False
                    resource['postExitTerminationActions'] = monitor.stop_owned(process, owned)
                    remaining = owned.wait_for_retirement()
                resource['ownedIdentities'] = owned.identities()
                resource['remainingOwnedIdentities'] = remaining
                resource['allOwnedIdentitiesRetired'] = not remaining
                if remaining:
                    resource['passed'] = False
            except BaseException as cleanup_error:
                resource['passed'] = False
                resource['retirementError'] = repr(cleanup_error)
        else:
            resource['ownedIdentities'] = resource['remainingOwnedIdentities'] = []
            resource['allOwnedIdentitiesRetired'] = not resource['childStarted']
        try:
            verify(prepared['bindings'])
            resource['sourcesUnchanged'] = True
            resource['savedProductUnchanged'] = product_snapshot() == prepared['productBefore']
            assert resource['savedProductUnchanged']
        except BaseException as error:
            resource['bindingError'] = repr(error)
            resource['passed'] = False
        if not resource['passed'] and resource['status'] == 'COMPLETED':
            resource['status'] = 'FAILED_FINALIZATION'
        resource['elapsedSeconds'] = time.monotonic() - began
        resource['completedAt'] = stamp()
        save(RUN / 'resource_receipt.json', resource)
    print(json.dumps({key: resource.get(key) for key in ('passed', 'status', 'error', 'elapsedSeconds')}), flush=True)
    return 0 if resource['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true')
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.child:
        raise SystemExit(child())
    else:
        raise SystemExit(run())
