"""One guarded reused-development UVY motion-fragment measurement, no models."""
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
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
DATASET = ROOT / 'data/external/uvy_tennis_videos'
CACHE = ROOT / 'outputs/vision_upgrade_audit/uvy_person_tracked640'
BASE = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe')
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
SEQUENCES = ('tennis_V01', 'tennis_V02', 'tennis_V03')
GUARDS = {'preflightAvailableBytes': 750000000, 'maximumOwnedRssBytes': 1000000000,
          'minimumAvailableBytes': 400000000, 'lowSamples': 5, 'sampleSeconds': .25, 'wallSeconds': 300}


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


def input_bindings():
    manifest = load(DATASET / 'manifest.json')
    cache_report = load(CACHE / 'report.json')
    protocol = load(UNIT / 'protocol.json')
    assert manifest['complete'] and set(manifest['sequences']) == set(SEQUENCES)
    assert cache_report['complete'] and cache_report['mode'] == 'tracked640'
    assert cache_report['qualification_evidence'] is False
    assert binding(DATASET / 'manifest.json')['sha256'] == cache_report['dataset_manifest_sha256']
    assert binding(ROOT / protocol['selector']['source'])['sha256'] == protocol['selector']['sha256']
    assert binding(ROOT / 'src/evaluation/detection_metrics.py')['sha256'] == protocol['scoring']['matcherSha256']
    assert binding(ROOT / 'scripts/evaluate/audit_uvy_videos.py')['sha256'] == protocol['scoring']['publisherParserSha256']
    assert binding(MONITOR)['sha256'] == protocol['guards']['monitorSha256']
    assert {key: protocol['guards'][key] for key in GUARDS} == GUARDS
    inputs = [Path(__file__), UNIT / 'protocol.json', BASE, MONITOR, DATASET / 'manifest.json',
              CACHE / 'report.json', ROOT / 'src/tracking/motion_supported_people.py',
              ROOT / 'src/evaluation/detection_metrics.py', ROOT / 'scripts/evaluate/audit_uvy_videos.py',
              ROOT / 'scripts/evaluate/audit_uvy_person_coverage.py',
              ROOT / 'scripts/evaluate/audit_product_motion_people.py',
              ROOT / 'artifacts/validation/vision_upgrade/uvy_tennis_timing_audit.json',
              ROOT / 'outputs/vision_upgrade_audit/uvy_video_alignment/audit.json',
              ROOT / 'outputs/vision_upgrade_audit/uvy_player_tracking_baseline/report.json',
              ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml']
    old = ROOT / 'artifacts/validation/product_execution/active_person_original_review/motion02'
    inputs.extend(old / leaf for leaf in ('audit_source.py', 'selector_source.py', 'motion_protocol.json',
                                         'motion_candidate.json', 'motion_candidate_selected.json',
                                         'source_observation_integrity.json'))
    inputs.extend(path for path in (ROOT / 'src').rglob('*.py'))
    for item in manifest['files']:
        path = DATASET / item['path']
        assert binding(path) == {'bytes': item['bytes'], 'sha256': item['sha256']}
        inputs.append(path)
    for sequence in SEQUENCES:
        old_record = cache_report['sequences'][sequence]
        path = CACHE / old_record['predictions_file']
        assert path.parent == CACHE and path.name == sequence + '_proposals.json'
        assert binding(path)['sha256'] == old_record['predictions_sha256']
        assert old_record['frames'] == manifest['sequences'][sequence]['publisher_frames']
        inputs.append(path)
    return {name(path): binding(path) for path in inputs}


def prepare():
    assert not RUN.exists(), 'This one-shot unit has already been consumed'
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    frozen = input_bindings()
    save(UNIT / 'preparation.json', {'preparedAt': stamp(), 'bindings': frozen,
         'productBefore': product_snapshot(), 'guards': GUARDS,
         'semanticGroundTruthRead': False, 'modelInferenceRun': False, 'videoDecodeRun': False,
         'qualificationEvidence': False, 'sourcePreparedOnly': True})
    print(json.dumps({'prepared': True, 'boundFiles': len(frozen), 'executionStarted': False}), flush=True)


def adapt_observations(cached, count):
    assert isinstance(cached, list) and len(cached) == count
    rows = []
    for frame in cached:
        assert isinstance(frame, list)
        people, identities = [], set()
        for item in frame:
            assert set(item) == {'id', 'box', 'confidence'}
            identity, box, confidence = item['id'], item['box'], item['confidence']
            assert type(identity) is int and identity >= 0 and identity not in identities
            assert isinstance(box, list) and len(box) == 4
            assert all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in box)
            assert box[0] < box[2] and box[1] < box[3]
            assert isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence)
            assert 0 < confidence <= 1
            identities.add(identity)
            people.append({'sourceTrackId': identity, 'bbox': list(box), 'confidence': confidence,
                           'courtRole': None, 'participantStatus': 'UNKNOWN'})
        rows.append(people)
    return rows


def write_row(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + '\n')
    stream.flush()


def camera_registration(cv2, np, image, people, index, fps, orb, matcher, anchor, protocol):
    height, width = image.shape[:2]
    reference_width = protocol['referenceWidth']
    reference_height = round(height * reference_width / width)
    gray = cv2.cvtColor(cv2.resize(image, (reference_width, reference_height)), cv2.COLOR_BGR2GRAY)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    box_scale = np.array([reference_width / width, reference_height / height,
                          reference_width / width, reference_height / height])
    for person in people:
        box = np.asarray(person['bbox'], dtype=float) * box_scale
        size = box[2:] - box[:2]
        lower = np.maximum(np.floor(box[:2] - .1 * size).astype(int), [0, 0])
        upper = np.minimum(np.ceil(box[2:] + .1 * size).astype(int), [reference_width - 1, reference_height - 1])
        if np.all(lower <= upper):
            cv2.rectangle(mask, tuple(int(value) for value in lower), tuple(int(value) for value in upper), 0, -1)
    points, descriptors = orb.detectAndCompute(gray, mask)
    audit = {'frame_index': index, 'timestamp_seconds': index / fps, 'sampled': True,
             'isValid': False, 'frameToAnchor': None, 'features': len(points),
             'matches': 0, 'inliers': 0, 'inlierFraction': None, 'meanResidualReferencePx': None,
             'matchedSpanFraction': None, 'referenceSize': [reference_width, reference_height],
             'nativeSize': [width, height], 'nativeBgrPixelSha256': hashlib.sha256(image.tobytes()).hexdigest(),
             'rejectionReason': 'INSUFFICIENT_BACKGROUND_MATCHES'}
    if index == 0:
        anchor = (points, descriptors)
        audit.update(isValid=True, frameToAnchor=np.eye(3).tolist(), rejectionReason=None,
                     inliers=len(points), meanResidualReferencePx=0,
                     anchorIdentityException=True, reducedTransform=np.eye(3).tolist())
        return audit, anchor
    anchor_points, anchor_descriptors = anchor
    if descriptors is None or anchor_descriptors is None:
        return audit, anchor
    pairs = matcher.knnMatch(descriptors, anchor_descriptors, k=2)
    good = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
            if first.distance < protocol['descriptorRatio'] * second.distance]
    audit['matches'] = len(good)
    if len(good) < protocol['minimumInliers']:
        return audit, anchor
    current_points = np.float32([points[match.queryIdx].pt for match in good])
    anchor_matches = np.float32([anchor_points[match.trainIdx].pt for match in good])
    transform, inliers = cv2.estimateAffinePartial2D(
        current_points, anchor_matches, method=cv2.RANSAC,
        ransacReprojThreshold=protocol['ransacThresholdReferencePx'],
        maxIters=protocol['ransacMaximumIterations'], confidence=protocol['ransacConfidence'])
    if transform is None or inliers is None:
        audit['rejectionReason'] = 'NO_PARTIAL_AFFINE_CONSENSUS'
        return audit, anchor
    keep = inliers.ravel().astype(bool)
    assert len(keep) == len(good)
    if not keep.any() or not np.isfinite(transform).all():
        audit['rejectionReason'] = 'EMPTY_OR_NONFINITE_PARTIAL_AFFINE_CONSENSUS'
        return audit, anchor
    transformed = current_points @ transform[:, :2].T + transform[:, 2]
    residual = np.linalg.norm(transformed[keep] - anchor_matches[keep], axis=1)
    span = np.ptp(current_points[keep], axis=0) / [reference_width, reference_height]
    fraction, mean_residual = float(keep.mean()), float(residual.mean())
    reduced = np.vstack([transform, [0, 0, 1]])
    audit.update(inliers=int(keep.sum()), inlierFraction=fraction,
                 meanResidualReferencePx=mean_residual, matchedSpanFraction=span.tolist(),
                 reducedTransform=reduced.tolist())
    valid = (keep.sum() >= protocol['minimumInliers'] and fraction >= protocol['minimumInlierFraction']
             and mean_residual <= protocol['maximumMeanInlierResidualReferencePx']
             and np.all(span >= protocol['minimumCurrentInlierSpanFraction']))
    if valid:
        native_scale = np.diag([width / reference_width, height / reference_height, 1])
        native_transform = native_scale @ reduced @ np.linalg.inv(native_scale)
        assert np.isfinite(native_transform).all()
        audit.update(isValid=True, frameToAnchor=native_transform.tolist(), rejectionReason=None,
                     nativeConjugationScale=native_scale.tolist())
    else:
        audit['rejectionReason'] = 'INSUFFICIENT_DISTRIBUTED_CAMERA_CONSENSUS'
    return audit, anchor


def select_sequence(cv2, np, selector, sequence, info, old_record, timing, protocol):
    count = info['publisher_frames']
    assert timing['timestamp_count'] == count and timing['strictly_increasing']
    fps = float(Fraction(timing['stream']['avg_frame_rate']))
    assert abs(fps - old_record['fps']) < 1e-9
    assert len(timing['frame_timestamps_seconds']) == count
    assert all(abs(value - index / fps) < .001 for index, value in enumerate(timing['frame_timestamps_seconds']))
    source = DATASET / info['video']
    cached = load(CACHE / old_record['predictions_file'])
    observations = adapt_observations(cached, count)
    directory = RUN / sequence
    directory.mkdir(exist_ok=False)
    camera, sample_indices = [None] * count, set(range(0, count, protocol['camera']['sampleFrameStride'])) | {count - 1}
    orb = cv2.ORB_create(nfeatures=protocol['camera']['orbFeatures'])
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    anchor = (None, None)
    capture = cv2.VideoCapture(str(source))
    decoded = 0
    try:
        assert capture.isOpened(), 'Cannot open exact source video'
        assert abs(capture.get(cv2.CAP_PROP_FPS) - fps) < 1e-6
        with (directory / 'camera.jsonl').open('x', encoding='utf-8', newline='\n') as camera_log:
            for index in range(count):
                ok, image = capture.read()
                assert ok and image is not None, f'{sequence}: sequential source frame {index} unavailable'
                assert [image.shape[1], image.shape[0]] == info['size']
                if index in sample_indices:
                    record, anchor = camera_registration(cv2, np, image, observations[index], index, fps,
                                                         orb, matcher, anchor, protocol['camera'])
                    camera[index] = record
                else:
                    record = {'frame_index': index, 'timestamp_seconds': index / fps, 'sampled': False,
                              'isValid': False, 'frameToAnchor': None, 'rejectionReason': 'NOT_SAMPLED_NO_INTERPOLATION'}
                write_row(camera_log, record)
                decoded += 1
            extra_ok, _ = capture.read()
            assert not extra_ok, 'Exact source exceeds frozen frame count'
    finally:
        capture.release()
    thresholds = {key: value for key, value in protocol['selector'].items() if key.startswith('minimum_')}
    selected, selection_audit = selector.select_motion_supported_people(observations, fps, camera, **thresholds)
    assert len(selected) == count
    selected_path = directory / 'selected.jsonl'
    with selected_path.open('x', encoding='utf-8', newline='\n') as selected_log:
        for index, people in enumerate(selected):
            assert len(people) <= 2 and len({person['sourceTrackId'] for person in people}) == len(people)
            assert all(person in observations[index] for person in people), 'Selected observation differs from exact cached source'
            assert all(person['courtRole'] is None and person['participantStatus'] == 'UNKNOWN' for person in people)
            write_row(selected_log, {'frame_index': index, 'timestamp_seconds': index / fps, 'people': people})
    save(directory / 'selection_audit.json', selection_audit)
    result = {'sequence': sequence, 'source': name(source), 'sourceBinding': binding(source),
              'cachedProposals': name(CACHE / old_record['predictions_file']),
              'cachedProposalsBinding': binding(CACHE / old_record['predictions_file']),
              'decodedFrames': decoded, 'fps': fps, 'fpsFraction': timing['stream']['avg_frame_rate'],
              'nativeSize': info['size'], 'cameraSamples': len(sample_indices),
              'acceptedCameraSamplesIncludingIdentityAnchor': sum(bool(row and row['isValid']) for row in camera),
              'camera': {'path': name(directory / 'camera.jsonl'), **binding(directory / 'camera.jsonl')},
              'selected': {'path': name(selected_path), **binding(selected_path)},
              'selectionAudit': {'path': name(directory / 'selection_audit.json'), **binding(directory / 'selection_audit.json')},
              'selectedActualObservations': sum(len(people) for people in selected),
              'sourceObservationIntegrity': True, 'semanticGroundTruthReadBeforeSelection': False,
              'qualificationEvidence': False}
    save(directory / 'selection_receipt.json', result)
    print(json.dumps({'sequence': sequence, 'stage': 'selection_saved_before_ground_truth',
                      'selected': result['selectedActualObservations'],
                      'acceptedCameraSamples': result['acceptedCameraSamplesIncludingIdentityAnchor']}), flush=True)
    return result


def jsonl_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def render_reviews(cv2, np, sequence, info, observations, selected, gt, review_reasons):
    directory = RUN / sequence
    review = []
    capture = cv2.VideoCapture(str(DATASET / info['video']))
    try:
        assert capture.isOpened()
        for index in range(max(review_reasons) + 1):
            ok, image = capture.read()
            assert ok and image is not None, 'Sequential review source frame unavailable'
            assert [image.shape[1], image.shape[0]] == info['size']
            if index not in review_reasons:
                continue
            original = directory / f'review_{index:06d}_original.png'
            overlay = directory / f'review_{index:06d}_overlay.png'
            assert not original.exists() and not overlay.exists()
            assert cv2.imwrite(str(original), image), 'Exact native original PNG write failed'
            canvas = image.copy()
            for person in observations[index]:
                x1, y1, x2, y2 = [round(value) for value in person['bbox']]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (150, 150, 150), 1)
            for item in gt[index]:
                if item['class'] not in (1, 4):
                    continue
                color = (255, 220, 40) if item['class'] == 1 else (230, 100, 200)
                x1, y1, x2, y2 = [round(value) for value in item['box']]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
                cv2.putText(canvas, f"GT class{item['class']} id{item['id']}", (max(0, x1), max(11, y1 - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
            for person in selected[index]:
                x1, y1, x2, y2 = [round(value) for value in person['bbox']]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 245, 60), 2)
                cv2.putText(canvas, f"moving fragment {person['sourceTrackId']}; role unknown", (max(0, x1), min(canvas.shape[0] - 4, y2 + 11)),
                            cv2.FONT_HERSHEY_SIMPLEX, .32, (50, 245, 60), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'{sequence} frame {index}; grey cached / cyan player GT / pink referee GT / green moving fragment',
                        (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .28, (255, 255, 255), 1, cv2.LINE_AA)
            assert cv2.imwrite(str(overlay), canvas), 'Native overlay PNG write failed'
            original_pixels = cv2.imread(str(original), cv2.IMREAD_UNCHANGED)
            assert original_pixels is not None and np.array_equal(original_pixels, image), 'Native original PNG not lossless'
            review.append({'frame_index': index, 'sourceFrameMethod': 'Exact sequential decode from frame zero; no seek',
                           'reasons': review_reasons[index], 'nativeSize': info['size'],
                           'nativeBgrPixelSha256': hashlib.sha256(image.tobytes()).hexdigest(),
                           'original': {'path': name(original), **binding(original)},
                           'overlay': {'path': name(overlay), **binding(overlay)}})
    finally:
        capture.release()
    assert {row['frame_index'] for row in review} == set(review_reasons)
    save(directory / 'review_receipt.json', {'reviews': review, 'humanVisualReviewComplete': False,
         'qualificationEvidence': False, 'originalPngsVerifiedLossless': True})
    return review


def score_sequence(cv2, np, metrics, gt_parser, sequence, info, old_record, result, protocol):
    directory = RUN / sequence
    count = info['publisher_frames']
    observations = adapt_observations(load(CACHE / old_record['predictions_file']), count)
    selected_records = jsonl_rows(directory / 'selected.jsonl')
    assert len(selected_records) == count
    assert all(row['frame_index'] == index for index, row in enumerate(selected_records))
    selected = [row['people'] for row in selected_records]
    selection_audit = load(directory / 'selection_audit.json')
    ambiguous = set(selection_audit['ambiguousFrames'])
    label_path = DATASET / 'UVY' / sequence / 'gt/labels.txt'
    labels = label_path.read_text(encoding='utf-8').splitlines()
    assert labels == ['player', 'sports ball', 'goal', 'referee', 'goalkeeper', 'pcamera', 'person playing']
    gt_path = DATASET / 'UVY' / sequence / 'gt/gt.txt'
    gt = gt_parser.read_gt(gt_path, count)
    assert len(old_record['per_frame_counts']) == count
    totals = {key: 0 for key in ('annotatedPlayers', 'proposalObservations', 'proposalMatched', 'proposalMissed',
                                'proposalPublisherUnmatched', 'selectedMatched', 'selectedPublisherUnmatched',
                                'selectedMissed', 'selectedRefereeOverlap', 'selectorLostMatched',
                                'selectedObservations', 'framesWithoutSelection', 'ambiguityFrames')}
    reasons = {0: ['FIRST_FRAME'], count // 2: ['MIDDLE_FRAME'], count - 1: ['LAST_FRAME']}
    for defect in protocol['scoring']['knownLabelDefectsRetained']:
        if defect['sequence'] == sequence:
            reasons.setdefault(defect['frameIndexZeroBased'], []).append('KNOWN_ORIGINAL_LABEL_DEFECT_RETAINED')
    first_extra = first_lost = first_ambiguity = None
    scoring_path = directory / 'scoring.jsonl'
    with scoring_path.open('x', encoding='utf-8', newline='\n') as scoring_log:
        for index in range(count):
            players = [row for row in gt[index] if row['class'] == 1]
            referees = [row for row in gt[index] if row['class'] == 4]
            targets = [row['box'] for row in players]
            proposal_match = metrics.match_detections([row['bbox'] for row in observations[index]], targets,
                                                       iou_threshold=protocol['scoring']['iouThreshold'])
            historical = {'matched': proposal_match.true_positives, 'missed': proposal_match.false_negatives,
                          'unmatched_proposals': proposal_match.false_positives}
            assert historical == old_record['per_frame_counts'][index], f'{sequence} frame {index}: cached metric replay differs'
            selected_match = metrics.match_detections([row['bbox'] for row in selected[index]], targets,
                                                       iou_threshold=protocol['scoring']['iouThreshold'])
            referee_match = metrics.match_detections([row['bbox'] for row in selected[index]], [row['box'] for row in referees],
                                                      iou_threshold=protocol['scoring']['iouThreshold'])
            lost = proposal_match.true_positives - selected_match.true_positives
            assert lost >= 0
            assert len(selected[index]) <= 2 and all(row in observations[index] for row in selected[index])
            assert index not in ambiguous or selected[index] == []
            current = {'annotatedPlayers': len(players), 'proposalObservations': len(observations[index]),
                       'proposalMatched': proposal_match.true_positives, 'proposalMissed': proposal_match.false_negatives,
                       'proposalPublisherUnmatched': proposal_match.false_positives,
                       'selectedMatched': selected_match.true_positives, 'selectedMissed': selected_match.false_negatives,
                       'selectedPublisherUnmatched': selected_match.false_positives,
                       'selectedRefereeOverlap': referee_match.true_positives, 'selectorLostMatched': lost,
                       'selectedObservations': len(selected[index]), 'framesWithoutSelection': int(not selected[index]),
                       'ambiguityFrames': int(index in ambiguous)}
            for key, value in current.items():
                totals[key] += value
            write_row(scoring_log, {'frame_index': index, 'timestamp_seconds': index / result['fps'],
                                   'observations': observations[index], 'selected': selected[index],
                                   'originalPublisherPlayers': players, 'originalPublisherReferees': referees,
                                   'counts': current, 'historicalProposalCountsReproduced': historical,
                                   'scope': 'REUSED_FLAWED_PUBLISHER_LABEL_AGREEMENT_NOT_VALIDATED_PARTICIPANT_PRECISION'})
            if selected_match.false_positives and first_extra is None:
                first_extra = index
                reasons.setdefault(index, []).append('FIRST_PUBLISHER_UNMATCHED_SELECTION_NOT_VALIDATED_FALSE_POSITIVE')
            if lost and first_lost is None:
                first_lost = index
                reasons.setdefault(index, []).append('FIRST_CACHED_PLAYER_MATCH_LOST_BY_SELECTOR')
            if index in ambiguous and first_ambiguity is None:
                first_ambiguity = index
                reasons.setdefault(index, []).append('FIRST_MORE_THAN_TWO_CANDIDATE_ABSTENTION')
    expected = {'annotatedPlayers': old_record['annotated_player_boxes'], 'proposalMatched': old_record['matched_player_boxes'],
                'proposalMissed': old_record['missed_player_boxes'], 'proposalObservations': old_record['person_proposals'],
                'proposalPublisherUnmatched': old_record['unmatched_proposals_not_false_positives']}
    assert all(totals[key] == value for key, value in expected.items()), 'Historical proposal totals changed'
    assert totals['selectedObservations'] == result['selectedActualObservations']
    assert totals['ambiguityFrames'] == len(ambiguous)
    defects = [{**row, 'retainedOriginalPlayerIds': [item['id'] for item in gt[row['frameIndexZeroBased']] if item['class'] == 1]}
               for row in protocol['scoring']['knownLabelDefectsRetained'] if row['sequence'] == sequence]
    summary = {'counts': totals, 'historicalProposalReplayPassedAllFrames': True,
               'knownLabelDefectsRetained': defects, 'groundTruth': {'path': name(gt_path), **binding(gt_path)},
               'scoring': {'path': name(scoring_path), **binding(scoring_path)},
               'qualificationEvidence': False, 'participantPrecision': None,
               'participantIdentityAccuracy': None, 'independentGeneralizationAccuracy': None}
    save(directory / 'scoring_summary.json', summary)
    reviews = render_reviews(cv2, np, sequence, info, observations, selected, gt, reasons)
    result.update(summary=summary, reviews=reviews, scoringComplete=True)
    print(json.dumps({'sequence': sequence, 'stage': 'scored', 'counts': totals}), flush=True)
    return result


def child():
    began = time.perf_counter()
    report = {'startedAt': stamp(), 'complete': False, 'qualificationEvidence': False,
              'productionPromoted': False, 'modelInferenceRun': False, 'sequences': {},
              'semanticGroundTruthRead': False, 'sourcesUnchanged': False, 'savedProductUnchanged': False,
              'scope': 'REUSED_DEVELOPMENT_MOVING_SOURCE_FRAGMENT_LABEL_AGREEMENT_ONLY'}
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
        selector = module_at('_uvy_motion_selector', ROOT / 'src/tracking/motion_supported_people.py')
        metrics = module_at('_uvy_motion_metrics', ROOT / 'src/evaluation/detection_metrics.py')
        gt_parser = module_at('_uvy_motion_gt_parser', ROOT / 'scripts/evaluate/audit_uvy_videos.py')
        assert not any(key == 'torch' or key.startswith(('torch.', 'ultralytics')) for key in sys.modules)
        report['environment'] = {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__,
                                 'scipy': scipy.__version__, 'psutil': psutil.__version__}
        report['ordinarySourcesBefore'] = ordinary_sources()
        protocol = load(UNIT / 'protocol.json')
        manifest = load(DATASET / 'manifest.json')
        cache_report = load(CACHE / 'report.json')
        timing = load(ROOT / 'artifacts/validation/vision_upgrade/uvy_tennis_timing_audit.json')
        for sequence in SEQUENCES:
            report['sequences'][sequence] = select_sequence(
                cv2, np, selector, sequence, manifest['sequences'][sequence],
                cache_report['sequences'][sequence], timing[sequence], protocol)
        assert set(report['sequences']) == set(SEQUENCES)
        assert sum(row['decodedFrames'] for row in report['sequences'].values()) == 2230
        assert all((RUN / sequence / 'selected.jsonl').is_file() for sequence in SEQUENCES)
        report['allThreeSelectionsPersistedBeforeSemanticGroundTruth'] = True
        atomic_save(RUN / 'selection_stage_receipt.json', {'completedAt': stamp(),
                    'semanticGroundTruthRead': False, 'sequences': report['sequences'], 'qualificationEvidence': False})
        report['semanticGroundTruthRead'] = True
        for sequence in SEQUENCES:
            score_sequence(cv2, np, metrics, gt_parser, sequence, manifest['sequences'][sequence],
                           cache_report['sequences'][sequence], report['sequences'][sequence], protocol)
        assert all(row['scoringComplete'] for row in report['sequences'].values())
        count_names = next(iter(report['sequences'].values()))['summary']['counts']
        report['pooledCounts'] = {key: sum(row['summary']['counts'][key] for row in report['sequences'].values())
                                 for key in count_names}
        report['pooledCountsScope'] = 'Three exposed clips from at most two additional match groups; frame-weighted publisher agreement only'
        report['participantPrecision'] = report['participantIdentityAccuracy'] = report['independentGeneralizationAccuracy'] = None
        report['ordinarySourcesAfter'] = ordinary_sources()
        verify(report['ordinarySourcesBefore'])
        verify(report['ordinarySourcesAfter'])
        assert all(report['ordinarySourcesAfter'].get(key) == value for key, value in report['ordinarySourcesBefore'].items())
        assert not any(key == 'torch' or key.startswith(('torch.', 'ultralytics')) for key in sys.modules)
        verify(before['bindings'])
        assert product_snapshot() == before['productBefore']
        report['sourcesUnchanged'] = report['savedProductUnchanged'] = True
        report['complete'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        # The process object has its own name; image/point arrays cannot replace it.
        # Final checks are isolated so a secondary failure cannot erase the primary report.
        finalization_errors = []
        try:
            if workload_process is not None:
                report['actualProcessMemoryBytes'] = workload_process.memory_info()._asdict()
        except BaseException as error:
            finalization_errors.append('process memory: ' + repr(error))
        try:
            if before is not None:
                verify(before['bindings'])
                report['sourcesUnchanged'] = True
                report['savedProductUnchanged'] = product_snapshot() == before['productBefore']
                assert report['savedProductUnchanged']
        except BaseException as error:
            finalization_errors.append('input/product preservation: ' + repr(error))
        report['finalizationErrors'] = finalization_errors
        if finalization_errors:
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
                'modelInferenceRun': False, 'qualificationEvidence': False}
    process = owned = monitor = None
    began = time.monotonic()
    try:
        assert resource['minimumAvailableRamBytes'] >= GUARDS['preflightAvailableBytes'], 'RAM preflight below750MB'
        monitor = module_at('_uvy_motion_owned_monitor', MONITOR)
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
                          'Wall above300seconds' if elapsed > GUARDS['wallSeconds'] else None)
                if reason:
                    resource['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(GUARDS['sampleSeconds'])
            resource['exitCode'] = process.returncode
        completed = load(RUN / 'report.json')
        assert process.returncode == 0 and completed['complete'], 'Incomplete motion measurement; retain partial evidence'
        assert completed['sourcesUnchanged'] and completed['savedProductUnchanged']
        assert completed['allThreeSelectionsPersistedBeforeSemanticGroundTruth']
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
