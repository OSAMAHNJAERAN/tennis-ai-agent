"""Join frozen original-image pilot labels to unchanged saved observations.

This does not run a model, alter a threshold or certify full-clip accuracy.
Annotations from a single AI reviewer per clip are diagnostic approximations.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'artifacts/validation/product_execution/public_ball_blind_review01'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_labels(protocol, source, labels):
    if (labels['protocolSha256'] != digest(TARGET / 'protocol.json')
            or labels['inputSha256'] != source['inputSha256']
            or labels['sourceId'] != source['id'] or labels['predictionsSeen'] is not False):
        raise ValueError('Label provenance disagrees with the frozen blind-review protocol.')
    records = labels['frames']
    if len(records) != len(source['centers']) or sorted(item['frame_index'] for item in records) != sorted(source['centers']):
        raise ValueError('Frozen center labels must be complete, unique and unchanged.')
    images = {item['frame_index']: item for item in source['frames']}
    for item in records:
        index = item['frame_index']
        if item['sha256'] != images[index]['sha256']:
            raise ValueError('A label refers to a different original image.')
        expected_context = sorted({max(0, min(source['decodedFrames'] - 1, index + delta)) for delta in (-3, 0, 3)})
        if sorted(item['context_frame_indices']) != expected_context:
            raise ValueError('Required exact-frame original context was not reviewed.')
        if item['visibility'] not in protocol['labels']['visibility']:
            raise ValueError('Unknown label visibility.')
        if not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Every visibility/absence judgment needs recorded original-image evidence.')
        center = item['center_px']
        if item['visibility'] == 'VISIBLE':
            radius = item['uncertainty_radius_px']
            if (not isinstance(center, list) or len(center) != 2
                    or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in center)
                    or not 0 <= center[0] < images[index]['width'] or not 0 <= center[1] < images[index]['height']
                    or isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0):
                raise ValueError('Visible center or approximate annotation uncertainty is invalid.')
        elif center is not None:
            raise ValueError('Unavailable/uncertain label must not invent a ball center.')
    return records


def main():
    protocol_path = TARGET / 'protocol.json'
    protocol = read(protocol_path)
    manifest_path = ROOT / 'data/demos/sources.json'
    if digest(manifest_path) != protocol['manifestSha256']:
        raise ValueError('Public source manifest changed after sample freeze.')
    manifest = read(manifest_path)
    frozen = []
    # All labels and original PNG identities are frozen before any result read.
    for source in protocol['sources']:
        labels_path = TARGET / source['id'] / 'labels.json'
        labels = read(labels_path)
        records = validate_labels(protocol, source, labels)
        if digest(ROOT / source['inputPath']) != source['inputSha256']:
            raise ValueError('Original input changed.')
        for original in source['frames']:
            if digest(ROOT / original['path']) != original['sha256']:
                raise ValueError('Original review pixels changed.')
        frozen.append((source, records, labels_path, digest(labels_path)))
    report = {'comparedAt': datetime.now(timezone.utc).isoformat(),
              'scope': '36_SYSTEMATIC_FRAMES; SINGLE_AI_APPROXIMATE_LABELS_PER_CLIP; NOT_FULL_CLIP_OR_QUALIFIED_ACCURACY',
              'protocolSha256': digest(protocol_path), 'comparisonScriptSha256': digest(Path(__file__)),
              'centerTolerancePx': protocol['comparison']['centerTolerancePx'],
              'modelInferenceRun': False, 'candidateFilterRun': False,
              'reasonCandidateNotRun': 'This label-protocol pilot does not supply dense contact/fast-travel/camera-cut controls for a detour promotion comparison.',
              'sources': []}
    aggregate = Counter()
    for source, records, labels_path, label_hash in frozen:
        entry = next(item for item in manifest['sources'] if item['id'] == source['id'])
        directory = ROOT / 'outputs/product/analyses' / entry['analysisId']
        protected = {name: digest(directory / name) for name in ('result.json', 'detections.json', 'trajectories.json')}
        raw = read(directory / 'detections.json')['frames']
        saved = read(directory / 'result.json')
        published = saved['detections']['frames']
        trajectory = saved['trajectories']['ballTrajectory']
        if (saved['summary']['id'] != entry['analysisId'] or len(raw) != source['decodedFrames']
                or len(published) != len(raw) or len(trajectory) != len(raw)):
            raise ValueError('Saved analysis identity or frame count disagrees.')
        for index, (actual, displayed, point) in enumerate(zip(raw, published, trajectory)):
            expected = json.loads(json.dumps(actual))
            ball = expected['ball']
            if ball['position_px'] is not None:
                ball['position_px'] = [round(value, 2) for value in ball['position_px']]
            confidence = ball.get('confidence')
            ball['confidence'] = round(confidence, 3) if confidence is not None and ball['state'] != 'MISSING' else None
            if (actual['frame_index'] != index or displayed != expected or point['frame_index'] != index
                    or displayed['ball']['state'] != point['state']
                    or displayed['ball']['position_px'] != ([point['x_px'], point['y_px']] if point['x_px'] is not None and point['y_px'] is not None else None)):
                raise ValueError('Unchanged control must reproduce every published state and rounded coordinate.')
        outcomes = []
        counts = Counter()
        for label in records:
            index = label['frame_index']
            observation = published[index]['ball']
            position = observation['position_px']
            visible = label['visibility'] == 'VISIBLE'
            eligible = label['visibility'] == 'CONFIRMED_ABSENT' or (visible and label['uncertainty_radius_px'] <= 4)
            distance = math.dist(label['center_px'], position) if visible and position is not None else None
            if not eligible:
                outcome = 'EXCLUDED_UNCERTAIN_OR_IMPRECISE_LABEL'
                counts['excluded'] += 1
            elif label['visibility'] == 'CONFIRMED_ABSENT':
                outcome = 'FP' if position is not None else 'TN'
                counts[outcome] += 1
                counts['confirmedAbsentControls'] += 1
            elif distance is not None and distance <= report['centerTolerancePx']:
                outcome = 'TP'
                counts['TP'] += 1
                counts['visibleControls'] += 1
            else:
                outcome = 'FP_AND_FN' if position is not None else 'FN'
                counts['FN'] += 1
                counts['FP'] += position is not None
                counts['visibleControls'] += 1
            counts['reviewedCenters'] += 1
            outcomes.append({'originalLabel': label, 'savedObservation': observation,
                             'eligibleForBoundedComparison': eligible, 'centerSeparationPx': distance, 'outcome': outcome})
        if protected != {name: digest(directory / name) for name in protected} or digest(labels_path) != label_hash:
            raise ValueError('Saved evidence or original labels changed during comparison.')
        aggregate.update(counts)
        report['sources'].append({'id': source['id'], 'analysisId': entry['analysisId'],
            'inputSha256': source['inputSha256'], 'labelsSha256': label_hash, 'protectedEvidenceHashes': protected,
            'allPublishedControlFramesReproduced': len(raw), 'counts': dict(counts), 'outcomes': outcomes})
    report['boundedCounts'] = dict(aggregate)
    report['noFullClipAccuracyExtrapolation'] = True
    destination = TARGET / 'saved_control_comparison.json'
    if destination.exists():
        raise ValueError('An existing comparison must be preserved; inspect before a new revision.')
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'boundedCounts': report['boundedCounts'], 'fullClipAccuracy': False,
                      'candidateFilterRun': False, 'report': destination.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
