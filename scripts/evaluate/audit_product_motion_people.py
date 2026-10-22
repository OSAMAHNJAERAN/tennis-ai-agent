"""Bounded developer-only motion selection on one reused diagnostic video."""
from pathlib import Path
import hashlib
import json
import argparse

import cv2
import numpy as np

from src.tracking.motion_supported_people import select_motion_supported_people
from src.tracking.player_fragment_linking import box_iou as iou

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / 'artifacts/validation/product_execution/active_person_original_review'
source = ROOT / 'artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv'
prediction_path = ROOT / 'outputs/product/analyses/match_1bb12f2aa52a438ebce8a9895fa2bd4f/image_people.json'
labels_path = base / 'labels.json'
parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='motion01')
args = parser.parse_args()
if not args.revision.isalnum():
    raise ValueError('Revision must be alphanumeric')
output = base / args.revision
output.mkdir(exist_ok=True)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


protocol = {'sourceSha256': digest(source), 'labelsSha256': digest(labels_path),
            'predictionsSha256': digest(prediction_path), 'scriptSha256': digest(Path(__file__)),
            'selectorSha256': digest(ROOT / 'src/tracking/motion_supported_people.py'),
            'camera': {'width': 960, 'sampleFrameStride': 12, 'orbFeatures': 4000, 'ratio': .75,
                       'minimumInliers': 20, 'minimumInlierFraction': .5, 'maximumMeanResidualPx': 2,
                       'minimumMatchedSpanFraction': [.4, .3], 'ransacThresholdPx': 2},
            'selector': {'minimum_observed_seconds': .5, 'minimum_motion_extent_heights': .6,
                         'minimum_camera_samples': 4, 'minimum_motion_span_seconds': .5},
            'selection': 'ALL_ELIGIBLE_OBSERVED_MOVING_FRAGMENTS; ABSTAIN_IF_MORE_THAN_TWO_COOCCUR',
            'independentHumanReview': False, 'qualificationEvidence': False,
            'productionDefaultsChanged': False, 'identityMerging': False, 'courtRolesAssigned': False}
protocol_path = output / 'motion_protocol.json'
if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
    raise ValueError('Frozen motion protocol changed; preserve old evidence and use a new revision')
protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
(output / 'selector_source.py').write_bytes((ROOT / 'src/tracking/motion_supported_people.py').read_bytes())
(output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
labels = json.loads(labels_path.read_text(encoding='utf-8'))
rows = [frame['people'] for frame in predictions['frames']]
samples = set(range(0, 600, 12)) | {frame['frame_index'] for frame in labels['frames']}
camera = [None] * 600
orb = cv2.ORB_create(nfeatures=4000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
cap = cv2.VideoCapture(str(source))
anchor_points = anchor_descriptors = None
camera_audit = []
try:
    for index in range(600):
        ok, image = cap.read()
        if not ok:
            raise ValueError(f'Original source failed to decode at frame {index}')
        if index not in samples:
            continue
        gray = cv2.cvtColor(cv2.resize(image, (960, 540)), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for person in rows[index]:
            box = np.array(person['bbox']) / 2
            size = box[2:] - box[:2]
            lower = np.floor(box[:2] - .1 * size).astype(int)
            upper = np.ceil(box[2:] + .1 * size).astype(int)
            cv2.rectangle(mask, tuple(np.maximum(lower, 0)), tuple(np.minimum(upper, [959, 539])), 0, -1)
        points, descriptors = orb.detectAndCompute(gray, mask)
        audit = {'frame_index': index, 'timestamp_seconds': index / 60, 'isValid': False,
                 'frameToAnchor': None, 'features': len(points), 'rejectionReason': 'INSUFFICIENT_BACKGROUND_MATCHES'}
        if index == 0:
            anchor_points, anchor_descriptors = points, descriptors
            audit.update(isValid=True, frameToAnchor=np.eye(3).tolist(), rejectionReason=None,
                         inliers=len(points), meanResidualPx=0)
        elif descriptors is not None and anchor_descriptors is not None:
            matches = matcher.knnMatch(descriptors, anchor_descriptors, k=2)
            good = [first for pair in matches if len(pair) == 2 for first, second in [pair]
                    if first.distance < .75 * second.distance]
            if len(good) >= 20:
                current = np.float32([points[match.queryIdx].pt for match in good])
                anchor = np.float32([anchor_points[match.trainIdx].pt for match in good])
                transform, inliers = cv2.estimateAffinePartial2D(current, anchor, method=cv2.RANSAC,
                                                               ransacReprojThreshold=2, maxIters=2000, confidence=.99)
                if transform is not None and inliers is not None:
                    keep = inliers.ravel().astype(bool)
                    transformed = current @ transform[:, :2].T + transform[:, 2]
                    residual = np.linalg.norm(transformed[keep] - anchor[keep], axis=1)
                    span = np.ptp(current[keep], axis=0) / [960, 540]
                    fraction = float(keep.mean())
                    audit.update(matches=len(good), inliers=int(keep.sum()), inlierFraction=fraction,
                                 meanResidualPx=float(residual.mean()), matchedSpanFraction=span.tolist())
                    valid = (keep.sum() >= 20 and fraction >= .5 and residual.mean() <= 2
                             and np.all(span >= [.4, .3]))
                    if valid:
                        # Convert reduced-image translation to native pixels.
                        native = np.vstack([transform, [0, 0, 1]])
                        native[:2, 2] *= 2
                        audit.update(isValid=True, frameToAnchor=native.tolist(), rejectionReason=None)
                    else:
                        audit['rejectionReason'] = 'INSUFFICIENT_DISTRIBUTED_CAMERA_CONSENSUS'
        camera[index] = audit
        camera_audit.append(audit)
finally:
    cap.release()
selected, selection_audit = select_motion_supported_people(rows, 60, camera, **protocol['selector'])
report = {'protocol': protocol, 'cameraSamples': camera_audit, 'selectionAudit': selection_audit,
          'frames': [], 'qualificationEvidence': False,
          'limitations': ['Single reused diagnostic clip and eight approximate agent-labeled frames.',
                          'Moving officials or spectators can qualify; motion does not prove participation.',
                          'No physical geometry, named identity, court roles or production promotion.',
                          'Sparse camera samples support offline fragment selection; no per-frame motion or speed estimate.']}
tp = fp = fn = ignored = 0
for labeled in labels['frames']:
    index = labeled['frame_index']
    observed = selected[index]
    used = set()
    active_matches = []
    for active in labeled['active']:
        options = sorted(((iou(active['bbox'], person['bbox']), proposal, person)
                          for proposal, person in enumerate(observed) if proposal not in used), key=lambda option: -option[0])
        best = options[0] if options else (0, None, None)
        match = best[0] >= .5
        if match:
            used.add(best[1])
        tp += match
        fn += not match
        active_matches.append({'labelIdentity': active['identity'], 'matched': match, 'maximumIoU': best[0],
                               'sourceTrackId': best[2]['sourceTrackId'] if match else None})
    extras = []
    for proposal, person in enumerate(observed):
        if proposal in used:
            continue
        if any(iou(person['bbox'], item['bbox']) >= .5 for item in labeled.get('ignoredActive', [])):
            ignored += 1
        else:
            fp += 1
            extras.append(person)
    report['frames'].append({'frame_index': index, 'selected': observed, 'matches': active_matches,
                             'unmatchedSelected': extras})
report['summary'] = {'sampledCameraFrames': len(camera_audit),
                     'acceptedCameraFrames': sum(frame['isValid'] for frame in camera_audit),
                     'selectedActualObservations': sum(len(frame) for frame in selected),
                     'labeledSubsetTP': tp, 'labeledSubsetFP': fp, 'labeledSubsetFN': fn,
                     'ignoredPartialSelections': ignored, 'productionQualified': False,
                     'identityAccuracy': None, 'independentGeneralizationAccuracy': None}
(output / 'motion_candidate.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
(output / 'motion_candidate_selected.json').write_text(json.dumps({'scope': selection_audit['selectionScope'],
                                                               'frames': selected}, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report['summary'], indent=2))
print(json.dumps(selection_audit['statistics'], indent=2))
