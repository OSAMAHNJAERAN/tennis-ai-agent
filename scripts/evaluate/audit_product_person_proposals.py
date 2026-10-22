"""Diagnostic active-body proposal coverage against frozen original-image labels.

Labels are never used to choose a deployable identity. Unmatched all-person
proposals are not detector false positives because spectator labels are partial.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
label_path = ROOT / 'artifacts/validation/product_execution/active_person_original_review/labels.json'
prediction_path = ROOT / 'outputs/product/analyses/match_1bb12f2aa52a438ebce8a9895fa2bd4f/image_people.json'
output = ROOT / 'artifacts/validation/product_execution/active_person_original_review'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def iou(first, second):
    low = [max(first[axis], second[axis]) for axis in (0, 1)]
    high = [min(first[axis + 2], second[axis + 2]) for axis in (0, 1)]
    overlap = max(0, high[0] - low[0]) * max(0, high[1] - low[1])
    a = (first[2] - first[0]) * (first[3] - first[1])
    b = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / (a + b - overlap) if a + b > overlap else 0


labels = json.loads(label_path.read_text(encoding='utf-8'))
# Freeze protocol before opening new model proposals.
protocol = {'labelSha256': digest(label_path), 'scriptSha256': digest(Path(__file__)),
            'iouThreshold': .5, 'purpose': 'FRAME_LOCAL_PERSON_PROPOSAL_COVERAGE_NOT_ACTIVE_SELECTION',
            'qualificationEvidence': False,
            'negativeScope': labels['negativeScope'],
            'unmatchedPersonProposalsCountedAsFalsePositives': False}
protocol_path = output / 'proposal_protocol.json'
if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
    raise ValueError('Frozen protocol changed; retain the old report and use a separate revision')
protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
report = {'protocol': protocol, 'analysisId': prediction_path.parent.name,
          'predictionsSha256': digest(prediction_path), 'sourceSha256': labels['sourceSha256'],
          'frames': [], 'qualificationEvidence': False}
matched = positives = negatives_observed = 0
for frame in labels['frames']:
    index = frame['frame_index']
    observed = predictions['frames'][index]['people']
    record = {'frame_index': index, 'personProposals': len(observed), 'active': [], 'negativeProposals': []}
    used = set()
    for active in frame['active']:
        options = sorted(((iou(active['bbox'], person['bbox']), proposal, person)
                          for proposal, person in enumerate(observed) if proposal not in used), reverse=True,
                         key=lambda option: option[0])
        score, proposal, person = options[0] if options else (0, None, None)
        is_match = score >= .5
        if is_match:
            used.add(proposal)
        positives += 1
        matched += is_match
        record['active'].append({'labelIdentity': active['identity'], 'labelBox': active['bbox'],
                                 'maximumIoU': score, 'matchedIoU05': is_match,
                                 'proposal': person})
    for negative in frame['negatives']:
        overlapping = [{'iou': iou(negative['bbox'], person['bbox']), 'proposal': person} for person in observed]
        overlapping = [person for person in overlapping if person['iou'] >= .5]
        negatives_observed += bool(overlapping)
        record['negativeProposals'].append({'kind': negative['kind'], 'matches': overlapping})
    report['frames'].append(record)
report['summary'] = {'visibleActiveBodyLabels': positives, 'matchedActiveBodyLabelsIoU05': matched,
                     'oracleProposalRecallPct': matched / positives * 100,
                     'explicitNegativeLabels': sum(len(frame['negatives']) for frame in labels['frames']),
                     'explicitNegativeLabelsWithProposalsIoU05': negatives_observed,
                     'activeSelectionPrecision': None, 'identityAccuracy': None,
                     'unmatchedAllPersonFalsePositives': None}
(output / 'proposal_coverage.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report['summary'], indent=2))
print(json.dumps([{'frame_index': frame['frame_index'], 'personProposals': frame['personProposals'],
                   'active': [{'identity': person['labelIdentity'], 'IoU': round(person['maximumIoU'], 3),
                               'sourceTrackId': person['proposal']['sourceTrackId'] if person['proposal'] else None}
                              for person in frame['active']]} for frame in report['frames']], indent=2))
