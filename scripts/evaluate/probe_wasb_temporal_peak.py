"""One frozen cached temporal-max feasibility probe, outside product inference.

All 44 cases are purposively reused reserved-training examples. Any favorable
result only motivates broader development evaluation; it cannot promote a model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from scripts.evaluate.verify_wasb_proposal_response import points
from src.detection.tiled_wasb_candidates import merge_by_confidence
from src.tracking.temporal_ball_tracker import BallObservation

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/vision_upgrade_audit/wasb_proposal_response01'
OUT = RUN / 'temporal_peak_probe01'
SOURCE = ROOT / 'artifacts/training/vision_upgrade/wasb_spaced_head_pilot05/manifest.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dependencies():
    return [Path(__file__), SOURCE, RUN / 'report.json', RUN / 'selection.json',
            RUN / 'independent_verification.json', RUN / 'independent_geometry_threshold_audit.json',
            RUN / 'visual_review/render_manifest.json', RUN / 'visual_review/visual_review.json',
            RUN / 'visual_review/VISUAL_REVIEW.md',
            RUN / 'visual_review/post_review_integrity.json',
            ROOT / 'docs/experiments/WASB_PROPOSAL_RESPONSE_PROTOCOL.md',
            ROOT / 'scripts/evaluate/verify_wasb_proposal_response.py',
            ROOT / 'src/detection/tiled_wasb_candidates.py',
            ROOT / 'src/tracking/temporal_ball_tracker.py',
            ROOT / 'src/utils/bbox_utils.py',
            ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar']


def snapshot(paths):
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def runtime_versions():
    return {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__}


def ready():
    report = read(RUN / 'report.json')
    selection = read(RUN / 'selection.json')
    audit = read(RUN / 'independent_geometry_threshold_audit.json')
    visual = read(RUN / 'visual_review/visual_review.json')
    assert report['complete'] and report['qualification_evidence'] is False
    assert selection['source_sha256'] == digest(SOURCE)
    assert report['selection_sha256'] == digest(RUN / 'selection.json')
    assert audit['reportSha256'] == digest(RUN / 'report.json')
    assert audit['all44WindowsRecomputed'] and audit['all44BoxSetsAndInverseTransformsVerified']
    assert audit['all44AveragedCandidatesAndTopOneReproduced']
    assert audit['targetSlotScalarVsDecoderThresholdDisagreements'] == []
    assert visual['visualInspectionComplete'] and visual['reviewRecordPersistenceComplete']
    assert visual['exactFrozenSubsetReviewedInOrder'] == [44, 566, 234, 317, 9, 20]
    assert len(report['entries']) == len(selection['entries']) == 44
    return report, selection


def freeze():
    if OUT.exists():
        raise FileExistsError('Preserve the frozen protocol and any prior probe.')
    report, _ = ready()
    paths = dependencies() + [RUN / entry['heatmaps'] for entry in report['entries']]
    protocol = {
        'frozenAt': datetime.now(timezone.utc).isoformat(),
        'purpose': 'ALL44_CACHED_PURPOSIVE_RESERVED_TRAINING_FEASIBILITY; NOT_QUALIFICATION',
        'candidateCount': 1, 'candidate': 'ELEMENTWISE_MAX_OVER_TARGET_ALIGNED_WINDOWS_PER_EXISTING_VIEW',
        'control': 'UNCHANGED_ELEMENTWISE_MEAN', 'activationThreshold': .2,
        'thresholdSemantics': 'Unchanged NumPy array comparison heatmap > .2; no Python scalar-peak surrogate.',
        'decoder': 'Unchanged mass-ranked connected components and float32 homogeneous-point inverse mapping.',
        'views': 5, 'merge': 'Unchanged confidence-ranked stable greedy merge, radius4 at512x288 reference.',
        'centerToleranceReferencePx': 4, 'referenceSize': [512, 288],
        'targetLabelsUsedDuringCandidateConstruction': False,
        'selection': 'All44 frozen rows, no category or error-based exclusions.',
        'gate': 'At least1 gained correct merged top-one, zero lost original correct controls, no total FP increase.',
        'gatePassingMeaning': 'Only feasibility for broader complete reserved-development evaluation; no promotion.',
        'modelInferenceRun': False, 'trainingRun': False, 'thresholdSweep': False,
        'runtimeVersions': runtime_versions(),
        'productionPromotionAllowed': False, 'protectedHashes': snapshot(paths),
        'limitations': ['Exposed purposive subset cannot estimate prevalence or general accuracy.',
                       'Publisher labels remain unchanged; tiny/blurred targets and absence control are not independent visual truth.',
                       'No physical measurements, identity, full-clip behavior or qualified accuracy is tested.']}
    OUT.mkdir()
    with (OUT / 'protocol.json').open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    print(json.dumps({'protocol': (OUT / 'protocol.json').relative_to(ROOT).as_posix(),
                      'candidateOutcomesComputed': False, 'protectedFiles': len(paths)}))


def decode(maps, views, reducer):
    proposals = []
    for view in views:
        values = reducer(maps[f"view{view['view']}"], axis=0)
        for rank, (_, x, y, confidence) in enumerate(points(values, np.asarray(view['inverse']), view['box'])):
            proposals.append({'view': view['view'], 'box': view['box'], 'mass_rank': rank,
                              'x': x, 'y': y, 'confidence': confidence})
    return proposals


def top_one(proposals, size):
    merged = merge_by_confidence([BallObservation(item['x'], item['y'], item['confidence']) for item in proposals], size)
    return [merged[0].x_px, merged[0].y_px] if merged else None


def score(target, prediction, size):
    if target is None:
        return ('FP', None, Counter(FP=1)) if prediction is not None else ('TN', None, Counter(TN=1))
    distance = math.hypot((prediction[0] - target[0]) * 512 / size[0],
                          (prediction[1] - target[1]) * 288 / size[1]) if prediction is not None else None
    if distance is not None and distance <= 4:
        return 'TP', distance, Counter(TP=1)
    return ('FP_AND_FN', distance, Counter(FP=1, FN=1)) if prediction is not None else ('FN', None, Counter(FN=1))


def run():
    destination = OUT / 'report.json'
    if destination.exists():
        raise FileExistsError('Preserve the completed probe.')
    protocol_path = OUT / 'protocol.json'
    protocol_hash = digest(protocol_path)
    protocol = read(protocol_path)
    protected = protocol['protectedHashes']
    assert protocol['runtimeVersions'] == runtime_versions()
    assert protected == snapshot([ROOT / name for name in protected])
    assert protocol['candidateCount'] == 1 and protocol['activationThreshold'] == .2
    assert protocol['centerToleranceReferencePx'] == 4 and protocol['productionPromotionAllowed'] is False
    report, selection = ready()
    selected = {(entry['clip'], entry['row']['frame']): entry for entry in selection['entries']}
    control_counts, candidate_counts, per_clip = Counter(), Counter(), {}
    records = []
    gained = lost = 0
    start = time.perf_counter()
    for entry in report['entries']:
        original = selected[(entry['clip'], entry['frame'])]
        row = original['row']
        size = (original['width'], original['height'])
        assert [view['view'] for view in entry['views']] == list(range(5))
        with np.load(RUN / entry['heatmaps'], allow_pickle=False) as maps:
            assert sorted(maps.files) == [f'view{i}' for i in range(5)]
            assert all(maps[name].dtype == np.float32 and np.isfinite(maps[name]).all()
                       and maps[name].shape == (len(row['windows']), 288, 512) for name in maps.files)
            control = decode(maps, entry['views'], np.mean)
            assert len(control) == len(row['candidates'])
            for actual, expected in zip(control, row['candidates']):
                assert all(actual[key] == expected[key] for key in ('view', 'box', 'mass_rank'))
                assert all(abs(actual[key] - expected[key]) <= 1e-5 for key in ('x', 'y', 'confidence'))
            before = top_one(control, size)
            assert (before is None) == (row['prediction_xy'] is None)
            if before is not None:
                assert max(abs(a - b) for a, b in zip(before, row['prediction_xy'])) <= 1e-5
            # Candidate construction reads arrays and view geometry only.
            candidate = decode(maps, entry['views'], np.max)
            after = top_one(candidate, size)
        before_outcome, before_error, bc = score(row['target_xy'], before, size)
        after_outcome, after_error, ac = score(row['target_xy'], after, size)
        gain, loss = before_outcome != 'TP' and after_outcome == 'TP', before_outcome == 'TP' and after_outcome != 'TP'
        gained += gain
        lost += loss
        control_counts.update(bc)
        candidate_counts.update(ac)
        clip = per_clip.setdefault(entry['clip'], {'control': Counter(), 'candidate': Counter(), 'gainedTP': 0, 'lostTP': 0})
        clip['control'].update(bc)
        clip['candidate'].update(ac)
        clip['gainedTP'] += gain
        clip['lostTP'] += loss
        records.append({'clip': entry['clip'], 'frame': entry['frame'], 'role': entry['role'],
                        'targetXY': row['target_xy'], 'size': size,
                        'controlXY': before, 'candidateXY': after,
                        'controlOutcome': before_outcome, 'candidateOutcome': after_outcome,
                        'controlReferenceError': before_error, 'candidateReferenceError': after_error,
                        'gainedTP': gain, 'lostTP': loss, 'candidateRawProposals': candidate})
    seconds_cached_decode_and_scoring = time.perf_counter() - start
    assert len(records) == 44 and control_counts['TP'] == 12
    assert protected == snapshot([ROOT / name for name in protected])
    assert digest(protocol_path) == protocol_hash
    feasible = gained >= 1 and lost == 0 and candidate_counts['FP'] <= control_counts['FP']
    result = {'completedAt': datetime.now(timezone.utc).isoformat(), 'complete': True,
              'protocolSha256': protocol_hash, 'scriptSha256': digest(Path(__file__)),
              'protocolUnchanged': True, 'runtimeVersions': runtime_versions(),
              'scope': protocol['purpose'], 'modelInferenceRun': False, 'productionChanged': False,
              'qualificationEvidence': False, 'protectedFilesUnchanged': True,
              'all44ControlRowsReproduced': True, 'controlCounts': dict(control_counts),
              'candidateCounts': dict(candidate_counts), 'gainedTP': gained, 'lostTP': lost,
              'secondsCachedDecodeAndScoring': seconds_cached_decode_and_scoring,
              'feasibleForBroaderDevelopmentEvaluation': feasible,
              'decision': 'NEEDS_COMPLETE_RESERVED_DEVELOPMENT_EVALUATION' if feasible else 'REJECT_CANDIDATE',
              'perClip': per_clip, 'rows': records, 'limitations': protocol['limitations']}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('controlCounts', 'candidateCounts', 'gainedTP', 'lostTP', 'decision')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    arguments = parser.parse_args()
    freeze() if arguments.freeze else run()
