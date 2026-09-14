"""Verify newly recorded candidates and complete attribution on all 900 labels."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluate.audit_spaced_ball_errors import distance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01'
    target = out/'completed_candidate_review.json'
    if target.exists():
        raise FileExistsError(target)
    initial = json.loads((out/'report.json').read_text())
    acquired = json.loads((out/'stride_one_candidates.json').read_text())
    assert acquired['complete'] and acquired['audit_sha256'] == digest(out/'report.json')
    assert initial['protocol_sha256'] == digest(out/'original_protocol.md')
    assert acquired['protocol_sha256'] == digest(ROOT/'docs/experiments/BALL_SPACED_ERROR_AUDIT_PROTOCOL.md')
    for path, checksum in acquired['code_hashes'].items():
        assert digest(ROOT/path) == checksum
    supplement = {(c['clip'],r['frame']):r for c in acquired['clips'] for r in c['rows']}
    assert len(supplement) == 250
    assert set(supplement) == {(r['clip'],r['frame']) for r in initial['rows'] if r['candidates'] is None}
    all_candidates = {key:value['candidates'] for key,value in supplement.items()}
    for group in ('expansion12','additional6'):
        path = ROOT/f'artifacts/validation/vision_upgrade/{group}_temporal_spacing_pilot01.json'
        assert digest(path) == initial['sources'][str(path.relative_to(ROOT))]
        sparse = json.loads(path.read_text())
        for clip in sparse['clips']:
            for detail in clip['candidate_details']:
                key = (clip['clip'],detail['frame'])
                assert key not in all_candidates
                all_candidates[key] = detail['candidates']
    assert len(all_candidates) == 900
    coverage, categories, errors, supplemental_misses = Counter(), Counter(), [], []
    for row in initial['rows']:
        info = row['candidates']
        if info is None:
            entry = supplement[row['clip'],row['frame']]
            point,reference = entry['raw_prediction_xy'],row['stages']['raw']
            assert point == reference or (point is not None and reference is not None and max(abs(a-b) for a,b in zip(point,reference)) <= .0001)
            ordered = sorted(entry['candidates'],key=lambda c:-c['confidence'])
            correct = [i for i,c in enumerate(ordered) if row['target_xy'] is not None and distance([c['x'],c['y']],row['target_xy'],row['width'],row['height']) <= 4]
            assert entry['has_correct_candidate'] == (bool(correct) if row['target_xy'] is not None else None)
            info = dict(count=len(ordered),correct_count=len(correct),first_correct_confidence_rank=correct[0]+1 if correct else None,
                        best_confidence=ordered[0]['confidence'] if ordered else None,
                        correct_candidate_confidence=ordered[correct[0]]['confidence'] if correct else None)
        if row['target_xy'] is not None:
            coverage['visible_frames'] += 1
            coverage['visible_with_correct_candidate'] += int(info['correct_count'] > 0)
            if row['outcomes']['raw'] == 'correct_ball':
                assert info['correct_count'] > 0
        # Independently replay greedy cross-view duplicate suppression to distinguish
        # lost proposals from surviving candidates that merely rank below top one.
        merged = []
        for candidate in sorted(all_candidates[row['clip'],row['frame']],key=lambda c:-c['confidence']):
            point = [candidate['x'],candidate['y']]
            if not any(distance(point,[kept['x'],kept['y']],row['width'],row['height']) <= 4 for kept in merged):
                merged.append(candidate)
        top = [merged[0]['x'],merged[0]['y']] if merged else None
        reference = row['stages']['raw']
        assert top == reference or (top is not None and reference is not None and max(abs(a-b) for a,b in zip(top,reference)) <= .0001)
        merged_correct = [i for i,c in enumerate(merged) if row['target_xy'] is not None and distance([c['x'],c['y']],row['target_xy'],row['width'],row['height']) <= 4]
        info = dict(**info,merged_count=len(merged),merged_correct_count=len(merged_correct),
                    first_correct_merged_rank=merged_correct[0]+1 if merged_correct else None)
        if row['target_xy'] is not None:
            coverage['visible_with_correct_merged_candidate'] += int(bool(merged_correct))
        category = row['category']
        if category == 'raw_miss_candidate_coverage_unknown':
            category = 'raw_miss_correct_candidate_available' if info['correct_count'] else 'raw_miss_no_correct_candidate'
            supplemental_misses.append(dict(clip=row['clip'],frame=row['frame'],category=category,candidates=info))
        if category == 'raw_miss_correct_candidate_available' and not merged_correct:
            category = 'raw_miss_correct_candidate_suppressed_by_merge'
        categories[category] += 1
        if category not in ('correct_ball','correct_absence'):
            errors.append(dict(clip=row['clip'],frame=row['frame'],category=category,candidates=info,
                               final_outcome=row['outcomes']['final']))
    assert sum(categories.values()) == 900 and len(errors) == 55
    report = dict(complete=True,qualification_evidence=False,
                  initial_audit_sha256=digest(out/'report.json'),acquisition_sha256=digest(out/'stride_one_candidates.json'),
                  script_sha256=digest(Path(__file__)),candidate_evidence_frames=900,
                  oracle_only=True,coverage=dict(coverage),categories=dict(categories),
                  five_previously_unknown_misses=supplemental_misses,all_error_frames=errors,
                  maximum_raw_difference_native_px=max(c['maximum_raw_difference_native_px'] for c in acquired['clips']),
                  acquisition_seconds=sum(c['seconds'] for c in acquired['clips']),
                  metrics_unchanged=initial['metrics'])
    target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'all_error_frames'},indent=2))


if __name__ == '__main__':
    main()
