"""Bound candidate-preserving filtering using the already verified sparse outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.review_wasb_selected_threshold import digest, read, error, outcome


def main():
    out = ROOT/'outputs/vision_upgrade_audit/wasb_head_external01'
    target = out/'filter_feasibility.json'
    if target.exists():
        raise FileExistsError(target)
    report, review = read(out/'report.json'), read(out/'review.json')
    assert report['complete'] and review['complete']
    assert digest(out/'report.json') == review['report_sha256']
    current_path = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01/report.json'
    current = read(current_path)
    assert current['complete'] and current['final_stream_reconstruction_exact']
    for path, checksum in current['sources'].items():
        assert digest(ROOT/path) == checksum
    original = {(r['clip'], r['frame']):r for r in current['rows']}
    seen, visible, covered, current_tp, unrecoverable, potential_gains = set(), 0, 0, 0, [], []
    for c in report['clips']:
        for r in c['rows']:
            key = c['clip'], r['frame']
            assert key not in seen
            seen.add(key)
            old = original[key]
            assert (old['width'], old['height'], old['target_xy'], old['dataset']) == (c['width'], c['height'], r['target_xy'], c['dataset'])
            if r['target_xy'] is None:
                continue
            visible += 1
            before = outcome(old['stages']['final'], r['target_xy'], c['width'], c['height']) == 'correct_ball'
            possible = any(error([p['x'], p['y']], r['target_xy'], c['width'], c['height']) <= 4 for p in r['adapted']['candidates'])
            current_tp += before
            covered += possible
            if before and not possible:
                unrecoverable.append(dict(clip=c['clip'], frame=r['frame']))
            if possible and not before:
                potential_gains.append(dict(clip=c['clip'], frame=r['frame']))
    assert seen == set(original) and len(seen) == 900
    assert current_tp == current['metrics']['tp'] == 803
    assert covered == review['correct_proposal_coverage']['adapted'] == 798
    assert visible == 829
    inspected = ['configs/phase6_analytics/wasb_spaced_residual_pose_small_validation.yaml',
                 'src/pipeline/phase6_pipeline.py', 'src/detection/spaced_ball_stream.py',
                 'src/detection/tiled_wasb_candidates.py', 'src/detection/wasb_ball_detector.py',
                 'src/tracking/stationary_candidate_filter.py', 'src/tracking/candidate_pixel_motion.py',
                 'src/tracking/selected_patch_persistence.py', 'src/tracking/selected_temporal_detours.py']
    result = dict(complete=True, qualification_evidence=False, report_sha256=digest(out/'report.json'),
                  review_sha256=digest(out/'review.json'), current_audit_sha256=digest(current_path),
                  script_sha256=digest(Path(__file__)), inspected_runtime_sources={p:digest(ROOT/p) for p in inspected},
                  labels_exact=True, visible_labels=visible, current_filtered_tp=current_tp,
                  candidate_only_tp_ceiling=covered, candidate_only_recall_ceiling=covered/visible,
                  current_filtered_recall=current_tp/visible, current_correct_without_adapted_proposal=unrecoverable,
                  possible_gains_over_current=potential_gains,
                  raw_external_gate_passed=review['advance_to_continuous'],
                  can_preserve_current_recall_by_candidate_selection=covered >= current_tp,
                  continuous_inference_executed=False,
                  scope='Upper bound for arbitrary selection/rejection from these saved raw candidate coordinates. Not a measured continuous filtered result.',
                  source_inspection='The configured model_top1 path selects original candidate coordinates; stationary/pixel/patch/detour stages reject candidates without synthesizing coordinates. The spaced/tiled stream averages the same real windows for phases of at least three frames.',
                  limitation='A changed model, threshold, coordinate correction or predictive tracker falls outside this bound. Continuous inference equivalence for this checkpoint has not been executed.',
                  decision='HOLD_CONTINUOUS_COMPUTE_FOR_RECALL_PRESERVING_PROMOTION; ADDRESS_PROPOSAL_COVERAGE')
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inspected_runtime_sources', 'current_correct_without_adapted_proposal', 'possible_gains_over_current')}, indent=2))
    print(json.dumps(dict(current_correct_uncovered=len(unrecoverable), possible_gains=len(potential_gains))))


if __name__ == '__main__':
    main()
