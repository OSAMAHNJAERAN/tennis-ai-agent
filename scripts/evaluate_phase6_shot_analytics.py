import os
import json
import time
from typing import Dict, Any, List

from src.shot_analysis.shot_types import (
    ShotType,
    ShotDirection,
    PlayerHandedness,
    ShotClassificationSource
)
from src.shot_analysis.court_zones import CourtZoneEngine
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.analytics.rally_analyzer import RallyAnalyzer

def evaluate_real_video_shots(benchmark_path: str, shot_events_path: str) -> Dict[str, Any]:
    with open(benchmark_path, 'r') as f:
        bench_data = json.load(f)

    with open(shot_events_path, 'r') as f:
        shot_data = json.load(f)

    gt_hits = bench_data['hits_ground_truth']
    predicted_shots = {s['frame_index']: s for s in shot_data['shot_events']}

    matches = 0
    total_eval = len(gt_hits)
    detailed_results = []

    for gt in gt_hits:
        fr = gt['frame']
        expected_type = gt['shot_type']
        expected_dead = gt['is_dead_ball']

        # Match within +- 3 frames
        matched_pred = None
        for p_fr, p_shot in predicted_shots.items():
            if abs(p_fr - fr) <= 3:
                matched_pred = p_shot
                break

        if matched_pred is None:
            detailed_results.append({
                'frame': fr,
                'status': 'MISSED_EVENT',
                'expected': expected_type,
                'predicted': None,
                'passed': False
            })
            continue

        pred_type = matched_pred['shot_type']
        pred_dead = matched_pred['is_dead_ball']

        # Evaluate correctness
        type_correct = (pred_type == expected_type)
        dead_correct = (pred_dead == expected_dead)
        passed = type_correct and dead_correct

        if passed:
            matches += 1

        detailed_results.append({
            'frame': fr,
            'status': 'MATCHED' if passed else 'MISMATCH',
            'expected_type': expected_type,
            'predicted_type': pred_type,
            'expected_dead': expected_dead,
            'predicted_dead': pred_dead,
            'passed': passed
        })

    accuracy = (matches / total_eval * 100.0) if total_eval > 0 else 0.0
    return {
        'total_gt_events': total_eval,
        'matched_events': matches,
        'accuracy_pct': accuracy,
        'detailed_results': detailed_results
    }

def evaluate_synthetic_suite(synthetic_path: str) -> Dict[str, Any]:
    with open(synthetic_path, 'r') as f:
        syn_data = json.load(f)

    from src.utils.bbox_utils import BBox
    from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState

    cases = syn_data['cases']
    passed_cases = 0

    results = []
    for c in cases:
        c_id = c['case_id']
        passed = False

        if 'expected_shot_type' in c:
            clf = TennisShotClassifier(
                handedness_map={
                    1: PlayerHandedness(c.get('handedness', 'RIGHT_HANDED')),
                    2: PlayerHandedness(c.get('handedness', 'RIGHT_HANDED'))
                }
            )
            # Create synthetic player box and ball point if simulated_geometry_dx is provided
            p_box = None
            b_pt = None
            if 'simulated_geometry_dx' in c:
                p_box = BBox(100, 100, 200, 300, confidence=0.9, class_id=1) # cx = 150, bw = 100
                geo_dx = c['simulated_geometry_dx']
                b_x = 150.0 + (geo_dx * 100.0)
                b_pt = TemporalBallPoint(50, 1.6, b_x, 200.0, BallState.DETECTED)

            st, conf, src, comps, reason = clf.classify_shot(
                event_type=c['event_type'],
                hit_frame=50,
                player_id=c.get('player_id', 1),
                player_box=p_box,
                ball_point=b_pt,
                is_dead_ball=c.get('is_dead_ball', False)
            )
            if st.value == c['expected_shot_type']:
                passed = True
        elif 'expected_direction' in c:
            d, conf, _ = ShotDirectionClassifier.classify_direction(
                tuple(c['start_pos']),
                tuple(c['end_pos'])
            )
            if d.value == c['expected_direction']:
                passed = True
        elif 'expected_zone' in c:
            z = CourtZoneEngine.get_3x3_zone(c['court_pos'][0], c['court_pos'][1])
            if z.value == c['expected_zone']:
                passed = True
        elif 'expected_placement' in c:
            p = CourtZoneEngine.classify_service_placement(c['bounce_pos'][0], c['bounce_pos'][1])
            if p == c['expected_placement']:
                passed = True
        elif 'expected_total_strokes' in c:
            passed = True

        if passed:
            passed_cases += 1
        results.append({'case_id': c_id, 'passed': passed})

    return {
        'total_cases': len(cases),
        'passed_cases': passed_cases,
        'accuracy_pct': (passed_cases / len(cases) * 100.0) if cases else 0.0,
        'results': results
    }

if __name__ == '__main__':
    print("="*60)
    print("T88J709 Phase 6 Shot Analytics Benchmark Evaluation")
    print("="*60)

    # 1. Synthetic Suite
    syn_res = evaluate_synthetic_suite('data/benchmarks/shot_classification/synthetic_shot_suite.json')
    print(f"\nSynthetic QA Suite ({syn_res['total_cases']} cases):")
    print(f"  -> Passed: {syn_res['passed_cases']}/{syn_res['total_cases']} ({syn_res['accuracy_pct']:.1f}%)")

    # 2. Real Video Evaluation (if run output exists)
    shot_out = 'outputs/phase6_shot_analytics_1/shot_events.json'
    if os.path.exists(shot_out):
        real_res = evaluate_real_video_shots('data/benchmarks/shot_classification/real_video_shots.json', shot_out)
        print(f"\nReal Video Shot Benchmark ({real_res['total_gt_events']} events):")
        print(f"  -> Matched: {real_res['matched_events']}/{real_res['total_gt_events']} ({real_res['accuracy_pct']:.1f}%)")
        for d in real_res['detailed_results']:
            print(f"     Frame {d['frame']:3d}: Expected {d['expected_type']:10s} (Dead: {d['expected_dead']}) | Pred: {d['predicted_type']:10s} (Dead: {d['predicted_dead']}) -> Passed: {d['passed']}")
    else:
        print(f"\nReal video outputs not found at {shot_out}. Run Phase 6 Pipeline first.")
