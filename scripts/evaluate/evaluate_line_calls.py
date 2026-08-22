import os
import json
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.line_calling.line_geometry import CourtLineGeometry, CourtLineType, ServiceBoxType
from src.line_calling.line_call_engine import TennisLineCallEngine, LineCallDecision, LineCallContext, LineCallEvidence
from src.line_calling.contact_refinement import BounceContactRefiner

def main():
    print("=" * 75)
    print("T88J709 PHASE 4: INDEPENDENT LINE CALLING & UNCERTAINTY BENCHMARK")
    print("=" * 75)

    gt_path = "data/benchmarks/line_calls_independent/ground_truth.json"
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)

    # 1. Load Trajectory & Homography for Real Video Bounces
    traj_path = "outputs/phase2_yolo11_final/trajectories.json"
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)['ball_trajectory']

    ball_points = [
        TemporalBallPoint(
            frame_index=p['frame_index'],
            timestamp_seconds=p['timestamp_seconds'],
            x_px=p['x_px'],
            y_px=p['y_px'],
            court_x_m=p['court_x_m'],
            court_y_m=p['court_y_m'],
            speed_kmh=p['speed_kmh'],
            confidence=p['confidence'],
            state=BallState(p['state'])
        )
        for p in traj_data
    ]

    # Reference Homography from 14 manual landmarks
    with open("data/benchmarks/tennis_events_independent/ground_truth.json", 'r') as f:
        events_gt = json.load(f)
    ref_landmarks = np.array([k['px'] for k in events_gt['manual_reference_court_landmarks_px']], dtype=np.float32)
    can_kps = TennisCourtGeometry.get_canonical_keypoints()
    H_ref, _ = compute_homography(ref_landmarks, can_kps)

    engine = TennisLineCallEngine()
    results = []

    # 2. Evaluate Real Video Bounces
    print("\n[1/2] Evaluating Real Video Bounces...")
    for idx, case in enumerate(gt_data['real_video_bounces']):
        f_idx = case['bounce_frame']
        ctx = LineCallContext(case['context'])
        target_box = ServiceBoxType(case['target_service_box']) if case.get('target_service_box') else None

        evidence = engine.evaluate_bounce(
            event_id=idx + 1,
            bounce_frame=f_idx,
            ball_trajectory=ball_points,
            homography_matrix=H_ref,
            context=ctx,
            target_service_box=target_box
        )
        results.append({
            "case_id": case['case_id'],
            "source": "REAL_VIDEO",
            "frame": f_idx,
            "context": ctx.value,
            "expected_decision": case['expected_decision'],
            "predicted_decision": evidence.decision.value,
            "nearest_line": evidence.nearest_line.value,
            "expected_nearest_line": case['nearest_line'],
            "margin_cm": evidence.ball_edge_margin_cm,
            "uncertainty_cm": evidence.position_uncertainty_cm,
            "spatial_tier": evidence.spatial_tier,
            "confidence": evidence.confidence,
            "is_correct": (evidence.decision.value == case['expected_decision']),
            "reason": evidence.reason
        })
        print(f"  Case: {case['case_id']:16s} | Pred: {evidence.decision.value:15s} | Exp: {case['expected_decision']:12s} | Margin: {evidence.ball_edge_margin_cm:+6.1f}cm | Unc: ±{evidence.position_uncertainty_cm:4.1f}cm | Tier: {evidence.spatial_tier}")

    # 3. Evaluate Synthetic Geometry Cases
    print("\n[2/2] Evaluating Deterministic Synthetic Geometry Suite...")
    for case in gt_data['synthetic_geometry_suite']:
        ctx = LineCallContext(case['context'])
        target_box = ServiceBoxType(case['target_service_box']) if case.get('target_service_box') else None
        c_x, c_y = case['court_position_m']
        st = BallState(case['tracker_state'])

        # Mock single bounce trajectory
        pts = [
            TemporalBallPoint(frame_index=0, timestamp_seconds=0.0, x_px=500.0, y_px=500.0, state=st, confidence=0.9),
            TemporalBallPoint(frame_index=1, timestamp_seconds=0.033, x_px=500.0, y_px=500.0, state=st, confidence=0.9),
            TemporalBallPoint(frame_index=2, timestamp_seconds=0.067, x_px=500.0, y_px=500.0, state=st, confidence=0.9)
        ]
        # Identity scaling
        H_syn = np.array([
            [c_x / 500.0, 0.0, 0.0],
            [0.0, c_y / 500.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        evidence = engine.evaluate_bounce(
            event_id=len(results) + 1,
            bounce_frame=1,
            ball_trajectory=pts,
            homography_matrix=H_syn,
            context=ctx,
            target_service_box=target_box
        )
        results.append({
            "case_id": case['case_id'],
            "source": "SYNTHETIC_SUITE",
            "frame": None,
            "context": ctx.value,
            "expected_decision": case['expected_decision'],
            "predicted_decision": evidence.decision.value,
            "nearest_line": evidence.nearest_line.value,
            "expected_nearest_line": case.get('nearest_line', 'N/A'),
            "margin_cm": evidence.ball_edge_margin_cm,
            "uncertainty_cm": evidence.position_uncertainty_cm,
            "spatial_tier": evidence.spatial_tier,
            "confidence": evidence.confidence,
            "is_correct": (evidence.decision.value == case['expected_decision']),
            "reason": evidence.reason
        })
        print(f"  Case: {case['case_id']:26s} | Pred: {evidence.decision.value:15s} | Exp: {case['expected_decision']:15s} | Tier: {evidence.spatial_tier:4s} | Correct: {evidence.decision.value == case['expected_decision']}")

    # 4. Compute Aggregate Metrics
    total_cases = len(results)
    correct_count = sum(1 for r in results if r['is_correct'])
    
    # Decisions breakdown
    automatic_decisions = [r for r in results if r['predicted_decision'] in ("IN", "OUT", "SERVE_IN", "SERVE_FAULT")]
    abstentions = [r for r in results if r['predicted_decision'] in ("REVIEW_REQUIRED", "UNKNOWN")]
    
    coverage_rate = len(automatic_decisions) / total_cases
    abstention_rate = len(abstentions) / total_cases

    # Accuracy among automatic decisions
    auto_correct = sum(1 for r in automatic_decisions if r['is_correct'])
    auto_accuracy = auto_correct / len(automatic_decisions) if automatic_decisions else 0.0

    # False-IN and False-OUT
    false_in = sum(1 for r in results if r['predicted_decision'] in ("IN", "SERVE_IN") and r['expected_decision'] in ("OUT", "SERVE_FAULT"))
    false_out = sum(1 for r in results if r['predicted_decision'] in ("OUT", "SERVE_FAULT") and r['expected_decision'] in ("IN", "SERVE_IN"))

    # Margin Buckets
    buckets = {
        ">20cm": {"total": 0, "correct": 0},
        "10-20cm": {"total": 0, "correct": 0},
        "5-10cm": {"total": 0, "correct": 0},
        "2-5cm": {"total": 0, "correct": 0},
        "0-2cm": {"total": 0, "correct": 0}
    }
    for r in results:
        m = abs(r['margin_cm'])
        if m > 20.0:
            b = ">20cm"
        elif m >= 10.0:
            b = "10-20cm"
        elif m >= 5.0:
            b = "5-10cm"
        elif m >= 2.0:
            b = "2-5cm"
        else:
            b = "0-2cm"
        buckets[b]["total"] += 1
        if r['is_correct']:
            buckets[b]["correct"] += 1

    summary = {
        "benchmark_id": "tennis_line_calls_independent_v1",
        "total_cases_evaluated": total_cases,
        "overall_accuracy": round(correct_count / total_cases, 4),
        "safety_metrics": {
            "automatic_call_coverage_pct": round(coverage_rate * 100.0, 2),
            "abstention_rate_pct": round(abstention_rate * 100.0, 2),
            "accuracy_among_automatic_calls_pct": round(auto_accuracy * 100.0, 2),
            "false_in_count": false_in,
            "false_out_count": false_out
        },
        "distance_to_line_performance": {
            b: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy_pct": round((stats["correct"] / stats["total"]) * 100.0, 1) if stats["total"] > 0 else 0.0
            }
            for b, stats in buckets.items()
        },
        "detailed_results": results
    }

    os.makedirs("experiments/phase4_line_calls", exist_ok=True)
    out_file = "experiments/phase4_line_calls/line_call_benchmark_results.json"
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print("LINE CALL EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Total Benchmark Cases:                 {total_cases}")
    print(f"Overall Accuracy:                      {summary['overall_accuracy']*100:.1f}%")
    print(f"Automatic-Call Coverage:               {summary['safety_metrics']['automatic_call_coverage_pct']:.1f}%")
    print(f"Abstention Rate (REVIEW_REQUIRED):     {summary['safety_metrics']['abstention_rate_pct']:.1f}%")
    print(f"Accuracy Among Automatic Calls:        {summary['safety_metrics']['accuracy_among_automatic_calls_pct']:.1f}%")
    print(f"False-IN Calls:                        {false_in}")
    print(f"False-OUT Calls:                       {false_out}")
    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()
