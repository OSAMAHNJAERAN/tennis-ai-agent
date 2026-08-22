"""
Phase 6.3 Upstream Failure Audit & Candidate Tracing Script.
Traces every ground-truth hit across video_02, video_03, video_04, video_05 stage-by-stage.
"""

import os
import cv2
import json
import math
from typing import Dict, Any, List
import numpy as np

from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.tracking.player_tracker import PlayerTracker
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.court_geometry import TennisCourtGeometry
from src.court.homography import compute_homography, validate_homography
from src.detection.player_detector import PlayerDetector
from src.tracking.player_tracker import PlayerTracker
from src.tracking.temporal_ball_tracker import TemporalBallTracker, BallState


def main():
    print("=" * 70)
    print("Phase 6.3 Upstream Failure & Stage-by-Stage Trace")
    print("=" * 70)

    with open("data/benchmarks/shot_classification_real_v2/videos.json") as f:
        videos_meta = json.load(f)["videos"]
    with open("data/benchmarks/shot_classification_real_v2/ground_truth.json") as f:
        gt_strokes = json.load(f)["strokes"]

    # Target diagnostic videos: video_02, video_03, video_04, video_05
    target_vids = ["video_02", "video_03", "video_04", "video_05"]
    target_strokes = [s for s in gt_strokes if s["video_id"] in target_vids]

    ball_detector = YOLO11BallDetector(imgsz=1024, high_conf=0.20, low_conf=0.01)
    player_detector = PlayerDetector(model_path="yolo11m.pt")
    court_detector = CourtKeypointDetector(model_path="models/keypoints_model.pth")
    temporal_tracker = TemporalBallTracker(high_conf_thresh=0.20, low_conf_thresh=0.02)

    results_per_stroke = []
    taxonomy_counts = {}

    for vid in target_vids:
        vpath = videos_meta[vid]["path"]
        print(f"\nProcessing {vid} ({vpath})...")
        cap = cv2.VideoCapture(vpath)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Step A: Court
        court_keypoints = court_detector.predict(frames[0])
        canonical_keypoints = TennisCourtGeometry.get_canonical_keypoints()
        h_mat, reproj_err = compute_homography(court_keypoints, canonical_keypoints)
        h_valid = validate_homography(h_mat, court_keypoints, canonical_keypoints)

        # Step B: Players
        all_player_dets = [player_detector.detect_and_track(f, persist=True) for f in frames]
        player_dict = PlayerTracker.choose_players(all_player_dets, court_keypoints)
        p1_boxes = player_dict[1]
        p2_boxes = player_dict[2]

        # Step C: Ball candidates (640 vs 1024)
        cands_1024 = [ball_detector.extract_candidates(f, imgsz=1024) for f in frames]
        cands_640 = [ball_detector.extract_candidates(f, imgsz=640) for f in frames]

        # Step D: Temporal ball track (Phase 6.2 baseline)
        ball_points = temporal_tracker.track_video_candidates(cands_1024, fps=30.0)

        # Trace each stroke for this video
        vid_strokes = [s for s in target_strokes if s["video_id"] in vid]
        for s in vid_strokes:
            sid = s["stroke_id"]
            f_hit = s["frame_hit"]
            stype = s["shot_type"]
            pid = s["player_id"]

            # 1. Window frames [f_hit-2 .. f_hit+2]
            win_frames = list(range(max(0, f_hit - 2), min(len(frames), f_hit + 3)))
            cands_in_win_1024 = [cands_1024[f] for f in win_frames]
            cands_in_win_640 = [cands_640[f] for f in win_frames]

            has_cand_1024 = any(len(c) > 0 for c in cands_in_win_1024)
            has_cand_640 = any(len(c) > 0 for c in cands_in_win_640)
            max_conf_1024 = max([c.confidence for sub in cands_in_win_1024 for c in sub], default=0.0)
            max_conf_640 = max([c.confidence for sub in cands_in_win_640 for c in sub], default=0.0)

            # 2. Tracker state at f_hit
            t_pt = ball_points[f_hit] if f_hit < len(ball_points) else None
            t_state = t_pt.state.value if t_pt else "MISSING"

            # 3. Player detection at f_hit
            p1_box = p1_boxes[f_hit] if f_hit < len(p1_boxes) else None
            p2_box = p2_boxes[f_hit] if f_hit < len(p2_boxes) else None
            target_pbox = p1_box if pid == 1 else p2_box
            player_detected = target_pbox is not None

            # 4. Failure Diagnosis
            if not has_cand_1024 and not has_cand_640:
                primary_failure = "BALL_NOT_DETECTED"
            elif has_cand_1024 and max_conf_1024 < 0.20 and t_state in ("MISSING", "PREDICTED"):
                primary_failure = "BALL_LOW_CONFIDENCE_REJECTED"
            elif t_state == "MISSING":
                primary_failure = "BALL_TRACK_LOST"
            elif not player_detected:
                primary_failure = "PLAYER_DETECTION_FAILED"
            elif not h_valid:
                primary_failure = "COURT_GEOMETRY_FAILED"
            elif t_state in ("TRACKED", "DETECTED") and player_detected:
                primary_failure = "HIT_CANDIDATE_NOT_GENERATED"
            else:
                primary_failure = "BALL_GATE_REJECTED"

            taxonomy_counts[primary_failure] = taxonomy_counts.get(primary_failure, 0) + 1

            record = {
                "stroke_id": sid,
                "video_id": vid,
                "frame_hit": f_hit,
                "shot_type": stype,
                "player_id": pid,
                "has_candidate_1024": has_cand_1024,
                "max_conf_1024": round(max_conf_1024, 4),
                "has_candidate_640": has_cand_640,
                "max_conf_640": round(max_conf_640, 4),
                "tracker_state_at_hit": t_state,
                "player_detected": player_detected,
                "court_valid": h_valid,
                "primary_failure_reason": primary_failure
            }
            results_per_stroke.append(record)
            print(f"  Stroke {sid:02d} (f={f_hit:03d}, {stype:8s}, P{pid}): "
                  f"Cand1024={has_cand_1024} (max_conf={max_conf_1024:.2f}), "
                  f"Track={t_state:10s}, Player={player_detected} -> Reason: {primary_failure}")

    print("\n" + "=" * 70)
    print("PRIMARY ROOT-CAUSE DISTRIBUTION ACROSS 23 REAL STROKES")
    print("=" * 70)
    for reason, count in sorted(taxonomy_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(target_strokes)) * 100.0
        print(f"  {reason:30s}: {count:2d} / {len(target_strokes)} ({pct:5.1f}%)")

    # Save to JSON
    os.makedirs("docs/audit", exist_ok=True)
    with open("outputs/phase6_3_upstream_audit.json", "w") as f:
        json.dump({"strokes": results_per_stroke, "taxonomy": taxonomy_counts}, f, indent=2)

    print("\nSaved upstream failure audit to outputs/phase6_3_upstream_audit.json")


if __name__ == "__main__":
    main()
