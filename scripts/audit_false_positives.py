"""Persist an auditable false-positive table for cross-match diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

try:
    from scripts.evaluate_phase6_4_cross_match import _event_type, _frame, one_to_one_matches
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_phase6_4_cross_match import _event_type, _frame, one_to_one_matches

from src.events.event_evaluator import frame_inside_annotation_coverage


def _load(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _classify(prediction, gt_events, rally_end, dead_ids, tolerance):
    frame = _frame(prediction)
    event_type = _event_type(prediction)
    evidence = prediction.get("evidence", {})
    if prediction.get("event_id") in dead_ids or frame > rally_end:
        return "POST_RALLY_BALL_NOISE"

    nearest = min(gt_events, key=lambda gt: abs(_frame(gt) - frame))
    difference = abs(_frame(nearest) - frame)
    nearest_type = _event_type(nearest)
    if difference <= tolerance and event_type != nearest_type:
        if event_type == "PLAYER_HIT" and nearest_type == "BOUNCE":
            return "BOUNCE_AS_HIT"
        if event_type == "BOUNCE" and nearest_type == "PLAYER_HIT":
            return "HIT_AS_BOUNCE"
        if event_type == "SERVE_CONTACT":
            return "FALSE_SERVE"
        return "UNKNOWN_CAUSE"
    if difference <= 20:
        return "TIMING_DRIFT"
    if event_type == "PLAYER_HIT":
        distances = [
            value for value in (
                evidence.get("player1_distance_px"), evidence.get("player2_distance_px")
            ) if value is not None
        ]
        if distances and min(distances) > 100:
            return "PLAYER_PROXIMITY_ONLY"
    if evidence.get("ball_state") in ("PREDICTED", "INTERPOLATED", "OCCLUDED"):
        return "LOW_CONF_BALL_ARTIFACT"
    if evidence.get("trajectory_continuity") is False:
        return "TRAJECTORY_DISCONTINUITY"
    return "BALL_NOISE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", default="outputs/phase6_4_qualification/final_cross_match_holdout"
    )
    parser.add_argument(
        "--output-json", default="artifacts/validation/phase6_4_false_positive_audit.json"
    )
    parser.add_argument(
        "--output-csv", default="artifacts/validation/phase6_4_false_positive_audit.csv"
    )
    args = parser.parse_args()

    events_gt = _load("data/benchmarks/cross_match_final_holdout/ground_truth_events.json")["events"]
    videos = _load("data/benchmarks/cross_match_final_holdout/videos.json")["videos"]
    coverage = _load(
        "data/benchmarks/cross_match_final_holdout/annotation_coverage.json"
    )
    rows = []
    outside_scope_count = 0
    for video_id in ("video_08", "video_09", "video_10"):
        output_dir = os.path.join(args.output_root, video_id)
        predictions = _load(os.path.join(output_dir, "match_events.json")).get("events", [])
        gt = events_gt[video_id]
        fps = float(videos[video_id]["fps"])
        tolerance = max(1, round(0.2 * fps))
        covered_predictions = [
            prediction
            for prediction in predictions
            if frame_inside_annotation_coverage(
                video_id, _frame(prediction), coverage
            )
        ]
        outside_scope_count += len(predictions) - len(covered_predictions)
        matches = one_to_one_matches(
            covered_predictions, gt, tolerance, require_event_type=True, fps=fps
        )
        matched_predictions = {pred_index for pred_index, _, _ in matches}
        scoring_path = os.path.join(output_dir, "scoring_events.json")
        scoring = _load(scoring_path).get("scoring_events", []) if os.path.exists(scoring_path) else []
        dead_ids = {
            item["event_id"] for item in scoring
            if item.get("outcome_type") == "DEAD_BALL_IGNORED"
        }
        rally_end = max(
            int(interval["end_frame"])
            for interval in coverage["videos"][video_id]["fully_reviewed_intervals"]
        )
        for pred_index, prediction in enumerate(covered_predictions):
            if pred_index in matched_predictions:
                continue
            frame = _frame(prediction)
            nearest = min(gt, key=lambda item: abs(_frame(item) - frame))
            rows.append({
                "video_id": video_id,
                "prediction_event_id": prediction.get("event_id"),
                "frame_index": frame,
                "timestamp_s": prediction.get("timestamp_s", frame / fps),
                "predicted_type": _event_type(prediction),
                "predicted_player_id": prediction.get("player_id"),
                "nearest_gt_event_id": nearest.get("event_id"),
                "nearest_gt_frame": _frame(nearest),
                "nearest_gt_type": _event_type(nearest),
                "timing_delta_frames": frame - _frame(nearest),
                "taxonomy": _classify(prediction, gt, rally_end, dead_ids, tolerance),
                "trajectory_state": prediction.get("trajectory_state"),
                "confidence": prediction.get("confidence"),
            })

    taxonomy = Counter(row["taxonomy"] for row in rows)
    report = {
        "schema_version": "1.0",
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC",
        "qualification_evidence": False,
        "matching": "PER_VIDEO_CANONICAL_TIMESTAMP_INTERVAL_ONE_TO_ONE",
        "tolerance_seconds": 0.2,
        "false_positive_count": len(rows),
        "outside_scope_prediction_count": outside_scope_count,
        "taxonomy_counts": dict(sorted(taxonomy.items())),
        "false_positives": rows,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["video_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"false_positive_count": len(rows), "taxonomy_counts": taxonomy}, indent=2))


if __name__ == "__main__":
    main()
