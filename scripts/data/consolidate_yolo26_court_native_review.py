"""Normalize three completed manual native court reviews into explicit decisions.

This reads recorded per-item visual observations; it performs no inference,
automatic geometric acceptance, image rendering, network access or export.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.data.prepare_yolo26_court_review import ROOT, context_components, load, relative, save, sha, validate_review


EVENT_ALIASES = {
    "Monte_Carlo": "MonteCarlo", "London_Finals": "LondonFinals",
    "Cincinnati_Western_Southern": "Cincinnati", "Indian_Wells": "IndianWells",
    # The reviewer confirmed literal MONTREAL 375 in source image100 on second look.
    "Canada_Rogers": "Montreal",
}
CLAY = {"Rome", "Madrid", "MonteCarlo", "BuenosAires", "Rio", "Barcelona", "Lyon", "Marrakech"}
GRASS = {"Halle", "Queens", "Mallorca"}


def consolidate(review: Path, detail: Path, output: Path):
    if output.exists():
        raise FileExistsError("Preserve prior decisions; write a fresh version")
    inventory_path = review / "cache_inventory.json"
    inventory = load(inventory_path)
    receipt_path = detail / "native_detail_receipt.json"
    receipt = load(receipt_path)
    if receipt["inventory_sha256"] != sha(inventory_path):
        raise ValueError("Manual review boards and native inventory disagree")
    inputs = [review / "native_review_head_observations.json", review / "native_review_middle.json",
              review / "native_review_tail.json"]
    head, middle, tail = map(load, inputs)
    observed = {}
    for number, event, nulls, decision, note in head["observations"]:
        observed[number] = {"number": number, "visual_event_family": event, "null_keypoint_indices": nulls,
                            "decision": decision, "review_note": note, "reviewer_type": head["reviewer_type"],
                            "manual_review_document": relative(inputs[0])}
    for path, document in zip(inputs[1:], (middle, tail)):
        for row in document["samples"]:
            if row["number"] in observed:
                raise ValueError("Repeated manual image decision")
            observed[row["number"]] = row | {"manual_review_document": relative(path)}
    if set(observed) != {sample["number"] for sample in inventory["samples"]}:
        raise ValueError("Require exhaustive 165-image manual review")
    board_by_id = {ident: board for board in receipt["boards"] for ident in board["sample_ids"]}
    rows = []
    for sample in inventory["samples"]:
        actual = observed[sample["number"]]
        nulls, decision = actual["null_keypoint_indices"], actual["decision"]
        if decision not in ("ACCEPT", "REJECT") or actual["reviewer_type"] != "AGENT_VISUAL_REVIEW":
            raise ValueError("Each image requires an explicit actual visual decision")
        event_original = actual["visual_event_family"]
        event = EVENT_ALIASES.get(event_original, event_original)
        points = ([None if index in nulls else point + [2]
                   for index, point in enumerate(sample["publisher_keypoints_xy_px"])] if decision == "ACCEPT" else None)
        if decision == "ACCEPT" and actual.get("reviewed_keypoints_px") is not None and actual["reviewed_keypoints_px"] != points:
            raise ValueError("Consolidation must preserve manual point decisions exactly")
        board = board_by_id[sample["id"]]
        proposed = sample["publisher_keypoints_xy_px"]
        bbox = [min(point[0] for point in proposed), min(point[1] for point in proposed),
                max(point[0] for point in proposed), max(point[1] for point in proposed)]
        rows.append({
            "number": sample["number"], "id": sample["id"], "source_video_id": sample["source_video_id"],
            "visual_event_family": event, "original_manual_event_label": event_original,
            "event_name_is_independently_verified_match_identity": False,
            "surface": "clay" if event in CLAY else "grass" if event in GRASS else "hard",
            "decision": decision, "reviewer_type": "AGENT_VISUAL_REVIEW",
            "scope": "elevated_end_court_singles" if decision == "ACCEPT" else "REJECTED_SCOPE_OR_LABELS",
            "review_note": actual["review_note"],
            "rejection_reason": actual.get("rejection_reason") or (actual["review_note"] if decision == "REJECT" else None),
            "semantic_order_reviewed": decision == "ACCEPT", "reviewed_keypoints_px": points,
            "null_keypoint_indices": nulls if decision == "ACCEPT" else [],
            "null_visibility_note": actual["review_note"] if nulls and decision == "ACCEPT" else None,
            "keypoint_visibility_decisions": ["UNKNOWN_NULL" if point is None else "VISIBLE_2" for point in points] if points is not None else None,
            "native_detail_board": board["image"], "native_detail_board_sha256": board["sha256"],
            "native_patch_reviewed_indices": list(range(14)),
            "manual_review_document": actual["manual_review_document"],
            "manual_review_document_sha256": sha(ROOT / actual["manual_review_document"]),
            "bbox_xyxy_px": bbox if decision == "ACCEPT" else None,
            "bbox_provenance": "PUBLISHER_OUTER_COURT_ENVELOPE_NATIVE_VISUAL_EXTENT_REVIEW; OCCLUDED_EXTENT_IS_A_LABELLED_ESTIMATE_NOT_METRIC_GROUND_TRUTH" if decision == "ACCEPT" else None,
            "coordinate_corrections_made": False,
        })
    by_video, contexts = context_components(inventory["samples"], rows)
    assignments = {group: "val" if set(context["visual_event_families"]) & {"MonteCarlo", "LondonFinals"} else "train"
                   for group, context in contexts.items()}
    for row in rows:
        row["context_group"] = by_video[row["source_video_id"]]
        # A rejected frame contributes lineage and gets its full group's assignment,
        # but its REJECT decision always prevents training/validation inclusion.
        row["development_split"] = assignments[row["context_group"]]
    decisions = {
        "schema_version": "1.1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EXHAUSTIVE_AGENT_NATIVE_REVIEW_AWAITING_ROOT_SECOND_PASS_NO_EXPORT",
        "inventory_sha256": sha(inventory_path), "native_detail_receipt": relative(receipt_path),
        "native_detail_receipt_sha256": sha(receipt_path),
        "manual_review_documents": [{"path": relative(path), "sha256": sha(path)} for path in inputs],
        "event_aliases": EVENT_ALIASES, "geography_is_visual_conservative_context_only": True,
        "context_assignment_rule": "WHOLE_TRANSITIVE_MONTECARLO_AND_LONDONFINALS_CONTEXTS_DEV_VAL; OTHER_CONTEXTS_TRAIN",
        "context_groups": contexts, "source_ids_and_events_from_rejections_retained": True,
        "independent_human_reviewed": False, "qualification_evidence": False, "final_test_images": 0,
        "model_predictions_used": False, "images_reviewed_at_native_patch_scale": 165,
        "samples": rows,
    }
    accepted, _ = validate_review(inventory, decisions)
    decisions["counts"] = {"images": len(rows), "accepted": len(accepted), "rejected": len(rows) - len(accepted),
                           "accepted_split_images": dict(Counter(row["development_split"] for row in accepted)),
                           "accepted_visible_keypoints": sum(point is not None for row in accepted for point in row["reviewed_keypoints_px"]),
                           "accepted_null_keypoints": sum(point is None for row in accepted for point in row["reviewed_keypoints_px"]),
                           "transitive_context_groups": len(contexts), "test": 0}
    save(output, decisions)
    return {"output": str(output), "sha256": sha(output), "counts": decisions["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(consolidate(args.review, args.detail, args.output)))


if __name__ == "__main__":
    main()
