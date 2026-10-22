"""Record the agent's actual local visual review, preserving every proposal."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "outputs/ball_yolo_author_native_audit32_v1/review_aids_manifest.json"
EXPECTED = "bab9ae8e201574ec443172f9607fd2e2bec796facc2f0d0c01f183127b8a924a"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if digest(MANIFEST) != EXPECTED:
        raise ValueError("frozen native review aids manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    decisions = {
        138: ("ACCEPT_VISIBLE_PUBLISHER_BOX_PENDING_CROP_CLOSURE", None, "VISIBLE_SMALL_BALL", "FAR_COURT", "LOW", "Small pale ball separates from the green rear wall and moves left/down across native frames137-139. The approximately5x5px proposal is centered on the visible blob with at most a small compression halo."),
        139: ("ACCEPT_VISIBLE_PUBLISHER_BOX_PENDING_CROP_CLOSURE", None, "VISIBLE_SMALL_BALL", "FAR_COURT", "LOW", "The small pale ball continues below the green rear wall in frames138-140. The approximately3x3px proposal follows the visible tiny blob. Adjacentframe138 is the same flight, not an independent source."),
        305: ("REJECT_UNRESOLVED_PLAYER_OVERLAP", None, "AMBIGUOUS_OR_OCCLUDED", "NEAR_COURT", "UNRESOLVED", "The proposal is on the near player's black dress/torso. The unmarked neighbors contain the player's pale limbs and racket, but do not resolve a separate visible ball-only extent at the proposed center. Do not infer the hidden ball or accept a black-dress box."),
        480: ("REJECT_UNRESOLVED_OFFCOURT_COLLECTION_OBJECT", None, "AMBIGUOUS_OFFCOURT_BLUR", "OUTSIDE_LEFT_REAR_COURT", "UNRESOLVED", "The pale proposal is beside the far-left ballkid's feet during a collection/dead-ball-looking interval, while other ballkids are moving. Native temporal patches show a pale streak but do not resolve a tight ball-only extent or an active-ball interpretation reliably. Exclusion is not an absence label."),
        706: ("ACCEPT_VISIBLE_PUBLISHER_BOX_PENDING_CROP_CLOSURE", None, "VISIBLE_SMALL_BALL", "FAR_COURT_NEAR_NET", "MODERATE", "A tiny pale vertical ball streak is distinct from the horizontal net tape in all three unmarked native patches and moves upward in image coordinates. The approximately3x5px publisher bounds align with the visible streak."),
        875: ("ACCEPT_REVIEWED_VISIBLE_EXTENT_PENDING_CROP_CLOSURE", [721, 362, 728, 376], "VISIBLE_VERTICAL_MOTION_BLUR", "NEAR_COURT", "HIGH", "A detached pale vertical ball streak moves upward across frames874-876 against clay below the net. Tightened the publisher approximately10x17px bounds to a7x14px visible streak envelope; this excludes surrounding clay while retaining the visible blur, not a guessed spherical or physical center."),
        1041: ("ACCEPT_REVIEWED_VISIBLE_EXTENT_PENDING_CROP_CLOSURE", [696, 204, 703, 210], "VISIBLE_SMALL_BLURRED_BALL", "FAR_COURT", "MODERATE", "A compact pale ball is isolated against clay, moving right/up across frames1040-1042. Tightened approximately9x8px publisher bounds to a7x6px visible blob envelope. The far player's racket is spatially separate."),
        1182: ("REJECT_UNRESOLVED_BALL_LINE_EXTENT", None, "AMBIGUOUS_LINE_OVERLAP", "FAR_COURT", "UNRESOLVED", "The proposed center lies on the slanted bright singles/service line. Neighbor patches show a bright spot along that line, but the current ball-only outline cannot be separated from the line at native resolution. Do not turn the line into a ball box or use temporal expectation as hidden ground truth."),
        1347: ("ACCEPT_VISIBLE_PUBLISHER_BOX_PENDING_CROP_CLOSURE", None, "VISIBLE_COMPACT_BALL_BLUR", "FAR_COURT", "MODERATE", "The pale ball is clearly detached from the far player/racket and travels right/up over clay in frames1346-1348. The approximately9x7px publisher proposal encloses its compact blur and small visible halo within manual boundary uncertainty."),
        1520: ("ACCEPT_REVIEWED_VISIBLE_EXTENT_PENDING_CROP_CLOSURE", [754, 274, 760, 280], "VISIBLE_SMALL_BLURRED_BALL", "FAR_COURT_NEAR_SERVICE_LINE", "MODERATE", "A compact pale moving ball is distinct from the farther horizontal line/net tape across frames1519-1521. Tightened approximately8x8px publisher bounds to a6x6px visible blob envelope; a nearby dark line/shadow is excluded."),
        1709: ("ACCEPT_REVIEWED_VISIBLE_EXTENT_PENDING_CROP_CLOSURE", [632, 188, 640, 194], "VISIBLE_COMPACT_BALL_BLUR", "FAR_COURT", "MODERATE", "A pale compact ball moves left/down over clay, clearly separated from the central Lacoste sign and far player. Tightened approximately10x8px publisher bounds to an8x6px visible blur envelope."),
        1878: ("REJECT_UNRESOLVED_OFFICIAL_SIGN_BACKGROUND", None, "AMBIGUOUS_SIGN_OR_BALL", "FAR_COURT_REAR_BOUNDARY", "UNRESOLVED", "The tiny proposal is against the central official/Lacoste sign. The unmarked temporal triplet does not show a distinct ball silhouette or displacement separable from the sign/official details. Preserve the proposal and reject rather than labeling a background feature or guessed hidden ball."),
    }
    samples = [sample for sample in manifest["samples"] if sample["alias"] == "France_Review_Red_Court_1" and sample["frame_index"] in decisions]
    if len(samples) != 12 or set(sample["frame_index"] for sample in samples) != set(decisions):
        raise ValueError("exact assigned twelve source frames are required")
    rows = []
    for sample in samples:
        frame = sample["frame_index"]
        decision, correction, state, region, blur, explanation = decisions[frame]
        for asset in [sample["native_overlay"], sample["unmarked_native_detail_board"], *sample["native_frames"]]:
            if digest(ROOT / asset["path"]) != asset["sha256"]:
                raise ValueError("review image/source hash changed")
        approved = decision.startswith("ACCEPT_")
        if len(sample["boxes"]) != 1:
            raise ValueError("assigned review expects one publisher proposal per frame")
        selected = correction if correction is not None else sample["boxes"][0]["publisher_box_xyxy_native_px"] if approved else None
        rows.append({
            "id": sample["id"], "source_alias": sample["alias"], "source_frame_index": frame,
            "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
            "source_timestamp_fraction": sample["source_timestamp_fraction"],
            "label_source_path": sample["label_source_path"], "label_sha256": sample["label_sha256"],
            "label_git_blob_sha1": sample["label_git_blob_sha1"], "native_width": 1280, "native_height": 720,
            "reviewer": "AGENT:/root/ball_bbox_inventory", "root_personally_viewed_by_this_receipt": False,
            "independent_human_review": False, "vision_model_or_model_predictions_used": False,
            "actually_viewed_native_overlay_locally": True, "actually_viewed_all_unmarked_native96_temporal_patches_locally": True,
            "native_neighbor_indices": sample["native_neighbor_indices"], "native_neighbor_pts": sample["native_neighbor_pts"],
            "native_overlay": sample["native_overlay"], "unmarked_native_detail_board": sample["unmarked_native_detail_board"],
            "native_frames_hash_verified_not_separate_full_neighbor_frame_view_claim": sample["native_frames"],
            "view_observation": "SUPPORTED_ELEVATED_BASELINE_SINGLES", "visual_event_family": "RolandGarros",
            "source_caption_lead_not_verified_match_identity": "2021_SF_Krejcikova_vs_Sakkari",
            "review_decision": decision, "observed_state": state, "near_far_region": region, "blur_observation": blur,
            "publisher_proposals_preserved": sample["boxes"], "selected_manual_box_xyxy_native_px": selected,
            "selected_box_source": "MANUAL_NATIVE_VISIBLE_EXTENT_CORRECTION" if correction else "PUBLISHER_BOX_VISUALLY_REVIEWED_RETAINED" if approved else None,
            "native_boundary_uncertainty_px_uncalibrated": 2 if approved else None,
            "center_definition": "midpoint_of_reviewed_visible_box_not_physical_3d_center" if approved else None,
            "observations": explanation,
            "full_frame_background_observation": "Source contains very bright yellow headline text, green score and speed displays, white court/net lines, people, rackets, towels and courtside equipment. These are potential distractors; no exhaustive negative annotation or spare-ball absence is established.",
            "visible_box_proposal_accepted_for_development_review": approved,
            "training_admitted": False, "full_training_crop_background_closure_required": True,
            "accepted_training_labels_exported": 0, "accepted_evaluation_labels": 0,
            "actual_match_id": None, "camera_session_id": None, "development_split": None,
        })
    receipt = {"schema_version": "1.0", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "status": "TWELVE_ASSIGNED_FRANCE_ROWS_ACTUALLY_REVIEWED_LOCALLY_EIGHT_VISIBLE_BOX_PROPOSALS_FOUR_REJECTED_NO_EXPORT",
               "review_aids_manifest": {"path": str(MANIFEST), "sha256": EXPECTED},
               "reviewer": "AGENT:/root/ball_bbox_inventory", "reviewed_frame_indices": sorted(decisions),
               "actual_local_view_image_calls": 24, "actual_native_overlays_viewed": 12,
               "actual_unmarked_native96_triplet_boards_viewed": 12, "native_temporal_patches_viewed": 36,
               "visible_box_proposals_accepted_for_development_review": 8, "publisher_boxes_retained": 4,
               "manual_visible_extent_corrections": 4, "rejected_unresolved_proposals": 4,
               "independent_human_review": False, "training_performed": False, "model_or_gpu_runs": 0,
               "training_export_or_split_assignment": False, "reserved_test_data_used": False,
               "external_frame_transmission": False, "network_requests_in_this_visual_review": 0,
               "manual_boundary_uncertainty_is_not_measured_calibration": True,
               "context_and_leakage_limit": "Every reviewed sample is from one multi-rally RolandGarros compilation. The source title/caption is a lead only; retain the complete compilation and source aliases in a common conservative context before any split. This receipt does not establish independent matches or absence of overlap with cached sources.",
               "background_and_training_limit": "These eight are visible-box review proposals, not supervised exports. Every eventual native training crop still needs full background closure to avoid missing spare-ball or equipment targets. Rejected proposals are unknown, never empty-negative labels.",
               "rows": rows}
    output = Path(__file__).resolve().parent / "france_tail12_agent_visual_review.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({"output": str(output), "sha256": digest(output), "rows": len(rows), "accepted_development_review_proposals": 8, "rejected": 4}, indent=2))


if __name__ == "__main__":
    main()
