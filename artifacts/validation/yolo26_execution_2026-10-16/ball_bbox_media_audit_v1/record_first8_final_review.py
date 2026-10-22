"""Record eight actual local source/visible-box reviews without altering prior receipts."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
MANIFEST = ROOT / "outputs/ball_yolo_author_native_audit32_v1/review_aids_manifest.json"
MANIFEST_SHA = "bab9ae8e201574ec443172f9607fd2e2bec796facc2f0d0c01f183127b8a924a"
PARTIAL = BASE / "partial_root_visual_review8.json"
PARTIAL_SHA = "ffbff8b0d47e0acf4afd515275d8fb6b50d70c16d0d5241ec1ab5b87fa614532"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if digest(MANIFEST) != MANIFEST_SHA or digest(PARTIAL) != PARTIAL_SHA:
        raise ValueError("frozen source or preserved partial review changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Coordinates are manual reviews of the actual native images, never scaled boxes.
    decisions = {
        "America_Best_10_1_f000001": ([846, 731, 853, 738], "VISIBLE_SMALL_BALL", "LOW", "A small dim gray/yellow ball is detached from the near player's arm/racket and is visible against the blue court in the middle native patch. The next patch shows its movement up/right. Tightened approximately11x10px publisher bounds to a7x7px visible blob envelope, preserving the low-contrast compression halo."),
        "America_Best_10_1_f000002": ([845, 724, 858, 737], "VISIBLE_DIAGONAL_BALL_BLUR", "MODERATE", "The pale/green ball streak moves up/right relative to frame1 and separates from the dark racket/arm. Retained the publisher's approximately13x13px diagonal visible-blur envelope, rounded to native integer edges. This is a blur box, not a sharp-ball diameter."),
        "America_Best_10_1_f000003": ([853, 717, 866, 729], "VISIBLE_DIAGONAL_BALL_BLUR", "MODERATE", "A distinct pale/green diagonal ball streak continues away from the near player's dark racket in frames2-4. Selected a13x12px native envelope around the visible streak; the publisher left/top bounds included a small additional dark background margin."),
        "America_Best_10_1_f000175": ([792, 664, 804, 687], "VISIBLE_LONG_BALL_BLUR", "HIGH", "The detached narrow pale ball streak moves down/left across the three unmarked native patches. Tightened approximately16x25px publisher bounds to a12x23px visible diagonal-blur envelope. Low contrast in the shaded court makes outer halo boundaries uncertain; no hidden spherical center is inferred."),
        "France_Review_Red_Court_1_f000000": ([571, 207, 578, 212], "VISIBLE_SMALL_BALL_BLUR", "MODERATE", "A small pale horizontal/diagonal ball is distinct from the far player's body, racket and shadow, and moves down/right in the next native patch. Retained the approximately7x5px publisher visible extent, rounded to integer native edges. The video begins here, so only current/next temporal patches exist."),
        "France_Review_Red_Court_1_f000030": (None, "UNRESOLVED_RACKET_PLAYER_ADJACENT_SWEEP", "UNRESOLVED", "The39x33px proposal follows a large pale triangular/oval sweep above the near player's raised hand/head, spatially compatible with the racket sweep. Neighbor frames do not resolve a separate ball-only contour inside it. Reject this proposal rather than using the racket/arm or a guessed ball position."),
        "France_Review_Red_Court_1_f000058": (None, "UNRESOLVED_MULTIPLE_BALLLIKE_OBJECTS", "UNRESOLVED", "Both proposals and all six unmarked patches were inspected. Box0 is a tiny pale near-hand object beside the near player; box1 is a far-player/racket-adjacent bright sweep. Their active-ball identity and full ball-only contours are unresolved. Neither one is admitted, and two simultaneous active balls are not assumed."),
        "France_Review_Red_Court_1_f000060": (None, "UNRESOLVED_MULTIPLE_BALLLIKE_OBJECTS", "UNRESOLVED", "Both proposals and all six unmarked patches were inspected. Box0 follows a bright far-player/racket-adjacent object; box1 follows a small near-player hand-adjacent bright object. This reverses publisher ordering relative to frame58 and provides no stable ball identity. Reject both proposals pending richer evidence; do not choose by index or size."),
    }
    samples = [row for row in manifest["samples"] if row["id"] in decisions]
    if len(samples) != 8:
        raise ValueError("exact assigned eight samples are required")
    rows = []
    accepted = rejected = patches = 0
    for sample in samples:
        selected, state, blur, reason = decisions[sample["id"]]
        for asset in [sample["native_overlay"], sample["unmarked_native_detail_board"], *sample["native_frames"]]:
            if digest(ROOT / asset["path"]) != asset["sha256"]:
                raise ValueError("source/native review image changed after actual local viewing")
        proposals = []
        for index, box in enumerate(sample["boxes"]):
            approve = selected is not None and index == 0 and len(sample["boxes"]) == 1
            if approve:
                accepted += 1
            else:
                rejected += 1
            proposals.append({
                "publisher_proposal_index": index, "publisher_proposal_preserved": box,
                "review_decision": "ACCEPT_REVIEWED_VISIBLE_BOX_PENDING_FULL_CROP_CLOSURE" if approve else "REJECT_UNRESOLVED_OBJECT_IDENTITY_OR_VISIBLE_EXTENT",
                "visible_box_proposal_accepted_for_development_review": approve,
                "selected_manual_box_xyxy_native_px": selected if approve else None,
                "selected_box_source": "MANUAL_NATIVE_VISIBLE_EXTENT_REVIEW" if approve else None,
                "boundary_uncertainty_px_uncalibrated": 2 if approve else None,
                "boundary_uncertainty_is_not_measured_calibration": True,
                "center_definition": "reviewed_visible_extent_midpoint_not_physical_ball_3d_center" if approve else None,
                "training_admitted": False, "empty_negative_label": False,
            })
        source_family = "USOpen" if sample["alias"] == "America_Best_10_1" else "RolandGarros"
        patch_count = len(sample["boxes"]) * len(sample["native_neighbor_indices"])
        patches += patch_count
        rows.append({
            "id": sample["id"], "source_alias": sample["alias"], "source_frame_index": sample["frame_index"],
            "native_width": sample["native_width"], "native_height": sample["native_height"],
            "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
            "source_timestamp_fraction": sample["source_timestamp_fraction"],
            "label_source_path": sample["label_source_path"], "label_sha256": sample["label_sha256"],
            "label_git_blob_sha1": sample["label_git_blob_sha1"],
            "reviewer": "AGENT:/root/ball_bbox_inventory", "independent_human_review": False,
            "root_personally_viewed_by_this_receipt": False, "model_or_gpu_used": False,
            "actually_viewed_original_resolution_overlay_locally": True,
            "actually_viewed_every_unmarked_native96_temporal_patch_locally": True,
            "actual_temporal_patches_viewed": patch_count,
            "native_neighbor_indices": sample["native_neighbor_indices"], "native_neighbor_pts": sample["native_neighbor_pts"],
            "native_overlay": sample["native_overlay"], "unmarked_native_detail_board": sample["unmarked_native_detail_board"],
            "native_frames_hash_verified_not_individual_full_neighbor_frame_view_claim": sample["native_frames"],
            "view_observation": "SUPPORTED_ELEVATED_BASELINE_SINGLES", "visual_event_family": source_family,
            "source_caption_lead_not_verified_match_identity": "AugerAliassime_DeMinaur_scoreboard" if source_family == "USOpen" else "2021_SF_Krejcikova_vs_Sakkari_caption",
            "observed_state": state, "blur_observation": blur, "manual_review_rationale": reason,
            "reviewed_proposals": proposals, "actual_match_id": None, "camera_session_id": None, "development_split": None,
            "full_training_crop_background_closure_required": True,
            "background_observation": "Native frame includes players/rackets, officials/ballkids, court/net lines, bright broadcast graphics and signage. No full-crop spare-ball/equipment closure or exhaustive negative annotation is established by this review.",
        })
    if (accepted, rejected, patches) != (5, 5, 29):
        raise ValueError("review totals differ from actual eight-sample observations")
    output = BASE / "first8_final_agent_visual_review.json"
    receipt = {"schema_version": "1.0", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "status": "FIRST8_ACTUALLY_REVIEWED_LOCALLY_FIVE_VISIBLE_BOX_PROPOSALS_FIVE_REJECTED_PROPOSALS_NO_EXPORT",
               "review_aids_manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA},
               "preserved_prior_root_partial_receipt": {"path": str(PARTIAL), "sha256": PARTIAL_SHA, "unchanged": True},
               "reviewer": "AGENT:/root/ball_bbox_inventory", "actual_local_view_image_calls": 16,
               "actual_native_overlays_viewed": 8, "actual_unmarked_temporal_boards_viewed": 8,
               "actual_unmarked_native96_patches_viewed": patches, "reviewed_samples": 8,
               "reviewed_publisher_proposals": 10, "visible_box_proposals_accepted_for_development_review": accepted,
               "unresolved_proposals_rejected": rejected, "accepted_sample_count": 5, "rejected_sample_count": 3,
               "selected_boxes_are_native_integer_xyxy": True, "independent_human_review": False,
               "qualification_evidence": False, "training_export_or_split_assigned": False,
               "training_or_gpu_runs": 0, "external_frame_transmission": False, "network_requests_for_this_review": 0,
               "compilation_and_overlap_limit": "The two aliases identify source compilations, not independent matches. Keep every original source/alias connection and conservative event context together for future leakage review. Captions/scoreboard names are unverified leads only; this receipt assigns no split and proves no source independence.",
               "speed_limit": "The127MPH USOpen graphic is broadcast text, not synchronized independent physical speed ground truth.",
               "training_limit": "Accepted boxes remain development annotation proposals until full native640 training-crop background closure. No ambiguous or rejected proposal is converted into an empty-negative label.",
               "rows": rows}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    if digest(PARTIAL) != PARTIAL_SHA or digest(MANIFEST) != MANIFEST_SHA:
        raise ValueError("input receipt changed")
    print(json.dumps({"path": str(output), "sha256": digest(output), "accepted": accepted, "rejected": rejected, "sample_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
