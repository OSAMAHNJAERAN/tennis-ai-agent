"""Record actual local full-crop background review; no dataset export."""
import datetime
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
MANIFEST = ROOT / "outputs/ball_yolo_author_fullcrop22_v1/crop_review_manifest.json"
EXPECTED = "cd2c7c7cef534a7d7f2e694cee838223a3df71d5e51dc16290f3277da3644dde"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if digest(MANIFEST) != EXPECTED:
        raise ValueError("frozen full-crop review manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    observations = {
        0: "The single visible ball lies over far clay court. Full crop includes the far player's racket/shadow, central officials/signs, net logos/lines and near player's legs/shoes. These other bright objects have resolved non-ball outlines. No additional distinct tennis ball or unresolved spare-ball-like object was observed inside this crop.",
        138: "Tiny target is visible against the far green wall, separated from the officials. Full crop includes the far-right red ballkid's arms/shoes, the near player's pale shoes and racket head, green speed/clock graphics, net/line marks and yellow headline. Those objects are distinguishable as people/footwear/racket/graphics; no extra visible ball target or unresolved spare-ball-like object was observed.",
        139: "Tiny target is below the rear-wall edge. The near player/racket and far-right ballkid/shoes remain visible in the full crop, alongside lettering/net hardware. No additional visible tennis-ball target or unresolved equipment ball-like object was observed; adjacent frame138 is the same scene, not independent evidence.",
        706: "One small target streak lies above the net tape. Yellow caption text, bright court/net lines, left umpire stand/towels, near player's racket/white shoes and rear official signs are present but distinguishable. No extra or ambiguous spare ball was observed within the actual640 crop.",
        875: "One clear vertical ball streak lies below the net. The separate pale blur beside the near player's head/hand at the bottom is associated with the racket/player rather than an isolated second ball. Rear officials' shoes, clock signage, net logos and court lines have recognizable context. No unresolved additional ball target was observed in this crop.",
        1041: "One isolated pale ball is over far clay. The yellow racket held by the partly cropped near-left player has a clear ring/handle; rear ballkids' pale shoes, scoreboard/advertising elements and net logos are identifiable non-ball objects. No unlabelled extra tennis ball or unresolved spare-ball-like object was observed.",
        1347: "One compact ball is detached from the far player. The near-left player's yellow racket ring, back-wall staff/shoes, bright headline/clock and net logos are resolved distractors. No additional visible ball target or unresolved ball-like equipment object was observed in the crop.",
        1520: "One small ball lies above the net on far court. The near player's yellow racket, pale shoes and clay scuffs, rear red ballkid/staff and logo/line marks have identifiable shapes. No extra tennis-ball target or unresolved spare-ball-like object was observed inside this640 crop.",
        1709: "One compact ball is detached from the central sign and far player. The far player's raised light racket/head area is recognizable as player/racket; rear staff/shoes, signs, net and near-player shoes are distinguishable. No additional visible ball target or unresolved spare-ball-like object was observed in this crop.",
    }
    rows = []
    for sample in manifest["samples"]:
        if sample["alias"] != "France_Review_Red_Court_1":
            continue
        if sample["source_frame_index"] not in observations:
            raise ValueError("unexpected assigned France crop")
        for field in ("unmarked_crop", "separate_overlay", "source_native_image"):
            asset = sample[field]
            if digest(ROOT / asset["path"]) != asset["sha256"]:
                raise ValueError("actually viewed crop/native source hash changed")
        box = sample["reviewed_box_xyxy_crop_px"]
        origin = sample["actual_crop_origin_native_px"]
        inverse = [box[0] + origin[0], box[1] + origin[1], box[2] + origin[0], box[3] + origin[1]]
        if inverse != sample["reviewed_box_xyxy_native_px"]:
            raise ValueError("reviewed native box round trip changed")
        rows.append({
            "id": sample["id"], "source_alias": sample["alias"], "source_frame_index": sample["source_frame_index"],
            "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
            "source_timestamp_fraction": sample["source_timestamp_fraction"],
            "reviewer": "AGENT:/root/ball_bbox_inventory", "root_personally_viewed_by_this_receipt": False,
            "actually_viewed_complete_unmarked640_crop_locally": True,
            "actually_viewed_complete_separate640_overlay_locally": True,
            "unmarked_crop": sample["unmarked_crop"], "separate_overlay": sample["separate_overlay"],
            "source_native_image": sample["source_native_image"], "source_review_receipt": sample["source_review_receipt"],
            "actual_crop_origin_native_px": origin, "crop_size_px": [640, 640],
            "origin_boundary_clamp_delta_px": sample["origin_boundary_clamp_delta_px"],
            "reviewed_box_xyxy_native_px": sample["reviewed_box_xyxy_native_px"], "reviewed_box_xyxy_crop_px": box,
            "reviewed_label_clamped_or_changed_by_crop_review": False,
            "coordinate_roundtrip_exact": True,
            "crop_closure_decision": "ACCEPT_REVIEWED_POSITIVE_CROP_ALL_OBSERVED_BALL_TARGETS_RESOLVED",
            "observed_visible_ball_target_count": 1, "additional_visible_ball_target_count": 0,
            "unresolved_spare_ball_or_equipment_target_observed": False,
            "background_review_observations": observations[sample["source_frame_index"]],
            "training_export_admitted_by_this_receipt": False, "labels_exported": 0,
            "empty_negative_label_assumption": False, "development_split": None,
            "actual_match_id": None, "camera_session_id": None, "independent_human_review": False,
            "view_observation": "SUPPORTED_ELEVATED_BASELINE_SINGLES", "visual_event_family": "RolandGarros",
            "caption_is_unverified_match_identity_lead": "2021_SF_Krejcikova_vs_Sakkari",
            "review_scope": "Only this exact unmarked640 crop and separately viewed overlay at these hashes/origin; no absence claim for original pixels outside the crop or other times, and no automatic extension to jittered/resized crops.",
        })
    if len(rows) != 9:
        raise ValueError("exactly nine actual France full-crop reviews required")
    receipt = {"schema_version": "1.0", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "status": "ALL_NINE_FRANCE_NATIVE640_POSITIVE_CROPS_ACTUALLY_REVIEWED_AND_ACCEPTED_NO_EXPORT",
               "crop_review_manifest": {"path": str(MANIFEST), "sha256": EXPECTED},
               "reviewer": "AGENT:/root/ball_bbox_inventory", "actual_local_view_image_calls": 18,
               "actual_complete_unmarked640_crops_viewed": 9, "actual_complete_separate640_overlays_viewed": 9,
               "accepted_positive_crops": 9, "rejected_positive_crops": 0,
               "full_crop_background_closure_complete_for_exact_hash_bound_crops": True,
               "source_grant": manifest["source_grant"], "rights_limitations": manifest["rights_limitations"],
               "independent_human_review": False, "qualification_evidence": False,
               "labels_exported": 0, "splits_assigned": False, "training_performed": False,
               "model_or_gpu_runs": 0, "external_frame_transmission": False, "network_requests_in_this_review": 0,
               "compilation_and_exposure_limit": "All nine crops remain one conservative RolandGarros compilation context. Captions do not prove independent match/session identity; this receipt assigns no train/development/test split and cannot override previous development exposure or source-rights limitations.",
               "rows": rows}
    output = BASE / "france_fullcrop9_agent_review.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({"path": str(output), "sha256": digest(output), "accepted": 9, "rejected": 0}, indent=2))


if __name__ == "__main__":
    main()
