"""Record the root agent's actually viewed native ball-box decisions.

This records visual observations, not an automatic label/accuracy test. It does
not export a training dataset, assign splits, or claim independent human GT.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/ball_yolo_author_native_audit32_v1/review_aids_manifest.json"
EXPECTED = "bab9ae8e201574ec443172f9607fd2e2bec796facc2f0d0c01f183127b8a924a"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert sha(MANIFEST) == EXPECTED
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    for relative, expected in manifest["input_bindings"].items():
        assert sha(ROOT / relative) == expected, relative
    decisions = {
        295: ([935, 264, 949, 278], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
              "Small light diagonal ball blur on far service court; three unmarked neighbors show coherent movement and separable extent."),
        423: ([1024, 198, 1030, 206], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
              "Tiny moving ball on green area ahead/right of far player, separated from the body and racket in the unmarked temporal patches."),
        517: (None, "REJECT_UNRESOLVED_BALL_ONLY_EXTENT",
              "Very faint long diagonal gray smear in near service court; movement is visible but the ball-only boundary cannot be separated reliably from compression/blur. No imagined tight box."),
        545: ([968, 282, 982, 292], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
              "Bright, short horizontal moving ball blur on the far service court, separated from lines in all three unmarked patches."),
        664: (None, "REJECT_UNRESOLVED_BALL_ONLY_EXTENT",
              "Very low contrast broad streak just above the near baseline beside the near player; native patches do not resolve a reliable ball-only extent."),
        784: ([610, 685, 626, 709], "MANUAL_NATIVE_VISIBLE_BLUR_EXTENT",
              "Moving diffuse diagonal ball blur in near court shadow; narrowed the loose publisher horizontal extent around the visible smear, preserving its visible trail. Boundary is uncalibrated and approximate."),
        919: ([865, 387, 877, 397], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
              "Clearly separated bright diagonal ball below the net tape, moving down/left across the unmarked temporal patches."),
        1030: ([935, 114, 945, 124], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
               "Small light oval ball in front of the dark far wall below the US Open lettering; three patches separate motion from the static sign."),
        1094: ([1141, 223, 1144, 230], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
               "Tiny narrow bright ball beside the far player; upward movement across three patches separates it from the player and lower racket outline."),
        1163: (None, "REJECT_PLAYER_RACKET_OVERLAP_AMBIGUOUS",
               "Publisher proposal overlaps the far player's lower body/hand-racket region; a distinct ball-only extent is unresolved in the three unmarked patches."),
        1317: ([1156, 654, 1166, 665], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
               "Distinct green round/short-blur ball above/right of the near player's head; visible separation and movement across the native patches."),
        1438: ([855, 403, 863, 410], "PUBLISHER_BOUNDS_RETAINED_INTEGER_ROUNDING",
               "Small bright green ball separated below the net tape in the service court; source framing has shifted relative to earlier samples, with no calibration transferred."),
    }
    rows = []
    for sample in manifest["samples"]:
        if sample["alias"] != "America_Best_10_1" or sample["frame_index"] not in decisions:
            continue
        box, method, note = decisions[sample["frame_index"]]
        assert len(sample["boxes"]) == 1
        for asset in [sample["native_overlay"], sample["unmarked_native_detail_board"], *sample["native_frames"]]:
            assert sha(ROOT / asset["path"]) == asset["sha256"], asset["path"]
        if box is not None:
            assert 0 <= box[0] < box[2] < sample["native_width"]
            assert 0 <= box[1] < box[3] < sample["native_height"]
        rows.append({
            "id": sample["id"], "alias": sample["alias"], "frame_index": sample["frame_index"],
            "reviewer": "ROOT_AGENT", "source_pts": sample["source_pts"],
            "source_time_base": sample["source_time_base"],
            "native_overlay": sample["native_overlay"], "unmarked_native_detail_board": sample["unmarked_native_detail_board"],
            "native_frames": sample["native_frames"],
            "actually_viewed_native_overlay": True,
            "actually_viewed_all_unmarked_native_neighbor_patches": True,
            "review_decision": "ACCEPT_VISIBLE_BOX_FOR_CROP_REVIEW" if box else "REJECT_UNRESOLVED_PROPOSAL",
            "publisher_box_xyxy_native_px": sample["boxes"][0]["publisher_box_xyxy_native_px"],
            "reviewed_box_xyxy_native_px": box, "box_method": method, "observations": note,
            "boundary_uncertainty_px": 2 if box else None,
            "boundary_uncertainty_is_calibrated": False,
            "visual_event_family": "USOpen", "caption_match_lead": "Auger-Aliassime / De Minaur",
            "view_observation": "SUPPORTED_ELEVATED_BASELINE_SINGLES",
            "match_id": None, "camera_session_id": None, "development_split": None,
            "native_fps_implies_verified_normal_speed": False,
            "full_640_training_crop_background_review_complete": False,
            "training_label_admitted": False, "independent_human_review": False,
            "qualification_ground_truth": False,
        })
    assert len(rows) == 12 and sum(r["reviewed_box_xyxy_native_px"] is not None for r in rows) == 9
    result = {
        "schema_version": "1.0", "status": "COMPLETE_USA_TAIL12_LOCAL_NATIVE_VISUAL_REVIEW_NOT_EXPORTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "review_aids_manifest": {"path": str(MANIFEST), "sha256": EXPECTED},
        "reviewer": "ROOT_AGENT", "review_method": "ACTUAL_LOCAL_VIEW_IMAGE_ORIGINAL_RESOLUTION",
        "source_video_compilation_alias": "America_Best_10_1",
        "frame_count_reviewed": 12, "original_overlays_viewed": 12,
        "unmarked_native_temporal_boards_viewed": 12, "native_temporal_patches_viewed": 36,
        "accepted_visible_box_proposals_pending_crop_review": 9, "rejected_ambiguous_proposals": 3,
        "rows": rows, "accepted_training_labels": 0, "export_performed": False,
        "split_assignment_performed": False, "model_or_gpu_used": False,
        "external_video_connector_used": False, "independent_human_review": False,
        "qualification_evidence": False,
        "rights_scope": "PRIVATE_NONCOMMERCIAL_ACADEMIC_AUDIT_UNDER_PUBLISHER_RESEARCH_GRANT; underlying broadcast ownership and public redistribution uncleared",
        "limitations": [
            "Agent observations and uncalibrated manual extents are development proposals, not independent human measurement truth.",
            "All accepted extents require full native640 training-crop background closure before any supervised export.",
            "Missing publisher labels are not negative frames; scoreboard speed text is not synchronized speed-reference evidence.",
            "Compilation/caption identity does not establish an independent match or camera session or eligibility during the entire video.",
        ],
    }
    output = BASE / "usa_tail12_root_visual_review.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "accepted_pending_crop": 9, "rejected": 3}))


if __name__ == "__main__":
    main()
