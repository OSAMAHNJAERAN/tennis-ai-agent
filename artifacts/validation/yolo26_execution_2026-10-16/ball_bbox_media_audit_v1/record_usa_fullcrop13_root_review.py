"""Record root's actual local visual background review of thirteen fixed crops."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
MANIFEST = ROOT / "outputs/ball_yolo_author_fullcrop22_v1/crop_review_manifest.json"
EXPECTED = "cd2c7c7cef534a7d7f2e694cee838223a3df71d5e51dc16290f3277da3644dde"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert sha(MANIFEST) == EXPECTED
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    for relative, expected in manifest["input_bindings"].items():
        assert sha(ROOT / relative) == expected, relative
    notes = {
        1: "One small reviewed ball beside near player; shoes, court marks, racket and speed graphic are visually distinct. No additional ball observed in crop.",
        2: "Same near-player shadow scene at adjacent source frame. One reviewed ball; no extra/spare ball observed; not an independent sample.",
        3: "Same initial rally with small near-player ball moving above/right; no extra ball observed in exact crop.",
        175: "One elongated light ball blur in shadow above near player; scoreboard yellow server marker is a broadcast graphic, not a second ball.",
        295: "One diagonal moving ball on far service court; far player and wall/signs do not supply additional ball targets in crop.",
        423: "One tiny ball separated from far player's body/racket; court/net logos and wall signs resolved as static context.",
        545: "One bright short ball blur on far court; player shoes and center mark are distinct, no extra ball observed.",
        784: "One faint elongated ball blur on near court; left-edge bags/straps, ball-kid clothing, scoreboard server marker and near player are visible context, no additional separable ball observed.",
        919: "One bright ball below net tape. The white far-baseline center mark and player shoes are static/structural objects, not additional ball targets.",
        1030: "One small ball below wall lettering. Players, court marks, net tape and dark-wall signs do not present extra ball targets.",
        1094: "One narrow bright ball right of far player, separated from body/racket; no extra ball observed in crop.",
        1317: "One green ball above/right of near player's head, spatially separated; no extra ball observed.",
        1438: "One small green ball below net tape; far player/center marks/signs do not supply extra ball targets. This exact shifted source framing is preserved.",
    }
    rows = []
    for sample in manifest["samples"]:
        if sample["alias"] != "America_Best_10_1":
            continue
        for key in ("unmarked_crop", "separate_overlay", "source_native_image", "source_review_receipt"):
            assert sha(ROOT / sample[key]["path"]) == sample[key]["sha256"], sample[key]["path"]
        frame = sample["source_frame_index"]
        assert frame in notes and sample["coordinate_roundtrip_exact"] is True
        assert sample["native_pixels_resized"] is False
        rows.append({"id": sample["id"], "source_frame_index": frame, "reviewer": "ROOT_AGENT",
            "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
            "unmarked_crop": sample["unmarked_crop"], "separate_overlay": sample["separate_overlay"],
            "source_native_image": sample["source_native_image"], "source_review_receipt": sample["source_review_receipt"],
            "actual_crop_origin_native_px": sample["actual_crop_origin_native_px"],
            "reviewed_box_xyxy_native_px": sample["reviewed_box_xyxy_native_px"],
            "reviewed_box_xyxy_crop_px": sample["reviewed_box_xyxy_crop_px"],
            "actually_viewed_entire_unmarked_native640_crop": True,
            "actually_viewed_entire_separate_native640_overlay": True,
            "review_decision": "ACCEPT_EXACT_CROP_VISIBLE_BALL_BACKGROUND_CLOSURE",
            "additional_visible_ball_targets_observed": 0, "unresolved_background_ball_proposal_observed": False,
            "observations": notes[frame], "development_split": None, "training_label_exported": False,
            "independent_human_review": False, "qualification_evidence": False})
    assert len(rows) == 13
    result = {"schema_version": "1.0", "status": "COMPLETE_USA13_EXACT_NATIVE640_CROP_BACKGROUND_REVIEW_NO_EXPORT",
        "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "ROOT_AGENT",
        "crop_review_manifest": {"path": str(MANIFEST), "sha256": EXPECTED},
        "actual_unmarked_full640_crops_viewed": 13, "actual_full640_overlays_viewed": 13,
        "accepted_exact_crops": 13, "rejected_exact_crops": 0, "rows": rows,
        "scope": "Only exact saved crop origins, pixels and reviewed boxes; no jittered/neighbor/full-frame background closure is implied.",
        "original_fps_is_not_normal_speed_verification": True,
        "independent_human_review": False, "qualification_evidence": False, "export_performed": False,
        "split_assignment_performed": False, "model_or_gpu_used": False, "external_connector_used": False,
        "rights_scope": "PRIVATE_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT; no public broadcast redistribution clearance"}
    output = BASE / "usa_fullcrop13_root_review.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(output), "sha256": sha(output), "accepted_exact_crops": 13}))


if __name__ == "__main__":
    main()
