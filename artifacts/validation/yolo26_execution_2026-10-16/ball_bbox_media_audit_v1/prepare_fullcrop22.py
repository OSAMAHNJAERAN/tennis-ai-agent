"""Prepare immutable native640 review crops; never export labels or assign splits."""
from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OUTPUT = ROOT / "outputs/ball_yolo_author_fullcrop22_v1"
INPUTS = {
    "outputs/ball_yolo_author_native_audit32_v1/review_aids_manifest.json": "bab9ae8e201574ec443172f9607fd2e2bec796facc2f0d0c01f183127b8a924a",
    "artifacts/validation/yolo26_execution_2026-10-16/ball_bbox_media_audit_v1/first8_final_agent_visual_review.json": "15a44a254622c6363a9f30d39bbee76707e535fd028172472b6b03620567ccfd",
    "artifacts/validation/yolo26_execution_2026-10-16/ball_bbox_media_audit_v1/france_tail12_agent_visual_review.json": "44a8947dd4d3384b6aa1dbbe8ab554e867ed872a959058807c6c5716170d4b7c",
    "artifacts/validation/yolo26_execution_2026-10-16/ball_bbox_media_audit_v1/usa_tail12_root_visual_review.json": "69b866736fdde80778d898e580e75ba0ca2f73890329f09da933b05404141880",
    "artifacts/validation/yolo26_execution_2026-10-16/ball_bbox_media_audit_v1/acquisition_receipt.json": "088586e374aa1d4b0b83e399b84e959e81338e31a08db043184fc06025b15791",
    "data/yolo26/ball_reviewed_development_v1/source_manifest.json": "b3fa03c171c797db9df52038fe1b984cd6aef6aafb942e6885207430832a25ab",
    "artifacts/validation/yolo26_execution_2026-10-15/ball_reviewed_crop_preparation_v1/derived_development_manifest.json": "b94a4e0d6cfc3504817b9223abc5e43a2252fd5d826ea95bf2dd14e129b41110",
}


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total_phys", "available_phys", "total_page", "available_page", "total_virtual", "available_virtual", "available_extended")]


def resources():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    values = {"available_ram_bytes": status.available_phys, "free_disk_bytes": shutil.disk_usage(ROOT).free}
    if values["available_ram_bytes"] < 1024**3 or values["free_disk_bytes"] < 2 * 1024**3:
        raise RuntimeError(f"native crop guards refused: {values}")
    return values


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def verify_inputs(bindings):
    for path, expected in bindings.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"frozen input changed: {path}")


def main():
    if OUTPUT.exists():
        raise ValueError("fresh immutable output path required")
    before = resources()
    verify_inputs(INPUTS)
    native_manifest = json.loads((ROOT / next(iter(INPUTS))).read_text(encoding="utf-8"))
    samples = {row["id"]: row for row in native_manifest["samples"]}
    acquisition = json.loads((BASE / "acquisition_receipt.json").read_text(encoding="utf-8"))
    bindings = dict(INPUTS)
    bindings.update(native_manifest["input_bindings"])
    bindings.update(acquisition["input_bindings"])
    review_files = list(INPUTS)[1:4]
    records = []
    for relative in review_files:
        review = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        for row in review["rows"]:
            accepted = []
            if "reviewed_proposals" in row:
                accepted = [item["selected_manual_box_xyxy_native_px"] for item in row["reviewed_proposals"] if item["visible_box_proposal_accepted_for_development_review"]]
            elif row.get("visible_box_proposal_accepted_for_development_review"):
                accepted = [row["selected_manual_box_xyxy_native_px"]]
            elif row.get("review_decision") == "ACCEPT_VISIBLE_BOX_FOR_CROP_REVIEW":
                accepted = [row["reviewed_box_xyxy_native_px"]]
            if not accepted:
                continue
            if len(accepted) != 1:
                raise ValueError("one reviewed visible target per accepted frame required")
            sample = samples[row["id"]]
            frame = next(item for item in sample["native_frames"] if item["frame_index"] == sample["frame_index"])
            bindings[frame["path"]] = frame["sha256"]
            x1, y1, x2, y2 = accepted[0]
            width, height = sample["native_width"], sample["native_height"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError("reviewed native box outside source; labels must not be clamped")
            proposed = [math.floor((x1 + x2) / 2 - 320), math.floor((y1 + y2) / 2 - 320)]
            origin = [min(max(0, proposed[0]), width - 640), min(max(0, proposed[1]), height - 640)]
            local_box = [x1 - origin[0], y1 - origin[1], x2 - origin[0], y2 - origin[1]]
            if not (0 <= local_box[0] < local_box[2] <= 640 and 0 <= local_box[1] < local_box[3] <= 640):
                raise ValueError("native reviewed target not fully inside fixed640 crop")
            roundtrip = [local_box[0] + origin[0], local_box[1] + origin[1], local_box[2] + origin[0], local_box[3] + origin[1]]
            if roundtrip != accepted[0]:
                raise ValueError("exact crop/native coordinate round trip failed")
            records.append({"id": sample["id"], "alias": sample["alias"], "source_frame_index": sample["frame_index"],
                            "native_width": width, "native_height": height, "source_pts": sample["source_pts"],
                            "source_time_base": sample["source_time_base"], "source_timestamp_fraction": sample["source_timestamp_fraction"],
                            "source_native_image": frame, "source_review_receipt": {"path": relative, "sha256": INPUTS[relative]},
                            "source_review_row_id": row["id"], "reviewed_box_xyxy_native_px": accepted[0],
                            "proposed_centered_crop_origin_native_px": proposed, "actual_crop_origin_native_px": origin,
                            "origin_boundary_clamp_delta_px": [origin[0] - proposed[0], origin[1] - proposed[1]],
                            "crop_size_px": [640, 640], "reviewed_box_xyxy_crop_px": local_box,
                            "reviewed_label_clamped_or_changed": False, "coordinate_roundtrip_exact": True,
                            "native_pixels_resized": False, "context_group": None, "development_split": None,
                            "actual_match_id": None, "camera_session_id": None, "full_crop_review_status": "PENDING_ACTUAL_UNMARKED640_AND_OVERLAY_REVIEW"})
    if len(records) != 22 or len({row["id"] for row in records}) != 22:
        raise ValueError("exact22 distinct reviewed positive samples required")
    if sum(row["alias"] == "France_Review_Red_Court_1" for row in records) != 9:
        raise ValueError("exact9France13USA split of review responsibility required")
    verify_inputs(bindings)
    OUTPUT.mkdir()
    (OUTPUT / "unmarked").mkdir()
    (OUTPUT / "overlays").mkdir()
    plan = {"status": "FROZEN22_NATIVE640_CROP_REVIEW_PLAN_NO_LABEL_EXPORT", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input_bindings": bindings, "implementation": {"path": str(Path(__file__)), "sha256": digest(Path(__file__))},
            "resource_guards": {"available_ram_bytes_minimum": 1024**3, "free_disk_bytes_minimum": 2 * 1024**3},
            "resources_before_pil": before, "samples": records, "sample_count": len(records),
            "source_grant": acquisition["publisher_grant"], "source_grant_input_bindings": acquisition["input_bindings"],
            "rights_limitations": acquisition["rights_limitation"], "source_training_admitted": False,
            "models_used": [], "yolo_labels_exported": 0, "training_performed": False, "final_test_used": False}
    write_new(OUTPUT / "frozen_crop_review_plan.json", plan)
    from PIL import Image, ImageDraw
    after = resources()
    for row in records:
        resources()
        path = ROOT / row["source_native_image"]["path"]
        left, top = row["actual_crop_origin_native_px"]
        with Image.open(path) as source:
            source.load()
            if source.mode != "RGB" or source.size != (row["native_width"], row["native_height"]):
                raise ValueError("native RGB shape changed; no silent conversion")
            crop = source.crop((left, top, left + 640, top + 640))
            expected_pixels = crop.tobytes()
            unmarked = OUTPUT / "unmarked" / (row["id"] + ".png")
            with unmarked.open("xb") as handle:
                crop.save(handle, format="PNG")
            with Image.open(unmarked) as reopened:
                reopened.load()
                if reopened.mode != "RGB" or reopened.size != (640, 640) or reopened.tobytes() != expected_pixels:
                    raise ValueError("unmarked PNG pixel ROI differs from exact native source")
            pixels_hash = hashlib.sha256(expected_pixels).hexdigest()
            overlay = crop.copy()
            drawing = ImageDraw.Draw(overlay)
            box = row["reviewed_box_xyxy_crop_px"]
            drawing.rectangle([math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])], outline=(255, 150, 0), width=1)
            drawing.rectangle([0, 0, 639, 15], fill=(0, 0, 0))
            drawing.text((3, 2), "REVIEW ONLY: orange visible-ball proposal; native640, no labels exported", fill=(255, 190, 0))
            overlay_path = OUTPUT / "overlays" / (row["id"] + ".png")
            with overlay_path.open("xb") as handle:
                overlay.save(handle, format="PNG")
            row.update(unmarked_crop={"path": str(unmarked.relative_to(ROOT)), "sha256": digest(unmarked), "native_rgb_pixels_sha256": pixels_hash},
                       separate_overlay={"path": str(overlay_path.relative_to(ROOT)), "sha256": digest(overlay_path)},
                       saved_unmarked_pixels_equal_native_roi=True)
            crop.close()
            overlay.close()
    verify_inputs(bindings)
    manifest = {"status": "COMPLETE22_UNMARKED_NATIVE640_CROP_AND_SEPARATE_OVERLAY_REVIEW_AIDS_NO_LABEL_EXPORT",
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "frozen_plan": {"path": str(OUTPUT / "frozen_crop_review_plan.json"), "sha256": digest(OUTPUT / "frozen_crop_review_plan.json")},
                "input_bindings": bindings, "resources_before_pil": before, "resources_after_pil": after, "resources_at_end": resources(),
                "sample_count": len(records), "france_count": 9, "usa_count": 13, "samples": records,
                "source_grant": acquisition["publisher_grant"], "rights_limitations": acquisition["rights_limitation"],
                "all_exact_native_roi_pixels_verified": True, "all_exact_coordinate_roundtrips_verified": True,
                "input_hashes_changed": [], "models_used": [], "yolo_labels_exported": 0, "training_performed": False,
                "splits_assigned": False, "final_test_used": False, "qualification_evidence": False}
    write_new(OUTPUT / "crop_review_manifest.json", manifest)
    old = json.loads((ROOT / "data/yolo26/ball_reviewed_development_v1/source_manifest.json").read_text(encoding="utf-8"))
    aliases = [alias for row in old["components"] for alias in row["aliases"]]
    exposures = {"status": "METADATA_ONLY_CONSERVATIVE_CONTEXT_EXPOSURE_NOT_A_NEW_SPLIT", "old_export_manifest_sha256": INPUTS["data/yolo26/ball_reviewed_development_v1/source_manifest.json"],
                 "old_context_components": old["components"], "new_compilation_aliases": sorted({row["alias"] for row in records}),
                 "new_visual_context_leads": ["USOpen", "RolandGarros"], "exact_new_context_token_present_in_old_aliases": {"usopen": "usopen" in aliases, "roland_garros": "roland_garros" in aliases},
                 "legacy_aliases_are_not_complete_original_match_identifiers": True,
                 "known_existing_development_components": [row for row in old["components"] if row["derived_split"] == "development"],
                 "conservative_compilation_link_requirements": ["Retain entire America_Best_10_1 compilation in one context; scoreboard names are source leads, not match IDs.", "Retain entire France_Review_Red_Court_1 compilation in one context;2021SF caption is a source lead, not independently verified match identity.", "Keep author America_Best_Move_1 and Federer_Top_40 aliases unresolved against USA compilation and all cached RacketVision event components until visual/source metadata review.", "Keep France/RolandGarros lead unresolved against existing paris quarantine and any unverified compilations; absence of identical event strings does not prove independent footage.", "Existing development components Miami/Washington, Dubai/Rome and Stockholm must never be moved into training through a new alias connection.", "No automatic claim of new TEST independence, train/development placement or normal-speed footage is made."],
                 "media_or_reserved_test_accessed_for_exposure_audit": False, "new_split_assigned": False}
    write_new(OUTPUT / "prior_development_context_exposure.json", exposures)
    print(json.dumps({"output": str(OUTPUT), "manifest_sha256": digest(OUTPUT / "crop_review_manifest.json"), "samples": len(records), "france": 9, "usa": 13,
                      "context_exposure_sha256": digest(OUTPUT / "prior_development_context_exposure.json")}, indent=2))


if __name__ == "__main__":
    main()
