"""Plan an explicitly approved, private101-sample native640 ball repair export.

Read-only planning uses metadata and hashes, reconstructs the old79 review graph,
and retains all32 new source-frame reviews, including rejected proposals. A
bounded stdlib PNG reader can recheck all22 native pixel ROIs before approval,
and execution always repeats that check. No image library, model, video decoder,
GPU, split search, jitter or resized image is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

from scripts.data import export_yolo26_reviewed_ball_wave as parent

ROOT = parent.ROOT
AUDIT = ROOT / "artifacts/validation/yolo26_execution_2026-10-16/ball_bbox_media_audit_v1"
OLD = ROOT / "data/yolo26/ball_reviewed_development_v1"
OUTPUT = ROOT / "data/yolo26/ball_author_repair_development_v2"
CROPS = ROOT / "outputs/ball_yolo_author_fullcrop22_v1/crop_review_manifest.json"
NATIVE = ROOT / "outputs/ball_yolo_author_native_audit32_v1/review_aids_manifest.json"
ALIASES = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/ball_bbox_source_followup_v1/publisher_tennis_inventory.json"
PINS = {
    OLD / "source_manifest.json": "b3fa03c171c797db9df52038fe1b984cd6aef6aafb942e6885207430832a25ab",
    CROPS: "cd2c7c7cef534a7d7f2e694cee838223a3df71d5e51dc16290f3277da3644dde",
    NATIVE: "bab9ae8e201574ec443172f9607fd2e2bec796facc2f0d0c01f183127b8a924a",
    AUDIT / "first8_final_agent_visual_review.json": "15a44a254622c6363a9f30d39bbee76707e535fd028172472b6b03620567ccfd",
    AUDIT / "france_tail12_agent_visual_review.json": "44a8947dd4d3384b6aa1dbbe8ab554e867ed872a959058807c6c5716170d4b7c",
    AUDIT / "usa_tail12_root_visual_review.json": "69b866736fdde80778d898e580e75ba0ca2f73890329f09da933b05404141880",
    AUDIT / "france_fullcrop9_agent_review.json": "eba61db537a6e340f67bab5785b834257f0ccca2f72d6e8a1ec25520fae3016b",
    AUDIT / "usa_fullcrop13_root_review.json": "32954b0c29e7aa01013e883a71a50ec31fa55fe687f7f7edaf59b284d2c89cf8",
    AUDIT / "acquisition_receipt.json": "088586e374aa1d4b0b83e399b84e959e81338e31a08db043184fc06025b15791",
    ALIASES: "1b89a2ed445538eae6387fca73783174cf35791a0e34b6bf75c2da3685adc227",
    ROOT / "scripts/data/export_yolo26_reviewed_ball_wave.py": "39d7148702004dd6a5c27b544725ba3d07fbc80f2d9c612274e52ecdb1d9aa00",
}
SCHEMA = "tennis.private-reviewed-ball-author-repair-export.v2"
APPROVAL_STATUS = "ROOT_APPROVED_PRIVATE_NC_BALL101_REPAIR_EXPORT_V2"
EXPECTED_COUNTS = {"images": 101, "positive_boxes": 97, "negative_tiles": 4, "train": 80, "val": 21, "test": 0,
                   "inherited_images": 79, "new_positive_images": 22}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def bind(bindings, path, expected):
    path = Path(path).resolve()
    if (not path.is_relative_to(ROOT.resolve()) or not isinstance(expected, str)
            or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)):
        raise ValueError("Bound source must remain inside workspace with an exact SHA256")
    key = str(path)
    if key in bindings and bindings[key] != expected:
        raise ValueError("Conflicting hashes for one source artifact")
    bindings[key] = expected


def check_bindings(bindings):
    changed = [path for path, expected in bindings.items() if parent.digest(path) != expected]
    if changed:
        raise ValueError(f"Pinned source or review changed: {changed}")


def validate_geometry(sample):
    """Fail on label clipping, resized crops, altered origin or lost source PTS."""
    width, height = sample["native_width"], sample["native_height"]
    if any(type(value) is not int or value < 640 for value in (width, height)):
        raise ValueError("Native dimensions must be exact integers admitting native640")
    origin = sample["actual_crop_origin_native_px"]
    if len(origin) != 2 or any(type(value) is not int for value in origin):
        raise ValueError("Native crop origin must be integer coordinates")
    if not (0 <= origin[0] <= width - 640 and 0 <= origin[1] <= height - 640):
        raise ValueError("Native crop leaves source image")
    if sample["crop_size_px"] != [640, 640] or sample.get("native_pixels_resized") is not False:
        raise ValueError("Only unresized native640 crops are permitted")
    if sample.get("reviewed_label_clamped_or_changed") is not False or sample.get("coordinate_roundtrip_exact") is not True:
        raise ValueError("Crop cannot clamp or alter a reviewed label")
    crop = parent._box(sample["reviewed_box_xyxy_crop_px"], 640)
    native = sample["reviewed_box_xyxy_native_px"]
    restored = [crop[0] + origin[0], crop[1] + origin[1], crop[2] + origin[0], crop[3] + origin[1]]
    if restored != native:
        raise ValueError("Reviewed native/crop box inverse differs")
    if type(sample["source_frame_index"]) is not int or sample["source_frame_index"] < 0:
        raise ValueError("Original zero-based frame index is required")
    pts, base = Fraction(sample["source_pts"]), Fraction(sample["source_time_base"])
    if pts.denominator != 1 or base <= 0 or pts * base != Fraction(sample["source_timestamp_fraction"]):
        raise ValueError("Source timestamp differs from actual PTS and time base")
    if sample["source_native_image"]["frame_index"] != sample["source_frame_index"] or str(sample["source_native_image"]["source_pts"]) != str(sample["source_pts"]):
        raise ValueError("Reviewed native image differs from source frame/PTS")
    return crop


def validate_closure(sample, review, *, france):
    for key in ("id", "source_frame_index", "source_pts", "source_time_base", "unmarked_crop", "separate_overlay", "source_native_image", "source_review_receipt", "actual_crop_origin_native_px", "reviewed_box_xyxy_native_px", "reviewed_box_xyxy_crop_px"):
        if review.get(key) != sample.get(key):
            raise ValueError(f"Complete crop review changed source/box/asset: {key}")
    if review.get("independent_human_review") is not False or review.get("development_split") is not None:
        raise ValueError("Agent development crop review cannot assert human qualification or a split")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        raise ValueError("Actual crop reviewer attribution is required")
    if france:
        if review.get("crop_closure_decision") != "ACCEPT_REVIEWED_POSITIVE_CROP_ALL_OBSERVED_BALL_TARGETS_RESOLVED":
            raise ValueError("France complete crop background closure is pending/rejected")
        flags = ("actually_viewed_complete_unmarked640_crop_locally", "actually_viewed_complete_separate640_overlay_locally")
        if review.get("observed_visible_ball_target_count") != 1 or review.get("additional_visible_ball_target_count") != 0 or review.get("unresolved_spare_ball_or_equipment_target_observed") is not False:
            raise ValueError("France crop contains unresolved or additional targets")
    else:
        if review.get("review_decision") != "ACCEPT_EXACT_CROP_VISIBLE_BALL_BACKGROUND_CLOSURE":
            raise ValueError("USA complete crop background closure is pending/rejected")
        flags = ("actually_viewed_entire_unmarked_native640_crop", "actually_viewed_entire_separate_native640_overlay")
        if review.get("additional_visible_ball_targets_observed") != 0 or review.get("unresolved_background_ball_proposal_observed") is not False:
            raise ValueError("USA crop contains unresolved or additional targets")
    if any(review.get(flag) is not True for flag in flags):
        raise ValueError("Every full unmarked crop and separate overlay must actually be viewed")


def validate_source_reviews(native_samples, source_reviews, crop_samples):
    """Reconstruct accepted extent decisions without discarding rejected rows."""
    native_by_id = {row["id"]: row for row in native_samples}
    if len(native_by_id) != len(native_samples):
        raise ValueError("Duplicate frozen native review ID")
    reviewed, accepted = {}, {}
    proposal_count = rejected_proposals = 0
    for review_path, receipt in source_reviews:
        for row in receipt["rows"]:
            sid = row["id"]
            if sid in reviewed or sid not in native_by_id:
                raise ValueError("Duplicate or extra source review ID")
            native = native_by_id[sid]
            reviewed[sid] = row
            if (row.get("source_alias", row.get("alias")) != native["alias"]
                    or row.get("source_frame_index", row.get("frame_index")) != native["frame_index"]
                    or row.get("source_pts") != native["source_pts"]
                    or row.get("source_time_base") != native["source_time_base"]
                    or row.get("native_overlay") != native["native_overlay"]
                    or row.get("unmarked_native_detail_board") != native["unmarked_native_detail_board"]):
                raise ValueError("Source review differs from frozen native frame, PTS or review assets")
            if row.get("independent_human_review") is not False or row.get("view_observation") != "SUPPORTED_ELEVATED_BASELINE_SINGLES":
                raise ValueError("Source review cannot assert independent human evidence or another camera view")
            overlay_flags = ("actually_viewed_original_resolution_overlay_locally", "actually_viewed_native_overlay_locally", "actually_viewed_native_overlay")
            temporal_flags = ("actually_viewed_every_unmarked_native96_temporal_patch_locally", "actually_viewed_all_unmarked_native96_temporal_patches_locally", "actually_viewed_all_unmarked_native_neighbor_patches")
            if not any(row.get(key) is True for key in overlay_flags) or not any(row.get(key) is True for key in temporal_flags):
                raise ValueError("Native overlay and every unmarked temporal patch need actual local review")
            if row.get("visual_event_family", "").lower() != {"America_Best_10_1": "usopen", "France_Review_Red_Court_1": "rolandgarros"}.get(native["alias"]):
                raise ValueError("Observed source event family changed")
            decisions = row.get("reviewed_proposals")
            selected = []
            if decisions is not None:
                if len(decisions) != len(native["boxes"]):
                    raise ValueError("Rejected or accepted publisher proposal was dropped")
                for index, (decision, publisher) in enumerate(zip(decisions, native["boxes"], strict=True)):
                    if decision.get("publisher_proposal_index") != index or decision.get("publisher_proposal_preserved") != publisher:
                        raise ValueError("Publisher proposal index or original box changed")
                    accepts = decision.get("visible_box_proposal_accepted_for_development_review") is True
                    if accepts != decision.get("review_decision", "").startswith("ACCEPT_"):
                        raise ValueError("Source proposal decision and admission flag disagree")
                    if accepts:
                        selected.append(decision["selected_manual_box_xyxy_native_px"])
                    else:
                        rejected_proposals += 1
            else:
                if len(native["boxes"]) != 1:
                    raise ValueError("Multi-proposal frame requires per-proposal adjudication")
                accepts = row.get("review_decision", "").startswith("ACCEPT_")
                if "visible_box_proposal_accepted_for_development_review" in row and accepts != row["visible_box_proposal_accepted_for_development_review"]:
                    raise ValueError("Source frame decision and admission flag disagree")
                if accepts:
                    selected.append(row.get("selected_manual_box_xyxy_native_px", row.get("reviewed_box_xyxy_native_px")))
                else:
                    rejected_proposals += 1
            proposal_count += len(native["boxes"])
            if len(selected) > 1:
                raise ValueError("Repair export permits one resolved ball per complete crop")
            if selected:
                if selected[0] is None:
                    raise ValueError("Accepted source proposal lacks a reviewed visible extent")
                accepted[sid] = (selected[0], review_path)
    if set(reviewed) != set(native_by_id):
        raise ValueError("Accepted/rejected source review lineage is incomplete")
    crop_by_id = {row["id"]: row for row in crop_samples}
    if len(crop_by_id) != len(crop_samples) or set(crop_by_id) != set(accepted):
        raise ValueError("Crops must contain every accepted source frame once and no rejected frame")
    for sid, sample in crop_by_id.items():
        selected, review_path = accepted[sid]
        native = native_by_id[sid]
        expected_receipt = {"path": Path(review_path).relative_to(ROOT).as_posix(), "sha256": PINS[review_path]}
        if sample.get("source_review_receipt") != expected_receipt or sample.get("source_review_row_id") != sid:
            raise ValueError("Crop points to another source adjudication receipt or row")
        if sample.get("reviewed_box_xyxy_native_px") != selected:
            raise ValueError("Crop changed the actually reviewed visible native extent")
        for key in ("alias", "native_width", "native_height", "source_pts", "source_time_base", "source_timestamp_fraction"):
            if sample.get(key) != native.get(key):
                raise ValueError("Crop changed original native source metadata")
        if sample.get("source_frame_index") != native["frame_index"] or sample.get("source_native_image") not in native["native_frames"]:
            raise ValueError("Crop is not from its frozen actual PTS frame")
    return {"reviewed_frames": len(reviewed), "reviewed_publisher_proposals": proposal_count,
            "accepted_positive_frames": len(accepted), "rejected_frames": len(reviewed) - len(accepted),
            "rejected_publisher_proposals": rejected_proposals}


def reconstruct_contexts(old_components, alias_rows, all_new_samples, extra_evidenced_links=()):
    """Union known source aliases and conservatively linked author compilations.

    Unknown common publisher does not create a match link; uncertainty remains
    explicit. Evidenced links to any old development/quarantined component make
    newTRAIN admission fail rather than moving old rows or hiding exposure.
    """
    roots = {}

    def find(node):
        roots.setdefault(node, node)
        while roots[node] != node:
            roots[node] = roots[roots[node]]
            node = roots[node]
        return node

    edges = []

    def union(left, right, reason):
        a, b = find(left), find(right)
        if a != b:
            roots[max(a, b)] = min(a, b)
        edges.append({"left": left, "right": right, "reason": reason})

    old_assignments = []
    for component in old_components:
        source_ids = component["source_ids"]
        if not source_ids or component["derived_split"] not in {"train", "development", "quarantine"}:
            raise ValueError("Old conservative source/split component is invalid")
        anchor = "legacy:" + source_ids[0]
        find(anchor)
        for source in source_ids:
            union(anchor, "legacy:" + source, "PRESERVED_OLD_FULL_REVIEW_CONTEXT_COMPONENT")
        for alias in component["aliases"]:
            union(anchor, "event:" + alias.lower(), "PRESERVED_OLD_EVENT_ALIAS_INCLUDING_REJECTED_SOURCE_LINKS")
        old_assignments.append((anchor, component["derived_split"]))
    aliases = {}
    for row in alias_rows:
        original, logical = row["publisher_original_alias"], row["logical_video_alias"]
        if original in aliases or logical in aliases.values():
            raise ValueError("Duplicate or ambiguous author source alias")
        aliases[original] = logical
        union("author:" + original, "author-logical:" + logical, "PUBLISHER_EXPLICIT_ORIGINAL_TO_DEIDENTIFIED_ALIAS")
    # Broad best-points/Federer compilations may share original matches. Keep
    # them together conservatively, including still-unseen aliases, without
    # asserting that any particular unseen match is present.
    possible = [name for name in ("America_Best_10_1", "America_Best_Move_1", "Federer_Top_40_1", "Federer_Top_40_2", "Federer_Top_40_3") if name in aliases]
    for name in possible[1:]:
        union("author:" + possible[0], "author:" + name, "CONSERVATIVE_POSSIBLE_COMPILATION_OVERLAP_UNVERIFIED")
    for names in (("AriyaHim_MostStd_forHand_1", "AriyaHim_MostStd_forHand_2", "AriyaHim_MostStd_forHand_3", "AriyaHim_MostStd_forHand_4"), ("ShunYi_Amaetur_4_0_0", "ShunYi_Amaetur_4_0_3")):
        existing = [name for name in names if name in aliases]
        for name in existing[1:]:
            union("author:" + existing[0], "author:" + name, "CONSERVATIVE_SAME_PUBLISHER_SOURCE_SERIES")
    sample_ids = set()
    for row in all_new_samples:
        if row["id"] in sample_ids or row["alias"] not in aliases:
            raise ValueError("Missing/duplicate new reviewed source or alias")
        sample_ids.add(row["id"])
        family = {"America_Best_10_1": "usopen", "France_Review_Red_Court_1": "rolandgarros"}.get(row["alias"])
        if family is None:
            raise ValueError("New source outside frozen two-compilation scope")
        union("author:" + row["alias"], "event:" + family, "VISUAL_EVENT_FAMILY_RETAINED_FOR_ALL_ACCEPTED_AND_REJECTED_FRAMES")
        union("author:" + row["alias"], "reviewed-frame:" + row["id"], "WHOLE_COMPILATION_AND_ALL_REJECTED_PROPOSAL_CONTEXT")
    for left, right in extra_evidenced_links:
        if left not in roots or right not in roots:
            raise ValueError("Evidenced source link must name existing graph vertices")
        union(left, right, "ADDITIONAL_EVIDENCED_SOURCE_LINK")
    assignments = defaultdict(set)
    for node, split in old_assignments:
        assignments[find(node)].add(split)
    proposed_sources = {"author:" + row["alias"] for row in all_new_samples}
    for node in proposed_sources:
        exposed = assignments.get(find(node), set())
        if exposed - {"train"}:
            raise ValueError("NewTRAIN compilation links to old development/quarantine exposure; preserve old split and repair proposal")
    groups = defaultdict(list)
    for node in roots:
        groups[find(node)].append(node)
    components = []
    for representative, members in sorted(groups.items()):
        splits = assignments.get(representative, set())
        if len(splits) > 1:
            raise ValueError("Context reconstruction crosses preserved old split boundaries")
        components.append({"component_id": "source_context_" + canonical_hash(sorted(members))[:16], "members": sorted(members),
                           "preserved_legacy_split": next(iter(splits)) if splits else None,
                           "new_proposed_split": "train" if any(node in proposed_sources for node in members) else None,
                           "independent_original_match_or_camera_session_verified": False})
    return {"components": components, "edges": edges, "new_proposed_split": "train", "test_count": 0,
            "same_publisher_is_not_proof_of_match_independence": True,
            "paris_quarantine_not_automatically_identified_as_rolandgarros": True,
            "unresolved_possible_overlap_with_other_legacy_matches_remains_unverified": True}


def validate_old_preservation(old_rows, combined_rows):
    inherited = combined_rows[:len(old_rows)]
    if len(inherited) != len(old_rows):
        raise ValueError("Old samples were dropped")
    for old, new in zip(old_rows, inherited, strict=True):
        expected = deepcopy(old)
        expected.update(image_path=new["image_path"], label_path=new["label_path"],
                        copy_source_path=old["image_path"], copy_label_source_path=old["label_path"],
                        inherited_from_old79=True)
        if new != expected:
            raise ValueError("Inherited old image/label/split/provenance was altered")


def native_source_metadata(sample, source, acquisition, probe_path, probe_sha256, probe):
    """Preserve probed rates and publisher timing without assuming normal play."""
    alias = sample["alias"]
    expected_path = (AUDIT / "native_probe" / (alias + ".json")).resolve()
    if Path(probe_path).resolve() != expected_path or Path(source["path"]).stem != alias:
        raise ValueError("Native probe/source path belongs to another compilation")
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Exactly one pinned native video stream is required")
    stream, container = streams[0], probe.get("format", {})
    if (type(stream.get("width")) is not int or type(stream.get("height")) is not int
            or (stream["width"], stream["height"]) != (sample["native_width"], sample["native_height"])):
        raise ValueError("Native probe shape differs from actual reviewed frame")
    if Fraction(stream["time_base"]) != Fraction(sample["source_time_base"]):
        raise ValueError("Native probe time base differs from actual reviewed PTS")
    for field in ("r_frame_rate", "avg_frame_rate", "time_base"):
        if Fraction(stream[field]) <= 0:
            raise ValueError("Native stream rate/time base must be a positive rational")
    frames = probe.get("frames", [])
    if stream.get("nb_frames") is not None and int(stream["nb_frames"]) != len(frames):
        raise ValueError("Native probe frame count differs from complete PTS inventory")
    if not frames or not 0 <= sample["source_frame_index"] < len(frames):
        raise ValueError("Reviewed frame is missing from the full native probe")
    timestamps = [row.get("pts") for row in frames]
    if any(type(pts) is not int for pts in timestamps) or any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Full native probe requires strictly increasing original integer PTS")
    if timestamps[sample["source_frame_index"]] != int(sample["source_pts"]):
        raise ValueError("Native probe frame PTS differs from the actually reviewed source frame")
    if container.get("size") is not None and int(container["size"]) != source["bytes"]:
        raise ValueError("Native probe file size belongs to another source video")
    for value in (stream.get("duration"), container.get("duration")):
        if value is not None and Fraction(value) <= 0:
            raise ValueError("Native duration metadata must remain positive")
    acquired = datetime.fromisoformat(acquisition["created_utc"])
    if acquired.tzinfo is None:
        raise ValueError("Acquisition receipt requires an explicit timezone")
    repository = "Zoutianjian/Ball-Yolo-A-Sport-Ball-Dataset"
    prefix = f"https://raw.githubusercontent.com/{repository}/{acquisition['pinned_commit']}"
    if source["url"] != f"{prefix}/tennis/Videos/{alias}.mp4":
        raise ValueError("Source URL differs from the pinned author repository and compilation")
    grants = [{"path": str((ROOT / path).resolve()), "sha256": sha256, "url": f"{prefix}/README.md"}
              for path, sha256 in acquisition["input_bindings"].items() if Path(path).name == "prerelease_root_readme.txt"]
    if len(grants) != 1:
        raise ValueError("Exact publisher research/academic grant attribution is required")
    return {
        "fps_rational": stream["r_frame_rate"],
        "fps_rational_definition": "NOMINAL_STREAM_R_FRAME_RATE_NOT_VERIFIED_NORMAL_SPEED_OR_CONSTANT_CADENCE",
        "source_stream_r_frame_rate": stream["r_frame_rate"],
        "source_stream_avg_frame_rate": stream["avg_frame_rate"],
        "normal_speed_verified": False, "constant_frame_cadence_verified": False,
        "source_native_probe": {"path": str(expected_path), "sha256": probe_sha256,
            "stream_metadata": {key: stream[key] for key in ("index", "codec_name", "width", "height", "pix_fmt", "r_frame_rate", "avg_frame_rate", "time_base", "start_pts", "start_time", "nb_frames", "duration") if key in stream},
            "container_metadata": {key: container[key] for key in ("format_name", "duration", "size") if key in container},
            "native_frame_pts_count": len(frames)},
        "source_publisher_repository": repository,
        "source_publisher_repository_url": f"https://github.com/{repository}",
        "source_publisher_is_not_verified_original_broadcast_rightsholder": True,
        "source_publisher_git_commit": acquisition["pinned_commit"],
        "source_publisher_git_blob_sha1": source["publisher_git_blob_sha1"],
        "source_acquisition_receipt_created_utc": acquisition["created_utc"],
        "source_acquisition_timestamp_definition": "ACQUISITION_RECEIPT_CREATED_UTC_NOT_CONTAINER_CREATION_TIME_OR_INDIVIDUAL_DOWNLOAD_START",
        "source_acquisition_receipt": {"path": str(AUDIT / "acquisition_receipt.json"), "sha256": PINS[AUDIT / "acquisition_receipt.json"]},
        "source_grant_attribution": grants,
    }


def verify_approval(approval, contract_hash, bindings, ids):
    if approval.get("status") != APPROVAL_STATUS or approval.get("reviewer_type") != "ROOT_AGENT":
        raise ValueError("Separate ROOT review of concrete101 proposal is required")
    if approval.get("proposal_sha256") != contract_hash or approval.get("input_bindings") != bindings or approval.get("approved_ordered_sample_ids") != ids:
        raise ValueError("ROOT approval differs from exact reviewed proposal, inputs or ordered samples")
    for key in ("qualification_evidence", "independent_human_review", "raw_media_public_redistribution_cleared", "weights_public_release_cleared", "commercial_clearance"):
        if approval.get(key) is not False:
            raise ValueError("Private agent development approval cannot assert broader qualification/rights")
    if type(approval.get("test_count")) is not int or approval["test_count"] != 0:
        raise ValueError("FinalTEST data are forbidden in this repair export")
    if approval.get("allowed_use") != "PRIVATE_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT" or not isinstance(approval.get("review_note"), str) or not approval["review_note"].strip():
        raise ValueError("Explicit private academic use and ROOT review rationale are required")


def prepare_plan(output=OUTPUT, approval_path=None, approval_sha256=None, *, allow_existing=False):
    output = parent.validate_destination(output, require_fresh=not allow_existing)
    bindings = {}
    for path, expected in PINS.items():
        bind(bindings, path, expected)
    bind(bindings, Path(__file__), parent.digest(__file__))
    check_bindings(bindings)
    old_result = parent.verify_export(OLD, expected_manifest_sha256=PINS[OLD / "source_manifest.json"])
    old = old_result["manifest"]
    if old_result["counts"] != {"images": 79, "positive_boxes": 75, "negative_tiles": 4, "train": 58, "val": 21, "test": 0, "positive_split_counts": {"train": 55, "development": 20}}:
        raise ValueError("Frozen old79 counts changed")
    for path, expected in old["input_bindings"].items():
        bind(bindings, path, expected)
    derived = read_json(parent.DERIVED_MANIFEST)
    for path, expected in derived["source_bindings"].items():
        bind(bindings, path, expected)
    crops, native, acquisition, alias_inventory = map(read_json, (CROPS, NATIVE, AUDIT / "acquisition_receipt.json", ALIASES))
    if len(native["samples"]) != 32 or sum(len(row["boxes"]) for row in native["samples"]) != 34:
        raise ValueError("All32 new reviewed frames and34 proposals, including rejected ones, are required")
    if crops["sample_count"] != 22 or crops["sample_count"] != len(crops["samples"]) or crops.get("all_exact_native_roi_pixels_verified") is not True:
        raise ValueError("Exact22 pixel-verified native crops are required")
    for path, expected in crops["input_bindings"].items():
        bind(bindings, path, expected)
    for path, expected in native["input_bindings"].items():
        bind(bindings, path, expected)
    for path, expected in acquisition["input_bindings"].items():
        bind(bindings, path, expected)
    closure_files = (AUDIT / "france_fullcrop9_agent_review.json", AUDIT / "usa_fullcrop13_root_review.json")
    closures = {}
    for path in closure_files:
        receipt = read_json(path)
        if receipt.get("crop_review_manifest") != {"path": str(CROPS), "sha256": PINS[CROPS]}:
            raise ValueError("Full crop receipt binds another rendered crop set")
        for row in receipt["rows"]:
            if row["id"] in closures:
                raise ValueError("Duplicate full crop closure review")
            closures[row["id"]] = (path, row)
    if set(closures) != {row["id"] for row in crops["samples"]}:
        raise ValueError("Complete crop closure has missing or additional accepted sample IDs")
    source_reviews = [(AUDIT / name, read_json(AUDIT / name)) for name in ("first8_final_agent_visual_review.json", "france_tail12_agent_visual_review.json", "usa_tail12_root_visual_review.json")]
    source_review_counts = validate_source_reviews(native["samples"], source_reviews, crops["samples"])
    if source_review_counts != {"reviewed_frames": 32, "reviewed_publisher_proposals": 34, "accepted_positive_frames": 22, "rejected_frames": 10, "rejected_publisher_proposals": 12}:
        raise ValueError("Frozen source adjudication counts changed")
    if len(alias_inventory["publisher_alias_mapping"]) != 16:
        raise ValueError("All16 original/logical author aliases must remain in context reconstruction")
    graph = reconstruct_contexts(old["components"], alias_inventory["publisher_alias_mapping"], native["samples"])
    proposed = []
    for original in old_result["samples"]:
        row = deepcopy(original)
        split = row["development_split"]
        row.update(image_path=str(output / "ball/images" / split / (row["sample_id"] + ".png")),
                   label_path=str(output / "ball/labels" / split / (row["sample_id"] + ".txt")),
                   copy_source_path=original["image_path"], copy_label_source_path=original["label_path"], inherited_from_old79=True)
        for key in ("image_path", "label_path"):
            bind(bindings, original[key], parent.digest(original[key]))
        proposed.append(row)
    source_by_alias = {Path(row["path"]).stem: row for row in acquisition["files"]}
    for sample in crops["samples"]:
        validate_geometry(sample)
        closure_path, closure = closures[sample["id"]]
        validate_closure(sample, closure, france=sample["alias"] == "France_Review_Red_Court_1")
        for key in ("unmarked_crop", "separate_overlay", "source_native_image"):
            bind(bindings, sample[key]["path"], sample[key]["sha256"])
        parent._native640_header(ROOT / sample["unmarked_crop"]["path"])
        line = parent.yolo_line(sample["reviewed_box_xyxy_crop_px"])
        source = source_by_alias[sample["alias"]]
        probe_path = (AUDIT / "native_probe" / (sample["alias"] + ".json")).resolve()
        probe_bindings = {str((ROOT / path).resolve()): sha256 for path, sha256 in native["input_bindings"].items() if "native_probe" in Path(path).parts}
        if str(probe_path) not in probe_bindings:
            raise ValueError("Native probe must come from the frozen rendered review input bindings")
        metadata = native_source_metadata(sample, source, acquisition, probe_path, probe_bindings[str(probe_path)], read_json(probe_path))
        sid = "author_" + sample["id"]
        graph_component = next(component for component in graph["components"] if "author:" + sample["alias"] in component["members"])
        proposed.append({"sample_id": sid, "source_id": "author:" + sample["alias"], "source_type": "AUTHOR_NATIVE_PTS_EXACT_REVIEWED640_CROP",
                         "development_split": "train", "derived_split": "train", "inherited_from_old79": False,
                         "image_source_path": str(ROOT / sample["unmarked_crop"]["path"]), "image_source_sha256": sample["unmarked_crop"]["sha256"],
                         "copy_source_path": str(ROOT / sample["unmarked_crop"]["path"]), "copy_label_source_path": None,
                         "image_path": str(output / "ball/images/train" / (sid + ".png")), "label_path": str(output / "ball/labels/train" / (sid + ".txt")),
                         "label_text": line, "label_sha256": hashlib.sha256(line.encode()).hexdigest(),
                         "box_xyxy_crop_px": sample["reviewed_box_xyxy_crop_px"], "box_xyxy_native_px": sample["reviewed_box_xyxy_native_px"],
                         "crop_xyxy_native_px": [*sample["actual_crop_origin_native_px"], sample["actual_crop_origin_native_px"][0] + 640, sample["actual_crop_origin_native_px"][1] + 640],
                         "crop_size_px": [640, 640], "resized": False, "negative_visible_absence_only": False,
                         "source_frame_index": sample["source_frame_index"], "source_pts": sample["source_pts"], "source_time_base": sample["source_time_base"],
                         "source_timestamp_fraction_seconds": sample["source_timestamp_fraction"],
                         "native_resolution": [sample["native_width"], sample["native_height"]], "source_video_path": str(ROOT / source["path"]),
                         "source_video_sha256": source["sha256"], "source_url": source["url"], "derived_component_id": graph_component["component_id"],
                         "match_id": None, "session_id": None, "actual_match_identity_verified": False, "actual_session_identity_verified": False,
                         "source_independence_verified": False, "independent_human_review": False, "qualification_evidence": False,
                         "allowed_use": "PRIVATE_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT", "training_allowed_after_root_export_approval": True,
                         "raw_broadcast_redistribution_cleared": False, "weights_public_release_cleared": False, "commercial_clearance": False,
                         "source_grant": acquisition["publisher_grant"], "source_rights_limitations": acquisition["rights_limitation"],
                         **metadata,
                         "review_evidence": {"source_box_review": sample["source_review_receipt"], "complete_crop_review_receipt": {"path": str(closure_path), "sha256": PINS[closure_path]},
                                             "complete_crop_review": closure, "native_crop_record": sample}})
    validate_old_preservation(old_result["samples"], proposed)
    counts = {"images": len(proposed), "positive_boxes": sum(not row["negative_visible_absence_only"] for row in proposed),
              "negative_tiles": sum(row["negative_visible_absence_only"] for row in proposed), "train": sum(row["development_split"] == "train" for row in proposed),
              "val": sum(row["development_split"] == "val" for row in proposed), "test": 0, "inherited_images": len(old_result["samples"]), "new_positive_images": len(crops["samples"])}
    if counts != EXPECTED_COUNTS or len({row["sample_id"] for row in proposed}) != 101:
        raise ValueError("Repair101 proposal count/duplicate contract changed")
    check_bindings(bindings)
    contract = {"schema_version": SCHEMA, "output": str(output), "counts": counts, "input_bindings": bindings,
                "parent_dataset": {"manifest_path": old_result["manifest_path"], "manifest_sha256": old_result["manifest_sha256"], "receipt_sha256": old_result["receipt_sha256"]},
                "inherited_sources_preserved": old_result["samples"], "preserved_old_components": old["components"], "context_graph": graph,
                "all32_new_source_review_rows_including_rejected": [row for _, review in source_reviews for row in review["rows"]],
                "source_review_counts": source_review_counts,
                "all34_publisher_proposals_preserved": native["samples"], "all_author_aliases_preserved": alias_inventory["publisher_alias_mapping"],
                "proposed_samples": proposed, "qualification_evidence": False, "independent_human_review": False,
                "final_test_used": False, "models_used": [], "training_performed": False,
                "scope": "PRIVATE_NONCOMMERCIAL_NATIVE640_AGENT_REVIEWED_DEVELOPMENT_REPAIR_ONLY",
                "normal_speed_and_original_match_independence_verified": False}
    contract_hash = canonical_hash(contract)
    ids = [row["sample_id"] for row in proposed]
    template = {"status": APPROVAL_STATUS, "reviewer_type": "ROOT_AGENT", "proposal_sha256": contract_hash, "input_bindings": bindings,
                "approved_ordered_sample_ids": ids, "test_count": 0, "allowed_use": "PRIVATE_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT",
                "qualification_evidence": False, "independent_human_review": False, "raw_media_public_redistribution_cleared": False,
                "weights_public_release_cleared": False, "commercial_clearance": False, "review_note": ""}
    approval = None
    if approval_path is not None or approval_sha256 is not None:
        if approval_path is None or approval_sha256 is None or parent.digest(approval_path) != approval_sha256:
            raise ValueError("Approval path and exact approval hash are both required")
        verify_approval(read_json(approval_path), contract_hash, bindings, ids)
        approval = {"path": str(Path(approval_path).resolve()), "sha256": approval_sha256}
    return {"status": "ROOT_APPROVED_READY_TO_EXPORT" if approval else "PROPOSED101_REQUIRES_ROOT_REVIEW_BEFORE_EXPORT",
            "created_utc": datetime.now(timezone.utc).isoformat(), "proposal_sha256": contract_hash,
            "contract": contract, "approval_template": template, "root_final_approval": approval}


def compare_roi_pixels(native_rgb, native_size, crop_rgb, origin, crop_size=(640, 640)):
    """Exact RGB row comparison; geometry is integral and never resampled."""
    width, height = native_size
    cw, ch = crop_size
    if any(type(v) is not int for v in (*native_size, *origin, *crop_size)) or min(width, height, cw, ch) <= 0:
        raise ValueError("Native ROI dimensions/origin must be positive integral geometry")
    if len(native_rgb) != width * height * 3 or len(crop_rgb) != cw * ch * 3:
        raise ValueError("RGB byte lengths differ from source/crop shapes")
    left, top = origin
    if left < 0 or top < 0 or left + cw > width or top + ch > height:
        raise ValueError("Native ROI leaves original image")
    for y in range(ch):
        start = ((top + y) * width + left) * 3
        if native_rgb[start:start + cw * 3] != crop_rgb[y * cw * 3:(y + 1) * cw * 3]:
            raise ValueError("Review crop pixels differ from exact native source ROI")
    return hashlib.sha256(crop_rgb).hexdigest()


def read_rgb8_png(path, *, max_pixels=1920 * 1080, max_file_bytes=32 * 1024 * 1024):
    """Decode the reviewed non-interlaced RGB8 PNG subset with bounded storage.

    Chunk CRCs, contiguous IDAT, exact scanline size and the zlib end marker are
    required. Unsupported formats fail rather than silently converting pixels.
    The decoded buffer is at most max_pixels * 3 bytes, independent of file data.
    """
    with Path(path).open("rb") as handle:
        raw = handle.read(max_file_bytes + 1)
    if len(raw) > max_file_bytes or raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("PNG signature/file size is outside the bounded RGB reader")
    offset, width, height = 8, None, None
    compressed = bytearray()
    idat_seen = idat_closed = ended = False
    while offset < len(raw):
        if len(raw) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", raw, offset)[0]
        kind = raw[offset + 4:offset + 8]
        finish = offset + 12 + length
        if finish > len(raw):
            raise ValueError("PNG chunk leaves bounded file")
        body = raw[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack_from(">I", raw, offset + 8 + length)[0]
        if zlib.crc32(kind + body) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if width is None and kind != b"IHDR":
            raise ValueError("PNG IHDR must be first")
        if kind == b"IHDR":
            if width is not None or length != 13:
                raise ValueError("Duplicate or malformed PNG IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", body)
            if not width or not height or width * height > max_pixels or (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ValueError("Only bounded non-interlaced native RGB8 PNG is permitted")
        elif kind == b"IDAT":
            if idat_closed:
                raise ValueError("PNG IDAT chunks must be contiguous")
            idat_seen = True
            compressed.extend(body)
        elif kind == b"IEND":
            if length or not idat_seen or finish != len(raw):
                raise ValueError("PNG IEND must close actual image data without trailing bytes")
            ended = True
            offset = finish
            break
        elif kind and kind[0] & 32:
            if idat_seen:
                idat_closed = True
        else:
            raise ValueError("Unsupported critical PNG chunk")
        offset = finish
    if not ended or width is None:
        raise ValueError("PNG is missing terminal image chunks")
    stride = width * 3
    expected_size = (stride + 1) * height
    inflater = zlib.decompressobj()
    try:
        packed = inflater.decompress(compressed, expected_size + 1)
    except zlib.error as error:
        raise ValueError("Invalid PNG zlib stream") from error
    if len(packed) != expected_size or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError("PNG decoded byte count or zlib stream boundary differs from IHDR")
    pixels = bytearray(stride * height)
    for y in range(height):
        start, out = y * (stride + 1), y * stride
        method = packed[start]
        source = packed[start + 1:start + 1 + stride]
        if method == 0:
            pixels[out:out + stride] = source
        elif method == 1:
            for x, value in enumerate(source):
                pixels[out + x] = (value + (pixels[out + x - 3] if x >= 3 else 0)) & 255
        elif method == 2:
            for x, value in enumerate(source):
                pixels[out + x] = (value + (pixels[out - stride + x] if y else 0)) & 255
        elif method in (3, 4):
            for x, value in enumerate(source):
                left = pixels[out + x - 3] if x >= 3 else 0
                above = pixels[out - stride + x] if y else 0
                if method == 3:
                    predictor = (left + above) // 2
                else:
                    upper_left = pixels[out - stride + x - 3] if y and x >= 3 else 0
                    p = left + above - upper_left
                    a, b, c = abs(p - left), abs(p - above), abs(p - upper_left)
                    predictor = left if a <= b and a <= c else above if b <= c else upper_left
                pixels[out + x] = (value + predictor) & 255
        else:
            raise ValueError("Unsupported PNG scanline filter")
    return bytes(pixels), (width, height)


def verify_actual_new_rois(contract):
    parent.guarded_resources()
    results = []
    for row in contract["proposed_samples"]:
        if row["inherited_from_old79"]:
            continue
        parent.guarded_resources()
        sample = row["review_evidence"]["native_crop_record"]
        native_rgb, native_size = read_rgb8_png(ROOT / sample["source_native_image"]["path"])
        crop_rgb, crop_size = read_rgb8_png(row["copy_source_path"], max_pixels=640 * 640)
        if native_size != tuple(row["native_resolution"]) or crop_size != (640, 640):
            raise ValueError("Native crop pixel shapes changed")
        pixel_hash = compare_roi_pixels(native_rgb, native_size, crop_rgb, sample["actual_crop_origin_native_px"])
        if pixel_hash != sample["unmarked_crop"]["native_rgb_pixels_sha256"]:
            raise ValueError("Native RGB pixel hash differs from actual reviewed640 crop")
        results.append({"sample_id": row["sample_id"], "native_crop_rgb_sha256": pixel_hash, "exact_native_roi_verified": True})
        del native_rgb, crop_rgb
    if len(results) != 22:
        raise ValueError("All22 appended native pixel ROIs must be checked")
    return results


def data_yaml(output):
    return "path: " + json.dumps(str(Path(output) / "ball")) + "\ntrain: images/train\nval: images/val\nnames:\n  0: tennis_ball\n"


def execute_export(plan):
    if plan.get("status") != "ROOT_APPROVED_READY_TO_EXPORT" or not plan.get("root_final_approval"):
        raise ValueError("Unapproved101 proposal must not write supervised data")
    contract = plan["contract"]
    output = parent.validate_destination(contract["output"])
    approval = plan["root_final_approval"]
    reconstructed = prepare_plan(output, approval["path"], approval["sha256"])
    if reconstructed["contract"] != contract or reconstructed["proposal_sha256"] != plan["proposal_sha256"]:
        raise ValueError("Caller altered the approved repair proposal")
    admission = parent.guarded_resources()
    rois = verify_actual_new_rois(contract)
    check_bindings(contract["input_bindings"])
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        (output / "ball/images" / split).mkdir(parents=True)
        (output / "ball/labels" / split).mkdir(parents=True)
    for row in contract["proposed_samples"]:
        parent.guarded_resources()
        parent.copy_and_verify(row["copy_source_path"], row["image_path"], row["image_source_sha256"])
        if row["inherited_from_old79"]:
            parent.copy_and_verify(row["copy_label_source_path"], row["label_path"], row["label_sha256"])
        else:
            with Path(row["label_path"]).open("xb") as handle:
                handle.write(row["label_text"].encode())
        if parent.digest(row["label_path"]) != row["label_sha256"]:
            raise ValueError("Exported label differs from exact old bytes or reviewed box serialization")
    with (output / "ball/data.yaml").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(data_yaml(output))
    check_bindings(contract["input_bindings"])
    manifest = {"schema_version": SCHEMA, "status": "EXPORTED_PRIVATE_BALL101_REPAIR_NOT_QUALIFICATION",
                "exported_utc": datetime.now(timezone.utc).isoformat(), "proposal_sha256": plan["proposal_sha256"],
                "contract": contract, "root_final_approval": approval, "actual22_native_pixel_roi_results": rois,
                "export_launch_resources": admission, "models_used": [], "training_performed": False, "qualification_evidence": False}
    write_new_json(output / "source_manifest.json", manifest)
    receipt = {"status": "PASS101_BYTE_PRESERVATION_NATIVE_ROI_AND_REVIEW_GRAPH_PRIVATE_EXPORT",
               "source_manifest_sha256": parent.digest(output / "source_manifest.json"), "data_yaml_sha256": parent.digest(output / "ball/data.yaml"),
               "proposal_sha256": plan["proposal_sha256"], "counts": contract["counts"], "root_final_approval": approval,
               "inherited79_images_and_labels_copied_byte_identically": True, "new22_exact_native_rois_rechecked": True,
               "input_files_changed": [], "qualification_evidence": False, "independent_human_review": False, "training_performed": False}
    write_new_json(output / "export_receipt.json", receipt)
    return receipt


def verify_export(output=OUTPUT, expected_manifest_sha256=None):
    output = parent.validate_destination(output, require_fresh=False)
    manifest_path = output / "source_manifest.json"
    manifest_hash = parent.digest(manifest_path)
    if expected_manifest_sha256 is not None and manifest_hash != expected_manifest_sha256:
        raise ValueError("Repair manifest differs from externally pinned export")
    manifest, receipt = read_json(manifest_path), read_json(output / "export_receipt.json")
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "EXPORTED_PRIVATE_BALL101_REPAIR_NOT_QUALIFICATION":
        raise ValueError("Repair dataset lacks terminal approved export contract")
    approval = manifest["root_final_approval"]
    expected = prepare_plan(output, approval["path"], approval["sha256"], allow_existing=True)
    contract = expected["contract"]
    if manifest.get("contract") != contract or manifest.get("proposal_sha256") != expected["proposal_sha256"]:
        raise ValueError("Repair dataset differs from full source/review reconstruction")
    for row in contract["proposed_samples"]:
        if parent.digest(row["image_path"]) != row["image_source_sha256"] or parent.digest(row["label_path"]) != row["label_sha256"]:
            raise ValueError("Repair image/label bytes changed")
        if Path(row["label_path"]).read_bytes() != row["label_text"].encode():
            raise ValueError("Repair label serialization differs from reviewed extent")
        parent._native640_header(row["image_path"])
    wanted = {str(Path(row[key]).relative_to(output)) for row in contract["proposed_samples"] for key in ("image_path", "label_path")}
    wanted.update({"source_manifest.json", "export_receipt.json", str(Path("ball/data.yaml"))})
    actual = {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    if actual != wanted or (output / "ball/images/test").exists() or (output / "ball/labels/test").exists():
        raise ValueError("Missing, unexpected or finalTEST file in repair dataset")
    if (output / "ball/data.yaml").read_text(encoding="utf-8") != data_yaml(output):
        raise ValueError("Repair YAML points outside exact train/val dataset")
    rois = manifest.get("actual22_native_pixel_roi_results", [])
    expected_rois = [{"sample_id": row["sample_id"], "native_crop_rgb_sha256": row["review_evidence"]["native_crop_record"]["unmarked_crop"]["native_rgb_pixels_sha256"], "exact_native_roi_verified": True} for row in contract["proposed_samples"] if not row["inherited_from_old79"]]
    if rois != expected_rois:
        raise ValueError("Terminal export lacks exact22 source/crop pixel evidence")
    if receipt.get("status") != "PASS101_BYTE_PRESERVATION_NATIVE_ROI_AND_REVIEW_GRAPH_PRIVATE_EXPORT" or receipt.get("counts") != contract["counts"] or receipt.get("input_files_changed") != []:
        raise ValueError("Repair export receipt is missing or inconsistent")
    if receipt.get("source_manifest_sha256") != manifest_hash or receipt.get("data_yaml_sha256") != parent.digest(output / "ball/data.yaml") or receipt.get("root_final_approval") != approval or receipt.get("proposal_sha256") != expected["proposal_sha256"]:
        raise ValueError("Repair receipt hashes or root approval changed")
    return {"status": "PASS_EXISTING_PRIVATE_BALL101_REPAIR_INTEGRITY_NOT_QUALIFICATION", "manifest": manifest,
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash, "counts": contract["counts"],
            "samples": contract["proposed_samples"], "receipt_path": str(output / "export_receipt.json"),
            "receipt_sha256": parent.digest(output / "export_receipt.json"), "data_yaml_path": str(output / "ball/data.yaml"),
            "data_yaml_sha256": parent.digest(output / "ball/data.yaml"), "qualification_evidence": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--verify-native-rois", action="store_true", help="Read-only bounded stdlib decode of all22 exact native pixel ROIs before ROOT review")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = prepare_plan(args.output, args.approval, args.approval_sha256)
    if args.verify_native_rois:
        plan["actual_readonly_native_pixel_roi_results"] = verify_actual_new_rois(plan["contract"])
    if args.plan_output:
        write_new_json(args.plan_output, plan)
    print(json.dumps(execute_export(plan) if args.execute else {"status": plan["status"], "proposal_sha256": plan["proposal_sha256"], "counts": plan["contract"]["counts"], "bound_inputs": len(plan["contract"]["input_bindings"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
