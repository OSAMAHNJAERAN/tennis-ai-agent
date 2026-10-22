"""Audit agent-reviewed development observations and prepare complete crop review.

This script creates NO supervised labels and imports NO model/video runtime.  It
preserves the original 600-item sampler, both 48-item visual reviews, and root
negative decisions.  Native 640-pixel crops require a separate complete-pixel
review before they may become supervised examples.
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
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/validation/yolo26_execution_2026-10-15"
MANIFEST = ROOT / "outputs/ball_data_repair_review_wave96_v3/review_aids_manifest.json"
SAMPLER = ROOT / "outputs/ball_data_repair_plan600_v2/frozen_sampler_source_groups.json"
HEAD = ARTIFACTS / "ball_review_head_v1/head_observations.json"
TAIL = ARTIFACTS / "ball_review_tail_v1/tail_observations.json"
NEGATIVES = ARTIFACTS / "ball_negative_review_root_v1.json"
NEGATIVE_REVISION = ARTIFACTS / "ball_negative_context_correction_v2.json"
PINNED = {
    MANIFEST: "c6ec5bd28d0a36af85afb5f8018b136f9bc9afbbae65996bf9aa938ab2ab4a17",
    SAMPLER: "920be9491390ef62327a19bf8c034a623ee14bd233c7801b93565af279eaf428",
    HEAD: "5be26332227cf40c0337d7b8fc08ab9858148e713bede57e747ec092360fbbf8",
    TAIL: "4819ae4d254fdebb58261d24468c8165e7e75dcb8e30780ec32787927cad2819",
    NEGATIVES: "4760f30b2eb6eed4faa79995022d87ecd8e4aa9c17309c5677e6291882b52d7b",
    NEGATIVE_REVISION: "7b621034f41df26f4ea3c24a635662943584271491568f5b5ee53ef409b1e6b9",
}
MIN_RAM = 1024 ** 3
MIN_DISK = 2 * 1024 ** 3
SEED = 709


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def guarded_resources():
    import psutil
    values = {"available_ram_bytes": int(psutil.virtual_memory().available),
              "free_disk_bytes": int(shutil.disk_usage(ROOT).free),
              "owned_rss_bytes": int(psutil.Process().memory_info().rss)}
    if values["available_ram_bytes"] < MIN_RAM or values["free_disk_bytes"] < MIN_DISK:
        raise RuntimeError(f"Unchanged 1 GiB RAM / 2 GiB disk preparation guard: {values}")
    return values


def _finite(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} requires {length} finite coordinates")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError(f"{name} coordinates must be finite numbers, not booleans")
    return list(values)


def _box(box):
    box = _finite(box, 4, "Box")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box requires positive visible extent")
    return box


def translate_box(box, origin, inverse=False):
    box, origin = _box(box), _finite(origin, 2, "Origin")
    sign = 1 if inverse else -1
    return [value + sign * origin[index % 2] for index, value in enumerate(box)]


def validate_manual_box(box, center, patch_box, patch_origin, width=1920, height=1080):
    box, center = _box(box), _finite(center, 2, "Center")
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Manual visible box outside native source geometry")
    expected_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    if center != expected_center:
        raise ValueError("Reviewed center must match its declared visible-extent midpoint")
    if translate_box(patch_box, patch_origin, inverse=True) != box:
        raise ValueError("Manual patch/native box inverse differs")
    return box


def jittered_crop(box, proposal_id, width=1920, height=1080, size=640, seed=SEED):
    """Hash-stable integer jitter, preserving all visible support without resize."""
    box = _box(box)
    if any(type(v) is not int or v <= 0 for v in (width, height, size)) or size > min(width, height):
        raise ValueError("Crop dimensions must be positive integer native geometry")
    if not isinstance(proposal_id, str) or not proposal_id or type(seed) is not int:
        raise ValueError("Crop requires a stable proposal ID and integer seed")
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError("Crop box outside native source geometry")
    ranges = [(max(0, math.ceil(box[2] - size)), min(width - size, math.floor(box[0]))),
              (max(0, math.ceil(box[3] - size)), min(height - size, math.floor(box[1])))]
    origin = []
    for axis, (lower, upper) in enumerate(ranges):
        if upper < lower:
            raise ValueError("Whole visible extent cannot fit inside a native crop")
        token = f"tennis-native640-jitter-v1|{seed}|{proposal_id}|{axis}".encode()
        value = int.from_bytes(hashlib.sha256(token).digest(), "big")
        origin.append(lower + value % (upper - lower + 1))
    return [origin[0], origin[1], origin[0] + size, origin[1] + size]


def observed_family(observation):
    if observation.get("observed_event_family"):
        return observation["observed_event_family"]
    if observation.get("observed_event_group"):
        return observation["observed_event_group"]
    identity = observation.get("observed_event_identity", "")
    for prefix, family in (("LIBEMA_OPEN", "libema_open"), ("MIAMI_OPEN", "miami"),
                           ("DUBAI_DUTY_FREE", "dubai"),
                           ("NITTO_ATP_FINALS_WITH_50_YEARS_1970_2020", "atp_finals_2020_city_unverified")):
        if identity.startswith(prefix):
            return family
    return None


def normalize_positive(proposal, observation, index):
    """Unify the two actual reviewer schemas without upgrading their evidence."""
    if type(index) is not int or index < 0:
        raise ValueError("Review order must be a nonnegative integer")
    for key in ("frame_index", "source_pts"):
        if type(proposal.get(key)) is not int or type(observation.get(key)) is not int:
            raise ValueError("Source frame and PTS must be integers, not booleans")
    for key in ("proposal_id", "source_id", "clip", "frame_index", "source_pts", "source_time_base"):
        if observation.get(key) != proposal.get(key):
            raise ValueError(f"Review/source mismatch for {key}: {proposal.get('proposal_id')}")
    observed_index = observation.get("selection_index", observation.get("manifest_positive_index"))
    if type(observed_index) is not int or observed_index != index:
        raise ValueError("Visual-review order differs from immutable 96-item manifest")
    if observation.get("frozen_conservative_event_group") != proposal["conservative_event_group"]:
        raise ValueError("Original event alias changed in visual review")
    if observation.get("frozen_proposed_split") != proposal["proposed_split"]:
        raise ValueError("Original split changed in visual review")
    decision = observation["decision"]
    if decision in ("ACCEPT_VISIBLE_BBOX", "ACCEPT_VISIBLE_BOX"):
        decision = "ACCEPT_VISIBLE_BOX"
    elif decision not in ("REJECT_AMBIGUOUS", "REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED"):
        raise ValueError("Unknown manual review decision")
    box = observation.get("bbox_xyxy_native_px", observation.get("accept_visible_bbox_xyxy_native_px"))
    extent = observation.get("observed_visible_extent_xyxy_native_px", box)
    center = observation.get("visible_center_xy_native_px", observation.get("observed_center_xy_native_px"))
    patch = observation.get("manual_bbox_xyxy_patch_px", observation.get("manually_selected_bbox_xyxy_patch_px"))
    origin = observation.get("native_patch_origin_xy_px")
    if origin != proposal["native_patch_origin_xy_px"]:
        raise ValueError("Manual review patch origin changed")
    if extent is not None:
        validate_manual_box(extent, center, patch, origin)
        if "MIDPOINT" not in observation.get("center_definition", ""):
            raise ValueError("Visible extent center definition is missing")
    if decision == "ACCEPT_VISIBLE_BOX" and (box is None or box != extent):
        raise ValueError("Accepted item lacks exact reviewed visible support")
    if decision != "ACCEPT_VISIBLE_BOX" and box is not None:
        raise ValueError("Rejected item cannot supply an accepted supervised box")
    if observation.get("independent_human_review") is not False:
        raise ValueError("Agent observation cannot become independent human truth")
    if observation.get("component_hypothesis_autoaccepted", observation.get("component_box_autoaccepted")) is not False:
        raise ValueError("Automatic component acceptance is forbidden")
    return {"proposal_id": proposal["proposal_id"], "manifest_positive_index": index,
            "source_id": proposal["source_id"], "clip": proposal["clip"],
            "frame_index": proposal["frame_index"], "source_pts": proposal["source_pts"],
            "source_time_base": proposal["source_time_base"],
            "timestamp_fraction_seconds": proposal["timestamp_fraction_seconds"],
            "frozen_conservative_event_group": proposal["conservative_event_group"],
            "frozen_proposed_split": proposal["proposed_split"],
            "decision": decision, "bbox_xyxy_native_px": box,
            "observed_visible_extent_xyxy_native_px": extent, "visible_center_xy_native_px": center,
            "center_definition": observation.get("center_definition"),
            "observed_event_family": observed_family(observation),
            "reviewer": observation["reviewer"], "reviewer_type": "AGENT",
            "independent_human_review": False, "original_observation": deepcopy(observation),
            "primary_camera_scope": observation.get("primary_camera_scope"),
            "supervised_export_approved": False, "complete_positive_crop_reviewed": False,
            "native_frame": deepcopy(proposal["images"]["native_frame"])}


def build_components(groups, sources, observations):
    """Transitive original-alias/source/observed-family graph, including rejects."""
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    memberships = {}
    for group in groups:
        for source in group["source_ids"]:
            if source in memberships:
                raise ValueError("Source occurs in multiple original groups")
            memberships[source] = group["group"]
            union("event:" + group["group"], "source:" + source)
    source_map = {source["source_id"]: source for source in sources}
    if len(source_map) != len(sources) or set(memberships) != set(source_map):
        raise ValueError("Original conservative groups must cover all source IDs once")
    for source in sources:
        if source.get("conservative_event_group") != memberships[source["source_id"]]:
            raise ValueError("Source/original-group membership differs")
        group = next(group for group in groups if group["group"] == memberships[source["source_id"]])
        if source.get("proposed_split") != group["proposed_split"]:
            raise ValueError("Source/original-group split differs; cannot demote development")
    for observation in observations:
        if observation["source_id"] not in memberships:
            raise ValueError("Observed source missing from immutable source accounting")
        family = observation.get("observed_event_family")
        if family:
            union("source:" + observation["source_id"], "event:" + family)
    nodes = defaultdict(list)
    for node in sorted(parents):
        nodes[find(node)].append(node)
    result = []
    for members in nodes.values():
        aliases = sorted(node.removeprefix("event:") for node in members if node.startswith("event:"))
        ids = sorted(node.removeprefix("source:") for node in members if node.startswith("source:"))
        original_groups = [group for group in groups if group["group"] in aliases]
        reasons = sorted({group["exclusion_reason"] for group in original_groups if group.get("exclusion_reason")})
        original_splits = {group["proposed_split"] for group in original_groups}
        if not original_splits <= {"train", "development"}:
            raise ValueError("Final TEST or unknown split is forbidden in development preparation")
        split = "quarantine" if reasons else "development" if "development" in original_splits else "train"
        token = "|".join(members).encode()
        result.append({"component_id": "context_" + hashlib.sha256(token).hexdigest()[:16],
                       "aliases": aliases, "source_ids": ids, "derived_split": split,
                       "original_splits": sorted(original_splits), "exclusion_reasons": reasons,
                       "actual_match_identity_verified": False, "actual_session_identity_verified": False,
                       "source_independence_verified": False})
    return sorted(result, key=lambda item: (item["aliases"], item["source_ids"]))


def prepare_audit():
    bindings = {}

    def verify(path, expected):
        path = Path(path)
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"Pinned input SHA256 mismatch: {path}")
        previous = bindings.setdefault(str(path), actual)
        if previous != actual:
            raise ValueError("Conflicting source file bindings")
        return actual

    for path, expected in PINNED.items():
        verify(path, expected)
    manifest, sampler, head, tail, negatives, revision = map(read_json, PINNED)
    if manifest["sampler_sha256"] != PINNED[SAMPLER] or manifest["frozen_sampler"]["sha256"] != PINNED[SAMPLER]:
        raise ValueError("Render manifest sampler lineage differs")
    if head["manifest"]["sha256"] != PINNED[MANIFEST] or tail["review_manifest_sha256"] != PINNED[MANIFEST]:
        raise ValueError("Agent review manifest lineage differs")
    if head["frozen_sampler"]["sha256"] != PINNED[SAMPLER] or tail["frozen_sampler_sha256"] != PINNED[SAMPLER]:
        raise ValueError("Agent review sampler lineage differs")
    verify(head["inspection_boards"]["path"], head["inspection_boards"]["sha256"])
    verify(tail["inspection_boards_manifest_path"], tail["inspection_boards_manifest_sha256"])
    verify(head["manual_observation_script"]["path"], head["manual_observation_script"]["sha256"])
    verify(revision["prior_negative_review"]["path"], revision["prior_negative_review"]["sha256"])
    if revision["manifest_sha256"] != PINNED[MANIFEST] or negatives["review_aids_manifest"]["sha256"] != PINNED[MANIFEST]:
        raise ValueError("Root negative review lineage differs")
    proposals = manifest["proposals"]["positive_center_proposals"]
    observations = head["observations"] + tail["records"]
    ids = [proposal["proposal_id"] for proposal in proposals]
    if len(ids) != 96 or len(set(ids)) != 96 or ids != sampler["wave_positive_ids"]:
        raise ValueError("Require the exact ordered immutable 96-item wave")
    if len(head["observations"]) != 48 or len(tail["records"]) != 48 or len(observations) != 96:
        raise ValueError("Both original visual reviews must cover exactly48 items")
    by_id = {proposal["proposal_id"]: proposal for proposal in proposals}
    panel_coverage = set()
    head_boards = read_json(head["inspection_boards"]["path"])
    tail_boards = read_json(tail["inspection_boards_manifest_path"])
    for board in head_boards["boards"]:
        verify(board["path"], board["sha256"])
        for mapping in board["maps"]:
            proposal = by_id[mapping["id"]]
            key = mapping.get("context", "native_frame")
            expected = proposal["images"][key]
            if mapping["source"] != expected:
                raise ValueError("Head inspection board source image map differs")
            if key != "native_frame":
                if mapping["native_patch_origin_xy_px"] != proposal["native_patch_origin_xy_px"]:
                    raise ValueError("Head inspection board native origin differs")
                if mapping["source_patch_xyxy"] == [0, 0, 96, 96]:
                    panel_coverage.add((proposal["proposal_id"], key))
    for board in tail_boards["boards"]:
        verify(board["board_path"], board["sha256"])
        for mapping in board["coordinate_maps"]:
            proposal = by_id[mapping["proposal_id"]]
            key = mapping["image_key"]
            if mapping["source_patch_origin_native_xy"] != proposal["native_patch_origin_xy_px"]:
                raise ValueError("Tail inspection board native origin differs")
            if "source_path" in mapping:
                expected = proposal["images"][key]
                if (mapping["source_path"], mapping["source_sha256"]) != (expected["path"], expected["sha256"]):
                    raise ValueError("Tail inspection board source image map differs")
                panel_coverage.add((proposal["proposal_id"], key))
    if panel_coverage != {(proposal["proposal_id"], key) for proposal in proposals for key in ("context0", "context1", "context2")}:
        raise ValueError("Review boards do not map every full current and temporal native patch")
    sources = {source["source_id"]: source for source in sampler["sources"]}
    if len(sources) != 60:
        raise ValueError("Require all60 original source records")
    for source in sources.values():
        verify(source["video_path"], source["video_sha256"])
        verify(source["publisher_csv_path"], source["publisher_csv_sha256"])
        card = source["dataset_card_audit"]
        verify(card["local_card_path"], card["local_raw_card_sha256"])
        if source.get("metadata_probe"):
            probe = source["metadata_probe"]
            verify(probe["receipt_path"], probe["receipt_sha256"])
            verify(probe["raw_pts_path"], probe["raw_pts_sha256"])
    decoded = {row["clip"]: row for row in manifest["decoded_sources"]}
    normalized = []
    for index, (proposal, observation) in enumerate(zip(proposals, observations, strict=True)):
        source = sources[proposal["source_id"]]
        frame = proposal["frame_index"]
        if type(frame) is not int or not 0 <= frame < len(source["source_pts"]):
            raise ValueError("Reviewed frame outside source native PTS")
        if proposal["source_pts"] != source["source_pts"][frame] or proposal["source_time_base"] != source["source_time_base"]:
            raise ValueError("Actual source PTS/timebase changed")
        if Fraction(proposal["timestamp_fraction_seconds"]) != proposal["source_pts"] * Fraction(proposal["source_time_base"]):
            raise ValueError("Source timestamp is not exact PTS multiplied by timebase")
        for key in ("video_sha256", "timeline_sha256"):
            if decoded[proposal["clip"]][key] != source[key] or observation["source_" + key] != source[key]:
                raise ValueError("Visual review source/video/timeline lineage differs")
        for item in proposal["images"].values():
            verify(item["path"], item["sha256"])
        for item in observation["images_actually_viewed"]:
            verify(item["path"], item["sha256"])
        if "context_images_seen_via_unmarked_board" in observation:
            expected_context = [proposal["images"][key] for key in ("context0", "context1", "context2")]
            if observation["context_images_seen_via_unmarked_board"] != expected_context:
                raise ValueError("Head review temporal image bindings differ")
        if "source_images" in observation:
            supplied = {image["key"]: (image["path"], image["sha256"]) for image in observation["source_images"]}
            expected = {key: (image["path"], image["sha256"]) for key, image in proposal["images"].items()}
            if supplied != expected:
                raise ValueError("Tail review source/native/temporal image bindings differ")
        item = normalize_positive(proposal, observation, index)
        native = observation.get("native_frame")
        if native is not None and native != proposal["images"]["native_frame"]:
            raise ValueError("Head review native image differs")
        if "source_native_frame_sha256" in observation and observation["source_native_frame_sha256"] != proposal["images"]["native_frame"]["sha256"]:
            raise ValueError("Tail review native image differs")
        normalized.append(item)
    decision_counts = Counter(item["decision"] for item in normalized)
    if decision_counts != {"ACCEPT_VISIBLE_BOX": 85, "REJECT_AMBIGUOUS": 8,
                            "REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED": 3}:
        raise ValueError(f"Original review decisions unexpectedly changed: {decision_counts}")
    components = build_components(sampler["groups"], sampler["sources"], normalized)
    by_source = {source: component for component in components for source in component["source_ids"]}
    derived_sources = []
    for source in sampler["sources"]:
        component = by_source[source["source_id"]]
        derived_sources.append({"source_id": source["source_id"], "clip": source["clip"],
            "video_path": source["video_path"], "video_sha256": source["video_sha256"],
            "frozen_event_alias": source["conservative_event_group"],
            "frozen_proposed_split": source["proposed_split"], "historical_split": source["historical_split"],
            "preserved_source_exclusion_reason": source.get("exclusion_reason"),
            "derived_component_id": component["component_id"],
            "derived_component_split": component["derived_split"],
            "derived_source_split": "quarantine" if source.get("exclusion_reason") else component["derived_split"],
            "actual_match_identity_verified": False, "actual_session_identity_verified": False})
    for item in normalized:
        component = by_source[item["source_id"]]
        source_exclusion = sources[item["source_id"]].get("exclusion_reason")
        item.update({"derived_component_id": component["component_id"],
                     "derived_split": "quarantine" if source_exclusion else component["derived_split"],
                     "source_exclusion_reason": source_exclusion})
        if item["decision"] == "ACCEPT_VISIBLE_BOX":
            if item["derived_split"] == "quarantine":
                raise ValueError("An accepted original observation is excluded by conservative provenance")
            crop = jittered_crop(item["bbox_xyxy_native_px"], item["proposal_id"])
            local = translate_box(item["bbox_xyxy_native_px"], crop[:2])
            if translate_box(local, crop[:2], inverse=True) != item["bbox_xyxy_native_px"]:
                raise ValueError("Jittered native crop inverse changed reviewed box")
            item["proposed_native_crop"] = {"xyxy_native_px": crop, "size_px": [640, 640],
                "resized": False, "seed": SEED, "policy": "SHA256_PER_PROPOSAL_CONSTRAINED_INTEGER_JITTER_V1",
                "bbox_xyxy_crop_px": local, "inverse_box_verified": True,
                "all_crop_pixels_reviewed": False, "additional_balls_reviewed": False}
    negative_proposals = manifest["proposals"]["negative_tile_proposals"]
    negative_ids = [item["proposal_id"] for item in negative_proposals]
    if len(negative_ids) != 16 or [item["proposal_id"] for item in negatives["samples"]] != negative_ids:
        raise ValueError("Root negative review must match all16 actual rendered tiles")
    revised_id = revision["tile_revision"]["proposal_id"]
    negative_records = []
    for proposal, observation in zip(negative_proposals, negatives["samples"], strict=True):
        for key in ("source_id", "clip", "frame_index", "source_pts", "source_time_base", "tile_origin_xy_px", "tile_size_px"):
            if proposal[key] != observation[key]:
                raise ValueError("Root negative review coordinate/source binding differs")
        native = observation["actually_viewed_full_native_tile"]
        if native != proposal["images"]["context1"]:
            raise ValueError("Root complete-tile visual review image differs")
        verify(native["path"], native["sha256"])
        source = sources[proposal["source_id"]]
        if proposal["source_pts"] != source["source_pts"][proposal["frame_index"]]:
            raise ValueError("Negative tile source PTS changed")
        decision = observation["decision"]
        if proposal["proposal_id"] == revised_id:
            if decision != revision["tile_revision"]["previous_decision"]:
                raise ValueError("Root revision does not match the original decision")
            decision = revision["tile_revision"]["new_decision"]
            for image in revision["tile_revision"]["images"].values():
                verify(image["path"], image["sha256"])
        component = by_source[proposal["source_id"]]
        negative_records.append({"proposal_id": proposal["proposal_id"], "source_id": proposal["source_id"],
            "clip": proposal["clip"], "frame_index": proposal["frame_index"], "source_pts": proposal["source_pts"],
            "source_time_base": proposal["source_time_base"], "frozen_event_alias": proposal["conservative_event_group"],
            "frozen_split": proposal["proposed_split"], "derived_component_id": component["component_id"],
            "derived_split": component["derived_split"], "effective_decision": decision,
            "whole_current_tile_reviewed_by_root_agent": True, "preparer_claims_new_visual_review": False,
            "independent_human_review": False, "original_observation": deepcopy(observation),
            "root_revision": deepcopy(revision["tile_revision"]) if proposal["proposal_id"] == revised_id else None,
            "supervised_export_approved": False})
    if sum(item["effective_decision"] == "ACCEPT_VISIBLE_ABSENCE" for item in negative_records) != 4:
        raise ValueError("Require exactly4 effective root negative acceptances")
    derived_all600 = []
    for proposal in sampler["positive_center_proposals"]:
        component = by_source[proposal["source_id"]]
        derived_all600.append({"proposal_id": proposal["proposal_id"], "source_id": proposal["source_id"],
            "frame_index": proposal["frame_index"], "frozen_event_alias": proposal["conservative_event_group"],
            "frozen_split": proposal["proposed_split"], "derived_component_id": component["component_id"],
            "derived_split": component["derived_split"], "supervised_annotation_accepted": False})
    if len(derived_all600) != 600:
        raise ValueError("Derived source accounting must preserve all600 frozen proposals")
    accepted = [item for item in normalized if item["decision"] == "ACCEPT_VISIBLE_BOX"]
    result = {"schema_version": "tennis.reviewed-ball-crop-preparation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PREPARATION_ONLY_85_COMPLETE_POSITIVE_CROPS_REQUIRE_ROOT_VISUAL_REVIEW",
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "source_bindings": bindings,
        "protocol": {"native_crop_size_px": 640, "seed": SEED, "resizing_permitted": False,
            "box_selection": "PRESERVED_MANUAL_VARIABLE_VISIBLE_BALL_BLUR_EXTENT",
            "full_positive_crop_review_required": True, "root_supervised_export_approval_required": True,
            "graph": "TRANSITIVE_ORIGINAL_ALIAS_SOURCE_AND_ALL_OBSERVED_EVENT_FAMILIES_INCLUDING_REJECTED_AND_LOW_VIEW",
            "split_precedence": ["ORIGINAL_GROUP_QUARANTINE", "ANY_ORIGINAL_DEVELOPMENT", "TRAIN"],
            "source_exclusion": "PRESERVED_ON_INDIVIDUAL_SOURCE;GROUP_EXCLUSIONS_APPLY_TO_WHOLE_COMPONENT",
            "review_board": "FOUR_640X640_NATIVE_UNMARKED_CROPS_2X2;LABELS_OUTSIDE_SOURCE_PIXELS",
            "no_test_split": True},
        "counts": {"original_sources": 60, "original_groups": len(sampler["groups"]),
            "derived_components": len(components), "all600_original_proposals": 600,
            "all600_derived_split_counts": dict(Counter(item["derived_split"] for item in derived_all600)),
            "original_visual_reviews": 96, "accepted_visible_box_proposals": 85,
            "rejected_ambiguous": 8, "rejected_unsupported_view": 3,
            "accepted_positive_derived_splits": dict(Counter(item["derived_split"] for item in accepted)),
            "root_negative_tiles_previously_reviewed": 16, "effective_root_accepted_visible_absence_tiles": 4,
            "bound_full_native96_temporal_patch_panels": len(panel_coverage),
            "positive_complete_crops_reviewed": 0, "labels_exported": 0},
        "components": components, "derived_source_provenance": derived_sources,
        "positive_observations": normalized,
        "effective_root_negative_observations": negative_records,
        "derived_all600_proposal_provenance": derived_all600,
        "independent_human_review": False, "source_independence_verified": False,
        "actual_match_identity_verified": False, "qualification_evidence": False,
        "supervised_export_approved": False, "model_used": False, "gpu_used": False,
        "training_performed": False, "yolo_labels_written": 0,
        "limitations": ["Agent development observations are not independent test ground truth.",
            "Every positive640crop needs full-pixel review for other active/spare balls before supervised export.",
            "Whole-source visible boxes are incomplete; crops prevent unlabeled source regions entering training only after crop review.",
            "Event-family union is conservative; unknown dates, players and actual matches remain unverified.",
            "Original Torino alias is retained and unioned with observed ATP Finals2020 city-unverified context.",
            "Original frozen plans and original decisions are immutable inputs, not rewritten corrections."]}
    return result


def render(audit, output):
    admission = guarded_resources()  # Must precede PIL import and any output directory.
    from PIL import Image, ImageDraw, __version__ as pillow_version
    output.mkdir(parents=True, exist_ok=False)
    (output / "native640").mkdir()
    (output / "boards").mkdir()
    audit["launch_resources"] = admission
    write_json(output / "derived_development_manifest.json", audit)
    crops, boards, observations = [], [], []
    accepted = [row for row in audit["positive_observations"] if row["decision"] == "ACCEPT_VISIBLE_BOX"]
    for start in range(0, len(accepted), 4):
        observations.append(guarded_resources())
        canvas = Image.new("RGB", (1280, 1336), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)
        cells = []
        for offset, row in enumerate(accepted[start:start + 4]):
            observations.append(guarded_resources())
            crop_box = row["proposed_native_crop"]["xyxy_native_px"]
            with Image.open(row["native_frame"]["path"]) as source:
                if source.size != (1920, 1080) or source.mode != "RGB":
                    raise ValueError("Require unchanged RGB1920x1080 native frame")
                native = source.crop(crop_box)
                pixel_sha = hashlib.sha256(native.tobytes()).hexdigest()
                path = output / "native640" / (row["proposal_id"] + ".png")
                native.save(path)
                col, grid_row = offset % 2, offset // 2
                canvas_xy = [col * 640, grid_row * 668 + 28]
                canvas.paste(native, tuple(canvas_xy))
            label = f"{start+offset:02d} {row['proposal_id']} {row['source_id']} f{row['frame_index']} {row['derived_split']} origin{crop_box[:2]}"
            draw.text((col * 640 + 4, grid_row * 668 + 7), label, fill=(240, 240, 240))
            record = {"proposal_id": row["proposal_id"], "path": str(path), "sha256": digest(path),
                "pixel_sha256": pixel_sha, "shape": [640, 640, 3],
                "native_frame": row["native_frame"], "source_crop_xyxy_native_px": crop_box,
                "bbox_xyxy_crop_px": row["proposed_native_crop"]["bbox_xyxy_crop_px"],
                "bbox_xyxy_native_px": row["bbox_xyxy_native_px"],
                "canvas_xy": canvas_xy, "scale": 1, "resized": False,
                "all_pixels_visually_reviewed": False, "extra_visible_balls_reviewed": False,
                "supervised_export_approved": False}
            crops.append(record)
            cells.append({"proposal_id": row["proposal_id"], "crop_path": str(path),
                          "crop_sha256": record["sha256"], "canvas_xy": canvas_xy,
                          "source_crop_xyxy_native_px": crop_box, "scale": 1})
        board_path = output / "boards" / f"board_{start//4:02d}.png"
        canvas.save(board_path)
        canvas.close()
        boards.append({"path": str(board_path), "sha256": digest(board_path),
                       "size_px": [1280, 1336], "unmarked_native_pixels": True,
                       "actually_viewed": False, "cells": cells})
    changed = [path for path, expected in audit["source_bindings"].items() if digest(path) != expected]
    if changed:
        raise ValueError(f"Input changed during preparation: {changed}")
    receipt = {"schema_version": "tennis.reviewed-ball-native-crop-boards.v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "85_NATIVE_CROPS_AND22_BOARDS_READY_FOR_COMPLETE_ROOT_REVIEW_NOT_LABEL_EXPORT",
        "derived_manifest": {"path": str(output / "derived_development_manifest.json"),
                             "sha256": digest(output / "derived_development_manifest.json")},
        "implementation": audit["implementation"], "pillow_version": pillow_version,
        "launch_resources": admission, "minimum_sampled_available_ram_bytes": min(v["available_ram_bytes"] for v in observations),
        "peak_sampled_owned_rss_bytes": max(v["owned_rss_bytes"] for v in observations),
        "minimum_sampled_free_disk_bytes": min(v["free_disk_bytes"] for v in observations),
        "source_inputs_changed": changed, "crops": crops, "boards": boards,
        "counts": {"positive_native640_crops": len(crops), "unmarked_2x2_native_scale_boards": len(boards),
                   "positive_full_crops_actually_reviewed": 0, "yolo_labels_written": 0},
        "model_used": False, "gpu_used": False, "training_performed": False,
        "qualification_evidence": False, "supervised_export_approved": False}
    write_json(output / "native_crop_review_boards.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "ball_reviewed_crop_preparation_v1")
    parser.add_argument("--render", action="store_true", help="Create unmarked native640crops and review boards; never labels")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ARTIFACTS.resolve()) or output == ARTIFACTS.resolve():
        raise ValueError("Fresh preparation output must remain under the named workspace validation directory")
    if args.render and output.exists():
        raise FileExistsError("Refusing to overwrite or reuse existing preparation artifacts")
    launch = guarded_resources()
    audit = prepare_audit()
    if args.render:
        try:
            result = render(audit, output)
        except Exception as error:
            if output.exists():
                write_json(output / "preparation_failure.json", {
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "TERMINAL_PREPARATION_FAILURE_PARTIAL_ARTIFACTS_PRESERVED",
                    "exception_type": type(error).__name__, "exception_message": str(error),
                    "implementation": audit["implementation"],
                    "completed_crop_files": len(list((output / "native640").glob("*.png"))),
                    "completed_board_files": len(list((output / "boards").glob("*.png"))),
                    "supervised_export_approved": False, "yolo_labels_written": 0})
            raise
        print(json.dumps({"output": str(output), "counts": result["counts"], "status": result["status"]}))
    else:
        print(json.dumps({"status": "AUDIT_ONLY_NO_OUTPUT_WRITES", "launch_resources": launch,
                          "counts": audit["counts"], "components": audit["components"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
