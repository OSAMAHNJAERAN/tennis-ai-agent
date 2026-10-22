"""Freeze 600 grouped publisher centers and render a bounded proposal review wave.

This private development workflow uses cached publisher TRAINING_ONLY sources.
It imports no model, writes no YOLO labels, and never certifies boxes/absence.
Prepare and render are separate steps so the sampler precedes image inspection.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import psutil

from scripts.data.yolo26_ball_review_metadata import (
    CACHES, CLIP_EXCLUSIONS, DEVELOPMENT_GROUPS, GROUP_EXCLUSIONS,
    GROUPS as PREVIOUS_GROUPS, REVISION, TILE_ORIGINS,
    allocate, digest, evenly_spaced, select_positive, native_timeline, metadata_resources, verify_dataset_card,
)

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {group: list(matches) for group, matches in PREVIOUS_GROUPS.items()}
# The earlier saved, unexecuted sampler omitted this source. Its event identity
# is unresolved; quarantining it restores complete accounting without guessing.
GROUPS["unresolved_event_quarantine"].append("match210")
MIN_AVAILABLE_RAM_BYTES = 1024 ** 3
MIN_FREE_DISK_BYTES = 2 * 1024 ** 3
PATCH_SIZE = 96
SCALE = 6


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def guarded_resources():
    resources = {"available_ram_bytes": psutil.virtual_memory().available,
                 "free_disk_bytes": shutil.disk_usage(ROOT).free,
                 "owned_rss_bytes": psutil.Process().memory_info().rss}
    if any(type(value) is not int or value < 0 for value in resources.values()):
        raise RuntimeError(f"Resource observations must be finite nonnegative integer byte counts: {resources}")
    if resources["available_ram_bytes"] < MIN_AVAILABLE_RAM_BYTES:
        raise RuntimeError(f"Available RAM below unchanged 1 GiB light-work guard: {resources}")
    if resources["free_disk_bytes"] < MIN_FREE_DISK_BYTES:
        raise RuntimeError(f"Available disk below 2 GiB output guard: {resources}")
    return resources


def verify_timeline(ticks, time_base):
    if not ticks or any(not isinstance(t, int) for t in ticks):
        raise ValueError("Require actual integer source PTS for every frame")
    if any(b <= a for a, b in zip(ticks, ticks[1:])) or Fraction(time_base) <= 0:
        raise ValueError("Require a positive time base and strictly increasing actual PTS")


def nearest_pts_index(ticks, target):
    """Choose a real PTS, with exact arithmetic and earlier-frame tie breaking."""
    index = bisect_left(ticks, target)
    candidates = {min(len(ticks) - 1, index), max(0, index - 1)}
    return min(candidates, key=lambda i: (abs(Fraction(ticks[i]) - target), i))


def exact_context(ticks, time_base, frame_index):
    verify_timeline(ticks, time_base)
    if not 0 <= frame_index < len(ticks):
        raise ValueError("Context source frame outside actual timeline")
    base = Fraction(time_base)
    selected = []
    for offset in (Fraction(-1, 10), Fraction(0), Fraction(1, 10)):
        index = nearest_pts_index(ticks, Fraction(ticks[frame_index]) + offset / base)
        actual_offset = (ticks[index] - ticks[frame_index]) * base
        selected.append({"frame_index": index, "source_pts": ticks[index],
                         "source_time_base": time_base,
                         "timestamp_fraction_seconds": str(ticks[index] * base),
                         "timestamp_seconds": float(ticks[index] * base),
                         "requested_offset_fraction_seconds": str(offset),
                         "offset_fraction_seconds": str(actual_offset),
                         "offset_seconds": float(actual_offset)})
    return selected


def patch_origin(center, width=1920, height=1080, size=PATCH_SIZE):
    if not (len(center) == 2 and 0 <= center[0] < width and 0 <= center[1] < height):
        raise ValueError("Patch center must lie inside native source geometry")
    return [min(width - size, max(0, int(center[0]) - size // 2)),
            min(height - size, max(0, int(center[1]) - size // 2))]


def native_to_patch_box(box, origin):
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or not (box[2] > box[0] and box[3] > box[1]):
        raise ValueError("Require finite positive native proposal edges")
    return [box[i] - origin[i % 2] for i in range(4)]


def validate_groups(groups, source_ids):
    flattened = [source for members in groups.values() for source in members]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(source_ids):
        raise ValueError("Conservative groups must cover each cached source exactly once")


def validate_group_split(proposals):
    group_splits = defaultdict(set)
    seen = set()
    for item in proposals:
        key = (item["clip"], item["frame_index"])
        if key in seen:
            raise ValueError("Duplicate source frame in proposal sampler")
        seen.add(key)
        group_splits[item["conservative_event_group"]].add(item["proposed_split"])
    if any(len(values) != 1 for values in group_splits.values()):
        raise ValueError("An event/source group spans training and development")


def verify_protected_inputs(plan):
    """Bind preserved input files without claiming an audit of every old artifact."""
    expected = dict(plan["cache_manifest_sha256"])
    expected["data/yolo26/broadcast_scope_review_v1/review_inventory.json"] = plan["scope_inventory_sha256"]
    expected["outputs/vision_upgrade_audit/ball_training_expansion60/report.json"] = plan["legacy_split_audit_sha256"]
    expected["scripts/data/prepare_ball_review_expansion.py"] = plan["helper_sha256"]
    expected["scripts/data/yolo26_ball_review_metadata.py"] = plan["metadata_helper_sha256"]
    for source in plan["sources"]:
        expected[str(Path(source["publisher_csv_path"]).relative_to(ROOT))] = source["publisher_csv_sha256"]
        expected[str(Path(source["dataset_card_audit"]["local_card_path"]).relative_to(ROOT))] = source["dataset_card_audit"]["local_raw_card_sha256"]
    for path, checksum in expected.items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Protected input changed: {path}")
    return expected


def validate_frozen_render_plan(plan):
    """Validate the existing sampler without selecting or accepting any item.

    An explicit byte hash supplied by the caller remains the authority for the
    frozen plan. These checks establish the contract understood by this newer
    renderer; they do not claim that its source is the historical sampler.
    """
    if plan.get("schema_version") != "tennis.ball-review-wave.v1" or plan.get("status") != "PLANNED_NOT_RENDERED":
        raise ValueError("Require an existing frozen, unrendered v1 sampler")
    if plan.get("pinned_hf_revision") != REVISION:
        raise ValueError("Frozen sampler publisher revision differs from the pinned cache")
    for key in ("qualification_evidence", "independent_human_review", "model_used", "training_performed"):
        if plan.get(key) is not False:
            raise ValueError(f"Frozen sampler must not claim {key}")
    for key in ("python_source_frames_decoded", "source_images_exported", "yolo_label_files_written"):
        if type(plan.get(key)) is not int or plan[key] != 0:
            raise ValueError("Frozen sampler must precede image decode and label export")
    guards = plan.get("resource_guards", {})
    if guards != {"minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES, "minimum_disk_bytes": MIN_FREE_DISK_BYTES}:
        raise ValueError("Frozen sampler resource guards must remain unchanged")
    for key in ("implementation_sha256", "helper_sha256", "metadata_helper_sha256"):
        value = plan.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Frozen sampler lacks an exact implementation/helper digest")
    group_rows = plan.get("groups", [])
    if len(group_rows) != len(GROUPS) or {item["group"]: item["source_ids"] for item in group_rows} != GROUPS:
        raise ValueError("Frozen conservative source groups differ from the retained grouping")
    sources = {item["clip"]: item for item in plan["sources"]}
    if len(plan["sources"]) != 60 or len(sources) != 60:
        raise ValueError("Frozen sampler must account for sixty distinct cached clips")
    group_by_source = {source: group for group, members in GROUPS.items() for source in members}
    validate_groups(GROUPS, {source["source_id"] for source in sources.values()})
    for source in sources.values():
        if source["clip"] != f"{source['source_id']}_{source['publisher_rally_id']}" or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in source["clip"]):
            raise ValueError("Frozen native clip identity is not a safe publisher source/rally ID")
        if source["conservative_event_group"] != group_by_source[source["source_id"]]:
            raise ValueError("Frozen clip differs from its conservative source group")
        expected_split = "development" if source["conservative_event_group"] in DEVELOPMENT_GROUPS else "train"
        if source["proposed_split"] != expected_split:
            raise ValueError("Frozen clip differs from its group split")
        if source["exclusion_reason"]:
            continue
        ticks = source["source_pts"]
        verify_timeline(ticks, source["source_time_base"])
        if any(type(tick) is not int for tick in ticks):
            raise ValueError("Actual source PTS cannot be booleans or manufactured fractions")
        if source["native_frame_count"] != len(ticks) or (source["native_width"], source["native_height"]) != (1920, 1080):
            raise ValueError("Frozen native timeline/geometry differs from the source inventory")
        if hashlib.sha256(json.dumps(ticks).encode()).hexdigest() != source["timeline_sha256"]:
            raise ValueError("Frozen actual source PTS digest mismatch")
    positive = plan["positive_center_proposals"]
    negatives = plan["negative_tile_proposals"]
    special = plan["special_unannotated_review_cases"]
    if len(positive) != 600 or len(negatives) != 120 or len(special) != 1:
        raise ValueError("Require the complete frozen600-center/120-tile/single-triage plan")
    validate_group_split(positive)
    if Counter(item["proposed_split"] for item in positive) != {"train": 480, "development": 120}:
        raise ValueError("Frozen600-center split must remain480train/120development")
    ids_by_kind = {}
    for kind, items in (("positive_center_proposals", positive), ("negative_tile_proposals", negatives), ("special_unannotated_review_cases", special)):
        ids = [item["proposal_id"] for item in items]
        if any(not isinstance(value, str) or not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in value) for value in ids):
            raise ValueError("Frozen proposal IDs must be safe unchanged filename identifiers")
        if len(ids) != len(set(ids)) or any(set(ids) & previous for previous in ids_by_kind.values()):
            raise ValueError("Frozen proposal IDs must be distinct across all proposal kinds")
        ids_by_kind[kind] = set(ids)
        for item in items:
            source = sources.get(item["clip"])
            if source is None or source["exclusion_reason"]:
                raise ValueError("Frozen review proposal references an excluded or unknown source")
            index = item["frame_index"]
            if type(index) is not int or not 0 <= index < source["native_frame_count"]:
                raise ValueError("Frozen proposal frame lies outside its native timeline")
            for field in ("source_id", "conservative_event_group", "proposed_split", "source_time_base"):
                if item[field] != source[field]:
                    raise ValueError("Frozen proposal source/group/split/time base changed")
            timestamp = source["source_pts"][index] * Fraction(source["source_time_base"])
            if type(item["source_pts"]) is not int or item["source_pts"] != source["source_pts"][index] or item["timestamp_seconds"] != float(timestamp):
                raise ValueError("Frozen proposal timing differs from actual source PTS")
            if "timestamp_fraction_seconds" in item and item["timestamp_fraction_seconds"] != str(timestamp):
                raise ValueError("Frozen proposal exact timestamp differs from actual source PTS")
            if item.get("accepted_ground_truth", False) is not False or item.get("accepted_negative", False) is not False:
                raise ValueError("Frozen proposals cannot contain accepted boxes or negatives")
            if kind == "positive_center_proposals":
                if item["publisher_visibility"] != 1 or item["review_status"] != "PENDING_PER_ITEM_VISUAL_REVIEW":
                    raise ValueError("Positive centers must remain unaccepted publisher-visible proposals")
                patch_origin(item["publisher_center_xy_px"])
            elif kind == "negative_tile_proposals":
                if item["tile_origin_xy_px"] not in [list(origin) for origin in TILE_ORIGINS] or item["tile_size_px"] != [640, 640]:
                    raise ValueError("Frozen complete-tile review geometry changed")
                if item["review_status"] != "PENDING_COMPLETE_NATIVE_TILE_REVIEW":
                    raise ValueError("Complete tiles must remain pending visual review")
    positive_ids, tile_ids = plan["wave_positive_ids"], plan["wave_tile_ids"]
    if len(positive_ids) not in (48, 96) or len(positive_ids) != len(set(positive_ids)) or not set(positive_ids) <= ids_by_kind["positive_center_proposals"]:
        raise ValueError("Frozen positive review-wave IDs are missing, duplicated or invalid")
    if len(tile_ids) != 16 or len(tile_ids) != len(set(tile_ids)) or not set(tile_ids) <= ids_by_kind["negative_tile_proposals"]:
        raise ValueError("Frozen complete-tile review-wave IDs are missing, duplicated or invalid")
    wave_counts = Counter(item["conservative_event_group"] for item in positive if item["proposal_id"] in positive_ids)
    tile_counts = Counter(item["conservative_event_group"] for item in negatives if item["proposal_id"] in tile_ids)
    if len(wave_counts) != 16 or set(wave_counts.values()) != {len(positive_ids) // 16} or tile_counts != dict.fromkeys(wave_counts, 1):
        raise ValueError("Frozen wave must preserve equal context coverage and one complete tile per group")
    return {"positive_center_proposals": len(positive), "complete_tile_proposals": len(negatives),
            "wave_positive_proposals": len(positive_ids), "wave_complete_tiles": len(tile_ids),
            "special_unannotated_cases": len(special), "wave_event_groups": dict(wave_counts)}


def read_frozen_render_plan(path, expected_sha256):
    """Read exact frozen bytes; neither this JSON nor its sidecar is rewritten."""
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Frozen sampler must be an existing private workspace file")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("Require an explicit lowercase SHA256 for the existing frozen sampler")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Frozen predecode sampler differs from the explicitly pinned hash")
    sidecar = path.with_name(path.name + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").split()
    if fields != [expected_sha256, path.name]:
        raise ValueError("Frozen sampler sidecar must bind its exact digest and filename")
    plan = json.loads(raw)
    counts = validate_frozen_render_plan(plan)
    return plan, {"path": str(path), "sha256": expected_sha256, "sidecar_path": str(sidecar),
                  "sidecar_sha256": digest(sidecar), "sampler_implementation_sha256": plan["implementation_sha256"],
                  "sampler_recomputed": False, "sampler_bytes_rewritten": False, "counts": counts}


def verify_frozen_render_inputs(plan, frozen):
    """Bind original plan, metadata receipts and every source used by the wave."""
    if digest(Path(frozen["path"])) != frozen["sha256"] or digest(Path(frozen["sidecar_path"])) != frozen["sidecar_sha256"]:
        raise ValueError("Frozen sampler or original sidecar changed during rendering")
    expected = verify_protected_inputs(plan)
    clips = {item["clip"] for key, selected in (("positive_center_proposals", set(plan["wave_positive_ids"])),
                                               ("negative_tile_proposals", set(plan["wave_tile_ids"])),
                                               ("special_unannotated_review_cases", {item["proposal_id"] for item in plan["special_unannotated_review_cases"]}))
             for item in plan[key] if item["proposal_id"] in selected}
    for source in plan["sources"]:
        if source["clip"] not in clips:
            continue
        protected = [(source["video_path"], source["video_sha256"])]
        probe = source["metadata_probe"]
        protected.extend((probe[field + "_path"], probe[field + "_sha256"]) for field in ("receipt", "raw_pts"))
        for path, checksum in protected:
            actual_path = Path(path).resolve()
            if not actual_path.is_relative_to(ROOT) or digest(actual_path) != checksum:
                raise ValueError("Frozen native video or original PTS metadata receipt changed")
            expected[str(actual_path.relative_to(ROOT))] = checksum
    return expected


def prepare(output, wave_size):
    (output / "metadata_probes").mkdir()
    scope_path = ROOT / "data/yolo26/broadcast_scope_review_v1/review_inventory.json"
    scope = json.loads(scope_path.read_text())
    legacy_path = ROOT / "outputs/vision_upgrade_audit/ball_training_expansion60/report.json"
    legacy = json.loads(legacy_path.read_text())
    history = {item["clip"]: item["split"] for item in legacy["clips"]}
    scope_ids = {item["source_id"] for item in scope["sources"]}
    if scope.get("final_test_used"):
        raise ValueError("Cached scope must exclude final test")
    validate_groups(GROUPS, scope_ids)
    group_by_source = {source: group for group, members in GROUPS.items() for source in members}
    sources, manifests, pools, not_visible = {}, {}, defaultdict(list), defaultdict(list)
    for cache in CACHES:
        directory = ROOT / "data/external" / cache
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["split"] != "TRAINING_ONLY" or manifest["revision"] != REVISION or manifest.get("final_test_downloaded"):
            raise ValueError("Only pinned cached publisher training sources may be opened")
        card_audit = verify_dataset_card(directory / "DATASET_CARD.md", manifest["dataset_card_sha256"])
        manifests[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
        files = {item["path"]: item for item in manifest["files"]}
        train_ids = {tuple(item) for item in json.loads((directory / "tennis/info/train.json").read_text())}
        for source_id, rally in manifest["selected_clips"]:
            if (source_id, rally) not in train_ids:
                raise ValueError("A selected clip is outside publisher training")
            clip, group = f"{source_id}_{rally}", group_by_source[source_id]
            if clip in sources:
                raise ValueError("Cached training sources must be distinct")
            exclusion = GROUP_EXCLUSIONS.get(group) or CLIP_EXCLUSIONS.get(source_id)
            if history[clip] != "train":
                exclusion = exclusion or "historical_internal_selection_clip_excluded"
            video_rel = f"tennis/videos/{clip}.mp4"
            csv_rel = f"tennis/all/{source_id}/csv/{rally}_ball.csv"
            video, labels = directory / video_rel, directory / csv_rel
            if digest(labels) != files[csv_rel]["sha256"]:
                raise ValueError("Original publisher CSV checksum mismatch")
            with labels.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            source = {"clip": clip, "source_id": source_id, "publisher_rally_id": rally,
                      "conservative_event_group": group, "original_match_identity_verified": False,
                      "historical_split": history[clip], "exclusion_reason": exclusion,
                      "proposed_split": "development" if group in DEVELOPMENT_GROUPS else "train",
                      "video_path": str(video), "video_sha256": files[video_rel]["sha256"], "video_url": files[video_rel]["url"],
                      "publisher_csv_path": str(labels), "publisher_csv_sha256": files[csv_rel]["sha256"],
                      "publisher_csv_url": files[csv_rel]["url"], "dataset_card_sha256": manifest["dataset_card_sha256"], "dataset_card_audit": card_audit,
                      "publisher_visible_rows": sum(row["Visibility"] == "1" for row in rows),
                      "publisher_not_visible_rows": sum(row["Visibility"] == "0" for row in rows),
                      "frame_count_historical_audit": next(item["frames"] for item in legacy["clips"] if item["clip"] == clip)}
            sources[clip] = source
            if exclusion:
                continue
            metadata_resources()
            if digest(video) != source["video_sha256"]:
                raise ValueError("Cached native source video checksum mismatch")
            stream, ticks, probe_receipt = native_timeline(video, output / "metadata_probes")
            verify_timeline(ticks, stream["time_base"])
            if (stream["width"], stream["height"]) != (1920, 1080):
                raise ValueError("Publisher centers require native 1920x1080 geometry")
            if len(ticks) != source["frame_count_historical_audit"]:
                raise ValueError("Timeline frame count differs from historic source audit")
            source.update(native_width=1920, native_height=1080, source_time_base=stream["time_base"],
                          average_frame_rate=stream["avg_frame_rate"], native_frame_count=len(ticks),
                          source_pts=ticks, timeline_sha256=hashlib.sha256(json.dumps(ticks).encode()).hexdigest(), metadata_probe=probe_receipt)
            seen, previous = set(), None
            for row in sorted(rows, key=lambda value: int(value["Frame"])):
                index, visibility = int(row["Frame"]), int(row["Visibility"])
                if index in seen or not 0 <= index < len(ticks) or visibility not in (0, 1):
                    raise ValueError("Duplicate/out-of-range/invalid sparse publisher row")
                seen.add(index)
                timestamp = ticks[index] * Fraction(source["source_time_base"])
                item = {"clip": clip, "source_id": source_id, "frame_index": index,
                        "source_pts": ticks[index], "source_time_base": source["source_time_base"],
                        "timestamp_fraction_seconds": str(timestamp), "timestamp_seconds": float(timestamp),
                        "conservative_event_group": group, "proposed_split": source["proposed_split"],
                        "publisher_visibility": visibility,
                        "publisher_center_xy_px": [int(row["X"]), int(row["Y"])] if visibility else None}
                if not visibility:
                    not_visible[group].append(item)
                    continue
                patch_origin(item["publisher_center_xy_px"])
                motion = math.dist(item["publisher_center_xy_px"], previous["publisher_center_xy_px"]) / (item["timestamp_seconds"] - previous["timestamp_seconds"]) if previous else 0.0
                item["labeled_image_motion_proxy_px_per_second"] = motion
                pools[group].append(item)
                previous = item
    groups = [group for group in GROUPS if pools[group]]
    if len(sources) != 60 or len(groups) != 16:
        raise ValueError("Expected sixty accounted cached clips and sixteen eligible conservative groups")
    train = [group for group in groups if group not in DEVELOPMENT_GROUPS]
    development = [group for group in groups if group in DEVELOPMENT_GROUPS]
    quotas = allocate(train, 480, {group: len(pools[group]) for group in train}) | allocate(development, 120, {group: len(pools[group]) for group in development})
    positive = [item for group in groups for item in select_positive(pools[group], quotas[group])]
    for index, item in enumerate(positive):
        item.update(proposal_id=f"positive_{index:04d}", review_status="PENDING_PER_ITEM_VISUAL_REVIEW", accepted_ground_truth=False)
    validate_group_split(positive)
    wave = []
    for group in groups:
        wave.extend(evenly_spaced([item for item in positive if item["conservative_event_group"] == group], wave_size // len(groups)))
    negative_quotas = allocate(groups, 120, {group: 120 for group in groups})
    negatives = []
    for group in groups:
        choices = evenly_spaced(not_visible[group], min(negative_quotas[group] // 2, len(not_visible[group])))
        choices += evenly_spaced([item for item in positive if item["conservative_event_group"] == group], negative_quotas[group] - len(choices))
        for position, choice in enumerate(choices):
            origins = [origin for origin in TILE_ORIGINS if choice["publisher_center_xy_px"] is None or not (origin[0] <= choice["publisher_center_xy_px"][0] < origin[0] + 640 and origin[1] <= choice["publisher_center_xy_px"][1] < origin[1] + 640)]
            negatives.append({key: choice[key] for key in choice if key not in ("proposal_id", "review_status", "accepted_ground_truth", "sampling_stratum", "labeled_image_motion_proxy_px_per_second")} |
                             {"proposal_id": f"tile_review_{len(negatives):04d}", "tile_origin_xy_px": list(origins[position % len(origins)]),
                              "tile_size_px": [640, 640], "selection_reason": "publisher_state_or_outside_known_center_is_not_absence",
                              "review_status": "PENDING_COMPLETE_NATIVE_TILE_REVIEW", "accepted_negative": False})
    beijing = sources["match172_000"]
    with Path(beijing["publisher_csv_path"]).open(newline="") as handle:
        if any(int(row["Frame"]) == 15 for row in csv.DictReader(handle)):
            raise ValueError("Expected missing publisher row in the special source-frame15 triage")
    special = {"proposal_id": "beijing_source_frame15_triage", "clip": "match172_000", "source_id": "match172",
               "frame_index": 15, "source_pts": beijing["source_pts"][15], "source_time_base": beijing["source_time_base"],
               "timestamp_seconds": float(beijing["source_pts"][15] * Fraction(beijing["source_time_base"])),
               "conservative_event_group": "beijing", "proposed_split": "train", "publisher_visibility": None,
               "publisher_center_xy_px": None, "root_reported_approximate_ball_xy_px": [847, 275],
               "root_reported_distractor_xy_px": [472.99, 976.44], "review_status": "UNANNOTATED_MANUAL_TRIAGE_ONLY",
               "accepted_ground_truth": False, "hypothesized_box_xyxy_native_px": None}
    first = next(i for i, item in enumerate(negatives) if item["conservative_event_group"] == "beijing")
    negatives[first] = {**special, "proposal_id": "tile_review_beijing_frame15_scoreboard", "tile_origin_xy_px": [0, 440],
                       "tile_size_px": [640, 640], "review_status": "PENDING_COMPLETE_NATIVE_TILE_REVIEW", "accepted_negative": False,
                       "selection_reason": "reported_scoreboard_false_proposal_requires_whole_native_tile_review"}
    tile_wave = [next(item for item in negatives if item["conservative_event_group"] == group) for group in groups]
    counts = {"cached_clips": len(sources), "cached_frames_historical_audit": sum(item["frame_count_historical_audit"] for item in sources.values()),
              "publisher_visible_rows": sum(item["publisher_visible_rows"] for item in sources.values()),
              "publisher_not_visible_rows": sum(item["publisher_not_visible_rows"] for item in sources.values())}
    counts["unannotated_source_frames"] = counts["cached_frames_historical_audit"] - counts["publisher_visible_rows"] - counts["publisher_not_visible_rows"]
    plan = {"schema_version": "tennis.ball-review-wave.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "PLANNED_NOT_RENDERED", "freeze_stage": "source_group_sampler_before_Python_image_decode_or_review", "pinned_hf_revision": REVISION,
            "implementation_sha256": digest(Path(__file__)), "helper_sha256": digest(ROOT / "scripts/data/prepare_ball_review_expansion.py"),
            "metadata_helper_sha256": digest(ROOT / "scripts/data/yolo26_ball_review_metadata.py"),
            "cache_manifest_sha256": manifests, "scope_inventory_sha256": digest(scope_path), "legacy_split_audit_sha256": digest(legacy_path),
            "publisher_license": "MIT publisher dataset card; private local academic development only; raw broadcast redistribution uncleared",
            "qualification_evidence": False, "independent_human_review": False, "model_used": False, "training_performed": False,
            "python_source_frames_decoded": 0, "source_images_exported": 0, "yolo_label_files_written": 0,
            "metadata_probe_policy": "Actual source PTS from bounded single-thread FFprobe; codec frame metadata may be inspected internally. Python imports no OpenCV/NumPy/Torch until the unchanged1GiB render guard passes.",
            "reserved_sources_not_accessed_by_this_workflow": ["NKUST outdoor v4", "CalTennis January28-29", "CalTennis October16"],
            "beijing_exception": "Explicitly authorized172/201 despite historic19 selection neighbor;19 excluded; no independence claim",
            "omitted_prior_source_repair": "match210 restored to unresolved_event_quarantine; original sampler retained unchanged",
            "resource_guards": {"minimum_available_ram_bytes": MIN_AVAILABLE_RAM_BYTES, "minimum_disk_bytes": MIN_FREE_DISK_BYTES},
            "counts": counts, "groups": [{"group": group, "source_ids": members, "exclusion_reason": GROUP_EXCLUSIONS.get(group),
                                              "proposed_split": "development" if group in DEVELOPMENT_GROUPS else "train",
                                              "selected_positive_centers": quotas.get(group, 0), "identity_verified": False} for group, members in GROUPS.items()],
            "sources": list(sources.values()), "positive_center_proposals": positive, "negative_tile_proposals": negatives,
            "special_unannotated_review_cases": [special], "wave_positive_ids": [item["proposal_id"] for item in wave],
            "wave_tile_ids": [item["proposal_id"] for item in tile_wave],
            "sampling_policy": "600 visible centers:480train/120DEV; whole conservative event groups; spatial/motion proxies and time-spread residual; first wave equally spaced six/group at96 or three/group at48",
            "review_policy": "Every rectangle remains a proposal. Review visible smear/native edges and active/spare-ball status per item. Visibility0 or missing CSV never implies absence. Every640px tile needs complete visual review. No labels exported."}
    write_json(output / "frozen_sampler_source_groups.json", plan)
    (output / "frozen_sampler_source_groups.json.sha256").write_text(digest(output / "frozen_sampler_source_groups.json") + "  frozen_sampler_source_groups.json\n")
    return plan


def image_record(path, pixels):
    import cv2
    if not cv2.imwrite(str(path), pixels):
        raise OSError("Failed to preserve source pixels")
    return {"path": str(path), "sha256": digest(path), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "shape": list(pixels.shape)}


def decode(plan, output):
    from scripts.data.prepare_ball_review_expansion import hypothesized_box
    sources = {source["clip"]: source for source in plan["sources"]}
    wanted, result = defaultdict(lambda: defaultdict(list)), {}
    for kind, ids in (("positive_center_proposals", plan["wave_positive_ids"]), ("negative_tile_proposals", plan["wave_tile_ids"]), ("special_unannotated_review_cases", [item["proposal_id"] for item in plan["special_unannotated_review_cases"]])):
        result[kind] = []
        for original in plan[kind]:
            if original["proposal_id"] not in ids:
                continue
            item = dict(original)
            source = sources[item["clip"]]
            item["context"] = exact_context(source["source_pts"], source["source_time_base"], item["frame_index"])
            item["images"] = {}
            if kind != "negative_tile_proposals":
                item["native_patch_origin_xy_px"] = patch_origin(item.get("publisher_center_xy_px") or item["root_reported_approximate_ball_xy_px"])
            result[kind].append(item)
            for position, context in enumerate(item["context"]):
                wanted[item["clip"]][context["frame_index"]].append((kind, item, position))
    decoded_sources, resources = [], []
    for clip, requests in wanted.items():
        resources.append(guarded_resources())
        source = sources[clip]
        if source["exclusion_reason"] or digest(Path(source["video_path"])) != source["video_sha256"] or digest(Path(source["publisher_csv_path"])) != source["publisher_csv_sha256"]:
            raise ValueError("Excluded or changed source cannot be decoded")
        capture = cv2.VideoCapture(source["video_path"])
        if not capture.isOpened():
            raise ValueError("Cached source cannot be opened")
        count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape != (1080, 1920, 3):
                    raise ValueError("Native source image geometry changed")
                native = None
                for kind, item, position in requests.get(count, []):
                    if position == 1:
                        if native is None:
                            native = image_record(output / "native" / f"{clip}_f{count:06d}.png", frame)
                        item["images"]["native_frame"] = native
                    if kind == "negative_tile_proposals":
                        x, y = item["tile_origin_xy_px"]
                        pixels, directory = frame[y:y + 640, x:x + 640], "tiles"
                    else:
                        x, y = item["native_patch_origin_xy_px"]
                        pixels, directory = frame[y:y + PATCH_SIZE, x:x + PATCH_SIZE], "patches"
                        if position == 1 and kind == "positive_center_proposals":
                            box, method = hypothesized_box(frame, item["publisher_center_xy_px"])
                            item["hypothesized_box_xyxy_native_px"] = box
                            item["hypothesized_box_xyxy_patch_px"] = native_to_patch_box(box, [x, y])
                            item["hypothesized_box_provenance"] = method
                    expected = (640, 640, 3) if directory == "tiles" else (96, 96, 3)
                    if pixels.shape != expected:
                        raise ValueError("Crop silently clipped native image pixels")
                    item["images"][f"context{position}"] = image_record(output / directory / f"{item['proposal_id']}_context{position}.png", pixels)
                count += 1
        finally:
            capture.release()
        if count != source["native_frame_count"]:
            raise ValueError("Sequential decoded frame count differs from the exact source PTS inventory")
        decoded_sources.append({"clip": clip, "native_frames": count, "requested_unique_frames": len(requests), "video_sha256": source["video_sha256"], "timeline_sha256": source["timeline_sha256"]})
        print(json.dumps(decoded_sources[-1]), flush=True)
    for items in result.values():
        for item in items:
            if set(item["images"]) != {"native_frame", "context0", "context1", "context2"}:
                raise ValueError("Review item lacks a source image or exact temporal context")
    return result, decoded_sources, resources


def caption(image, text, x=8, y=23, scale=.46):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)


def positive_panel(item):
    panel = np.full((680, 1056, 3), 24, np.uint8)
    caption(panel, f"PROPOSAL {item['proposal_id']} {item['clip']} f{item['frame_index']} {item['conservative_event_group']} {item['proposed_split']}")
    caption(panel, f"PTS {item['source_pts']} @ {item['source_time_base']} CSV visibility {item['publisher_visibility']} | NO ACCEPTED BOX", y=43)
    enlarged = cv2.resize(cv2.imread(item["images"]["context1"]["path"]), (576, 576), interpolation=cv2.INTER_NEAREST)
    box = item.get("hypothesized_box_xyxy_patch_px")
    if box:
        cv2.rectangle(enlarged, (round(box[0] * SCALE), round(box[1] * SCALE)), (round(box[2] * SCALE), round(box[3] * SCALE)), (0, 75, 255), 1)
    panel[62:638, 8:584] = enlarged
    for index, key in enumerate(("context0", "context2")):
        panel[62:254, 600 + index * 220:792 + index * 220] = cv2.resize(cv2.imread(item["images"][key]["path"]), (192, 192), interpolation=cv2.INTER_NEAREST)
        caption(panel, f"offset{item['context'][index * 2]['offset_seconds']:+.3f}s", x=600 + index * 220, y=278, scale=.43)
    panel[308:551, 600:1032] = cv2.resize(cv2.imread(item["images"]["native_frame"]["path"]), (432, 243))
    caption(panel, "Native96px x6 NEAREST; red edges are PROPOSAL", x=600, y=580, scale=.38)
    caption(panel, "Unmarked source/patch PNGs retained", x=600, y=604, scale=.4)
    caption(panel, item.get("sampling_stratum", "Missing CSV: approximate ROI triage; NO BOX"), x=600, y=630, scale=.38)
    return panel


def tile_panel(item):
    panel = np.full((820, 1056, 3), 24, np.uint8)
    caption(panel, f"COMPLETE TILE PENDING {item['proposal_id']} {item['clip']} f{item['frame_index']}")
    caption(panel, "640px NATIVE SOURCE TILE | NOT-VISIBLE/MISSING != ABSENT | NO EMPTY LABEL", y=46, scale=.43)
    panel[66:706, 8:648] = cv2.imread(item["images"]["context1"]["path"])
    for index, key in enumerate(("context0", "context2")):
        panel[66 + index * 328:386 + index * 328, 672:992] = cv2.resize(cv2.imread(item["images"][key]["path"]), (320, 320))
    caption(panel, f"origin {item['tile_origin_xy_px']} PTS {item['source_pts']} {item['source_time_base']} split {item['proposed_split']}", y=744)
    caption(panel, "Review active/spare balls across EVERY pixel; reject ambiguity before negative export.", y=774, scale=.44)
    return panel


def boards(result, output):
    inventory = []
    for kind, size, factory in (("positive_center_proposals", 6, positive_panel), ("special_unannotated_review_cases", 6, positive_panel), ("negative_tile_proposals", 4, tile_panel)):
        for start in range(0, len(result[kind]), size):
            guarded_resources()
            items = result[kind][start:start + size]
            panels = [factory(item) for item in items]
            if len(panels) % 2:
                panels.append(np.full_like(panels[0], 24))
            canvas = np.vstack([np.hstack(panels[index:index + 2]) for index in range(0, len(panels), 2)])
            path = output / "boards" / f"{kind}_{start // size:03d}.jpg"
            if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError("Review board write failed")
            inventory.append({"path": str(path), "sha256": digest(path), "proposal_ids": [item["proposal_id"] for item in items], "purpose": "PENDING_VISUAL_REVIEW_ONLY_NO_GROUND_TRUTH"})
    return inventory


def import_image_stack():
    """Called only after the unchanged media guard admits the render."""
    global cv2, np
    import cv2
    import numpy as np
    cv2.setNumThreads(1)


def _publish_render_report(plan, frozen, output, renderer_sha256, protected,
                           launch_resources, result, decoded_sources, resources, inventory):
    report = {"schema_version": "tennis.ball-review-aids-wave.v1", "completed_at": datetime.now(timezone.utc).isoformat(),
              "sampler_sha256": frozen["sha256"], "frozen_sampler": frozen,
              "sampler_implementation_sha256": frozen["sampler_implementation_sha256"],
              "implementation_sha256": renderer_sha256, "python_executable": sys.executable,
              "private_local_only": True, "qualification_evidence": False, "independent_human_review": False,
              "model_used": False, "training_performed": False, "accepted_visible_boxes": 0, "accepted_complete_tile_negatives": 0,
              "yolo_label_files_written": 0, "positive_center_proposals_rendered": len(result["positive_center_proposals"]),
              "negative_tile_proposals_rendered": len(result["negative_tile_proposals"]), "special_unannotated_cases": len(result["special_unannotated_review_cases"]),
              "first_wave_event_groups": dict(Counter(item["conservative_event_group"] for item in result["positive_center_proposals"])),
              "first_wave_split_counts": dict(Counter(item["proposed_split"] for item in result["positive_center_proposals"])),
              "protected_input_hashes_unchanged": protected, "prior_artifact_mutations_performed": False,
              "reserved_sources_not_accessed": plan["reserved_sources_not_accessed_by_this_workflow"],
              "launch_resources": launch_resources, "predecode_resources": resources,
              "decoded_sources": decoded_sources, "review_boards": inventory, "proposals": result,
              "limitations": "All600 selected centers and120 tile hypotheses remain unaccepted. Only the bounded first wave is rendered. Event branding is conservative, original-match identity unresolved. Previous inspected data and first-wave sources are DEV; no independently frozen TEST."}
    write_json(output / "review_aids_manifest.json", report)
    (output / "review_aids_manifest.json.sha256").write_text(digest(output / "review_aids_manifest.json") + "  review_aids_manifest.json\n")
    write_json(output / "render_execution_receipt.json", {
        "schema_version": "tennis.ball-frozen-plan-render-execution.v1", "status": "COMPLETED_PENDING_VISUAL_REVIEW_ONLY",
        "completed_at": datetime.now(timezone.utc).isoformat(), "sampler_sha256": frozen["sha256"],
        "renderer_implementation_sha256": renderer_sha256,
        "review_aids_manifest_sha256": digest(output / "review_aids_manifest.json"),
        "prior_artifact_mutations_performed": False, "accepted_visible_boxes": 0,
        "accepted_complete_tile_negatives": 0, "yolo_label_files_written": 0,
        "model_used": False, "training_performed": False, "qualification_evidence": False})
    return report


def render_frozen_plan(frozen_path, expected_sha256, output):
    """Render one exact old sampler into a new directory, never resume in place.

    Old partial artifacts are deliberately neither consumed nor repaired. The
    current renderer is identified separately from the historical sampler; no
    frozen proposal, group, source PTS, tile selection or split is regenerated.
    """
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain in the private workspace")
    if output.exists():
        raise FileExistsError("Preserve prior artifacts; rendering requires a fresh output directory")
    if output.is_relative_to(frozen_path.resolve().parent):
        raise ValueError("Fresh rendering must remain outside the original sampler artifact directory")
    launch_resources = guarded_resources()
    plan, frozen = read_frozen_render_plan(frozen_path, expected_sha256)
    protected = verify_frozen_render_inputs(plan, frozen)
    renderer_sha256 = digest(Path(__file__))
    # No image imports, source decode or output mutation precedes this guard.
    preimport_resources = guarded_resources()
    import_image_stack()
    # Importing the media stack also consumes memory; check before mkdir/write.
    postimport_resources = guarded_resources()
    if output.exists():
        raise FileExistsError("Fresh output appeared during render preflight; preserve it")
    output.mkdir(parents=True)
    launch = {"schema_version": "tennis.ball-frozen-plan-render.v1", "status": "ADMITTED_NOT_COMPLETED",
              "created_at": datetime.now(timezone.utc).isoformat(), "frozen_sampler": frozen,
              "renderer_implementation_sha256": renderer_sha256,
              "sampler_and_renderer_implementation_match": frozen["sampler_implementation_sha256"] == renderer_sha256,
              "renderer_evolution_policy": "Explicit exact frozen-plan input; no sampling code rerun, no proposal acceptance, fresh output only",
              "output_directory": str(output), "python_executable": sys.executable,
              "launch_resources": launch_resources, "preimport_resources": preimport_resources,
              "postimport_resources": postimport_resources, "resource_guards": plan["resource_guards"],
              "protected_input_hashes": protected, "prior_artifact_mutations_performed": False,
              "qualification_evidence": False, "model_used": False, "training_performed": False,
              "accepted_visible_boxes": 0, "accepted_complete_tile_negatives": 0, "yolo_label_files_written": 0}
    phase = "render_launch_manifest_publication"
    try:
        write_json(output / "render_launch_manifest.json", launch)
        phase = "creating_fresh_render_directories"
        for directory in ("native", "patches", "tiles", "boards"):
            (output / directory).mkdir()
        phase = "native_decode_and_exact_context_crops"
        result, decoded_sources, resources = decode(plan, output)
        phase = "pending_review_board_render"
        inventory = boards(result, output)
        phase = "postrender_frozen_input_verification"
        if verify_frozen_render_inputs(plan, frozen) != protected or digest(Path(__file__)) != renderer_sha256:
            raise ValueError("Protected inputs or current renderer changed during review preparation")
        phase = "review_manifest_publication"
        report = _publish_render_report(plan, frozen, output, renderer_sha256, protected,
                                        launch_resources, result, decoded_sources, resources, inventory)
    except BaseException as error:
        try:
            write_json(output / "render_execution_receipt.json", {
                "schema_version": "tennis.ball-frozen-plan-render-execution.v1", "status": "TERMINAL_RENDER_FAILURE_PARTIAL_ARTIFACTS_RETAINED",
                "completed_at": datetime.now(timezone.utc).isoformat(), "failed_phase": phase,
                "exception_type": type(error).__name__, "exception_message": str(error),
                "sampler_sha256": expected_sha256, "renderer_implementation_sha256": renderer_sha256,
                "completed_review_manifest_written": False, "prior_artifact_mutations_performed": False,
                "partial_review_manifest_file_exists": (output / "review_aids_manifest.json").exists(),
                "accepted_visible_boxes": 0, "accepted_complete_tile_negatives": 0,
                "yolo_label_files_written": 0, "model_used": False, "training_performed": False,
                "qualification_evidence": False})
        except Exception as receipt_error:
            error.add_note(f"Could not save the new attempt's failure receipt: {type(receipt_error).__name__}: {receipt_error}")
        raise
    print(json.dumps({key: value for key, value in report.items() if key not in ("review_boards", "proposals", "decoded_sources", "predecode_resources", "limitations")}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", "--plan-only", dest="prepare", action="store_true")
    action.add_argument("--render", action="store_true")
    parser.add_argument("--wave-size", type=int, choices=(48, 96), default=None,
                        help="Prepare only: choose the first wave before sampling is frozen (default96)")
    parser.add_argument("--frozen-plan", type=Path, help="Render only: existing immutable frozen sampler JSON")
    parser.add_argument("--expected-plan-sha256", help="Render only: explicitly pinned SHA256 of the existing sampler")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain in the private workspace")
    if args.prepare:
        if args.frozen_plan is not None or args.expected_plan_sha256 is not None:
            parser.error("Prepare cannot consume an existing frozen plan; use --render with a fresh --output")
        metadata_resources()
        if output.exists():
            raise FileExistsError("Preserve prior artifacts; choose a fresh output directory")
        output.mkdir(parents=True)
        plan = prepare(output, args.wave_size or 96)
        print(json.dumps({"status": "PLANNED_NOT_RENDERED", "sampler": str(output / "frozen_sampler_source_groups.json"), "counts": plan["counts"], "positive_centers": len(plan["positive_center_proposals"]), "wave_centers": len(plan["wave_positive_ids"]), "complete_tile_proposals": len(plan["negative_tile_proposals"]), "wave_tiles": len(plan["wave_tile_ids"]), "accepted_boxes": 0}), flush=True)
        return
    if args.frozen_plan is None or args.expected_plan_sha256 is None:
        parser.error("Render requires --frozen-plan and --expected-plan-sha256; --output must be a fresh directory")
    if args.wave_size is not None:
        parser.error("Rendering cannot change --wave-size; the existing frozen IDs determine the wave")
    render_frozen_plan(args.frozen_plan, args.expected_plan_sha256, output)


if __name__ == "__main__":
    main()
