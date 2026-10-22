"""Prepare training-only ball proposals from cached raw sources; export no labels."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from scripts.data.review_racketvision_ball_training import native_timeline, digest


ROOT = Path(__file__).resolve().parents[2]
REVISION = "85157ca21faa2abca96d837dd2b963738029bcc8"
CACHES = ("racketvision_training20", "racketvision_training_additional20", "racketvision_training_third20")
GROUPS = {
    "beijing": ["match172", "match19", "match201"],
    "shanghai": ["match173", "match18", "match2"],
    "halle": ["match174", "match187", "match212"],
    "barcelona": ["match175", "match176", "match178", "match183", "match185", "match202"],
    "washington": ["match177", "match200", "match215", "match221"],
    "queens": ["match179"],
    "indian_wells": ["match180", "match194", "match216"],
    "miami": ["match181", "match186"],
    "st_petersburg": ["match182"],
    "rome": ["match184", "match217"],
    "torino": ["match188", "match189", "match224"],
    "paris": ["match190", "match207"],
    "sydney": ["match191", "match21", "match214", "match222"],
    "monte_carlo": ["match192", "match204", "match218", "match223"],
    "perth": ["match193", "match195"],
    "dubai": ["match196"],
    "melbourne": ["match198"],
    "montpellier": ["match199", "match211"],
    "rotterdam": ["match203", "match209"],
    "madrid": ["match208"],
    "montreal": ["match213", "match220", "match225"],
    "stockholm": ["match219"],
    "unresolved_event_quarantine": ["match20", "match197", "match205", "match206", "match22"],
}
GROUP_EXCLUSIONS = {
    "barcelona": "mixed_historical_train_selection_event_group",
    "paris": "mixed_historical_train_selection_event_group",
    "montreal": "mixed_historical_train_selection_event_group",
    "indian_wells": "historical_selection_event_group",
    "montpellier": "historical_selection_event_group",
    "st_petersburg": "entire_existing_native_yolo_development_group_untouched",
    "unresolved_event_quarantine": "unresolved_possible_neighboring_event_identity",
}
CLIP_EXCLUSIONS = {"match174": "doubles", "match187": "doubles", "match213": "doubles", "match209": "low_closeup_outside_primary_view"}
DEVELOPMENT_GROUPS = {"washington", "dubai", "stockholm"}
TILE_ORIGINS = [(x, y) for y in (0, 440) for x in (0, 512, 1024, 1280)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def closest_index(times, target):
    position = bisect_left(times, target)
    options = {min(len(times)-1, position), max(0, position-1)}
    return min(options, key=lambda index: (abs(times[index]-target), index))


def evenly_spaced(values, count):
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[len(values)//2]]
    return [values[round(index*(len(values)-1)/(count-1))] for index in range(count)]


def select_positive(pool, count):
    """Spatial and image-motion proxies guide review, never establish a label."""
    pool = sorted(pool, key=lambda item: (item["clip"], item["frame_index"]))
    spatial = sorted(pool, key=lambda item: (item["publisher_center_xy_px"][1], item["clip"], item["frame_index"]))
    third = max(1, math.ceil(len(pool)/3))
    motion = sorted(pool, key=lambda item: (-item["labeled_image_motion_proxy_px_per_second"], item["clip"], item["frame_index"]))
    strata = [("upper_image_far_proxy", spatial[:third]), ("lower_image_near_proxy", spatial[-third:]),
              ("fast_image_motion_blur_proxy", motion[:third])]
    chosen, seen = [], set()
    quota = count//4
    for name, values in strata:
        available = [item for item in values if (item["clip"], item["frame_index"]) not in seen]
        for item in evenly_spaced(available, quota):
            chosen.append({**item, "sampling_stratum": name})
            seen.add((item["clip"], item["frame_index"]))
    available = [item for item in pool if (item["clip"], item["frame_index"]) not in seen]
    chosen.extend({**item, "sampling_stratum": "temporally_spread_residual"} for item in evenly_spaced(available, count-len(chosen)))
    return sorted(chosen, key=lambda item: (item["clip"], item["frame_index"]))


def allocate(groups, total, caps):
    quotas = {group: 0 for group in groups}
    while sum(quotas.values()) < total:
        progressed = False
        for group in groups:
            if quotas[group] < caps[group] and sum(quotas.values()) < total:
                quotas[group] += 1
                progressed = True
        if not progressed:
            raise ValueError("Insufficient explicit publisher-visible centers for the declared sample")
    return quotas


def prepare_plan(output):
    import csv
    cache_manifests, sources = {}, {}
    legacy = json.loads((ROOT / "outputs/vision_upgrade_audit/ball_training_expansion60/report.json").read_text())
    historical_split = {c["clip"]: c["split"] for c in legacy["clips"]}
    scope_path = ROOT / "data/yolo26/broadcast_scope_review_v1/review_inventory.json"
    scope = json.loads(scope_path.read_text())
    if scope["final_test_used"] or {s["source_id"] for s in scope["sources"]} != {m for values in GROUPS.values() for m in values}:
        raise ValueError("Scope/group metadata mismatch or reserved source exposure")
    group_by_source = {match: group for group, matches in GROUPS.items() for match in matches}
    for name in CACHES:
        directory = ROOT / "data/external" / name
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["split"] != "TRAINING_ONLY" or manifest["revision"] != REVISION or manifest.get("final_test_downloaded"):
            raise ValueError("Only the existing pinned TRAINING_ONLY caches are allowed")
        if digest(directory / "DATASET_CARD.md") != manifest["dataset_card_sha256"]:
            raise ValueError("Publisher dataset card changed")
        cache_manifests[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
        files = {item["path"]: item for item in manifest["files"]}
        publisher_train = {tuple(x) for x in json.loads((directory / "tennis/info/train.json").read_text())}
        for match, rally in manifest["selected_clips"]:
            group, clip = group_by_source[match], f"{match}_{rally}"
            exclusion = GROUP_EXCLUSIONS.get(group) or CLIP_EXCLUSIONS.get(match)
            if historical_split[clip] != "train":
                exclusion = exclusion or "historical_internal_selection_clip_not_decoded"
            label_rel, video_rel = f"tennis/all/{match}/csv/{rally}_ball.csv", f"tennis/videos/{clip}.mp4"
            if (match, rally) not in publisher_train:
                raise ValueError("Selected cache clip is outside publisher training split")
            label_path, video_path = directory / label_rel, directory / video_rel
            if digest(label_path) != files[label_rel]["sha256"]:
                raise ValueError("Original publisher CSV changed")
            rows = list(csv.DictReader(label_path.open(newline="", encoding="utf-8")))
            sources[clip] = {"clip": clip, "source_id": match, "publisher_rally_id": rally,
                "conservative_visual_event_group": group, "group_identity_independence_verified": False,
                "proposed_development_split": "development" if group in DEVELOPMENT_GROUPS else "train",
                "historical_clip_split": historical_split[clip], "exclusion_reason": exclusion,
                "video_path": str(video_path), "video_url": files[video_rel]["url"], "video_sha256": files[video_rel]["sha256"],
                "publisher_csv_path": str(label_path), "publisher_csv_url": files[label_rel]["url"],
                "publisher_csv_sha256": files[label_rel]["sha256"], "raw_rows": rows,
                "publisher_visible_centers": sum(r["Visibility"] == "1" for r in rows),
                "publisher_not_visible_states": sum(r["Visibility"] == "0" for r in rows)}
            if exclusion:
                continue
            if digest(video_path) != files[video_rel]["sha256"]:
                raise ValueError("Selected cached raw video changed")
            stream, ticks = native_timeline(video_path)
            if (stream["width"], stream["height"]) != (1920, 1080):
                raise ValueError("Require native publisher coordinate resolution")
            times = [float(Fraction(tick)*Fraction(stream["time_base"])) for tick in ticks]
            seen = set()
            for row in rows:
                index = int(row["Frame"])
                if index in seen or not 0 <= index < len(ticks) or row["Visibility"] not in ("0", "1"):
                    raise ValueError("Invalid sparse source label")
                seen.add(index)
                if row["Visibility"] == "1" and not (0 <= int(row["X"]) < 1920 and 0 <= int(row["Y"]) < 1080):
                    raise ValueError("Invalid native publisher center")
            sources[clip].update(native_width=1920, native_height=1080, average_frame_rate=stream["avg_frame_rate"],
                                 source_time_base=stream["time_base"], native_frame_count=len(ticks), ticks=ticks, times=times)
    pools, not_visible = defaultdict(list), defaultdict(list)
    for clip, source in sources.items():
        if source["exclusion_reason"]:
            continue
        previous = None
        for row in sorted(source["raw_rows"], key=lambda r: int(r["Frame"])):
            index = int(row["Frame"])
            base = {"clip": clip, "source_id": source["source_id"], "frame_index": index,
                    "source_pts": source["ticks"][index], "source_time_base": source["source_time_base"],
                    "timestamp_seconds": source["times"][index], "conservative_event_group": source["conservative_visual_event_group"],
                    "proposed_split": source["proposed_development_split"]}
            if row["Visibility"] == "0":
                not_visible[base["conservative_event_group"]].append({**base, "publisher_visibility": 0, "publisher_center_xy_px": None})
                continue
            center = [int(row["X"]), int(row["Y"])]
            motion = 0.0
            if previous is not None and base["timestamp_seconds"] > previous["timestamp_seconds"]:
                motion = math.dist(center, previous["publisher_center_xy_px"])/(base["timestamp_seconds"]-previous["timestamp_seconds"])
            item = {**base, "publisher_visibility": 1, "publisher_center_xy_px": center,
                    "labeled_image_motion_proxy_px_per_second": motion}
            pools[base["conservative_event_group"]].append(item)
            previous = item
    ordered = [g for g in GROUPS if g in pools]
    train = [g for g in ordered if g not in DEVELOPMENT_GROUPS]
    development = [g for g in ordered if g in DEVELOPMENT_GROUPS]
    quotas = allocate(train, 480, {g: len(pools[g]) for g in train}) | allocate(development, 120, {g: len(pools[g]) for g in development})
    positive = [item for group in ordered for item in select_positive(pools[group], quotas[group])]
    for index, item in enumerate(positive):
        item.update(proposal_id=f"positive_{index:04d}", review_status="PENDING_MANUAL_AGENT_BOX_REVIEW", accepted_ground_truth=False)
    negative_quotas = allocate(ordered, 120, {g: 120 for g in ordered})
    negatives = []
    for group in ordered:
        choices = evenly_spaced(sorted(not_visible[group], key=lambda i: (i["clip"], i["frame_index"])), min(negative_quotas[group]//2, len(not_visible[group])))
        positives_for_group = [p for p in positive if p["conservative_event_group"] == group]
        choices += evenly_spaced(positives_for_group, negative_quotas[group]-len(choices))
        for index, choice in enumerate(choices):
            origins = [origin for origin in TILE_ORIGINS if choice["publisher_center_xy_px"] is None or not
                       (origin[0] <= choice["publisher_center_xy_px"][0] < origin[0]+640 and origin[1] <= choice["publisher_center_xy_px"][1] < origin[1]+640)]
            origin = origins[index % len(origins)]
            negatives.append({k: choice[k] for k in ("clip", "source_id", "frame_index", "source_pts", "source_time_base", "timestamp_seconds", "conservative_event_group", "proposed_split", "publisher_visibility", "publisher_center_xy_px")} |
                             {"proposal_id": f"tile_review_{len(negatives):04d}", "tile_origin_xy_px": list(origin), "tile_size_px": [640, 640],
                              "selection_reason": "publisher_not_visible_requires_full_tile_adjudication" if choice["publisher_visibility"] == 0 else "fixed_tile_outside_known_center_requires_complete_tile_review",
                              "review_status": "PENDING_COMPLETE_TILE_MANUAL_REVIEW", "accepted_negative": False})
    source = sources["match172_000"]
    special = {"proposal_id": "beijing_run4_source_frame15_triage", "clip": "match172_000", "source_id": "match172",
               "frame_index": 15, "source_pts": source["ticks"][15], "source_time_base": source["source_time_base"],
               "timestamp_seconds": source["times"][15], "conservative_event_group": "beijing", "proposed_split": "train",
               "publisher_visibility": None, "publisher_center_xy_px": None, "root_reported_approximate_ball_xy_px": [847, 275],
               "root_reported_distractor_xy_px": [473, 976], "review_status": "MISSING_PUBLISHER_ROW_MANUAL_TRIAGE_ONLY", "accepted_ground_truth": False}
    if any(int(r["Frame"]) == 15 for r in source["raw_rows"]):
        raise ValueError("Special case metadata expected an unannotated source frame")
    # Mandatory root-reported failure replaces one Beijing tile proposal; it
    # is reviewed against the unsullied source pixels, never the HUD image.
    first_beijing = next(i for i, item in enumerate(negatives) if item["conservative_event_group"] == "beijing")
    negatives[first_beijing] = {**special, "proposal_id": "tile_review_beijing_frame15_scoreboard", "tile_origin_xy_px": [0, 440],
                               "tile_size_px": [640, 640], "selection_reason": "root_reported_scoreboard_digit_error_full_source_tile_review_required",
                               "review_status": "PENDING_COMPLETE_TILE_MANUAL_REVIEW", "accepted_negative": False}
    groups = [{"group": g, "source_ids": ids, "proposed_split": "development" if g in DEVELOPMENT_GROUPS else "train",
               "exclusion_reason": GROUP_EXCLUSIONS.get(g), "visible_candidates_available": len(pools[g]),
               "visible_candidates_selected": quotas.get(g, 0), "negative_tile_proposals": negative_quotas.get(g, 0),
               "group_identity_verified": False} for g, ids in GROUPS.items()]
    serial_sources = [{k: v for k, v in source.items() if k not in ("raw_rows", "ticks", "times")} for source in sources.values()]
    plan = {"schema_version": "tennis.ball-review-proposals.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "freeze_stage": "source_group_and_sparse_label_sampling_before_new_frame_decode_or_box_proposal", "pinned_hf_revision": REVISION,
            "private_local_noncommercial_academic": True, "qualification_evidence": False, "independent_human_review": False,
            "publisher_license": "MIT dataset-card declaration; original raw broadcast public redistribution not separately verified",
            "cache_manifest_sha256": cache_manifests, "cached_scope_inventory_sha256": digest(scope_path),
            "legacy_split_audit_sha256": digest(ROOT / "outputs/vision_upgrade_audit/ball_training_expansion60/report.json"),
            "implementation_sha256": digest(Path(__file__)), "all_reserved_final_test_sources_unopened": True,
            "scope": "cached TRAINING_ONLY media; source grouping is conservative visual event branding, not verified original match/session identity",
            "sampler": {"positive_frames": 600, "proposed_train": 480, "proposed_development": 120, "native_patch_px": 96,
                        "patch_display_nearest_neighbor_scale": 6, "context_seconds": 0.10, "negative_tile_proposals": 120,
                        "strata": "one quarter upper-image third, lower-image third, high labeled image-motion third; remaining temporally spread; overlapping picks deduplicated",
                        "blur_far_near_status": "proxies only; manual visual labels required", "negative_policy": "No CSV visibility0, missing label, or detector output implies absence. Full native tile must be reviewed before an empty label is allowed."},
            "beijing_exception": "Root explicitly retained Beijing training-development group despite historic match19 selection neighbor; decode172/201 only, never19; no independence or historical-score comparison claimed",
            "hypothesized_box_policy": "PROPOSAL ONLY: bright/yellow local connected component near publisher center; fallback16px square is an explicitly hypothetical display aid. No accepted box or YOLO label export.",
            "groups": groups, "sources": serial_sources, "positive_center_proposals": positive,
            "negative_tile_proposals": negatives, "special_unannotated_review_cases": [special]}
    write_json(output / "frozen_sampler_source_groups.json", plan)
    (output / "frozen_sampler_source_groups.json.sha256").write_text(digest(output / "frozen_sampler_source_groups.json")+"  frozen_sampler_source_groups.json\n")
    return plan, sources


def hypothesized_box(frame, center):
    x, y = center
    left, top, right, bottom = max(0, x-24), max(0, y-24), min(frame.shape[1], x+25), min(frame.shape[0], y+25)
    local = frame[top:bottom, left:right]
    gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
    blue, green, red = [local[:, :, i].astype(np.float32) for i in range(3)]
    mask = ((gray.astype(np.float32) >= min(235, float(np.median(gray))+30)) |
            ((green > 110) & (red > 90) & (green > blue*1.3) & (red > blue*1.15))).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    choices = []
    for index in range(1, count):
        a, b, w, h, area = map(int, stats[index])
        distance = math.dist([centroids[index][0]+left, centroids[index][1]+top], center)
        if 3 <= area <= 400 and 2 <= w <= 36 and 2 <= h <= 36 and distance <= 10:
            choices.append((distance, area, [left+a, top+b, left+a+w, top+b+h]))
    if choices:
        return min(choices)[2], "classical_bright_yellow_component_PROPOSAL_NOT_GROUND_TRUTH"
    return [max(0,x-8),max(0,y-8),min(frame.shape[1],x+8),min(frame.shape[0],y+8)], "fixed16px_fallback_HYPOTHESIS_NOT_GROUND_TRUTH"


def image_record(path, pixels):
    if not cv2.imwrite(str(path), pixels):
        raise OSError("Failed to preserve source pixels")
    return {"path": str(path), "sha256": digest(path), "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            "shape": list(pixels.shape)}


def decode_review(plan, sources, output):
    wanted = defaultdict(lambda: defaultdict(list))
    result = {"positive_center_proposals": [], "negative_tile_proposals": [], "special_unannotated_review_cases": []}
    for kind in result:
        for original in plan[kind]:
            item = dict(original)
            source = sources[item["clip"]]
            context = [closest_index(source["times"], item["timestamp_seconds"]+offset) for offset in (-.10, 0, .10)]
            item["context"] = [{"frame_index": index, "source_pts": source["ticks"][index], "timestamp_seconds": source["times"][index],
                                "offset_seconds": source["times"][index]-item["timestamp_seconds"]} for index in context]
            item["images"] = {}
            if kind != "negative_tile_proposals":
                center = item.get("publisher_center_xy_px") or item["root_reported_approximate_ball_xy_px"]
                origin = [min(1920-96,max(0,center[0]-48)), min(1080-96,max(0,center[1]-48))]
                item["native_patch_origin_xy_px"] = origin
            result[kind].append(item)
            for position, index in enumerate(context):
                wanted[item["clip"]][index].append((kind, item, position))
    for clip, requests in wanted.items():
        source = sources[clip]
        capture = cv2.VideoCapture(source["video_path"])
        if not capture.isOpened():
            raise ValueError("Cached source cannot be opened")
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (1080, 1920):
                    raise ValueError("Native source geometry changed")
                native_record = None
                for kind, item, position in requests.get(index, []):
                    if position == 1:
                        if native_record is None:
                            native_path = output / "native" / f"{clip}_f{index:06d}.png"
                            native_record = image_record(native_path, frame)
                        item["images"]["native_frame"] = native_record
                    if kind == "negative_tile_proposals":
                        x, y = item["tile_origin_xy_px"]
                        pixels = frame[y:y+640, x:x+640]
                        path = output / "tiles" / f"{item['proposal_id']}_context{position}.png"
                    else:
                        x, y = item["native_patch_origin_xy_px"]
                        pixels = frame[y:y+96, x:x+96]
                        path = output / "patches" / f"{item['proposal_id']}_context{position}.png"
                        if position == 1:
                            center = item.get("publisher_center_xy_px") or item["root_reported_approximate_ball_xy_px"]
                            box, method = hypothesized_box(frame, center)
                            item["hypothesized_box_xyxy_native_px"] = box
                            item["hypothesized_box_provenance"] = method
                    item["images"][f"context{position}"] = image_record(path, pixels)
                index += 1
        finally:
            capture.release()
        if index != source["native_frame_count"]:
            raise ValueError("Decoded source frame count differs from the exact PTS inventory")
        print(json.dumps({"decoded_training_clip": clip, "native_frames": index, "requested_review_frames": len(requests)}), flush=True)
    return result


def caption(image, text, x=8, y=23, scale=.48, color=(235,235,235)):
    cv2.putText(image, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_boards(result, output):
    board_inventory = []
    for kind in ("positive_center_proposals", "special_unannotated_review_cases"):
        items = result[kind]
        panels = []
        for item in items:
            panel = np.full((680, 1056, 3), 24, np.uint8)
            caption(panel, f"PROPOSAL {item['proposal_id']} {item['clip']} frame{item['frame_index']} {item['conservative_event_group']} {item['proposed_split']}")
            caption(panel, f"PTS {item['source_pts']} at {item['source_time_base']} | CSV visibility {item['publisher_visibility']} | NO ACCEPTED BOX", y=43)
            patch = cv2.imread(item["images"]["context1"]["path"])
            enlarged = cv2.resize(patch, (576,576), interpolation=cv2.INTER_NEAREST)
            box = item["hypothesized_box_xyxy_native_px"]
            origin = item["native_patch_origin_xy_px"]
            corners = [(round((box[i]-origin[i%2])*6), round((box[i+1]-origin[(i+1)%2])*6)) for i in (0,2)]
            cv2.rectangle(enlarged, corners[0], corners[1], (0,75,255), 1)
            panel[62:638, 8:584] = enlarged
            for index, key in enumerate(("context0", "context2")):
                pixels = cv2.imread(item["images"][key]["path"])
                panel[62:254, 600+index*220:792+index*220] = cv2.resize(pixels,(192,192),interpolation=cv2.INTER_NEAREST)
                caption(panel, f"offset{item['context'][index*2]['offset_seconds']:+.3f}s", x=600+index*220, y=278, scale=.43)
            native = cv2.imread(item["images"]["native_frame"]["path"])
            panel[308:551,600:1032] = cv2.resize(native,(432,243))
            caption(panel, "Native96px patch x6 NEAREST | Red rectangle PROPOSAL", x=600,y=580,scale=.38)
            caption(panel, "Unmarked source/patch PNGs retained for review", x=600,y=604,scale=.38)
            caption(panel, f"{item.get('sampling_stratum','special_missing_label_triage')}",x=600,y=630,scale=.4)
            panels.append((item, panel))
        for start in range(0,len(panels),6):
            page = panels[start:start+6]
            rows = []
            for index in range(0,len(page),2):
                pair = page[index:index+2]
                if len(pair)==1:
                    pair.append((None,np.full_like(pair[0][1],24)))
                rows.append(np.hstack([p[1] for p in pair]))
            path=output/"boards"/f"{kind}_{start//6:03d}.jpg"
            canvas=np.vstack(rows)
            if not cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,94]):
                raise OSError("Review board write failed")
            ids=[p[0]["proposal_id"] for p in page]
            board_inventory.append({"path":str(path),"sha256":digest(path),"proposal_ids":ids,"purpose":"PROPOSAL_REVIEW_ONLY_NO_GT"})
    for start in range(0,len(result["negative_tile_proposals"]),4):
        items=result["negative_tile_proposals"][start:start+4]
        panels=[]
        for item in items:
            panel=np.full((820,1056,3),24,np.uint8)
            caption(panel,f"PENDING COMPLETE TILE {item['proposal_id']} {item['clip']} f{item['frame_index']} {item['conservative_event_group']}")
            caption(panel,"640px SOURCE TILE AT NATIVE SCALE | NOT VISIBLE/MISSING != ABSENT | NO EMPTY LABEL",y=46,scale=.44)
            panel[66:706,8:648]=cv2.imread(item["images"]["context1"]["path"])
            for index,key in enumerate(("context0","context2")):
                panel[66+index*328:386+index*328,672:992]=cv2.resize(cv2.imread(item["images"][key]["path"]),(320,320))
            caption(panel,f"origin {item['tile_origin_xy_px']} PTS {item['source_pts']} {item['source_time_base']} split {item['proposed_split']}",y=744)
            caption(panel,"Review every pixel for active/spare balls and uncertainty; reject ambiguity before any negative export.",y=774,scale=.43)
            panels.append(panel)
        while len(panels)%2: panels.append(np.full_like(panels[0],24))
        canvas=np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])
        path=output/"boards"/f"complete_tile_review_{start//4:03d}.jpg"
        if not cv2.imwrite(str(path),canvas,[cv2.IMWRITE_JPEG_QUALITY,94]): raise OSError("Tile board write failed")
        board_inventory.append({"path":str(path),"sha256":digest(path),"proposal_ids":[i["proposal_id"] for i in items],"purpose":"COMPLETE_TILE_REVIEW_PENDING_NOT_NEGATIVE_GT"})
    return board_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/ball_data_repair_review600_v1")
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Require a fresh workspace-contained proposal directory")
    output.mkdir(parents=True)
    for name in ("native","patches","tiles","boards"):(output/name).mkdir()
    plan,sources=prepare_plan(output)
    print(json.dumps({"frozen_sampler":str(output/"frozen_sampler_source_groups.json"),"visible_center_proposals":len(plan["positive_center_proposals"]),"complete_tile_proposals":len(plan["negative_tile_proposals"]),"accepted_labels":0}),flush=True)
    result=decode_review(plan,sources,output)
    boards=render_boards(result,output)
    report={"schema_version":"tennis.ball-review-aids.v1","completed_at":datetime.now(timezone.utc).isoformat(),
            "sampler_sha256":digest(output/"frozen_sampler_source_groups.json"),"implementation_sha256":digest(Path(__file__)),
            "private_local_only":True,"qualification_evidence":False,"independent_human_review":False,
            "training_performed":False,"model_used":False,"accepted_visible_boxes":0,"accepted_complete_tile_negatives":0,
            "yolo_label_files_written":0,"source_labels_or_prior_manifests_changed":False,
            "positive_center_proposals":len(result["positive_center_proposals"]),"negative_tile_proposals":len(result["negative_tile_proposals"]),
            "special_unannotated_cases":len(result["special_unannotated_review_cases"]),
            "candidate_event_groups":dict(Counter(i["conservative_event_group"] for i in result["positive_center_proposals"])),
            "proposed_split_counts":dict(Counter(i["proposed_split"] for i in result["positive_center_proposals"])),
            "native_full_frame_files":len(list((output/"native").glob('*.png'))),"review_boards":boards,"proposals":result,
            "limitations":"All boxes and tiles remain unaccepted proposals. Visual branding groups are not verified original sessions. CSV0 is not-visible; unannotated frames are unknown. Classical component rectangles are display hypotheses, not ball annotations."}
    write_json(output/"review_aids_manifest.json",report)
    (output/"review_aids_manifest.json.sha256").write_text(digest(output/"review_aids_manifest.json")+"  review_aids_manifest.json\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ('review_boards','proposals','limitations')}),flush=True)


if __name__=="__main__":
    cv2.setNumThreads(1)
    main()
