"""Print existing pinned CalTennis tree candidates; does not acquire or modify data."""
from __future__ import annotations

import json
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[2]
tree_path = root / "data/external/caltennis_diagnostic/tree.json"
tree = json.loads(tree_path.read_text(encoding="utf-8"))
by_path = {record["path"]: record for record in tree}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--write-manifest", action="store_true")
parser.add_argument("--refresh-candidate-manifest", action="store_true", help="Replace only an unfrozen candidate manifest created by this helper")
parser.add_argument("--related-stem", help="Print only records containing this source stem")
args = parser.parse_args()
groups: dict[str, list[dict]] = defaultdict(list)
for record in tree:
    if record.get("type") == "file" and record["path"].endswith(".mp4"):
        groups[record["path"].split("/")[0]].append(record)
if args.related_stem:
    print(json.dumps([record for record in tree if args.related_stem in record["path"]], indent=2))
elif not args.write_manifest:
    for group, records in sorted(groups.items()):
        print(json.dumps({
            "session_path": group,
            "videos": len(records),
            "bytes": sum(record["size"] for record in records),
            "sources": [{key: record.get(key) for key in ("path", "size", "lfs")} for record in records],
        }, ensure_ascii=False))
else:
    revision = "6c1c10b2e6c16d46ab1a83614f439d2fb91e9d7d"
    publisher = "https://huggingface.co/datasets/demalenk/caltennis"
    chosen = [
        ("2026-01-23", "01_23_2026_17_00_court2/01_23_2026_17_00_17_000_2_W_02_18_10_24_610.mp4"),
        ("2026-02-25", "02_25_2026_16_30_court1/02_25_2026_16_27_25_000_1_W_01_16_32_23_111.mp4"),
        ("2026-02-26", "02_26_2026_16_30_court1/02_26_2026_17_45_02_000_1_E_04_18_27_05_613.mp4"),
        ("2026-03-04", "03_04_2026_16_30_court1/03_04_2026_17_42_50_000_1_SE_03_18_24_53_868.mp4"),
        ("2026-03-05", "03_05_2026_16_30_court1/03_05_2026_17_29_42_000_1_NE_02_18_21_44_096.mp4"),
        ("2026-03-09", "03_09_2026_17_00_court1/03_09_2026_17_16_43_000_3_SW_02_17_39_13_016.mp4"),
    ]
    acquired_manifests = list((root / "data/external").glob("*/manifest.json"))
    acquired_contents = {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in acquired_manifests}

    def source_record(path: str) -> dict:
        entry = by_path.get(path)
        return {
            "path": path,
            "present_in_pinned_tree": entry is not None,
            "expected_bytes": entry.get("size") if entry else None,
            "publisher_sha256": entry.get("lfs", {}).get("oid") if entry else None,
            "publisher_git_blob_oid": entry.get("oid") if entry else None,
            "source_url": f"{publisher}/resolve/{revision}/{path}",
        }

    candidates = []
    for number, (date, path) in enumerate(chosen, 1):
        entry = by_path[path]
        sha = entry["lfs"]["oid"]
        session = path.split("/")[0]
        stem = Path(path).stem
        timestamp = path.removesuffix(".mp4") + "_timestamps.npy"
        if timestamp not in by_path:
            matches = [key for key in by_path if key.endswith(stem + "_timestamps.npy")]
            if len(matches) == 1:
                timestamp = matches[0]
        calibration = f"camera_calibration/{session}/{stem}/calib.json"
        candidates.append({
            "candidate_group_id": f"caltennis-{date}",
            "publisher_session_path": session,
            "recording_date_from_path": date,
            "date_verified_from_timestamps": False,
            "independence_basis": "Different date and immutable source hash; same venue/players may remain correlated",
            "status": "NOT_ACQUIRED_CAMERA_SINGLES_AND_OVERLAP_REVIEW_PENDING",
            "license": "CC-BY-NC-4.0",
            "intended_use": "Private noncommercial academic qualification only",
            "public_product_demo_permission": False,
            "pretrained_exposure": "UNKNOWN",
            "video": source_record(path),
            "timestamps": source_record(timestamp),
            "calibration": source_record(calibration),
            "exact_sha256_in_existing_acquired_manifests": [name for name, content in acquired_contents.items() if sha in content],
            "clip_slots": [{"candidate_clip_id": f"q{number:02d}-{suffix}", "trim_source_pts_start": None, "trim_source_pts_end_exclusive": None, "source_frame_indices": None, "clip_sha256": None, "label_sha256": None} for suffix in ("a", "b")],
        })
    output = {
        "schema_version": 1,
        "created_date": "2026-09-30",
        "status": "CANDIDATE_ACQUISITION_PLAN_NOT_A_FROZEN_TEST_SET",
        "qualification_evidence": False,
        "certified_clips": 0,
        "certified_sessions": 0,
        "source": publisher,
        "source_revision": revision,
        "local_primary_tree": str(tree_path.relative_to(root)).replace("\\", "/"),
        "local_primary_tree_sha256": hashlib.sha256(tree_path.read_bytes()).hexdigest(),
        "planned_clip_slots": 12,
        "candidate_date_groups": 6,
        "expected_video_bytes": sum(candidate["video"]["expected_bytes"] for candidate in candidates),
        "exclusions": ["09_18_2025_16_15_court5", "10_02_2025_08_00_court5", "01_28_2026_16_30_ath", "01_28_2026_19_00_ath"],
        "merge_required": ["10_16_2025_16_00_court5 and 10_16_2025_16_51_court2 contain identical SHA256 videos"],
        "candidates": candidates,
    }
    destination = root / "docs/evaluation/TENNIS_QUALIFICATION_ACQUISITION_CANDIDATES_2026-09-30.json"
    if args.refresh_candidate_manifest and destination.exists():
        old_manifest = json.loads(destination.read_text(encoding="utf-8"))
        if old_manifest.get("qualification_evidence") is not False or old_manifest.get("status") != output["status"]:
            raise ValueError("Refusing to overwrite qualification evidence or a frozen manifest")
    with destination.open("w" if args.refresh_candidate_manifest else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"saved": str(destination), "candidate_groups": len(candidates), "expected_video_bytes": output["expected_video_bytes"], "hash_overlap": [c["exact_sha256_in_existing_acquired_manifests"] for c in candidates], "missing_related_assets": [c["candidate_group_id"] for c in candidates if not c["calibration"]["present_in_pinned_tree"] or not c["timestamps"]["present_in_pinned_tree"]]}))
