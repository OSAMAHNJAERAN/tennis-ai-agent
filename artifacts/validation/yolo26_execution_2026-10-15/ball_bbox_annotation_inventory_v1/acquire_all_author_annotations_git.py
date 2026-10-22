"""Fetch only immutable tennis annotation bodies through public sparse partial Git.

No complete repository archive, media checkout, authentication, or Dryad bypass.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import zipfile

from acquire_annotations import admission

BASE = Path(__file__).resolve().parent
CHECKOUT = BASE / "author_sparse_checkout"
REMOTE = "https://github.com/Zoutianjian/Ball-Yolo-A-Sport-Ball-Dataset.git"
EXPECTED_LABEL_TREE = "8103a5b69699fc32d7230fafb214052b14dc292d"
MAX_GIT_LOGICAL_BYTES = 15 * 1024**2
MAX_COMMAND_SECONDS = 180
GIT = ["git", "-c", "credential.helper=", "-c", "core.askpass=", "-c", "protocol.version=2",
       "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-c", "http.maxRequests=1"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes() -> int:
    directory = CHECKOUT / ".git"
    if not directory.exists():
        return 0
    size = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                size += (Path(root) / name).stat().st_size
            except FileNotFoundError:
                pass
    return size


def execute(arguments: list[str], name: str, receipt: dict, cwd: Path = BASE, trace: bool = False) -> str:
    environment = dict(os.environ)
    environment.update(GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    if trace:
        environment["GIT_TRACE_PACKET"] = str(BASE / "public_server_capabilities.log")
    admission()
    log = BASE / f"git_{name}.log"
    started = time.monotonic()
    peak_bytes = git_bytes()
    with log.open("xb") as output:
        process = subprocess.Popen(GIT + arguments, cwd=cwd, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        record = {"command": GIT + arguments, "owned_pid": process.pid, "cwd": str(cwd), "log": str(log)}
        receipt["commands"].append(record)
        reason = None
        while process.poll() is None:
            peak_bytes = max(peak_bytes, git_bytes())
            if peak_bytes > MAX_GIT_LOGICAL_BYTES:
                reason = "owned Git directory exceeds fixed logical-byte ceiling"
            if time.monotonic() - started > MAX_COMMAND_SECONDS:
                reason = "owned Git command exceeds fixed timeout"
            if reason:
                # This handle belongs to the process created immediately above.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=output, stderr=subprocess.STDOUT, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW, check=False)
                break
            time.sleep(0.1)
        process.wait(timeout=15)
        record.update(return_code=process.returncode, elapsed_seconds=time.monotonic() - started,
                      peak_sampled_git_logical_bytes=max(peak_bytes, git_bytes()), termination_reason=reason)
        if process.returncode or reason:
            raise RuntimeError(f"owned {name} failed: {record}")
    text = log.read_text(encoding="utf-8", errors="replace")
    if "filtering not recognized by server" in text.lower() or "filtering not supported" in text.lower():
        raise RuntimeError("server did not honor partial clone filtering")
    return text.strip()


def main() -> int:
    receipt = {"schema_version": 1, "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "source": REMOTE, "expected_label_git_tree": EXPECTED_LABEL_TREE,
               "expected_selected_label_files": 12415, "expected_selected_label_bytes": 480966,
               "repo_metadata_size_kib": 1623305, "whole_archive_downloaded": False,
               "max_git_directory_logical_bytes": MAX_GIT_LOGICAL_BYTES,
               "ceiling_scope": "sampled owned Git directory bytes, not a measured network byte counter",
               "max_command_seconds": MAX_COMMAND_SECONDS, "commands": [], "status": "RUNNING",
               "images_or_videos_requested": False, "images_or_videos_viewed": 0,
               "test_or_train_split_assigned": False, "training_performed": False}
    try:
        receipt["launch"] = admission()
        if CHECKOUT.exists():
            raise ValueError("fresh immutable attempt required; checkout path already exists")
        refs = execute(["ls-remote", REMOTE, "refs/heads/main"], "remote_ref", receipt, trace=True)
        match = re.fullmatch(r"([0-9a-f]{40})\s+refs/heads/main", refs)
        if not match:
            raise ValueError("unexpected remote main reference response")
        commit = match[1]
        receipt["pinned_commit"] = commit
        trace = (BASE / "public_server_capabilities.log").read_text(encoding="utf-8", errors="replace")
        if not re.search(r"fetch=[^\r\n]*\bfilter\b", trace):
            raise ValueError("public Git server did not advertise fetch filter; no clone attempted")
        receipt["server_filter_capability_verified"] = True
        execute(["clone", "--filter=blob:none", "--depth=1", "--single-branch", "--branch=main", "--no-checkout", REMOTE, str(CHECKOUT)], "blobless_clone", receipt)
        actual = execute(["rev-parse", "HEAD"], "commit_pin", receipt, CHECKOUT)
        if actual != commit:
            raise ValueError("remote commit changed after freeze; no checkout attempted")
        label_tree = execute(["rev-parse", "HEAD:tennis/Labels"], "label_tree_pin", receipt, CHECKOUT)
        if label_tree != EXPECTED_LABEL_TREE:
            raise ValueError("exact frozen tennis annotation tree changed; no checkout attempted")
        execute(["sparse-checkout", "set", "--no-cone", "/tennis/Labels/"], "sparse_path", receipt, CHECKOUT)
        execute(["checkout", "--detach", commit], "annotation_checkout", receipt, CHECKOUT)
        directory = json.loads((BASE / "author_prerelease_label_tree.json").read_text(encoding="utf-8"))
        expected = {item["path"]: item for item in directory["tree"] if item["type"] == "blob"}
        allowed = {item["sha"] for item in expected.values()}
        object_text = execute(["cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)"], "received_objects", receipt, CHECKOUT)
        objects = []
        for line in object_text.splitlines():
            if line.startswith("warning:"):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[1] not in ("blob", "commit", "tree", "tag"):
                raise ValueError(f"unexpected stored object inventory line: {line[:100]}")
            object_id, kind, size = fields[0], fields[1], int(fields[2])
            if kind == "blob" and object_id not in allowed:
                raise ValueError(f"unexpected non-tennis-label blob received: {object_id}, {size} bytes")
            objects.append((object_id, kind, size))
        receipt["received_unique_blob_count"] = sum(kind == "blob" for _, kind, _ in objects)
        receipt["received_unique_blob_bytes"] = sum(size for _, kind, size in objects if kind == "blob")
        receipt["all_received_blob_ids_belong_to_exact_selected_tennis_label_tree"] = True
        actual_paths = {path.relative_to(CHECKOUT).as_posix() for path in CHECKOUT.rglob("*") if path.is_file() and ".git" not in path.relative_to(CHECKOUT).parts}
        wanted_paths = {"tennis/Labels/" + name for name in expected}
        if actual_paths != wanted_paths:
            raise ValueError("sparse checkout contains missing, additional, or non-label files")
        counts = collections.defaultdict(lambda: collections.Counter())
        maximum_boxes = 0
        coordinate_ranges = [[math.inf, -math.inf] for _ in range(4)]
        archive_path = BASE / "complete_author_tennis_labels.zip"
        with (BASE / "complete_annotation_inventory.jsonl").open("x", encoding="utf-8") as manifest, zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(expected):
                item = expected[relative]
                data = (CHECKOUT / "tennis" / "Labels" / relative).read_bytes()
                object_id = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
                if object_id != item["sha"] or len(data) != item["size"]:
                    raise ValueError("checkout label does not reproduce exact frozen Git blob ID/size")
                folder = relative.split("/")[0]
                frame = int(re.search(r"_(\d+)\.txt$", relative)[1])
                boxes = []
                for line in data.decode("utf-8").splitlines():
                    fields = line.split()
                    if len(fields) != 5 or fields[0] != "0":
                        raise ValueError("unexpected full-corpus class or annotation schema")
                    coords = list(map(float, fields[1:]))
                    if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                        raise ValueError("invalid full-corpus normalized box")
                    x, y, width, height = coords
                    outside = x - width / 2 < 0 or x + width / 2 > 1 or y - height / 2 < 0 or y + height / 2 > 1
                    boxes.append({"class_id": 0, "xywh_normalized": coords, "extends_beyond_unit_image": outside})
                    counts[folder]["boxes"] += 1
                    counts[folder]["boxes_extending_beyond_unit_image"] += outside
                    for index, value in enumerate(coords):
                        coordinate_ranges[index][0] = min(coordinate_ranges[index][0], value)
                        coordinate_ranges[index][1] = max(coordinate_ranges[index][1], value)
                counts[folder]["label_files"] += 1
                counts[folder]["bytes"] += len(data)
                counts[folder]["empty_label_files"] += not boxes
                counts[folder]["multi_box_frames"] += len(boxes) > 1
                counts[folder][f"frames_with_{len(boxes)}_boxes"] += 1
                maximum_boxes = max(maximum_boxes, len(boxes))
                record = {"source_path": relative, "frame_index": frame, "git_blob_sha": object_id,
                          "sha256": sha256(data), "bytes": len(data), "boxes": boxes}
                manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                entry = zipfile.ZipInfo("tennis/Labels/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data, compresslevel=6)
        summary = {folder: dict(value) for folder, value in sorted(counts.items())}
        total = collections.Counter()
        for value in counts.values():
            total.update(value)
        if total["label_files"] != 12415 or total["bytes"] != 480966:
            raise ValueError("actual complete labels differ from frozen source metadata totals")
        receipt.update(status="COMPLETE_AUTHOR_TENNIS_ANNOTATIONS_ACQUIRED_AND_PARSED_NO_MEDIA", counts=dict(total),
                       per_alias=summary, maximum_boxes_in_one_frame=maximum_boxes,
                       coordinate_ranges_xywh_normalized=coordinate_ranges,
                       archive_bytes=archive_path.stat().st_size, archive_sha256=sha256(archive_path.read_bytes()),
                       inventory_bytes=(BASE / "complete_annotation_inventory.jsonl").stat().st_size,
                       inventory_sha256=sha256((BASE / "complete_annotation_inventory.jsonl").read_bytes()),
                       final_git_directory_logical_bytes=git_bytes(), native_dimensions_fps_pts_verified=False,
                       human_annotation_or_visible_only_policy_verified=False, source_training_admission=False,
                       author_prerelease_is_not_byte_proven_equivalent_to_inaccessible_official_dryad_zip=True,
                       annotation_rights="Author separately published pre-release research/academic notice; official Dryad annotation CC0 tag retained separately, exact archive equivalence unverified")
        return_code = 0
    except Exception as exc:
        receipt.update(status="COMPLETE_AUTHOR_ANNOTATION_ACQUISITION_NOT_PROVEN", error=type(exc).__name__, detail=str(exc))
        return_code = 1
    with (BASE / "complete_annotation_acquisition_receipt.json").open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("commands", "per_alias")}, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
