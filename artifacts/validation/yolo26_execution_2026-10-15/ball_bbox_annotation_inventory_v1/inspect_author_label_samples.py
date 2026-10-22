"""Bounded annotation-only sample inspection using immutable author Git blob IDs."""
from __future__ import annotations

import base64
import collections
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request

BASE = Path(__file__).resolve().parent
TREE = BASE / "author_prerelease_label_tree.json"
TREE_SHA256 = "af4fad7eaed7d4d040cf0f15e16ce4567059d85ccf9f19d5904de96c88b2eaa4"
EXPECTED_TREE = "8103a5b69699fc32d7230fafb214052b14dc292d"
MAX_REQUESTS = 48
MAX_TOTAL_RESPONSE_BYTES = 192 * 1024


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def frame_index(row: dict) -> int:
    matched = re.search(r"_(\d+)\.txt$", row["path"])
    if not matched:
        raise ValueError(f"unexpected annotation path: {row['path']}")
    return int(matched[1])


def main() -> int:
    tree_bytes = TREE.read_bytes()
    if sha256(tree_bytes) != TREE_SHA256:
        raise ValueError("frozen author metadata changed")
    tree = json.loads(tree_bytes)
    if tree["sha"] != EXPECTED_TREE or tree["truncated"]:
        raise ValueError("tree mismatch or incomplete metadata")
    folders: dict[str, list[dict]] = collections.defaultdict(list)
    for row in tree["tree"]:
        if row["type"] != "blob":
            continue
        if row["mode"] != "100644" or not row["path"].endswith(".txt"):
            raise ValueError("non-regular annotation in tennis label subtree")
        path = PurePosixPath(row["path"])
        if len(path.parts) != 2 or any(part in (".", "..") for part in path.parts):
            raise ValueError("unexpected label path layout")
        folders[path.parts[0]].append(row)
    proposed: list[dict] = []
    summary = []
    for folder in sorted(folders):
        rows = sorted(folders[folder], key=frame_index)
        indices = [frame_index(row) for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate source frame index")
        stat = {"original_alias": folder, "label_files": len(rows),
                "publisher_git_blob_bytes": sum(row["size"] for row in rows),
                "empty_label_files": sum(row["size"] == 0 for row in rows),
                "minimum_labeled_frame": min(indices), "maximum_labeled_frame": max(indices),
                "unlabeled_indices_between_extrema": max(indices) - min(indices) + 1 - len(indices),
                "size_histogram": dict(sorted(collections.Counter(row["size"] for row in rows).items())),
                "frames_are_not_known_total_video_frames": True}
        summary.append(stat)
        selected = [rows[0], rows[len(rows) // 2], sorted(rows, key=lambda row: (-row["size"], frame_index(row)))[0]]
        for row in selected:
            if not any(prior["path"] == row["path"] for prior in proposed):
                proposed.append(row)
    if len(proposed) > MAX_REQUESTS:
        raise ValueError("fixed sample request ceiling exceeded")
    receipt = {"schema_version": 1, "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "policy": "first labeled frame, median labeled frame, and largest label file per original alias; duplicate selections removed",
               "tree_sha256": TREE_SHA256, "git_tree_sha": EXPECTED_TREE, "max_requests": MAX_REQUESTS,
               "max_total_response_bytes": MAX_TOTAL_RESPONSE_BYTES, "media_acquired_bytes": 0,
               "images_viewed": 0, "model_imports": [], "actual_requests": 0, "actual_response_bytes": 0,
               "summary": summary, "samples": [], "status": "RUNNING"}
    output = BASE / "author_label_samples"
    output.mkdir(exist_ok=False)
    try:
        for row in proposed:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "TennisAcademicDatasetAudit/1.0", "Accept": "application/vnd.github+json"})
            receipt["actual_requests"] += 1
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read(8193)
                if len(raw) > 8192:
                    raise ValueError("single annotation metadata exceeded fixed ceiling")
                receipt["actual_response_bytes"] += len(raw)
                if receipt["actual_response_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
                    raise ValueError("total annotation metadata ceiling exceeded")
                blob = json.loads(raw)
                if blob.get("sha") != row["sha"] or blob.get("encoding") != "base64":
                    raise ValueError("publisher annotation Git object or encoding changed")
                content = base64.b64decode(blob["content"], validate=False)
                object_id = hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()
                if object_id != row["sha"] or len(content) != row["size"]:
                    raise ValueError("decoded annotation does not reproduce publisher Git blob ID and size")
                local_path = output / row["path"].replace("/", "__")
                with local_path.open("xb") as handle:
                    handle.write(content)
                boxes, errors = [], []
                for line_index, line in enumerate(content.decode("utf-8").splitlines()):
                    fields = line.split()
                    if len(fields) != 5:
                        errors.append({"line": line_index, "error": "expected class plus four YOLO coordinates"})
                        continue
                    try:
                        klass = int(fields[0])
                        box = list(map(float, fields[1:]))
                        x, y, w, h = box
                        if klass != 0 or not all(0 <= value <= 1 for value in box) or w <= 0 or h <= 0:
                            errors.append({"line": line_index, "error": "invalid class/coordinate domain"})
                        boxes.append({"class": klass, "xywh_normalized": box,
                                      "extends_beyond_unit_image": x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1})
                    except (ValueError, OverflowError):
                        errors.append({"line": line_index, "error": "coordinate parse failure"})
                receipt["samples"].append({"source_path": row["path"], "frame_index": frame_index(row),
                                           "url": row["url"], "git_blob_sha": row["sha"], "bytes": len(content),
                                           "sha256": sha256(content), "local_path": str(local_path),
                                           "boxes": boxes, "errors": errors})
        receipt.update(status="BOUNDED_ANNOTATION_SAMPLE_INSPECTION_COMPLETED", actual_sample_labels=len(receipt["samples"]),
                       actual_sample_boxes=sum(len(item["boxes"]) for item in receipt["samples"]),
                       actual_sample_parse_errors=sum(len(item["errors"]) for item in receipt["samples"]),
                       total_metadata_label_files=sum(item["label_files"] for item in summary),
                       total_metadata_label_bytes=sum(item["publisher_git_blob_bytes"] for item in summary),
                       exact_full_corpus_box_count=None,
                       visible_or_occluded_state_semantics_verified=False, native_dimensions_verified=False,
                       native_fps_or_pts_verified=False, annotation_process_verified=False)
        return_code = 0
    except Exception as exc:
        receipt.update(status="PARTIAL_ANNOTATION_SAMPLE_INSPECTION", error=type(exc).__name__, detail=str(exc))
        return_code = 1
    with (BASE / "author_label_sample_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ("samples", "summary")}, indent=2))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
