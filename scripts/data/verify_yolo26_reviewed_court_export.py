"""Verify a court development export using bounded reads and PNG headers only."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def peak_working_set():
    if not hasattr(ctypes, "WinDLL"):
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return info.PeakWorkingSetSize


def verify(export: Path, frozen_path: Path, decisions_path: Path, inventory_path: Path, approval_path: Path, output: Path):
    if output.exists():
        raise FileExistsError("Preserve previous verification receipts")
    manifest_path = export / "source_manifest.json"
    manifest, frozen, decisions, inventory = map(load, (manifest_path, frozen_path, decisions_path, inventory_path))
    paths = {"frozen_manifest_sha256": frozen_path, "decisions_sha256": decisions_path,
             "inventory_sha256": inventory_path, "root_approval_sha256": approval_path}
    for key, path in paths.items():
        if manifest["lineage"][key] != digest(path):
            raise ValueError("Export lineage changed: " + key)
    if manifest["qualification_evidence"] or manifest["source_independence_verified"] or manifest["final_test_used"]:
        raise ValueError("Development export cannot claim qualification or a final test")
    selected = {row["id"]: row for row in frozen["selected"]}
    inv = {row["id"]: row for row in inventory["samples"]}
    reviewed = {row["id"]: row for row in decisions["samples"]}
    sources = {row["source_id"]: row for row in manifest["sources"]}
    if len(selected) != 120 or len(manifest["samples"]) != 120 or {row["id"] for row in manifest["samples"]} != set(selected):
        raise ValueError("Export must contain the exact 120 approved unique selected images")
    counts, slots, bytes_total, file_hashes = Counter(), Counter(), 0, []
    contexts, videos = {"train": set(), "val": set()}, {"train": set(), "val": set()}
    for record in manifest["samples"]:
        ident, split = record["id"], record["split"]
        row, sample, source = selected[ident], inv[ident], sources[record["source_id"]]
        if row != reviewed[ident] or split != row["development_split"] or split not in ("train", "val"):
            raise ValueError("Exported decision/split differs from approved native review")
        if source["source_type"] != "PUBLISHER_STILL_IMAGE" or source["match_id"] is not None or source["fps"] is not None:
            raise ValueError("Do not invent original match identity or frame rate")
        if any(record[key] is not None for key in ("timestamp_seconds", "source_pts", "source_time_base")):
            raise ValueError("Static publisher images have unknown source timing")
        image_path, label_path = ROOT / record["image"], ROOT / record["label"]
        with image_path.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR" or struct.unpack(">II", header[16:24]) != (1280, 720):
            raise ValueError("Native PNG header/resolution changed")
        image_sha, original_sha, label_sha = digest(image_path), digest(ROOT / sample["source_image"]), digest(label_path)
        if image_sha != original_sha or image_sha != sample["sha256"] or label_sha != record["label_sha256"]:
            raise ValueError("Copied image identity or label checksum failed")
        lines = label_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            raise ValueError("One court instance per reviewed image is required")
        values = list(map(float, lines[0].split()))
        if len(values) != 47 or values[0] != 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("Court YOLO pose row requires exactly 47 finite fields")
        if record["keypoints_px"] != row["reviewed_keypoints_px"] or record["bbox_xyxy_px"] != row["bbox_xyxy_px"]:
            raise ValueError("Exported native annotation changed")
        x1, y1, x2, y2 = row["bbox_xyxy_px"]
        expected_box = [(x1 + x2) / 2 / 1280, (y1 + y2) / 2 / 720, (x2 - x1) / 1280, (y2 - y1) / 720]
        if any(abs(a - b) > 1e-9 for a, b in zip(values[1:5], expected_box)):
            raise ValueError("Court bounding extent normalization failed")
        for index, point in enumerate(row["reviewed_keypoints_px"]):
            triple = values[5 + index * 3:8 + index * 3]
            if point is None:
                if triple != [0, 0, 0]:
                    raise ValueError("Unknown court point gained inferred coordinates/visibility")
                slots["unknown_null"] += 1
            else:
                if triple[2] != 2 or abs(triple[0] * 1280 - point[0]) > 1e-6 or abs(triple[1] * 720 - point[1]) > 1e-6:
                    raise ValueError("Native semantic point roundtrip failed")
                slots["visible"] += 1
        counts[split] += 1
        bytes_total += image_path.stat().st_size
        contexts[split].add(row["context_group"])
        videos[split].add(sample["source_video_id"])
        file_hashes.append({"id": ident, "image_sha256": image_sha, "source_image_sha256": original_sha,
                            "label_sha256": label_sha, "split": split})
    if contexts["train"] & contexts["val"] or videos["train"] & videos["val"]:
        raise ValueError("Publisher videos or complete observed contexts crossed splits")
    counts["test"] = 0
    if dict(counts) != frozen["counts"] or dict(counts) != manifest["counts"]:
        raise ValueError("Declared image counts differ from actual exported files")
    if slots != Counter({"visible": 1657, "unknown_null": 23}):
        raise ValueError("Visible/null slot counts differ from approved review")
    actual_images = set((export / "court" / "images").rglob("*.png"))
    actual_labels = set((export / "court" / "labels").rglob("*.txt"))
    if len(actual_images) != 120 or len(actual_labels) != 120:
        raise ValueError("Unregistered or missing image/label files")
    plan = load(export / "training_plan.json")
    if plan["augmentation"] != {"fliplr": 0, "flipud": 0, "mosaic": 0, "mixup": 0} or plan["training_executed"]:
        raise ValueError("Initial court augmentation/training declaration changed")
    yaml = (export / "court" / "data.yaml").read_text(encoding="utf-8")
    if "kpt_shape: [14, 3]" not in yaml or "flip_idx: [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]" not in yaml:
        raise ValueError("Data YAML changed landmark shape/semantic flip mapping")
    receipt = {"status": "PASS_EXACT_PRIVATE_DEVELOPMENT_EXPORT_NOT_QUALIFICATION",
               "completed_utc": datetime.now(timezone.utc).isoformat(), "counts": dict(counts),
               "keypoint_slots": dict(slots), "image_bytes_copied": bytes_total,
               "source_manifest_sha256": digest(manifest_path), "lineage_hashes_after_export": manifest["lineage"],
               "source_image_copy_identity": True, "label_fields_per_instance": 47,
               "normalized_to_native_roundtrip_max_allowed_pixels": 1e-6,
               "selected_context_counts": {split: len(values) for split, values in contexts.items()},
               "selected_video_id_counts": {split: len(values) for split, values in videos.items()},
               "video_id_overlap": [], "context_overlap": [], "native_images_decoded": False,
               "gpu_used": False, "training_executed": False, "independent_human_review": False,
               "qualification_evidence": False, "verification_process_peak_working_set_bytes": peak_working_set(),
               "files": file_hashes}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    return {key: value for key, value in receipt.items() if key != "files"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("export", "frozen", "decisions", "inventory", "approval", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.export, args.frozen, args.decisions, args.inventory, args.approval, args.output)))


if __name__ == "__main__":
    main()
