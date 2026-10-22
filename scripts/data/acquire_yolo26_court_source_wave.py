"""Acquire only the frozen 64-image court wave with strict per-member guards.

The module imports only standard-library code until actual media admission has
passed. It preserves partial datasets and immutable attempt/member receipts.
PNG structure and dimensions are checked without opening/decoding image pixels.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import zlib


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/court_source_wave_metadata_v2/source_wave_plan.json"
PLAN_SHA256 = "27d72148f4173ab476e9dbaaa5df6b64ce392864fd7150e6d7c0b59e893af65a"
DATASET = ROOT / "data/external/court_source_wave64_v1"
RAM_FLOOR = 1024**3
DISK_FLOOR = 2 * 1024**3
NETWORK_CEILING = 80_000_000
MAX_MEMBER = 5_000_000
HTTP_HELPER_SHA256 = "95e6d7153f4afb88b9291525c581fa546d4e5cd9290a6175a12d032be9d0947b"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, document):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def within_workspace(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def current_resources(disk_path: Path) -> dict:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("size", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in
            ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile", "private")]

    status = MemoryStatus()
    status.size = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    process = ProcessMemory()
    process.cb = ctypes.sizeof(process)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(process), process.cb):
        raise OSError("GetProcessMemoryInfo failed")
    existing = disk_path
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError("No existing parent for disk resource check")
        existing = existing.parent
    return {"available_physical_ram_bytes": status.available,
            "free_disk_bytes": shutil.disk_usage(existing).free,
            "owned_rss_bytes": process.rss, "peak_owned_rss_bytes": process.peak_rss}


def enforce_resources(resources: dict):
    if resources["available_physical_ram_bytes"] < RAM_FLOOR:
        raise ResourceAdmissionError("Available physical RAM is below unchanged 1 GiB media floor")
    if resources["free_disk_bytes"] < DISK_FLOOR:
        raise ResourceAdmissionError("Free disk is below unchanged 2 GiB media floor")


class ResourceAdmissionError(RuntimeError):
    pass


def png_geometry(data: bytes) -> tuple[int, int]:
    """Verify PNG chunk CRCs and native IHDR geometry; never decode pixels."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid native PNG signature")
    position, kinds, geometry = 8, [], None
    while position < len(data):
        if position + 12 > len(data):
            raise ValueError("Truncated PNG chunk")
        size = struct.unpack_from(">I", data, position)[0]
        kind = data[position + 4:position + 8]
        end = position + 12 + size
        if end > len(data):
            raise ValueError("PNG chunk leaves file")
        payload = data[position + 8:position + 8 + size]
        expected_crc = struct.unpack_from(">I", data, position + 8 + size)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if not kinds and kind != b"IHDR":
            raise ValueError("PNG must begin with IHDR")
        if kind == b"IHDR":
            if kinds or size != 13:
                raise ValueError("Invalid or repeated PNG IHDR")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            legal = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
            if color not in legal or depth not in legal[color] or compression or filtering or interlace not in (0, 1):
                raise ValueError("Unsupported PNG IHDR fields")
            if (width, height) != (1280, 720):
                raise ValueError("PNG is not native 1280x720")
            geometry = width, height
        kinds.append(kind)
        position = end
        if kind == b"IEND":
            if size or position != len(data) or b"IDAT" not in kinds:
                raise ValueError("Invalid PNG ending or missing IDAT")
            return geometry
    raise ValueError("PNG is missing IEND")


def zip64_sizes(extra: bytes, compressed: int, uncompressed: int) -> tuple[int, int]:
    position, zip64 = 0, None
    while position < len(extra):
        if position + 4 > len(extra):
            raise ValueError("Truncated ZIP extra field")
        kind, size = struct.unpack_from("<HH", extra, position)
        position += 4
        if position + size > len(extra):
            raise ValueError("ZIP extra field leaves header")
        if kind == 1:
            if zip64 is not None:
                raise ValueError("Repeated ZIP64 extra field")
            zip64 = extra[position:position + size]
        position += size
    if compressed == 0xFFFFFFFF or uncompressed == 0xFFFFFFFF:
        if zip64 is None:
            raise ValueError("ZIP64 placeholder has no size extra")
        offset = 0
        if uncompressed == 0xFFFFFFFF:
            if offset + 8 > len(zip64):
                raise ValueError("ZIP64 uncompressed size missing")
            uncompressed = struct.unpack_from("<Q", zip64, offset)[0]
            offset += 8
        if compressed == 0xFFFFFFFF:
            if offset + 8 > len(zip64):
                raise ValueError("ZIP64 compressed size missing")
            compressed = struct.unpack_from("<Q", zip64, offset)[0]
    return compressed, uncompressed


def verify_descriptor(tail: bytes, entry: dict):
    if tail[:4] == b"PK\x07\x08":
        tail = tail[4:]
    expected = (entry["crc32"], entry["compressed_bytes"], entry["bytes"])
    for fmt in ("<III", "<IQQ"):
        if len(tail) >= struct.calcsize(fmt) and struct.unpack_from(fmt, tail) == expected:
            return
    raise ValueError("ZIP data descriptor sizes/CRC do not match pinned index")


def verified_png_member(block: bytes, entry: dict) -> tuple[bytes, dict]:
    if len(block) != entry["planned_range_bytes_upper_bound"] or len(block) > MAX_MEMBER:
        raise ValueError("Downloaded member range length differs from frozen budget")
    if len(block) < 30:
        raise ValueError("Truncated ZIP local header")
    signature, version, flags, method, _, _, crc, compressed, uncompressed, name_size, extra_size = struct.unpack_from("<4s5H3I2H", block)
    if signature != b"PK\x03\x04" or not 10 <= version <= 63 or flags & ~0x080E or method != 8:
        raise ValueError("Unsupported/encrypted ZIP local header")
    if extra_size > 1024 or name_size != len(entry["name"].encode("utf-8")):
        raise ValueError("ZIP header exceeds frozen name/extra allowance")
    start = 30 + name_size + extra_size
    if start > len(block) or block[30:30 + name_size] != entry["name"].encode("utf-8"):
        raise ValueError("ZIP member name mismatch")
    compressed, uncompressed = zip64_sizes(block[30 + name_size:start], compressed, uncompressed)
    if flags & 8:
        if crc not in (0, entry["crc32"]) or compressed not in (0, entry["compressed_bytes"]) or uncompressed not in (0, entry["bytes"]):
            raise ValueError("Deferred ZIP header differs from pinned sizes/CRC")
    elif (crc, compressed, uncompressed) != (entry["crc32"], entry["compressed_bytes"], entry["bytes"]):
        raise ValueError("ZIP local header sizes/CRC differ from pinned index")
    if not 0 < entry["bytes"] <= MAX_MEMBER or not 0 < entry["compressed_bytes"] <= MAX_MEMBER:
        raise ValueError("Pinned member exceeds fixed limit")
    end = start + entry["compressed_bytes"]
    if end > len(block):
        raise ValueError("Compressed payload leaves frozen range")
    if flags & 8:
        verify_descriptor(block[end:], entry)
    decoder = zlib.decompressobj(-15)
    data = decoder.decompress(block[start:end], entry["bytes"] + 1)
    if len(data) != entry["bytes"] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Inflated PNG size or deflate boundary mismatch")
    if zlib.crc32(data) & 0xFFFFFFFF != entry["crc32"]:
        raise ValueError("PNG publisher ZIP CRC mismatch")
    width, height = png_geometry(data)
    return data, {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                  "publisher_crc32_verified": True, "png_all_chunk_crcs_verified": True,
                  "native_png_geometry": [width, height], "pixel_decoding_performed": False,
                  "zip_flags": flags, "zip_version_needed": version,
                  "zip_local_header_sizes_crc_verified": True}


def load_frozen_plan() -> dict:
    if digest(PLAN) != PLAN_SHA256:
        raise ValueError("Only the exact root-reviewed v2 court plan is authorized")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if len(plan["samples"]) != 64 or len({row["source_video_id"] for row in plan["samples"]}) != 64:
        raise ValueError("Frozen 64-image selection changed")
    if plan["archive"]["planned_range_bytes_upper_bound"] != 52_987_635 or plan["archive"]["fixed_range_budget_bytes"] != NETWORK_CEILING:
        raise ValueError("Frozen range budget changed")
    for item in plan["source_inputs"]:
        if digest(within_workspace(ROOT / item["path"])) != item["sha256"]:
            raise ValueError("Root-reviewed source metadata changed")
    if digest(ROOT / "scripts/data/inspect_court_training_archive.py") != HTTP_HELPER_SHA256:
        raise ValueError("Bounded HTTP range helper changed since acquisition review")
    return plan


def acquire(attempt: Path) -> dict:
    attempt = within_workspace(attempt)
    dataset = within_workspace(DATASET)
    if attempt.exists():
        raise FileExistsError("Use a fresh immutable attempt directory")
    if dataset.exists():
        raise FileExistsError("Preserve existing/partial dataset; never overwrite or silently resume")
    attempt.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "status": "STARTED", "plan": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
               "plan_sha256": PLAN_SHA256, "source_script_sha256": digest(Path(__file__)),
               "http_range_helper_sha256": HTTP_HELPER_SHA256,
               "dataset": str(dataset.relative_to(ROOT)).replace("\\", "/"),
               "ram_floor_bytes": RAM_FLOOR, "disk_floor_bytes": DISK_FLOOR,
               "fixed_network_ceiling_bytes": NETWORK_CEILING, "max_inflight_members": 1,
               "completed_images": 0, "requested_member_ranges": 0, "range_bytes_received": 0,
               "qualification_evidence": False, "all_development_splits_unassigned": True}
    save(attempt / "attempt_start.json", receipt)
    files = []
    try:
        plan = load_frozen_plan()
        receipt["resources_before_admission"] = current_resources(dataset)
        enforce_resources(receipt["resources_before_admission"])
        # Deliberately deferred until the unchanged actual-media guard passes.
        from scripts.data.inspect_court_training_archive import BoundedHTTPFile

        receipt["resources_after_http_import"] = current_resources(dataset)
        enforce_resources(receipt["resources_after_http_import"])
        (dataset / "images").mkdir(parents=True)
        save(dataset / "acquisition_context.json", {"status": "PARTIAL_UNTIL_COMPLETE_MANIFEST_EXISTS",
                                                   "plan_sha256": PLAN_SHA256, "attempt": str(attempt.relative_to(ROOT)).replace("\\", "/"),
                                                   "selected_ids": [row["id"] for row in plan["samples"]],
                                                   "complete": False, "qualification_evidence": False})
        for sample in plan["samples"]:
            before = current_resources(dataset)
            receipt["last_runtime_resources"] = before
            enforce_resources(before)
            entry = sample["archive_member"]
            length = entry["planned_range_bytes_upper_bound"]
            if receipt["range_bytes_received"] + length > NETWORK_CEILING:
                raise ValueError("Next range exceeds fixed aggregate network ceiling")
            remote = BoundedHTTPFile(plan["archive"]["url"], plan["archive"]["params"],
                                     plan["archive"]["archive_bytes"], budget=length)
            try:
                receipt["requested_member_ranges"] += 1
                with remote:
                    remote.seek(entry["planned_range_start"])
                    block = remote.read(length)
            finally:
                receipt["range_bytes_received"] += remote.received
            data, checks = verified_png_member(block, entry)
            path = dataset / "images" / f"{sample['id']}.png"
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if digest(path) != checks["sha256"]:
                raise ValueError("Saved native image bytes changed")
            file = {"id": sample["id"], "source_video_id": sample["source_video_id"],
                    "path": str(path.relative_to(dataset)).replace("\\", "/"), **checks,
                    "range_bytes": length, "resources_before_member": before,
                    "development_split": None, "context_group": None,
                    "visual_review_status": "PENDING_ACTUAL_NATIVE_IMAGE_CONTEXT_INTERSECTION_REVIEW"}
            files.append(file)
            save(attempt / f"member_{len(files):03}.json", file)
            receipt["completed_images"] = len(files)
            del block, data
            if len(files) % 8 == 0:
                print(json.dumps({"completed_images": len(files), "total": 64, "range_bytes_received": receipt["range_bytes_received"]}), flush=True)
        if len(files) != 64 or receipt["range_bytes_received"] != 52_987_635:
            raise ValueError("Incomplete frozen wave")
        if any(digest(dataset / item["path"]) != item["sha256"] for item in files):
            raise ValueError("An acquired file changed before completion")
        load_frozen_plan()
        manifest = {"status": "COMPLETE_ACQUISITION_PENDING_NATIVE_REVIEW_NO_EXPORT", "complete": True,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": PLAN_SHA256,
                    "whole_archive_checksum_verified": False, "qualification_evidence": False,
                    "source_inputs": plan["source_inputs"], "rights": plan["rights"],
                    "context_review_constraints": plan["context_review_constraints"],
                    "samples": plan["samples"], "files": files,
                    "range_bytes_received": receipt["range_bytes_received"]}
        save(dataset / "manifest.json", manifest)
        receipt["manifest_sha256"] = digest(dataset / "manifest.json")
        receipt["status"] = "COMPLETED_64_NATIVE_PNGS_PENDING_VISUAL_REVIEW"
    except ResourceAdmissionError as error:
        receipt["status"] = "REFUSED_RESOURCE_ADMISSION_BEFORE_NETWORK" if receipt["requested_member_ranges"] == 0 else "STOPPED_RUNTIME_RESOURCE_GUARD_PARTIAL_PRESERVED"
        receipt["exception_type"] = type(error).__name__
        receipt["reason"] = str(error)
    except Exception as error:
        receipt["status"] = "FAILED_PARTIAL_PRESERVED_NO_COMPLETE_MANIFEST"
        receipt["exception_type"] = type(error).__name__
        receipt["reason"] = str(error)
    receipt["dataset_exists"] = dataset.exists()
    receipt["complete_manifest_exists"] = (dataset / "manifest.json").exists()
    receipt["requests_imported"] = "requests" in sys.modules
    receipt["media_or_model_modules_imported"] = sorted(name for name in sys.modules if name.split(".")[0] in ("PIL", "cv2", "numpy", "torch", "ultralytics"))
    receipt["elapsed_seconds"] = time.monotonic() - started
    receipt["resources_at_end"] = current_resources(dataset)
    save(attempt / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path)
    args = parser.parse_args()
    receipt = acquire(args.attempt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt["status"].startswith("COMPLETED_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
