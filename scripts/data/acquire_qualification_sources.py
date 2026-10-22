"""Acquire immutable CalTennis qualification candidates; never runs inference.

Private noncommercial research only. Candidate acquisition is not test certification.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/evaluation/TENNIS_QUALIFICATION_ACQUISITION_CANDIDATES_2026-09-30.json"
DEST = ROOT / "data/external/caltennis_qualification"
MARGIN = 2 * 1024**3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_path(relative: str) -> Path:
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or any(":" in p for p in part.parts):
        raise ValueError(f"Unsafe publisher path: {relative}")
    result = (DEST / Path(*part.parts)).resolve()
    if not result.is_relative_to(DEST.resolve()):
        raise ValueError(f"Path leaves acquisition directory: {relative}")
    return result


def file_hashes(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            sha.update(chunk)
            git.update(chunk)
    return sha.hexdigest(), git.hexdigest()


def validate(path: Path, source: dict) -> dict:
    size = path.stat().st_size
    if size != source["expected_bytes"]:
        raise ValueError(f"Wrong bytes for {path.name}: {size}")
    sha, git = file_hashes(path)
    expected_sha = source.get("publisher_sha256")
    expected_git = source.get("publisher_git_blob_oid")
    if expected_sha and sha != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path.name}")
    if not expected_sha and expected_git and git != expected_git:
        raise ValueError(f"Git blob mismatch for {path.name}")
    if not expected_sha and not expected_git:
        raise ValueError(f"No immutable publisher digest for {path.name}")
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": size, "sha256": sha, "git_blob_oid": git, "publisher_digest_verified": True, "last_verified_at": now()}


def acquire(source: dict) -> dict:
    destination = safe_path(source["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        record = validate(destination, source)
        return {**record, "source_url": source["source_url"], "status": "EXISTING_VERIFIED"}
    partial = destination.with_suffix(destination.suffix + ".part")
    expected = source["expected_bytes"]
    last_error = None
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise ValueError(f"Oversized partial download: {partial.name}")
            if offset < expected:
                headers = {"User-Agent": "TennisResearchQualification/1.0", "Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                request = urllib.request.Request(source["source_url"], headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if offset and response.status != 206:
                        raise ValueError("Server did not honor resume range; preserving partial")
                    if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Unexpected resume range")
                    if not offset and response.status != 200:
                        raise ValueError(f"Unexpected response status {response.status}")
                    started = time.monotonic()
                    reported = started
                    with partial.open("ab" if offset else "wb") as stream:
                        while chunk := response.read(1024**2):
                            offset += len(chunk)
                            if offset > expected:
                                raise ValueError("Response exceeds pinned size; preserving partial")
                            stream.write(chunk)
                            if time.monotonic() - reported >= 20:
                                print(json.dumps({"download": source["path"], "bytes": offset, "expected_bytes": expected}), flush=True)
                                reported = time.monotonic()
            record = validate(partial, source)
            partial.replace(destination)
            record["path"] = str(destination.relative_to(ROOT)).replace("\\", "/")
            print(json.dumps({"verified": source["path"], "bytes": expected}), flush=True)
            return {**record, "source_url": source["source_url"], "status": "ACQUIRED_VERIFIED", "acquired_at": now()}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
            print(json.dumps({"download_error": source["path"], "attempt": attempt + 1, "reason": last_error}), flush=True)
            if isinstance(exc, ValueError):
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Acquisition failed for {source['path']}: {last_error}")


def inspect_metadata(candidate: dict) -> dict:
    import numpy as np

    timestamp_path = safe_path(candidate["timestamps"]["path"])
    stamps = np.load(timestamp_path, allow_pickle=False, mmap_mode="r")
    if stamps.dtype.hasobject or stamps.ndim != 1 or not np.issubdtype(stamps.dtype, np.number):
        raise ValueError("Timestamp array must be one-dimensional numeric data without pickle")
    finite = bool(np.isfinite(stamps).all())
    differences = np.diff(stamps.astype(np.float64))
    calibration = json.loads(safe_path(candidate["calibration"]["path"]).read_text(encoding="utf-8"))
    return {
        "timestamp_shape": list(stamps.shape), "timestamp_dtype": str(stamps.dtype),
        "timestamp_finite": finite, "timestamp_strictly_increasing": bool((differences > 0).all()),
        "timestamp_first_raw": float(stamps[0]), "timestamp_last_raw": float(stamps[-1]),
        "timestamp_delta_median_raw": float(np.median(differences)),
        "timestamp_delta_min_raw": float(differences.min()), "timestamp_delta_max_raw": float(differences.max()),
        "timestamp_units_verified": False, "video_frame_alignment_verified": False,
        "calibration_keys": sorted(calibration), "calibration_coordinate_convention_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Inspect safe timestamp arrays/calibration after acquisition")
    parser.add_argument("--group", action="append", help="Only these exact candidate group IDs")
    args = parser.parse_args()
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    candidates = [c for c in plan["candidates"] if not args.group or c["candidate_group_id"] in args.group]
    if not candidates or args.group and len(candidates) != len(set(args.group)):
        raise ValueError("Unknown or empty candidate selection")
    sources = [candidate[kind] for candidate in candidates for kind in ("timestamps", "calibration", "video")]
    missing = sum(s["expected_bytes"] for s in sources if not safe_path(s["path"]).exists())
    disk = shutil.disk_usage(ROOT)
    print(json.dumps({"free_disk_bytes": disk.free, "planned_missing_bytes": missing, "required_with_margin": missing + MARGIN, "numpy_available": importlib.util.find_spec("numpy") is not None}), flush=True)
    if not args.acquire and not args.inspect:
        return
    if args.acquire and disk.free < missing + MARGIN:
        raise RuntimeError("Insufficient free disk for pinned acquisition plus 2 GiB margin")
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot = DEST / "candidate_plan.json"
    if snapshot.exists() and snapshot.read_bytes() != plan_bytes:
        raise ValueError("Preserved acquisition plan differs from current candidate plan")
    if not snapshot.exists():
        snapshot.write_bytes(plan_bytes)
    diagnostic = ROOT / "data/external/caltennis_diagnostic"
    for name in ("README.md", "tree.json", "metadata_mini.jsonl"):
        source = diagnostic / name
        target = DEST / "publisher_metadata" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    license_notice = DEST / "LICENSE_AND_USAGE.md"
    if not license_notice.exists():
        license_notice.write_text("# CalTennis candidate qualification acquisition\n\nPublisher: https://huggingface.co/datasets/demalenk/caltennis\n\nLicense: CC BY-NC 4.0 — https://creativecommons.org/licenses/by-nc/4.0/\n\nPrivate noncommercial academic evaluation only. Attribute CalTennis / demalenk and preserve source/card notices. No public product demo use is authorized by this acquisition. Source videos remain unchanged; any review frames/excerpts are identified derivatives. No model inference is permitted on future holdout before protocol/labels/bundle freeze. Acquisition is not qualification evidence.\n", encoding="utf-8")
    manifest_path = DEST / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    files = {f["source_url"]: f for f in previous.get("files", [])}
    errors = []
    if args.acquire:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(acquire, source): source for source in sources}
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    files[result["source_url"]] = {**files.get(result["source_url"], {}), **result}
                except Exception as exc:
                    errors.append({"path": futures[future]["path"], "reason": str(exc)})
                write_json(manifest_path, {**previous, "source": plan["source"], "revision": plan["source_revision"], "candidate_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "license": "CC-BY-NC-4.0", "public_product_demo_permission": False, "qualification_evidence": False, "certified_clips": 0, "certified_sessions": 0, "files": list(files.values()), "errors": errors, "complete": len(files) == 18 and not errors, "updated_at": now()})
    if args.inspect:
        observations = []
        for candidate in candidates:
            try:
                observations.append({"candidate_group_id": candidate["candidate_group_id"], **inspect_metadata(candidate)})
            except Exception as exc:
                observations.append({"candidate_group_id": candidate["candidate_group_id"], "metadata_error": str(exc)})
        write_json(DEST / "metadata_inspection.json", {"inspected_at": now(), "allow_pickle": False, "qualification_evidence": False, "observations": observations})
        print(json.dumps({"metadata_inspection": observations}), flush=True)
    print(json.dumps({"verified_files": len(files), "errors": errors, "qualification_evidence": False}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
