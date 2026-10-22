"""Small metadata-only helpers; importing this module loads no image/model stack."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT / "scripts/data/prepare_ball_review_expansion.py"


def legacy_constant(name):
    # Read literal declarations without importing the earlier OpenCV module.
    for statement in ast.parse(PREVIOUS.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise ValueError(f"Missing retained sampler declaration: {name}")


CACHES = legacy_constant("CACHES")
CLIP_EXCLUSIONS = legacy_constant("CLIP_EXCLUSIONS")
DEVELOPMENT_GROUPS = legacy_constant("DEVELOPMENT_GROUPS")
GROUP_EXCLUSIONS = legacy_constant("GROUP_EXCLUSIONS")
GROUPS = legacy_constant("GROUPS")
REVISION = legacy_constant("REVISION")
TILE_ORIGINS = [(x, y) for y in (0, 440) for x in (0, 512, 1024, 1280)]
MAX_PROBE_RSS_BYTES = 64 * 1024 ** 2
MIN_METADATA_AVAILABLE_RAM_BYTES = 128 * 1024 ** 2


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_card(path, publisher_hash):
    raw = Path(path).read_bytes()
    normalized = raw.replace(b"\r\n", b"\n")
    raw_hash = hashlib.sha256(raw).hexdigest()
    normalized_hash = hashlib.sha256(normalized).hexdigest()
    if publisher_hash not in (raw_hash, normalized_hash):
        raise ValueError("Publisher card text changed beyond original Windows CRLF translation")
    return {"local_card_path": str(path), "local_raw_card_sha256": raw_hash,
            "publisher_declared_card_sha256": publisher_hash,
            "strict_crlf_to_lf_sha256": normalized_hash,
            "checksum_reconciliation": "exact_raw" if publisher_hash == raw_hash else "Windows_acquisition_CRLF_only_to_original_LF_text"}


def metadata_resources():
    value = {"available_ram_bytes": psutil.virtual_memory().available,
             "owned_rss_bytes": psutil.Process().memory_info().rss,
             "scope": "CSV_hash_PTS_metadata_only_no_image_imports"}
    if value["available_ram_bytes"] < MIN_METADATA_AVAILABLE_RAM_BYTES:
        raise RuntimeError(f"Insufficient headroom for bounded metadata-only work: {value}")
    return value


def evenly_spaced(values, count):
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return list(values)
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def allocate(groups, total, caps):
    quotas = {group: 0 for group in groups}
    while sum(quotas.values()) < total:
        progressed = False
        for group in groups:
            if quotas[group] < caps[group] and sum(quotas.values()) < total:
                quotas[group] += 1
                progressed = True
        if not progressed:
            raise ValueError("Insufficient explicit publisher-visible centers")
    return quotas


def select_positive(pool, count):
    pool = sorted(pool, key=lambda item: (item["clip"], item["frame_index"]))
    spatial = sorted(pool, key=lambda item: (item["publisher_center_xy_px"][1], item["clip"], item["frame_index"]))
    third = max(1, math.ceil(len(pool) / 3))
    motion = sorted(pool, key=lambda item: (-item["labeled_image_motion_proxy_px_per_second"], item["clip"], item["frame_index"]))
    strata = [("upper_image_far_proxy", spatial[:third]), ("lower_image_near_proxy", spatial[-third:]),
              ("fast_image_motion_blur_proxy", motion[:third])]
    chosen, seen = [], set()
    for name, values in strata:
        for item in evenly_spaced([item for item in values if (item["clip"], item["frame_index"]) not in seen], count // 4):
            chosen.append({**item, "sampling_stratum": name})
            seen.add((item["clip"], item["frame_index"]))
    chosen.extend({**item, "sampling_stratum": "temporally_spread_residual"} for item in evenly_spaced([item for item in pool if (item["clip"], item["frame_index"]) not in seen], count - len(chosen)))
    return sorted(chosen, key=lambda item: (item["clip"], item["frame_index"]))


def native_timeline(video, probe_directory):
    """Collect actual frame PTS through one bounded FFprobe metadata process.

    FFprobe may inspect codec frames internally. No image pixels are exported,
    loaded in Python, rendered, reviewed, or accepted by this metadata stage.
    """
    executable = os.getenv("TENNIS_FFPROBE") or shutil.which("ffprobe")
    if not executable:
        local = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Links/ffprobe.exe"
        executable = str(local) if local.is_file() else None
    if not executable:
        raise RuntimeError("Require existing FFprobe; no synthetic FPS-derived PTS")
    launch = metadata_resources()
    raw = probe_directory / f"{video.stem}_pts.json"
    error = probe_directory / f"{video.stem}_ffprobe.stderr.txt"
    audit = {"status": "RUNNING_METADATA_ONLY", "created_at": datetime.now(timezone.utc).isoformat(),
             "video_path": str(video), "launch": launch, "maximum_allowed_probe_rss_bytes": MAX_PROBE_RSS_BYTES,
             "peak_sampled_probe_rss_bytes": 0, "minimum_sampled_available_ram_bytes": launch["available_ram_bytes"],
             "source_pixels_exported": 0}
    audit_path = probe_directory / f"{video.stem}_probe_receipt.json"
    with raw.open("w", encoding="utf-8") as out, error.open("w", encoding="utf-8") as err:
        process = subprocess.Popen([executable, "-v", "error", "-threads", "1", "-select_streams", "v:0", "-show_entries",
                                    "frame=pts:stream=width,height,avg_frame_rate,time_base", "-of", "json", str(video)],
                                   stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        owned = psutil.Process(process.pid)
        audit.update(pid=process.pid, process_create_time=owned.create_time())
        try:
            while process.poll() is None:
                available = psutil.virtual_memory().available
                audit["minimum_sampled_available_ram_bytes"] = min(audit["minimum_sampled_available_ram_bytes"], available)
                try:
                    rss = owned.memory_info().rss
                    audit["peak_sampled_probe_rss_bytes"] = max(audit["peak_sampled_probe_rss_bytes"], rss)
                except psutil.NoSuchProcess:
                    break
                if rss > MAX_PROBE_RSS_BYTES or available < MIN_METADATA_AVAILABLE_RAM_BYTES:
                    process.terminate()
                    process.wait(timeout=10)
                    audit["status"] = "REFUSED_OWNED_METADATA_PROBE_RESOURCE_LIMIT"
                    raise RuntimeError("Bounded owned FFprobe metadata resource limit exceeded")
                time.sleep(.02)
            code = process.wait(timeout=10)
            if code:
                audit["status"] = "FAILED_METADATA_PROBE"
                raise RuntimeError(f"FFprobe metadata failed with code {code}; retained stderr")
            audit["status"] = "COMPLETED_PTS_METADATA_ONLY_NOT_RENDERED"
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            audit["return_code"] = process.returncode
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    value = json.loads(raw.read_text(encoding="utf-8"))
    return value["streams"][0], [int(frame["pts"]) for frame in value["frames"]], {
        "receipt_path": str(audit_path), "receipt_sha256": digest(audit_path),
        "raw_pts_path": str(raw), "raw_pts_sha256": digest(raw), **audit}
