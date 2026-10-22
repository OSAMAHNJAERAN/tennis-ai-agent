"""Read-only inventory of product cleanup candidates and available evidence.

Run from the repository root. Outputs a snapshot JSON; never removes files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "execution_baseline" / "2026-09-30" / "data_cleanup_inventory.json"


def footprint(path: Path) -> dict:
    size = 0
    count = 0
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(base) / name).is_symlink()]
        for name in files:
            p = Path(base) / name
            if not p.is_symlink():
                size += p.stat().st_size
                count += 1
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": size, "files": count}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    if "--check-temp" in sys.argv:
        print(json.dumps([dict(footprint(ROOT / name), exists=(ROOT / name).exists()) for name in ("outputs/product_regression_tmp", "outputs/product_regression_tmp2", "outputs/product_review_tests", "outputs/product_test_final", "outputs/product_test_tmp")], indent=2))
        return
    aggregate = [footprint(ROOT / name) for name in ("artifacts", "outputs", "data", "frontend")]
    candidates = []
    for name in ("artifacts", "outputs"):
        for path in sorted((ROOT / name).iterdir()):
            if path.is_dir():
                candidates.append(footprint(path))
    weights = []
    for name in ("yolo11m.pt", "yolo11s-pose.pt", "yolo11n-pose.pt", "yolov8x.pt", "yolov8n.pt", "models/keypoints_model.pth", "models/yolo5_last.pt", "artifacts/models/ball/wasb_tennis_best.pth.tar"):
        p = ROOT / name
        if p.is_file():
            weights.append({"path": name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifests = []
    external = ROOT / "data/external"
    for p in sorted(external.rglob("manifest.json")):
        item = {"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p)}
        try:
            manifest = json.loads(p.read_text(encoding="utf-8"))
            for field in ("source", "revision", "declared_license", "split", "selected_clips", "qualification_evidence", "source_broadcast_independence", "final_test_downloaded"):
                if field in manifest:
                    item[field] = manifest[field]
        except (ValueError, UnicodeError) as exc:
            item["parse_error"] = str(exc)
        manifests.append(item)
    videos = []
    for directory in ("data/holdout_videos", "data/validation_videos"):
        for p in sorted((ROOT / directory).glob("*")):
            if p.suffix.lower() in {".mp4", ".webm", ".avi", ".mov"}:
                videos.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})
    result = {"schema_version": "1", "snapshot_date": "2026-09-30", "scope": "Read-only path/size/hash inventory before cleanup; no deletion executed", "aggregate": aggregate, "candidate_directories": candidates, "weights": weights, "external_manifests": manifests, "available_diagnostic_videos": videos}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve the first audit snapshot: {OUTPUT}")
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "aggregate": aggregate, "candidate_directories": candidates, "weights": weights, "manifest_count": len(manifests), "diagnostic_videos": videos}, indent=2))


if __name__ == "__main__":
    main()
