"""Verify or restore the supplemental project files stored in project_snapshot/."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("project_snapshot"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    index = json.loads((snapshot / "index.json").read_text(encoding="utf-8"))
    manifest = snapshot / "manifest.jsonl.gz"
    if digest(manifest) != index["manifest_sha256"]:
        raise ValueError("Manifest SHA-256 mismatch; run git lfs pull first.")
    for part in index["parts"]:
        path = snapshot / part["name"]
        if path.name != part["name"] or digest(path) != part["sha256"]:
            raise ValueError(f"Archive SHA-256 mismatch: {part['name']}")
    print(f"Verified {len(index['parts'])} archive parts.", flush=True)

    destination = args.destination.resolve() if args.destination else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    archives: dict[str, zipfile.ZipFile] = {}
    verified: set[tuple[str, str]] = set()
    part_names = {part["name"] for part in index["parts"]}
    count = 0
    try:
        with gzip.open(manifest, "rt", encoding="utf-8") as rows:
            for line in rows:
                row = json.loads(line)
                relative = PurePosixPath(row["path"])
                if relative.is_absolute() or ".." in relative.parts or ":" in row["path"]:
                    raise ValueError(f"Unsafe manifest path: {row['path']}")
                part_name, sha = row["part"], row["sha256"]
                if part_name not in part_names:
                    raise ValueError(f"Unknown archive: {part_name}")
                if part_name not in archives:
                    archives[part_name] = zipfile.ZipFile(snapshot / part_name)
                archive = archives[part_name]
                member = f"objects/{sha}"
                if archive.getinfo(member).file_size != row["size"]:
                    raise ValueError(f"Size mismatch: {row['path']}")
                key = (part_name, sha)
                if key not in verified:
                    with archive.open(member) as stream:
                        actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    if actual != sha:
                        raise ValueError(f"Object SHA-256 mismatch: {row['path']}")
                    verified.add(key)
                if destination:
                    target = (destination / Path(*relative.parts)).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError(f"Path escapes destination: {row['path']}")
                    if target.exists() and not args.overwrite:
                        if target.is_file() and digest(target) == sha:
                            count += 1
                            continue
                        raise FileExistsError(f"Use --overwrite to replace {target}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    handle, temporary = tempfile.mkstemp(dir=target.parent)
                    try:
                        with os.fdopen(handle, "wb") as output, archive.open(member) as source:
                            shutil.copyfileobj(source, output, 1024 * 1024)
                        if digest(Path(temporary)) != sha:
                            raise ValueError(f"Restored SHA-256 mismatch: {row['path']}")
                        os.replace(temporary, target)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                count += 1
                if count % 25000 == 0:
                    print(f"Processed {count:,} manifest files.", flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    if count != index["file_count"] or len(verified) != index["unique_objects"]:
        raise ValueError("Snapshot object or file count mismatch.")
    print(f"{'Restored' if destination else 'Verified'} {count:,} files ({len(verified):,} unique objects).")


if __name__ == "__main__":
    main()
