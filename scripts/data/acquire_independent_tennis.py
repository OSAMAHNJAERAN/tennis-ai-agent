"""Bounded publisher acquisition; inspection never runs a model or displays frames.

The default MUVY inventory reads ZIP metadata using exact HTTP byte ranges. A
selected member is saved only after a pre-existing original-match reservation.
Publisher archive checksums are recorded, never claimed verified by ranges.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[2]
RECORD_ID = 13883315
ARCHIVE_NAME = "MUVY_v01.zip"
ARCHIVE_BYTES = 7758643869
ARCHIVE_MD5 = "607051346013728c1e14dcdd94f319c1"
RECORD_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files/{ARCHIVE_NAME}/content"
PUBLISHER_PAGE = "https://zenodo.org/records/13883315"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_destination(output: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or any(":" in p for p in path.parts):
        raise ValueError("Unsafe publisher member path")
    target = output.joinpath(*path.parts).resolve()
    if not target.is_relative_to(output.resolve()):
        raise ValueError("Member leaves acquisition directory")
    return target


class RangeReader(io.RawIOBase):
    """Seekable read-only ZIP stream with a cumulative network transfer budget."""

    def __init__(self, session, *, url: str = ARCHIVE_URL, size: int = ARCHIVE_BYTES,
                 budget: int = 32 * 1024**2):
        self.session, self.url, self.size, self.budget = session, url, size, budget
        self.position = 0
        self.received = 0
        self.ranges: list[dict] = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else (
            self.position + offset if whence == io.SEEK_CUR else self.size + offset)
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or not 0 <= position <= self.size:
            raise ValueError("Invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if not size:
            return b""
        if self.received + size > self.budget:
            raise ValueError("Archive read exceeds cumulative byte budget")
        chunks = []
        remaining = size
        while remaining:
            count = min(remaining, 1024**2)
            start, end = self.position, self.position + count - 1
            with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                  "Accept-Encoding": "identity"}, stream=True,
                                  timeout=(15, 45)) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                    raise ValueError("Publisher did not honor the exact bounded range")
                block = response.raw.read(count + 1)
                if len(block) != count:
                    raise ValueError("Truncated or oversized publisher range")
            self.ranges.append({"start": start, "end": end, "bytes": count, "sha256": sha256(block)})
            self.received += count
            self.position += count
            remaining -= count
            chunks.append(block)
        return b"".join(chunks)


def check_record(record: dict):
    if record.get("id") != RECORD_ID or record.get("metadata", {}).get("license", {}).get("id") != "cc-by-4.0":
        raise ValueError("Publisher record or dataset license differs from rights review")
    items = [f for f in record.get("files", []) if f.get("key") == ARCHIVE_NAME]
    if len(items) != 1 or items[0].get("size") != ARCHIVE_BYTES or items[0].get("checksum") != "md5:" + ARCHIVE_MD5:
        raise ValueError("Publisher archive differs from pinned revision")


def check_reservation(path: Path, name: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("qualification_evidence") is not False or document.get("reserved_before_visual_review") is not True:
        raise ValueError("Require an explicit pre-review candidate reservation, not a qualification claim")
    groups = [group for group in document.get("groups", []) if name in group.get("members", [])]
    if len(groups) != 1 or not groups[0].get("original_match_id") or not groups[0].get("source_video_ids"):
        raise ValueError("Member lacks one original-match group with full source IDs")
    if groups[0].get("split") not in ("validation_candidate", "frozen_test_candidate", "outside_scope_negative"):
        raise ValueError("Unknown candidate reservation split")
    if groups[0].get("split") != "outside_scope_negative" and (
            groups[0].get("original_match_previously_used_for_development") is not False or
            not groups[0].get("original_match_overlap_audit")):
        raise ValueError("Held-out content requires an untouched original-match overlap audit")
    return groups[0]


def inspect_pts(path: Path) -> dict:
    """Decode source timestamps only. Never derives frame time from nominal FPS."""
    import av
    import numpy as np
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stamps = [float(frame.pts * frame.time_base) if frame.pts is not None else None
                  for frame in container.decode(stream)]
        finite = [t for t in stamps if t is not None]
        differences = np.diff(finite)
        return {"width": stream.width, "height": stream.height,
                "source_time_base": str(stream.time_base), "nominal_average_rate": str(stream.average_rate),
                "decoded_frames": len(stamps), "missing_pts": len(stamps) - len(finite),
                "pts_seconds": stamps,
                "pts_strictly_increasing": bool(len(finite) == len(stamps) and (differences > 0).all()),
                "duration_between_first_last_pts_seconds": finite[-1] - finite[0] if len(finite) > 1 else None,
                "median_pts_delta_seconds": float(np.median(differences)) if len(differences) else None,
                "view_eligibility": "not_visually_reviewed", "annotation_status": "not_independently_labeled"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member", help="Exact inventoried tennis archive member; omitted lists names only")
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--inspect-pts", action="store_true", help="Inspect saved video PTS without displaying images")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Output must remain inside this workspace")
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(RECORD_URL, timeout=30)
    response.raise_for_status()
    record = response.json()
    check_record(record)
    publisher_bytes = json.dumps(record, indent=2).encode("utf-8")
    (output / "publisher_record.json").write_bytes(publisher_bytes)
    receipt = {"acquired_at": now(), "source_record": PUBLISHER_PAGE, "license": "CC-BY-4.0",
               "rights_scope": "Publisher dataset grant; original uploader attribution must be retained",
               "qualification_evidence": False, "whole_archive_checksum_verified": False,
               "publisher_archive_md5": ARCHIVE_MD5, "publisher_record_sha256": sha256(publisher_bytes),
               "script_sha256": sha256(Path(__file__).read_bytes()), "complete": False, "files": []}
    reader = RangeReader(session, budget=64 * 1024**2)
    try:
        with zipfile.ZipFile(reader) as archive:
            entries = [info for info in archive.infolist() if re.search("tennis", info.filename, re.IGNORECASE)]
            receipt["tennis_inventory"] = [{"name": i.filename, "bytes": i.file_size,
                "compressed_bytes": i.compress_size, "crc32": i.CRC} for i in entries]
            if args.member:
                if not args.reservation:
                    raise ValueError("Selected content requires --reservation created before review")
                group = check_reservation(args.reservation, args.member)
                matches = [i for i in entries if i.filename == args.member]
                if len(matches) != 1 or matches[0].is_dir() or not 0 < matches[0].file_size <= 20 * 1024**2:
                    raise ValueError("Member is absent, duplicate, a directory, or above 20 MiB")
                target = safe_destination(output, args.member)
                if target.exists():
                    raise FileExistsError(target)
                with archive.open(matches[0]) as stream:
                    content = stream.read(matches[0].file_size + 1)
                if len(content) != matches[0].file_size:
                    raise ValueError("Publisher member differs from pinned size")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                file_record = {"member": args.member, "bytes": len(content), "sha256": sha256(content),
                               "publisher_zip_crc32_verified": True, "group": group,
                               "reservation_sha256": sha256(args.reservation.read_bytes())}
                if args.inspect_pts and target.suffix.lower() in (".mp4", ".mov", ".avi"):
                    file_record["video_metadata"] = inspect_pts(target)
                receipt["files"].append(file_record)
            receipt["complete"] = True
    finally:
        receipt["ranges"] = reader.ranges
        receipt["transferred_bytes"] = reader.received
        (output / "acquisition_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": receipt["complete"], "tennis_members": len(receipt.get("tennis_inventory", [])),
                      "transferred_bytes": reader.received, "saved_files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
