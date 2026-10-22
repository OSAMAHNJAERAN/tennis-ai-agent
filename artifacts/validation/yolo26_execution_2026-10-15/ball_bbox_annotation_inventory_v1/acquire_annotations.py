"""One bounded public Dryad annotation fetch; no media, model, or extraction."""
from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request
import zipfile

BASE = Path(__file__).resolve().parent
EXPECTED_SIZE = 2_377_146
EXPECTED_SHA256 = "e889a4badd6eac089f46cf249e054c82bb66908dc4a715cfa14c973d668c279f"
URL = "https://datadryad.org/api/v2/files/4736299/download"


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong)
        for name in ("total_phys", "available_phys", "total_page", "available_page", "total_virtual", "available_virtual", "available_extended")
    ]


def admission() -> dict:
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise ctypes.WinError()
    value = {"available_ram_bytes": memory.available_phys, "free_disk_bytes": shutil.disk_usage(BASE).free}
    if value["available_ram_bytes"] < 128 * 1024**2 or value["free_disk_bytes"] < 512 * 1024**2:
        raise RuntimeError(f"metadata-only admission refused: {value}")
    return value


def main() -> int:
    receipt = {"schema_version": 1, "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "url": URL, "expected_bytes": EXPECTED_SIZE, "expected_sha256": EXPECTED_SHA256,
               "maximum_response_bytes": 4 * 1024**2, "attempts": 1,
               "images_or_videos_acquired": 0, "models_imported": False}
    try:
        receipt["launch"] = admission()
        digest = hashlib.sha256()
        count = 0
        request = urllib.request.Request(URL, headers={"User-Agent": "TennisAcademicDatasetAudit/1.0", "Accept": "application/zip, application/octet-stream"})
        with urllib.request.urlopen(request, timeout=30) as response, (BASE / "tennis_download.payload").open("xb") as target:
            receipt.update(http_status=response.status, final_url=response.url, content_type=response.headers.get("Content-Type"))
            while chunk := response.read(64 * 1024):
                count += len(chunk)
                if count > receipt["maximum_response_bytes"]:
                    raise ValueError("response exceeds fixed annotation ceiling")
                target.write(chunk)
                digest.update(chunk)
        receipt.update(actual_bytes=count, actual_sha256=digest.hexdigest())
        if count != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError("payload differs from publisher exact size/hash; rejected, preserved without archive admission")
        archive = BASE / "tennis_download.payload"
        with zipfile.ZipFile(archive) as opened:
            members = opened.infolist()
            directory = [{"name": row.filename, "bytes": row.file_size, "compressed_bytes": row.compress_size,
                          "crc32": f"{row.CRC:08x}", "method": row.compress_type, "flags": row.flag_bits}
                         for row in members]
            if len(members) > 100_000 or sum(row.file_size for row in members) > 64 * 1024**2:
                raise ValueError("archive metadata exceeds fixed member or inflation ceiling")
            (BASE / "archive_directory.json").write_text(json.dumps(directory, indent=2), encoding="utf-8")
            receipt.update(archive_members=len(members), declared_uncompressed_bytes=sum(row.file_size for row in members),
                           status="EXACT_PUBLISHER_ANNOTATION_ARCHIVE_ACQUIRED_NOT_EXTRACTED")
        print(json.dumps({"receipt": receipt, "first_members": directory[:12]}, indent=2))
        return 0
    except Exception as exc:
        receipt.update(status="ANNOTATION_ARCHIVE_NOT_ADMITTED", error=type(exc).__name__, detail=str(exc))
        print(json.dumps(receipt, indent=2))
        return 1
    finally:
        with (BASE / "annotation_acquisition_receipt.json").open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
