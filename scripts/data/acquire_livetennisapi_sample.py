"""Acquire only the pinned public academic score sample after protocol freeze."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


RECORD_URL = "https://zenodo.org/api/records/22048731"
EXPECTED = {
    "README.md": (3229, "bce7ac58772bb0730c7f9b1716e423fa"),
    "DATA_DICTIONARY.md": (2363, "acf482ddc8d2f67e07d796a6aef9c68f"),
    "points_sample_2026-06.csv.gz": (9989821, "f0df31a54a7bc87f1ce46c167e1a3c3e"),
    "matches.csv.gz": (2496391, "7434c59a3e74f2dca5310e51fb3236ac"),
    "players.csv.gz": (452561, "6dab301c38ab44a18fe949dab4aa1018"),
}


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(output, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    started = datetime.now(timezone.utc)
    frozen = datetime.fromisoformat(protocol["frozen_at"].replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen >= started or protocol.get("freeze_stage") != "before_public_sample_acquisition_or_score_row_inspection":
        raise ValueError("A dated pre-acquisition protocol is required")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "acquisition_receipt.json").exists():
        raise ValueError("Acquisition receipts are immutable; select a new version")
    receipt = {"schema_version": "tennis.publisher-acquisition.v1", "publisher_record": RECORD_URL,
               "record_id": 22048731, "publisher_version": "1.0.0", "license": "CC-BY-NC-4.0",
               "usage": "private_noncommercial_academic_research", "raw_redistribution": False,
               "vision_qualification_evidence": False, "scores_inspected": False, "evaluations_executed": False,
               "protocol_path": str(protocol_path), "protocol_sha256": digest_file(protocol_path),
               "started_at": started.isoformat(), "files": [], "status": "INCOMPLETE"}
    try:
        with requests.Session() as session:
            session.headers["User-Agent"] = "T88J709-private-academic-acquisition/1.0"
            response = session.get(RECORD_URL, timeout=(15, 45))
            response.raise_for_status()
            if len(response.content) > 1024 * 1024:
                raise ValueError("Publisher metadata exceeded acquisition bound")
            metadata = response.json()
            if metadata["metadata"]["license"]["id"] != "cc-by-nc-4.0" or metadata["metadata"]["version"] != "1.0.0":
                raise ValueError("Publisher license/version changed")
            files = {entry["key"]: entry for entry in metadata["files"]}
            for name, (size, md5) in EXPECTED.items():
                if files[name]["size"] != size or files[name]["checksum"] != "md5:" + md5:
                    raise ValueError("Pinned publisher sample size/checksum changed")
            (output / "publisher_metadata.json").write_bytes(response.content)
            receipt["publisher_metadata_sha256"] = hashlib.sha256(response.content).hexdigest()
            receipt["publisher_revision"] = metadata["revision"]
            for name, (size, md5) in EXPECTED.items():
                destination = output / name
                source_url = files[name]["links"]["self"]
                expected_url = f"https://zenodo.org/api/records/22048731/files/{name}/content"
                if source_url != expected_url:
                    raise ValueError("Publisher file URL differs from the pinned record")
                timestamp = datetime.now(timezone.utc).isoformat()
                if destination.exists():
                    if destination.stat().st_size != size or digest_file(destination, "md5") != md5:
                        raise ValueError("Retained partial source mismatches publisher; choose a new acquisition directory")
                    reused = True
                else:
                    reused = False
                    count = 0
                    with session.get(source_url, stream=True, timeout=(15, 45)) as download:
                        download.raise_for_status()
                        with destination.open("xb") as stream:
                            for block in download.iter_content(1024 * 1024):
                                count += len(block)
                                if count > size:
                                    raise ValueError("Publisher transfer exceeds declared byte bound")
                                stream.write(block)
                    if count != size or digest_file(destination, "md5") != md5:
                        raise ValueError("Publisher size/checksum failed")
                entry = {"path": str(destination), "name": name, "source_url": source_url, "bytes": size,
                         "sha256": digest_file(destination), "publisher_checksum": "md5:" + md5,
                         "publisher_checksum_verified": True, "acquired_at": timestamp, "reused_retained_file": reused}
                receipt["files"].append(entry)
                print(json.dumps({"acquired": name, "bytes": size, "sha256": entry["sha256"]}), flush=True)
        receipt["status"] = "ACQUIRED_PINNED_PUBLIC_SAMPLE"
    except (requests.RequestException, OSError, KeyError, ValueError) as exc:
        receipt["failure"] = str(exc)
    receipt["completed_at"] = datetime.now(timezone.utc).isoformat()
    receipt["total_downloaded_file_bytes"] = sum(entry["bytes"] for entry in receipt["files"])
    path = output / ("acquisition_receipt.json" if receipt["status"] == "ACQUIRED_PINNED_PUBLIC_SAMPLE" else "failed_acquisition_" + started.strftime("%Y%m%dT%H%M%S") + ".json")
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(digest_file(path) + "  " + path.name + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/external/livetennisapi_public_sample_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/prediction/livetennisapi_2026_06_protocol_v1.json"))
    args = parser.parse_args()
    receipt = acquire(args.output, args.protocol)
    print(json.dumps({"status": receipt["status"], "bytes": receipt["total_downloaded_file_bytes"], "failure": receipt.get("failure")}))
    return 0 if receipt["status"] == "ACQUIRED_PINNED_PUBLIC_SAMPLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
