"""Derive a bounded NKUST DEV acquisition inventory from retained metadata only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


requests = read_json("requests.json")
for request in requests:
    path = ROOT / request["saved"]
    assert request["status"] == 200
    assert path.stat().st_size == request["bytes"]
    assert digest(path) == request["sha256"]

folders = {item["id"]: item for item in read_json("folders_v4.json")}


def ancestry(folder_id: str):
    chain = []
    seen = set()
    while folder_id:
        assert folder_id not in seen
        seen.add(folder_id)
        folder = folders[folder_id]
        chain.append(folder)
        folder_id = folder.get("parent_id")
    return list(reversed(chain))


root_files = read_json("root_files_v4.json")
top_files = read_json("indoor_straight_top_files_v4.json")
side_files = read_json("indoor_straight_side_files_v4.json")
assert len(top_files) == len(side_files) == 99
selected = []
for records, filename, expected_names, expected_id in (
    (root_files, "Indoor Field - Straight Shot.csv", [], "9991b6fc-1eea-476e-83ac-a0af8ef424e5"),
    (top_files, "IST01.mp4", ["Indoor Field", "Straight Shot", "Top-View"], "d644feac-66b1-40dc-ac9e-c276db3cd831"),
    (side_files, "ISS01.mp4", ["Indoor Field", "Straight Shot", "Side-View"], "c8de595e-512a-46b4-8de8-4a0cfa9a8819"),
):
    matches = [item for item in records if item["filename"] == filename]
    assert len(matches) == 1
    item = matches[0]
    assert item["id"] == expected_id
    assert item["status"] == "COMPLETED"
    assert item["size"] == item["content_details"]["size"]
    chain = ancestry(item["folder_id"]) if item.get("folder_id") else []
    assert [folder["name"] for folder in chain] == expected_names
    selected.append({"record": item, "ancestry": chain})

bundle_path = ROOT / "publisher_bundle.js"
bundle = bundle_path.read_text(encoding="utf-8")
routes = []
for needle in ("folder_id=", "/folders/${n}"):
    offset = bundle.index(needle)
    begin = max(0, offset - 250)
    routes.append({
        "needle": needle,
        "character_offset": offset,
        "snippet": bundle[begin:begin + 800],
    })

receipt = {
    "schema_version": 1,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Official v4 metadata-only inventory of one indoor DEV reference CSV and matched straight-shot views",
    "dataset": {
        "id": "75m8vz7jr2",
        "version": 4,
        "doi": "10.17632/75m8vz7jr2.4",
        "publisher_page": "https://data.mendeley.com/datasets/75m8vz7jr2/4",
        "published": "2024-05-01",
        "license_observed_on_publisher_page": "CC BY 4.0",
        "attribution": ["Kalin Guanlun Lai", "Hsu-Chun Huang", "Shang-Yi Lin", "Wei-Ting Lin", "Kawuu Weicheng Lin"],
    },
    "metadata_request_accept": "application/vnd.mendeley-public-dataset.1+json",
    "requests": requests,
    "publisher_route_evidence": {
        "page_observed_script_url": "https://data.mendeley.com/datasets/bundle.js?8533f74fd5ffa7f3d44f",
        "bundle_sha256": digest(bundle_path),
        "route_excerpts": routes,
        "note": "Routes copied from the publisher's own linked client bundle; no file download/view URL was requested.",
    },
    "inventory_counts": {"root_file_records": len(root_files), "indoor_straight_top_records": len(top_files), "indoor_straight_side_records": len(side_files)},
    "selected_dev_assets": selected,
    "selected_publisher_total_bytes": sum(item["record"]["size"] for item in selected),
    "view_pairing_basis": {
        "publisher_instruction": "The v4 Steps to reproduce requires the same shot category and matching top/side view numbers.",
        "ancestry_verified": True,
        "numeric_suffix": "01",
        "csv_identifier_join": "UNRESOLVED: no row index, identifier or ordering equivalence inferred",
        "frame_exact_synchronization": "UNRESOLVED: matched clip numbering is insufficient",
    },
    "reference_readiness": {
        "baseline_identity": "UNRESOLVED",
        "sideline_identity_and_global_axes": "UNRESOLVED",
        "actual_indoor_court_dimensions": "UNRESOLVED",
        "line_edge_or_center_convention": "UNRESOLVED",
        "measurement_uncertainty": "UNQUANTIFIED in reviewed author text",
        "practice_landing_evaluation_available": False,
    },
    "bounds": {
        "metadata_and_javascript_only": True,
        "media_downloads": 0,
        "csv_content_downloads": 0,
        "video_frames_opened": 0,
        "outdoor_target_values_opened": 0,
        "model_inference": False,
        "training": False,
        "future_acquisition_scope": "Reserved indoor DEV assets only; body acquisition and convention review remain separate work",
    },
    "producer": {"path": "make_receipt.py", "sha256": digest(Path(__file__).resolve())},
}
assert receipt["selected_publisher_total_bytes"] == 13186692
out = ROOT / "inventory_receipt.json"
out.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"receipt": str(out), "sha256": digest(out), "selected_assets": len(selected), "publisher_total_bytes": receipt["selected_publisher_total_bytes"]}))
