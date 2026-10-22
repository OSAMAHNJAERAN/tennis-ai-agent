"""Correct a retained v2 suffix audit's doubles exclusions without rerunning metrics."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path

from scripts.data.acquire_livetennisapi_sample import digest_file
from scripts.evaluate.evaluate_publisher_snapshots import read_receipt, write_receipt


def corrected_audit(directory, old_import, protocol_path):
    old = read_receipt(old_import)
    protocol = read_receipt(protocol_path)
    if protocol["protocol_id"] != "livetennisapi-june2026-conditional-suffix-v2b-player-audit":
        raise ValueError("A frozen v2b player-join correction protocol is required")
    if protocol["parent_import_sha256"] != digest_file(old_import):
        raise ValueError("Parent import binding changed")
    acquisition = read_receipt(directory / "acquisition_receipt.json")
    if acquisition["status"] != "ACQUIRED_PINNED_PUBLIC_SAMPLE" or old["acquisition_sha256"] != digest_file(directory / "acquisition_receipt.json"):
        raise ValueError("Parent acquisition binding changed")
    for file in acquisition["files"]:
        if digest_file(directory / file["name"]) != file["sha256"]:
            raise ValueError("Publisher retained file hash mismatch")
    players, matches = {}, {}
    with gzip.open(directory / "players.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            players[row["player_id"]] = row
    with gzip.open(directory / "matches.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            matches[row["match_id"]] = row
    retained, rejected = [], []
    changed, player_doubles = [], []
    source = [{"match_id": m["match_id"], "score_rows": m["original_score_rows"], "reason": None} for m in old["matches"]] + old["excluded_groups"]
    for group in source:
        mid = group["match_id"]
        metadata = matches.get(mid, {})
        participants = [players.get(metadata.get(key)) for key in ("player1_id", "player2_id")]
        reason = group["reason"]
        # Preserve the v2 ordering: format and event status were checked first.
        if metadata.get("best_of") == "BO3" and not metadata.get("event_status"):
            if any(not p or not p.get("name", "").strip() for p in participants):
                reason = "missing_joined_player_metadata"
            elif any("/" in p["name"] for p in participants):
                reason = "doubles_from_joined_players"
                player_doubles.append(mid)
        if reason != group["reason"]:
            changed.append({"match_id": mid, "previous_reason": group["reason"], "corrected_reason": reason})
        if reason is None:
            retained.append(next(m for m in old["matches"] if m["match_id"] == mid))
        else:
            rejected.append({**group, "reason": reason})
    if len(source) != old["original_match_groups"] or len({g["match_id"] for g in source}) != len(source):
        raise ValueError("Parent audit does not account for every original group")
    return {**old, "schema_version": "tennis.publisher-conditional-player-correction.v1",
            "created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest_file(protocol_path),
            "parent_import_sha256": digest_file(old_import), "parent_protocol_sha256": old["protocol_sha256"],
            "eligible_match_groups": len(retained), "excluded_match_groups": len(rejected),
            "eligible_split_counts": dict(Counter(m["split"] for m in retained)),
            "selected_split_counts": dict(Counter(m["split"] for m in retained)),
            "exclusion_counts": dict(Counter(m["reason"] for m in rejected)),
            "joined_player_doubles_groups": len(player_doubles), "changed_group_exclusions": changed,
            "matches": retained, "excluded_groups": rejected,
            "forecast_metrics_executed": False,
            "limitations": "Corrects doubles/player metadata only. V2 requires inferred point transitions and lacks explicit event-family rule provenance; two retained suffix groups do not independently qualify forecasts. V3 uses direct states without point inference."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external/livetennisapi_public_sample_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/prediction/livetennisapi_2026_06_protocol_v2b.json"))
    parser.add_argument("--output", type=Path, default=Path("data/prediction/livetennisapi_2026_06_conditional_player_audit_v2b.json"))
    args = parser.parse_args()
    report = corrected_audit(args.input, args.input / "conditional_import_v2.json", args.protocol)
    write_receipt(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("matches", "excluded_groups", "changed_group_exclusions")}))


if __name__ == "__main__":
    main()
