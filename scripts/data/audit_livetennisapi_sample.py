"""Stream a publisher score inventory without running forecasts or inventing points."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from scripts.data.acquire_livetennisapi_sample import digest_file


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Publisher dictionary explicitly declares these naive capture fields UTC.
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def assign_split(timestamp, protocol):
    for name, interval in protocol["splits"].items():
        if utc(interval["scheduled_start_utc_inclusive"]) <= timestamp < utc(interval["scheduled_end_utc_exclusive"]):
            return name
    return None


def score_tuple(row):
    games1, games2 = json.loads(row["games_p1"]), json.loads(row["games_p2"])
    if (not isinstance(games1, list) or not isinstance(games2, list) or len(games1) != len(games2)
            or any(type(value) is not int or value < 0 for value in games1 + games2)):
        raise ValueError("Invalid per-set games array")
    points = tuple("AD" if row[key] in ("A", "AD") else row[key] for key in ("points_p1", "points_p2"))
    return int(row["sets_p1"]), int(row["sets_p2"]), tuple(games1), tuple(games2), *points, row["is_tiebreak"]


def audit(directory, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    acquisition = json.loads((directory / "acquisition_receipt.json").read_text(encoding="utf-8"))
    if acquisition["status"] != "ACQUIRED_PINNED_PUBLIC_SAMPLE" or acquisition["protocol_sha256"] != digest_file(protocol_path):
        raise ValueError("Acquisition does not bind this frozen protocol")
    for entry in acquisition["files"]:
        if digest_file(directory / entry["name"]) != entry["sha256"]:
            raise ValueError("Retained publisher file checksum mismatch")
    matches, metadata_count, headers = {}, 0, {}
    with gzip.open(directory / "matches.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers["matches"] = reader.fieldnames
        for row in reader:
            metadata_count += 1
            try:
                date = utc(row["scheduled_time_utc"])
            except (ValueError, TypeError):
                continue
            split = assign_split(date, protocol)
            if split:
                matches[row["match_id"]] = {**row, "split": split}
    summaries, rows, unregistered = {}, 0, 0
    with gzip.open(directory / "points_sample_2026-06.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers["points"] = reader.fieldnames
        for row in reader:
            rows += 1
            identifier = row["match_id"]
            if identifier not in summaries:
                summaries[identifier] = {"match_id": identifier, "rows": 0, "first": row, "last": row,
                                          "missing_server_rows": 0, "repeated_score_rows": 0,
                                          "capture_reversals": 0, "malformed_rows": 0, "outside_split_rows": 0}
            summary = summaries[identifier]
            summary["rows"] += 1
            if row["server"] not in ("1", "2"):
                summary["missing_server_rows"] += 1
            try:
                timestamp, score = utc(row["timestamp_utc"]), score_tuple(row)
                if "last_capture" in summary:
                    if timestamp < summary["last_capture"]:
                        summary["capture_reversals"] += 1
                    if score == summary["last_score"]:
                        summary["repeated_score_rows"] += 1
                summary["last_capture"], summary["last_score"] = timestamp, score
                match = matches.get(identifier)
                if match is not None and assign_split(timestamp, protocol) != match["split"]:
                    summary["outside_split_rows"] += 1
            except (ValueError, TypeError, KeyError):
                summary["malformed_rows"] += 1
            summary["last"] = row
            if identifier not in matches:
                unregistered += 1
    summary_counts = Counter()
    group_inventory = []
    for identifier, summary in summaries.items():
        match = matches.get(identifier)
        first, last = summary["first"], summary["last"]
        reasons = []
        if match is None:
            reasons.append("outside_scheduled_june_window_or_missing_metadata")
        else:
            if "double" in match["round"].lower() or "double" in match["tournament"].lower():
                reasons.append("doubles")
            if match["best_of"] not in ("BO3", "BO5"):
                reasons.append("unsupported_match_format")
            if match["event_status"]:
                reasons.append("nonstandard_terminal_status")
        try:
            first_score = score_tuple(first)
            is_zero = first_score[:2] == (0, 0) and not any(first_score[2] + first_score[3]) and first_score[4:6] == ("0", "0")
            if not is_zero:
                reasons.append("initial_score_not_zero")
        except (ValueError, TypeError):
            reasons.append("invalid_initial_score")
        if first["server"] not in ("1", "2"):
            reasons.append("initial_server_missing")
        if summary["missing_server_rows"]:
            reasons.append("missing_observed_server")
        for key in ("capture_reversals", "malformed_rows", "outside_split_rows"):
            if summary[key]:
                reasons.append(key)
        required_sets = 3 if match and match["best_of"] == "BO5" else 2
        try:
            final_sets = (int(last["sets_p1"]), int(last["sets_p2"]))
            if max(final_sets) != required_sets or min(final_sets) >= required_sets:
                reasons.append("no_terminal_match_score")
        except ValueError:
            reasons.append("invalid_terminal_score")
        # This publisher schema has no separate winner/result column. Do not
        # counterfeit an independent result by copying the last tape state.
        if not any(key in headers["matches"] for key in ("winner_id", "winner", "result")):
            reasons.append("independent_publisher_result_not_released")
        summary_counts.update(set(reasons))
        group_inventory.append({key: summary[key] for key in ("match_id", "rows", "missing_server_rows", "repeated_score_rows", "capture_reversals", "malformed_rows", "outside_split_rows")}
                               | {"split": match["split"] if match else None, "scheduled_time_utc": match["scheduled_time_utc"] if match else None,
                                  "initial_source_utc": first["timestamp_utc"], "final_source_utc": last["timestamp_utc"],
                                  "structural_exclusion_reasons": sorted(set(reasons)), "single_point_transitions_audited": False})
    return {"schema_version": "tennis.publisher-score-inventory.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "data_kind": "publisher_live_observed_score_states", "scope": "private_academic_statistical_inventory_only",
            "protocol_sha256": digest_file(protocol_path), "acquisition_sha256": digest_file(directory / "acquisition_receipt.json"),
            "headers": headers, "metadata_matches": metadata_count, "scheduled_june_matches": len(matches),
            "observed_score_rows": rows, "observed_match_groups": len(summaries), "rows_without_june_scheduled_metadata": unregistered,
            "split_group_counts": dict(Counter(item["split"] for item in group_inventory)),
            "exclusion_counts_overlapping": dict(summary_counts), "full_v1_eligible_matches": sum(not item["structural_exclusion_reasons"] for item in group_inventory),
            "vision_qualification_evidence": False, "forecast_evaluation_executed": False, "human_independent_review": False,
            "groups": sorted(group_inventory, key=lambda item: item["match_id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external/livetennisapi_public_sample_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/prediction/livetennisapi_2026_06_protocol_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Inventory versions are immutable")
    report = audit(args.input, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest_file(args.output) + "  " + args.output.name + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}))


if __name__ == "__main__":
    main()
