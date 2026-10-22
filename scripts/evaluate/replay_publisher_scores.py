"""Private causal replay of observed score suffixes; never assert human qualification."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import gzip
from itertools import groupby
import json
from pathlib import Path
import random

from scripts.data.acquire_livetennisapi_sample import digest_file
from scripts.data.audit_livetennisapi_sample import assign_split, score_tuple, utc
from src.prediction.evaluation import _summarize, _quantile
from src.prediction.tennis import TennisWinProbability
from src.scoring.match_state import (BallPlayState, CompletedSetScore, MatchFormat,
                                    MatchState, PointState, ServerCourtEnd,
                                    ServiceSide, SetFormat)
from src.scoring.scoring_rules import TennisScoringRules


class TapeRejected(ValueError):
    pass


def _integer(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as exc:
        raise TapeRejected("malformed_score") from exc
    if result < 0:
        raise TapeRejected("negative_score")
    return result


def _raw(row):
    try:
        signature = score_tuple(row)
        if row["is_tiebreak"] not in ("True", "False"):
            raise TapeRejected("undocumented_tiebreak_flag")
        if any(value < 0 for value in signature[:2]):
            raise TapeRejected("negative_set_score")
        return (*signature[:6], row["is_tiebreak"] == "True")
    except (ValueError, TypeError, KeyError) as exc:
        raise TapeRejected("malformed_score") from exc


def _signature(state):
    first = [item.games_p1 for item in state.completed_sets]
    second = [item.games_p2 for item in state.completed_sets]
    if not state.match_complete:
        first.append(state.games_p1)
        second.append(state.games_p2)
    return (state.sets_won_p1, state.sets_won_p2, tuple(first), tuple(second),
            state.get_points_display_p1(), state.get_points_display_p2(), state.tie_break_active)


def _matches(state, row):
    observed = _raw(row)
    expected = _signature(state)
    if state.match_complete:
        # A finished feed can clear point/serve indicators; they are not used
        # for forecasts, and full completed set scores still match exactly.
        return expected[:4] == observed[:4]
    return expected == observed and row["server"] == str(state.server_id)


def initial_observed_state(metadata, row):
    if metadata.get("best_of") != "BO3":
        raise TapeRejected("unsupported_match_format")
    if metadata.get("event_status"):
        raise TapeRejected("nonstandard_terminal_status")
    if ("double" in metadata.get("round", "").lower() or "double" in metadata.get("tournament", "").lower()
            or not metadata.get("player1_id") or not metadata.get("player2_id") or metadata["player1_id"] == metadata["player2_id"]):
        raise TapeRejected("doubles_or_invalid_participants")
    if row.get("server") not in ("1", "2"):
        raise TapeRejected("initial_server_missing")
    score = _raw(row)
    sets1, sets2, first, second, point1, point2, tiebreak = score
    if max(sets1, sets2) >= 2:
        raise TapeRejected("initial_score_already_terminal")
    if not first and not second and sets1 == sets2 == 0:
        first, second = (0,), (0,)
    if len(first) != sets1 + sets2 + 1:
        raise TapeRejected("initial_set_arrays_inconsistent")
    completed, winners = [], Counter()
    for index, (a, b) in enumerate(zip(first[:-1], second[:-1])):
        legal = (max(a, b) == 6 and min(a, b) <= 4) or (max(a, b) == 7 and min(a, b) in (5, 6))
        if not legal:
            raise TapeRejected("unsupported_completed_set")
        winner = 1 if a > b else 2
        winners[winner] += 1
        completed.append(CompletedSetScore(index + 1, a, b, winner_id=winner))
    if (winners[1], winners[2]) != (sets1, sets2):
        raise TapeRejected("initial_completed_sets_disagree")
    server = int(row["server"])
    state = MatchState(match_id=str(metadata["match_id"]), format=MatchFormat.BEST_OF_3, set_format=SetFormat.TIE_BREAK_SET,
                       server_id=server, receiver_id=3-server, initial_server_id=server,
                       sets_won_p1=sets1, sets_won_p2=sets2, completed_sets=completed,
                       current_set=sets1+sets2+1, games_p1=first[-1], games_p2=second[-1],
                       current_game=first[-1]+second[-1]+1, score_source="publisher_observed_rule_checked_not_human_reviewed")
    if tiebreak:
        if state.current_set == 3:
            raise TapeRejected("undocumented_deciding_set_tiebreak")
        state.tie_break_active = True
        state.tie_break_points_p1, state.tie_break_points_p2 = _integer(point1), _integer(point2)
        block = (state.tie_break_points_p1 + state.tie_break_points_p2 + 1) // 2
        state.initial_server_id = server if block % 2 == 0 else 3-server
    else:
        display = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        if point1 not in display or point2 not in display:
            raise TapeRejected("undocumented_standard_point_score")
        state.points_p1, state.points_p2 = display[point1], display[point2]
    if TennisScoringRules.validate_invariants(state):
        raise TapeRejected("invalid_initial_rules_state")
    if not _matches(state, {**row, "games_p1": json.dumps(first), "games_p2": json.dumps(second)}):
        raise TapeRejected("initial_state_not_roundtrip_consistent")
    return state


def parse_tape(metadata, rows, protocol):
    """No forecasts here: eligibility/selection must finish before performance."""
    if not rows:
        raise TapeRejected("empty_tape")
    state = initial_observed_state(metadata, rows[0])
    initial = state.to_dict()
    initial_time = previous_time = utc(rows[0]["timestamp_utc"])
    split = metadata["split"]
    if assign_split(initial_time, protocol) != split:
        raise TapeRejected("capture_crosses_reserved_split")
    points, duplicates, original_rows = [], 0, len(rows)
    for source_index, row in enumerate(rows[1:], 1):
        timestamp = utc(row["timestamp_utc"])
        if timestamp < previous_time:
            raise TapeRejected("capture_clock_reversed")
        if assign_split(timestamp, protocol) != split:
            raise TapeRejected("capture_crosses_reserved_split")
        previous_time = timestamp
        if _matches(state, row):
            duplicates += 1
            continue
        if state.match_complete:
            raise TapeRejected("score_changed_after_terminal_result")
        candidates = [(winner, TennisScoringRules.award_point(state, winner, "Observed publisher transition")) for winner in (1, 2)]
        valid = [(winner, candidate) for winner, candidate in candidates if _matches(candidate, row)]
        if len(valid) != 1:
            raise TapeRejected("skipped_ambiguous_or_invalid_point_transition")
        winner, candidate = valid[0]
        if candidate.current_set == 3 and candidate.tie_break_active:
            raise TapeRejected("undocumented_deciding_set_tiebreak")
        points.append({"point_id": f"publisher-{metadata['match_id']}-{source_index}", "winner": winner,
                       "server": state.server_id, "timestamp_seconds": (timestamp-initial_time).total_seconds(),
                       "source_utc": row["timestamp_utc"], "source_row_index": source_index,
                       "provenance": "publisher_observed_unique_rules_transition_not_human_reviewed"})
        state = candidate
    if not state.match_complete or state.winner_id not in (1, 2):
        raise TapeRejected("no_rules_complete_terminal_score")
    if not points:
        raise TapeRejected("no_causal_point_transitions")
    return {"match_id": metadata["match_id"], "source_group_id": "livetennisapi-match-" + metadata["match_id"],
            "scheduled_time_utc": metadata["scheduled_time_utc"], "split": split,
            "initial_state": initial, "initial_source_utc": rows[0]["timestamp_utc"], "points": points,
            "outcome": {"winner_id": state.winner_id, "provenance": "single_publisher_observed_terminal_score_not_independent_human_result"},
            "original_score_rows": original_rows, "duplicate_score_rows": duplicates, "initial_historical_points_unknown": True}


def import_tapes(directory, protocol_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "livetennisapi-june2026-conditional-suffix-v2":
        raise ValueError("Conditional replay requires the explicitly amended frozen v2 protocol")
    if utc(protocol["frozen_at"]) >= datetime.now(timezone.utc):
        raise ValueError("Protocol must predate import/evaluation")
    acquisition = json.loads((directory / "acquisition_receipt.json").read_text())
    if acquisition["status"] != "ACQUIRED_PINNED_PUBLIC_SAMPLE":
        raise ValueError("Publisher acquisition is incomplete")
    for file in acquisition["files"]:
        if digest_file(directory / file["name"]) != file["sha256"]:
            raise ValueError("Publisher content hash mismatch")
    metadata = {}
    with gzip.open(directory / "matches.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                split = assign_split(utc(row["scheduled_time_utc"]), protocol)
            except (ValueError, TypeError):
                continue
            if split:
                metadata[row["match_id"]] = {**row, "split": split}
    eligible, exclusions, seen, rows_count, group_count = [], Counter(), set(), 0, 0
    exclusion_groups = []
    with gzip.open(directory / "points_sample_2026-06.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for identifier, iterator in groupby(csv.DictReader(stream), key=lambda row: row["match_id"]):
            if identifier in seen:
                raise ValueError("Publisher tapes are interleaved; preserve capture order in a versioned indexed import")
            seen.add(identifier)
            rows = []
            for row in iterator:
                rows_count += 1
                if len(rows) >= 8192:
                    raise ValueError("Score tape exceeds bounded import size")
                rows.append(row)
            group_count += 1
            try:
                if identifier not in metadata:
                    raise TapeRejected("missing_reserved_metadata")
                eligible.append(parse_tape(metadata[identifier], rows, protocol))
            except (TapeRejected, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
                exclusions[reason] += 1
                exclusion_groups.append({"match_id": identifier, "reason": reason, "score_rows": len(rows)})
    selected = []
    for split, interval in protocol["splits"].items():
        candidates = sorted((item for item in eligible if item["split"] == split), key=lambda item: (utc(item["scheduled_time_utc"]), item["match_id"]))
        selected.extend(candidates[:interval["maximum_evaluation_matches"]])
    return {"schema_version": "tennis.publisher-conditional-import.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest_file(protocol_path), "acquisition_sha256": digest_file(directory / "acquisition_receipt.json"),
            "data_kind": "real_publisher_observed_unambiguous_score_suffixes", "human_independent_review": False,
            "qualification_status": "UNQUALIFIED_PRIVATE_PUBLISHER_SCORE_REPLAY", "score_rows_audited": rows_count,
            "original_match_groups": group_count, "eligible_match_groups": len(eligible), "excluded_match_groups": len(exclusion_groups),
            "exclusion_counts": dict(exclusions), "eligible_split_counts": dict(Counter(item["split"] for item in eligible)),
            "selected_split_counts": dict(Counter(item["split"] for item in selected)),
            "selection": "frozen earliest UTC scheduled timestamp, then match ID; no performance-based selection",
            "forecast_metrics_executed": False, "matches": selected, "excluded_groups": exclusion_groups}


def _restore_initial(values):
    # to_dict includes presentation/derived fields; retain dataclass fields only.
    from dataclasses import fields
    allowed = {item.name for item in fields(MatchState)}
    data = {key: value for key, value in values.items() if key in allowed}
    data["format"], data["set_format"] = MatchFormat(data["format"]), SetFormat(data["set_format"])
    data["completed_sets"] = [CompletedSetScore(**item) for item in data["completed_sets"]]
    for player in (1, 2):
        observed = values[f"player_{player}"]
        data[f"player_{player}_id"] = observed["id"]
        data[f"sets_won_p{player}"] = observed["sets_won"]
        data[f"games_p{player}"] = observed["games"]
        key = f"tie_break_points_p{player}" if data["tie_break_active"] else f"points_p{player}"
        data[key] = observed["points_raw"]
    for key, enum in (("service_side", ServiceSide), ("point_state", PointState),
                      ("ball_state", BallPlayState), ("player_1_court_end", ServerCourtEnd)):
        data[key] = enum(data[key])
    if data.get("deciding_set_format"):
        data["deciding_set_format"] = SetFormat(data["deciding_set_format"])
    state = MatchState(**data)
    if TennisScoringRules.validate_invariants(state) or state.to_dict() != values:
        raise ValueError("Imported initial observed state failed lossless restoration")
    return state


def _forecast_match(match):
    """Build causal prefixes without passing terminal labels to either engine."""
    state = _restore_initial(match["initial_state"])
    statistical, baseline = TennisWinProbability(), TennisWinProbability()
    snapshots = []

    def snapshot(timestamp, point_id):
        forecast = statistical.snapshot(state, timestamp_seconds=timestamp)
        reference = baseline.snapshot(state, timestamp_seconds=timestamp)
        if forecast["status"] not in ("prior_only", "uncalibrated") or reference["status"] != "prior_only":
            raise ValueError("Nonterminal publisher forecasts cannot be silently removed")
        snapshots.append({"match_id": match["match_id"], "point_id": point_id, "timestamp_seconds": timestamp,
                          "probability": forecast["probabilities"]["match"]["player_1"],
                          "baseline_probability": reference["probabilities"]["match"]["player_1"],
                          "past_observed_points": forecast["sample_count"],
                          "source_utc": match["initial_source_utc"] if not timestamp else current_source_utc,
                          "actual_winner_p1": int(match["outcome"]["winner_id"] == 1)})

    snapshot(0., "initial-publisher-score")
    for point in match["points"]:
        if state.server_id != point["server"]:
            raise ValueError("Imported serving order changed before replay")
        statistical.observe_point(point_id=point["point_id"], server_id=state.server_id,
                                  winner_id=point["winner"], timestamp_seconds=point["timestamp_seconds"])
        state = TennisScoringRules.award_point(state, point["winner"], "Publisher observed transition, not human qualification")
        current_source_utc = point["source_utc"]
        if not state.match_complete:
            snapshot(point["timestamp_seconds"], point["point_id"])
    if not state.match_complete or state.winner_id != match["outcome"]["winner_id"]:
        raise ValueError("Publisher replay failed to preserve the recorded terminal outcome")
    return {"match_id": match["match_id"], "split": match["split"], "snapshots": snapshots}


def evaluate_import(imported, protocol, *, selected_split=None):
    records = []
    for match in imported["matches"]:
        if selected_split is None or match["split"] == selected_split:
            records.append(_forecast_match(match))
    splits, rng = {}, random.Random(protocol["uncertainty"]["seed"])
    for split in protocol["splits"]:
        if selected_split is not None and split != selected_split:
            continue
        matches = [item for item in records if item["split"] == split]
        if not matches:
            splits[split] = {"match_count": 0, "metrics": None, "reason": "no_eligible_observed_suffixes"}
            continue
        metrics = _summarize(matches, 10)
        intervals = {}
        if len(matches) >= 2:
            samples = [_summarize([rng.choice(matches) for _ in matches], 10) for _ in range(protocol["uncertainty"]["bootstrap_resamples"])]
            for model in ("statistical", "score_only_baseline", "improvement_against_baseline"):
                intervals[model] = {metric: [_quantile([item[model][metric] for item in samples], q) for q in (.025, .975)] for metric in ("brier", "log_loss")}
        splits[split] = {"match_count": len(matches), "snapshot_count": sum(len(item["snapshots"]) for item in matches),
                         "metrics": metrics, "match_bootstrap_95_percent_intervals": intervals or None}
    return {"schema_version": "tennis.publisher-conditional-evaluation.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "data_kind": imported["data_kind"], "qualification_status": "UNQUALIFIED_PRIVATE_PUBLISHER_SCORE_REPLAY",
            "calibration_status": "not_fitted_or_qualified", "vision_qualification_evidence": False,
            "human_independent_review": False, "outcome_reference": protocol["outcome_reference"],
            "scope": protocol["scope"], "weighting": protocol["metric_weighting"], "splits": splits,
            "selected_match_groups": len(records), "excluded_match_groups": imported["excluded_match_groups"],
            "original_match_groups": imported["original_match_groups"], "exclusion_counts": imported["exclusion_counts"],
            "matches": records,
            "notes": "No fitting or test tuning. Prior starts at first captured score; earlier points are unknown. Terminal outcomes never enter forecast features."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external/livetennisapi_public_sample_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/prediction/livetennisapi_2026_06_protocol_v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reuse-import", action="store_true", help="Evaluate an existing immutable, hash-verified import")
    parser.add_argument("--split", choices=("development", "calibration", "test", "all"), default="all")
    args = parser.parse_args()
    if args.reuse_import:
        expected = args.output.with_suffix(args.output.suffix + ".sha256").read_text().split()[0]
        if digest_file(args.output) != expected:
            parser.error("Retained import hash mismatch")
        report = json.loads(args.output.read_text())
        if report["protocol_sha256"] != digest_file(args.protocol) or report["acquisition_sha256"] != digest_file(args.input / "acquisition_receipt.json"):
            parser.error("Retained import protocol/acquisition binding changed")
    elif args.output.exists():
        parser.error("Imported versions are immutable")
    else:
        report = import_tapes(args.input, args.protocol)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest_file(args.output) + "  " + args.output.name + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("matches", "excluded_groups")}))
    if args.evaluate:
        protocol = json.loads(args.protocol.read_text())
        path = args.output.with_name(args.output.stem + ("" if args.split == "all" else "_" + args.split) + "_evaluation.json")
        if path.exists():
            parser.error("Evaluation receipts are immutable")
        evaluation = evaluate_import(report, protocol, selected_split=None if args.split == "all" else args.split)
        evaluation.update(import_sha256=digest_file(args.output), protocol_sha256=digest_file(args.protocol),
                          implementation_sha256={str(path): digest_file(path) for path in (Path(__file__), Path("src/prediction/tennis.py"), Path("src/prediction/evaluation.py"), Path("src/scoring/scoring_rules.py"), Path("src/scoring/match_state.py"))})
        path.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        path.with_suffix(path.suffix + ".sha256").write_text(digest_file(path) + "  " + path.name + "\n")
        print(json.dumps({"evaluation": str(path), "selected_match_groups": evaluation["selected_match_groups"], "splits": {key: {"match_count": value["match_count"], "metrics": value["metrics"]} for key, value in evaluation["splits"].items()}}))


if __name__ == "__main__":
    main()
