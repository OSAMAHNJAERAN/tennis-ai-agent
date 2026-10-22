"""Private direct-state publisher evaluation; missing points are never invented."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import csv
import gzip
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import random

from scripts.data.acquire_livetennisapi_sample import digest_file
from scripts.data.audit_livetennisapi_sample import assign_split, utc
from scripts.evaluate.replay_publisher_scores import TapeRejected, _raw
from src.prediction.evaluation import _loss, _quantile
from src.prediction.tennis import TennisWinProbability
from src.scoring.match_state import CompletedSetScore, MatchFormat, MatchState, ServiceSide, SetFormat
from src.scoring.scoring_rules import TennisScoringRules


PROTOCOL_ID = "livetennisapi-june2026-direct-snapshot-v3"
FEATURE_KEYS = ("sets_p1", "sets_p2", "games_p1", "games_p2", "points_p1", "points_p2", "server", "is_tiebreak")


def load_protocol(path):
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    if digest_file(path) != expected:
        raise ValueError("Frozen protocol hash changed")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or utc(protocol["frozen_at"]) >= datetime.now(timezone.utc):
        raise ValueError("A previously frozen v3 direct-snapshot protocol is required")
    return protocol


def metadata_reason(metadata, players, protocol):
    if not metadata:
        return "missing_reserved_metadata"
    if metadata.get("best_of") != "BO3":
        return "unsupported_or_unknown_match_format"
    if metadata.get("event_status"):
        return "retirement_walkover_or_nonstandard_terminal_status"
    participants = [players.get(metadata.get(key)) for key in ("player1_id", "player2_id")]
    if metadata.get("player1_id") == metadata.get("player2_id") or any(not p or not p.get("name", "").strip() for p in participants):
        return "missing_or_invalid_participant_metadata"
    if (any("/" in p["name"] for p in participants)
            or "double" in (metadata.get("round", "") + metadata.get("tournament", "")).lower()):
        return "doubles_from_joined_players_or_event_metadata"
    tour = metadata.get("round", "").split(" ", 1)[0]
    if metadata.get("tournament") not in protocol["rules_family"].get(tour, []):
        return "unknown_or_excluded_event_rules_family"
    return None


def _completed(first, second):
    completed, winners = [], Counter()
    for index, (a, b) in enumerate(zip(first, second)):
        if not ((max(a, b) == 6 and min(a, b) <= 4) or (max(a, b) == 7 and min(a, b) in (5, 6))):
            raise TapeRejected("invalid_or_unsupported_completed_set")
        winner = 1 if a > b else 2
        winners[winner] += 1
        if index < len(first) - 1 and winners[winner] >= 2:
            raise TapeRejected("completed_set_after_match_already_won")
        completed.append(CompletedSetScore(index + 1, a, b, winner_id=winner))
    return completed, (winners[1], winners[2])


def terminal_identity(row):
    """The target is one observed terminal score, never an input to a forecast."""
    sets1, sets2, first, second, *_ = _raw(row)
    if max(sets1, sets2) < 2:
        return None
    if max(sets1, sets2) != 2 or min(sets1, sets2) >= 2 or len(first) != sets1 + sets2:
        raise TapeRejected("invalid_terminal_set_counts_or_arrays")
    completed, counts = _completed(first, second)
    if counts != (sets1, sets2) or completed[-1].winner_id != (1 if sets1 == 2 else 2):
        raise TapeRejected("terminal_arrays_and_winner_disagree")
    return counts, first, second


def observed_state(row, *, match_id="direct-state"):
    """Map only the current observed fields to the existing engine's state."""
    sets1, sets2, first, second, point1, point2, tiebreak = _raw(row)
    if max(sets1, sets2) >= 2:
        raise TapeRejected("terminal_state_not_forecast")
    if row.get("server") not in ("1", "2"):
        raise TapeRejected("missing_observed_server")
    if not first and sets1 == sets2 == 0:
        first, second = (0,), (0,)
    if len(first) != sets1 + sets2 + 1:
        raise TapeRejected("inconsistent_nonterminal_set_arrays")
    completed, counts = _completed(first[:-1], second[:-1])
    if counts != (sets1, sets2):
        raise TapeRejected("completed_sets_disagree_with_observed_set_count")
    server = int(row["server"])
    state = MatchState(match_id=match_id, format=MatchFormat.BEST_OF_3,
                       set_format=SetFormat.TIE_BREAK_SET, server_id=server, receiver_id=3-server,
                       initial_server_id=server, sets_won_p1=sets1, sets_won_p2=sets2,
                       completed_sets=completed, current_set=sets1+sets2+1,
                       games_p1=first[-1], games_p2=second[-1], current_game=first[-1]+second[-1]+1,
                       score_source="direct_publisher_observation_not_independent_review")
    if tiebreak:
        if not point1.isdigit() or not point2.isdigit():
            raise TapeRejected("invalid_tiebreak_point_counts")
        state.tie_break_active = True
        state.tie_break_points_p1, state.tie_break_points_p2 = int(point1), int(point2)
        if max(state.tie_break_points_p1, state.tie_break_points_p2) >= 7 and abs(state.tie_break_points_p1-state.tie_break_points_p2) >= 2:
            raise TapeRejected("completed_tiebreak_still_displayed_as_active")
        block = (state.tie_break_points_p1 + state.tie_break_points_p2 + 1) // 2
        # The first server of THIS currently observed tiebreak is determined
        # by its observed current server and the documented service sequence.
        state.initial_server_id = server if block % 2 == 0 else 3-server
        total = state.tie_break_points_p1 + state.tie_break_points_p2
    else:
        display = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
        if point1 not in display or point2 not in display:
            raise TapeRejected("invalid_standard_point_display")
        state.points_p1, state.points_p2 = display[point1], display[point2]
        if point1 == "AD" and point2 != "40" or point2 == "AD" and point1 != "40":
            raise TapeRejected("ambiguous_advantage_score")
        if (state.games_p1, state.games_p2) == (6, 6):
            raise TapeRejected("six_all_without_tiebreak_flag")
        total = state.points_p1 + state.points_p2
    state.service_side = ServiceSide.DEUCE if total % 2 == 0 else ServiceSide.AD
    if TennisScoringRules.validate_invariants(state):
        raise TapeRejected("invalid_observed_rules_state")
    return state


def _progress_signature(state):
    return (tuple((s.games_p1, s.games_p2) for s in state.completed_sets), state.games_p1, state.games_p2)


def progress_valid(previous, current):
    before, games1, games2 = previous
    after, next1, next2 = current
    if len(after) < len(before) or after[:len(before)] != before:
        return False
    if len(after) == len(before):
        return next1 >= games1 and next2 >= games2
    completed_current = after[len(before)]
    return completed_current[0] >= games1 and completed_current[1] >= games2


def parse_snapshots(metadata, rows, protocol):
    snapshots, seen, row_counts = [], set(), Counter()
    previous_time, previous_progress, terminal = None, None, None
    previous_capture_time, previous_capture_signature = None, None
    initial_time = utc(rows[0]["timestamp_utc"])
    for index, row in enumerate(rows):
        timestamp = utc(row["timestamp_utc"])
        if previous_time is not None and timestamp < previous_time:
            raise TapeRejected("capture_clock_reversed")
        if assign_split(timestamp, protocol) != metadata["split"]:
            raise TapeRejected("capture_crosses_reserved_split")
        previous_time = timestamp
        try:
            raw = _raw(row)
            # Check BEFORE terminal branching and row-level state exclusions.
            # Terminal display/serve cleanup has no forecasting meaning; its
            # sets and full completed game arrays remain the score identity.
            capture_signature = raw[:4] if max(raw[:2]) >= 2 else (*raw, row.get("server"))
            if timestamp == previous_capture_time and capture_signature != previous_capture_signature:
                raise TapeRejected("conflicting_scores_at_same_capture_time")
            previous_capture_time, previous_capture_signature = timestamp, capture_signature
            identity = terminal_identity(row)
            if terminal is not None and identity != terminal:
                raise TapeRejected("conflicting_postterminal_score")
            if identity is not None:
                if terminal is not None and identity != terminal:
                    raise TapeRejected("conflicting_postterminal_score")
                if previous_progress is not None:
                    counts, first, second = identity
                    if not progress_valid(previous_progress, (tuple(zip(first, second)), 0, 0)):
                        raise TapeRejected("terminal_score_reverses_observed_progress")
                terminal = identity
                row_counts["terminal_or_duplicate_terminal_rows"] += 1
                continue
            if terminal is not None:
                raise TapeRejected("nonterminal_score_after_observed_result")
            state = observed_state(row, match_id=metadata["match_id"])
        except TapeRejected as exc:
            if terminal is not None or str(exc) in ("conflicting_scores_at_same_capture_time", "conflicting_postterminal_score", "terminal_score_reverses_observed_progress", "nonterminal_score_after_observed_result"):
                raise
            row_counts[str(exc)] += 1
            continue
        progress = _progress_signature(state)
        if previous_progress is not None and not progress_valid(previous_progress, progress):
            raise TapeRejected("completed_sets_or_games_reversed_or_changed")
        previous_progress = progress
        features = {key: row[key] for key in FEATURE_KEYS}
        signature = (*_raw(row), row["server"])
        if signature in seen:
            row_counts["duplicate_identical_score_server_rows"] += 1
            continue
        seen.add(signature)
        snapshots.append({"features": features, "source_utc": row["timestamp_utc"], "source_row_index": index,
                          "timestamp_seconds": (timestamp-initial_time).total_seconds(), "signature": signature})
        row_counts["eligible_unique_nonterminal_snapshots"] += 1
    if terminal is None:
        raise TapeRejected("no_observed_rules_complete_terminal_target")
    if not snapshots:
        raise TapeRejected("no_valid_nonterminal_snapshots")
    for snapshot in snapshots:
        snapshot.pop("signature")
    return {"match_id": metadata["match_id"], "split": metadata["split"], "scheduled_time_utc": metadata["scheduled_time_utc"],
            "rules_family": metadata["round"].split(" ", 1)[0] + " " + metadata["tournament"],
            "snapshots": snapshots, "outcome": {"winner_id": 1 if terminal[0][0] == 2 else 2,
                "provenance": "same_publisher_observed_rules_complete_terminal_score_not_independent_result"},
            "original_score_rows": len(rows), "row_audit": dict(row_counts)}, row_counts


def import_snapshots(directory, protocol_path):
    protocol = load_protocol(protocol_path)
    acquisition_path = directory / "acquisition_receipt.json"
    acquisition = json.loads(acquisition_path.read_text())
    if acquisition["status"] != "ACQUIRED_PINNED_PUBLIC_SAMPLE":
        raise ValueError("Publisher acquisition is incomplete")
    for file in acquisition["files"]:
        if digest_file(directory / file["name"]) != file["sha256"]:
            raise ValueError("Publisher content hash mismatch")
    players = {}
    with gzip.open(directory / "players.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            players[row["player_id"]] = row
    metadata = {}
    with gzip.open(directory / "matches.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                split = assign_split(utc(row["scheduled_time_utc"]), protocol)
            except (ValueError, TypeError):
                continue
            if split:
                metadata[row["match_id"]] = {**row, "split": split}
    eligible, audits, seen, reasons, row_counts = [], [], set(), Counter(), Counter()
    with gzip.open(directory / "points_sample_2026-06.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for identifier, iterator in groupby(csv.DictReader(stream), key=lambda row: row["match_id"]):
            if identifier in seen:
                raise ValueError("Publisher score groups interleaved")
            seen.add(identifier)
            rows = []
            for row in iterator:
                if len(rows) >= 8192:
                    raise ValueError("Score tape exceeds frozen8192-row bound")
                rows.append(row)
            item = metadata.get(identifier)
            reason = metadata_reason(item, players, protocol)
            try:
                if reason:
                    raise TapeRejected(reason)
                match, counts = parse_snapshots(item, rows, protocol)
                eligible.append(match)
                row_counts.update(counts)
            except (TapeRejected, ValueError, TypeError, KeyError) as exc:
                reason = str(exc)
                reasons[reason] += 1
                row_counts["rows_in_excluded_match_groups"] += len(rows)
            audits.append({"match_id": identifier, "split": item["split"] if item else None,
                           "score_rows": len(rows), "eligible": reason is None, "reason": reason})
    selected = []
    for split, interval in protocol["splits"].items():
        candidates = sorted((m for m in eligible if m["split"] == split), key=lambda m: (utc(m["scheduled_time_utc"]), m["match_id"]))
        selected.extend(candidates[:interval["maximum_evaluation_matches"]])
    chosen = {m["match_id"] for m in selected}
    for audit in audits:
        audit["selected"] = audit["match_id"] in chosen
    return {"schema_version": "tennis.publisher-snapshot-import.v1", "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest_file(protocol_path), "acquisition_sha256": digest_file(acquisition_path),
            "data_kind": "real_publisher_observed_direct_score_snapshots", "qualification_status": "UNQUALIFIED_PRIVATE_PUBLISHER_SCORE_ENGINEERING",
            "human_independent_review": False, "forecast_metrics_executed": False,
            "original_match_groups": len(seen), "score_rows_audited": sum(a["score_rows"] for a in audits),
            "eligible_match_groups": len(eligible), "excluded_match_groups": sum(reasons.values()),
            "eligible_split_counts": dict(Counter(m["split"] for m in eligible)),
            "selected_split_counts": dict(Counter(m["split"] for m in selected)),
            "exclusion_counts": dict(reasons), "row_audit": dict(row_counts), "groups": audits, "matches": selected}


@lru_cache(maxsize=16384)
def _raw_probability(feature_values):
    # This cache key contains only the current direct state; no target, match
    # result or final set array can enter the forecast computation.
    state = observed_state(dict(zip(FEATURE_KEYS, feature_values)))
    forecast = TennisWinProbability().snapshot(state)
    if forecast["status"] != "prior_only" or forecast["sample_count"] != 0:
        raise ValueError("Direct-state evaluation cannot fabricate performance counts")
    return forecast["probabilities"]["match"]["player_1"]


def forecast_match(match):
    rows = []
    for snapshot in match["snapshots"]:
        probability = _raw_probability(tuple(snapshot["features"][key] for key in FEATURE_KEYS))
        rows.append({"probability": probability, "baseline_probability": probability,
                     "unconditional_probability": 0.5, "source_row_index": snapshot["source_row_index"],
                     "source_utc": snapshot["source_utc"], "timestamp_seconds": snapshot["timestamp_seconds"],
                     "past_observed_points": 0})
    # Target is attached only after every forecast in this group is complete.
    for row in rows:
        row["actual_winner_p1"] = int(match["outcome"]["winner_id"] == 1)
    return {"match_id": match["match_id"], "split": match["split"], "snapshots": rows}


def fit_calibration(records, settings):
    if len(records) < settings["minimum_development_match_groups"]:
        return {"status": "insufficient_development_match_groups", "parameters": None,
                "match_count": len(records), "minimum_match_groups": settings["minimum_development_match_groups"]}
    import numpy as np
    from scipy.optimize import minimize
    x, y, weight = [], [], []
    clip = settings["logit_probability_clip"]
    for match in records:
        if match["split"] != "development":
            raise ValueError("Only development groups may fit calibration")
        for row in match["snapshots"]:
            p = min(1-clip, max(clip, row["probability"]))
            x.append(math.log(p)-math.log1p(-p))
            y.append(row["actual_winner_p1"])
            weight.append(1/len(records)/len(match["snapshots"]))
    x, y, weight = np.asarray(x), np.asarray(y), np.asarray(weight)
    strength = settings["regularization_lambda"]

    def objective(theta):
        a, b = theta
        z = a + b*x
        probability = 1/(1+np.exp(-z))
        loss = np.sum(weight*(np.logaddexp(0, z)-y*z)) + strength/2*(a*a+(b-1)*(b-1))
        residual = weight*(probability-y)
        gradient = np.array([residual.sum()+strength*a, (residual*x).sum()+strength*(b-1)])
        return float(loss), gradient

    result = minimize(objective, settings["initial_parameters"], method="L-BFGS-B", jac=True,
                      bounds=settings["parameter_bounds"], options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-9})
    success = bool(result.success and np.all(np.isfinite(result.x)))
    return {"status": "fitted_development_only" if success else "optimizer_failed", "parameters": result.x.tolist() if success else None,
            "match_count": len(records), "snapshot_count": len(x), "regularized_training_objective": float(result.fun),
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message), "optimizer_iterations": int(result.nit),
            "settings": settings, "calibration_status": "unqualified_same_publisher_engineering_only"}


def add_calibrated(records, fit, settings):
    if fit.get("parameters") is None:
        return
    a, b = fit["parameters"]
    clip = settings["logit_probability_clip"]
    for match in records:
        for row in match["snapshots"]:
            p = min(1-clip, max(clip, row["probability"]))
            z = a+b*(math.log(p)-math.log1p(-p))
            row["calibrated_probability"] = 1/(1+math.exp(-z))


def summarize(records, *, bootstrap=1000, seed=20261009):
    """Compute per-match contributions before grouped resampling."""
    models = {"raw_neutral_rules": "probability", "neutral_score_only_baseline": "baseline_probability",
              "unconditional_secondary_baseline": "unconditional_probability"}
    if records and "calibrated_probability" in records[0]["snapshots"][0]:
        models["development_calibrated_rules"] = "calibrated_probability"
    contributions = {name: [] for name in models}
    reliability = {name: [] for name in models}
    for name, key in models.items():
        for match in records:
            losses = [_loss(row[key], row["actual_winner_p1"]) for row in match["snapshots"]]
            contributions[name].append(tuple(sum(v[index] for v in losses)/len(losses) for index in (0, 1)))
        for index in range(10):
            members = [(row, 1/len(records)/len(match["snapshots"]), match["match_id"])
                       for match in records for row in match["snapshots"] if min(9, int(row[key]*10)) == index]
            mass = sum(w for _, w, _ in members)
            reliability[name].append({"lower": index/10, "upper": (index+1)/10, "snapshot_count": len(members),
                "match_count": len({mid for _, _, mid in members}), "weight": mass,
                "mean_probability": sum(row[key]*w for row, w, _ in members)/mass if mass else None,
                "observed_win_frequency": sum(row["actual_winner_p1"]*w for row, w, _ in members)/mass if mass else None})
    metrics = {name: {"brier": sum(v[0] for v in values)/len(values), "log_loss": sum(v[1] for v in values)/len(values),
                      "reliability": reliability[name]} for name, values in contributions.items()}
    deltas = {name: {metric: metrics["neutral_score_only_baseline"][metric]-values[metric] for metric in ("brier", "log_loss")}
              for name, values in metrics.items() if name != "neutral_score_only_baseline"}
    intervals, delta_intervals = None, None
    if len(records) >= 2:
        rng = random.Random(seed)
        samples = {name: [[], []] for name in models}
        paired = {name: [[], []] for name in deltas}
        for _ in range(bootstrap):
            indices = [rng.randrange(len(records)) for _ in records]
            means = {name: [sum(values[i][metric] for i in indices)/len(indices) for metric in (0, 1)] for name, values in contributions.items()}
            for name, values in means.items():
                for index in (0, 1):
                    samples[name][index].append(values[index])
                    if name in paired:
                        paired[name][index].append(means["neutral_score_only_baseline"][index]-values[index])
        intervals = {name: {metric: [_quantile(values[index], q) for q in (.025, .975)] for index, metric in enumerate(("brier", "log_loss"))} for name, values in samples.items()}
        delta_intervals = {name: {metric: [_quantile(values[index], q) for q in (.025, .975)] for index, metric in enumerate(("brier", "log_loss"))} for name, values in paired.items()}
    return {"match_count": len(records), "snapshot_count": sum(len(r["snapshots"]) for r in records), "metrics": metrics,
            "improvement_against_neutral_score_only": deltas, "match_bootstrap_95_percent_intervals": intervals,
            "paired_improvement_95_percent_intervals": delta_intervals,
            "interval_status": "estimated_from_match_groups_frozen_calibrator" if intervals else "insufficient_match_groups"}


def write_receipt(path, value):
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise ValueError("Receipts are immutable; select a fresh version")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(digest_file(path)+"  "+path.name+"\n", encoding="utf-8")


def read_receipt(path):
    if digest_file(path) != path.with_suffix(path.suffix + ".sha256").read_text().split()[0]:
        raise ValueError("Retained receipt hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external/livetennisapi_public_sample_v1"))
    parser.add_argument("--protocol", type=Path, default=Path("data/prediction/livetennisapi_2026_06_protocol_v3.json"))
    parser.add_argument("--import-output", type=Path, default=Path("data/prediction/livetennisapi_2026_06_snapshot_import_v3.json"))
    parser.add_argument("--evaluate-split", choices=("development", "calibration", "test"))
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.import_output.exists():
        imported = read_receipt(args.import_output)
        if imported["protocol_sha256"] != digest_file(args.protocol) or imported["acquisition_sha256"] != digest_file(args.input / "acquisition_receipt.json"):
            raise ValueError("Retained import source/protocol binding changed")
    else:
        imported = import_snapshots(args.input, args.protocol)
        write_receipt(args.import_output, imported)
    print(json.dumps({key: value for key, value in imported.items() if key not in ("groups", "matches")}))
    if not args.evaluate_split:
        return
    records = [forecast_match(m) for m in imported["matches"] if m["split"] == args.evaluate_split]
    fit_path = args.import_output.with_name(args.import_output.stem + "_development_fit.json")
    if args.evaluate_split == "development":
        fit = fit_calibration(records, protocol["calibration"])
        fit.update(protocol_sha256=digest_file(args.protocol), import_sha256=digest_file(args.import_output),
                   created_at=datetime.now(timezone.utc).isoformat())
        write_receipt(fit_path, fit)
    else:
        fit = read_receipt(fit_path)
        if fit["protocol_sha256"] != digest_file(args.protocol) or fit["import_sha256"] != digest_file(args.import_output):
            raise ValueError("Development fit source/protocol changed")
    add_calibrated(records, fit, protocol["calibration"])
    result = summarize(records, bootstrap=protocol["uncertainty"]["bootstrap_resamples"], seed=protocol["uncertainty"]["seed"]) if records else {"match_count": 0, "snapshot_count": 0, "metrics": None, "reason": "no_eligible_match_groups"}
    output = args.import_output.with_name(args.import_output.stem+"_"+args.evaluate_split+"_evaluation.json")
    report = {"schema_version": "tennis.publisher-snapshot-evaluation.v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": digest_file(args.protocol), "import_sha256": digest_file(args.import_output),
              "development_fit_sha256": digest_file(fit_path), "calibration": fit, "split": args.evaluate_split,
              "qualification_status": "UNQUALIFIED_PRIVATE_PUBLISHER_SCORE_ENGINEERING", "human_independent_review": False,
              "vision_qualification_evidence": False, "production_promotion": False,
              "outcome_reference": "same_publisher_observed_rules_complete_terminal_score_not_independent_result",
              "original_match_groups": imported["original_match_groups"], "excluded_match_groups": imported["excluded_match_groups"],
              "exclusion_counts": imported["exclusion_counts"], "metrics": result, "matches": records,
              "implementation_sha256": {str(path): digest_file(path) for path in (Path(__file__), Path("src/prediction/tennis.py"), Path("src/prediction/evaluation.py"), Path("src/scoring/scoring_rules.py"), Path("src/scoring/match_state.py"))}}
    write_receipt(output, report)
    print(json.dumps({"evaluation": str(output), "split": args.evaluate_split, "match_count": result["match_count"],
                      "snapshot_count": result["snapshot_count"], "calibration_status": fit["status"]}))


if __name__ == "__main__":
    main()
