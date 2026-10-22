"""Chronological match-grouped forecast evaluation without fitting on the test split."""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import date
from pathlib import Path
from typing import Any

from src.prediction.tennis import TennisWinProbability, replay_predictions
from src.scoring.match_state import MatchFormat, MatchState, SetFormat
from src.scoring.scoring_rules import TennisScoringRules


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "1.0" or manifest.get("usage_rights") != "authorized_for_evaluation":
        raise ValueError("Versioned manifest must declare authorized evaluation usage rights.")
    matches = manifest.get("matches", [])
    if not matches:
        raise ValueError("No independently reviewed point matches supplied.")
    ids, groups, dates = set(), {}, {"development": [], "calibration": [], "test": []}
    for match in matches:
        ident, group, split = match.get("match_id"), match.get("source_group_id"), match.get("split")
        if not ident or ident in ids or not group or split not in dates:
            raise ValueError("Matches need unique IDs, source groups and declared chronological splits.")
        ids.add(ident)
        if group in groups and groups[group] != split:
            raise ValueError("A source/camera group crosses independent splits.")
        groups[group] = split
        dates[split].append(date.fromisoformat(match["match_date"]))
        reference = match.get("ground_truth", {})
        if reference.get("status") != "independently_reviewed" or not reference.get("source") or reference.get("winner_id") not in (1, 2):
            raise ValueError("Each actual winner needs independently reviewed reference provenance.")
        if not match.get("points"):
            raise ValueError("Continuous reviewed point records are required.")
        point_ids, previous_time = set(), -1
        for point in match["points"]:
            if not point.get("point_id") or "timestamp_seconds" not in point or point.get("status") != "confirmed":
                raise ValueError("Evaluation points require stable IDs, source seconds and confirmed status.")
            if point.get("provenance") not in ("manual", "human_review"):
                raise ValueError("Prediction evaluation cannot use unreviewed automatic outputs as reference points.")
            timestamp = point['timestamp_seconds']
            if point['point_id'] in point_ids or not math.isfinite(timestamp) or timestamp < previous_time:
                raise ValueError("Point IDs must be unique and source times finite and chronological.")
            point_ids.add(point['point_id'])
            previous_time = timestamp
    ordered = [(name, values) for name, values in dates.items() if values]
    if not dates["calibration"] or not dates["test"]:
        raise ValueError("Independent chronological calibration and final-test matches are required.")
    for (_, previous), (_, current) in zip(ordered, ordered[1:]):
        if max(previous) >= min(current):
            raise ValueError("Chronological splits overlap or are out of order.")
    return matches


def initial_state(match):
    values = dict(match.get("initial_state", {}))
    allowed = {"format", "set_format", "server_id", "initial_server_id", "current_set", "current_game",
               "sets_won_p1", "sets_won_p2", "games_p1", "games_p2", "points_p1", "points_p2",
               "tie_break_active", "tie_break_points_p1", "tie_break_points_p2", "tie_break_points_to_win",
               "deciding_set_tie_break_points_to_win", "deciding_set_format"}
    if set(values) - allowed:
        raise ValueError("Unsupported initial score fields; future outcomes are never forecast inputs.")
    if "server_id" not in values or "format" not in values:
        raise ValueError("Evaluation initial score requires an explicit format and server.")
    values["format"] = MatchFormat(values["format"])
    values["set_format"] = SetFormat(values.get("set_format", "TIE_BREAK_SET"))
    if values.get("deciding_set_format"):
        values["deciding_set_format"] = SetFormat(values["deciding_set_format"])
    values.setdefault("initial_server_id", values["server_id"])
    values["receiver_id"] = 3-values["server_id"]
    state = MatchState(match_id=match["match_id"], score_source="independent_evaluation_reference", **values)
    issues = TennisScoringRules.validate_invariants(state)
    if issues:
        raise ValueError("Invalid independent initial score: " + " ".join(issues))
    return state


def _loss(p, y):
    clipped = min(1-1e-12, max(1e-12, p))
    return (p-y)**2, -(y*math.log(clipped) + (1-y)*math.log1p(-clipped))


def _summarize(matches, bins):
    # Every match contributes equal total weight, irrespective of duration.
    weighted = []
    for match in matches:
        for item in match["snapshots"]:
            weighted.append((item, 1 / len(matches) / len(match["snapshots"])))
    metrics = {}
    for label, key in (("statistical", "probability"), ("score_only_baseline", "baseline_probability")):
        brier = sum(_loss(row[key], row["actual_winner_p1"])[0]*weight for row, weight in weighted)
        log_loss = sum(_loss(row[key], row["actual_winner_p1"])[1]*weight for row, weight in weighted)
        reliability = []
        for index in range(bins):
            members = [(row, weight) for row, weight in weighted if min(bins-1, int(row[key]*bins)) == index]
            mass = sum(weight for _, weight in members)
            reliability.append({"lower": index/bins, "upper": (index+1)/bins,
                "snapshot_count": len(members), "match_count": len({row["match_id"] for row, _ in members}),
                "mean_probability": sum(row[key]*weight for row, weight in members)/mass if mass else None,
                "observed_win_frequency": sum(row["actual_winner_p1"]*weight for row, weight in members)/mass if mass else None})
        metrics[label] = {"brier": brier, "log_loss": log_loss, "reliability": reliability}
    metrics["improvement_against_baseline"] = {name: metrics["score_only_baseline"][name] - metrics["statistical"][name]
                                               for name in ("brier", "log_loss")}
    return metrics


def _quantile(values, fraction):
    values = sorted(values)
    index = (len(values)-1)*fraction
    lower = int(index)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(index-lower)


def evaluate_manifest(manifest: dict[str, Any], *, bins=10, bootstrap_samples=1000, seed=20261008):
    if not 2 <= bins <= 20 or bootstrap_samples < 0:
        raise ValueError("Reliability needs 2–20 fixed bins; bootstrap count cannot be negative.")
    records = []
    for match in validate_manifest(manifest):
        state = initial_state(match)
        replay = replay_predictions(state, match["points"])
        if not replay["state"]["match_complete"] or replay["state"]["winner_id"] != match["ground_truth"]["winner_id"]:
            raise ValueError("Reviewed point ledger does not reach the independently confirmed full match result.")
        # Baseline uses the same causal score/server and fixed neutral prior.
        # It does not learn serve performance from any previous point.
        baseline = TennisWinProbability()
        snapshots = []
        for point, forecast in zip(match["points"], replay["history"]):
            state = TennisScoringRules.award_point(state, point["winner"], "Independent reviewed point")
            if forecast["status"] == "confirmed_result":
                continue  # Outcome display is not a successful prediction.
            if forecast["status"] != "uncalibrated":
                raise ValueError("Unavailable/unstable predictions cannot be silently removed from evaluation.")
            reference = baseline.snapshot(state)
            snapshots.append({"match_id": match["match_id"], "point_id": point["point_id"],
                              "timestamp_seconds": point["timestamp_seconds"],
                              "probability": forecast["probabilities"]["match"]["player_1"],
                              "baseline_probability": reference["probabilities"]["match"]["player_1"],
                              "actual_winner_p1": int(match["ground_truth"]["winner_id"] == 1)})
        if not snapshots:
            raise ValueError("Match has no pre-outcome forecasts to evaluate.")
        records.append({"match_id": match["match_id"], "source_group_id": match["source_group_id"],
                        "match_date": match["match_date"], "split": match["split"], "snapshots": snapshots})
    splits = {}
    rng = random.Random(seed)
    for split in ("development", "calibration", "test"):
        matches = [record for record in records if record["split"] == split]
        if not matches:
            continue
        metrics = _summarize(matches, bins)
        intervals = {}
        if len(matches) >= 2 and bootstrap_samples:
            sample_metrics = [_summarize([rng.choice(matches) for _ in matches], bins) for _ in range(bootstrap_samples)]
            for model in ("statistical", "score_only_baseline", "improvement_against_baseline"):
                intervals[model] = {name: [_quantile([row[model][name] for row in sample_metrics], .025),
                                         _quantile([row[model][name] for row in sample_metrics], .975)]
                                    for name in ("brier", "log_loss")}
                if model != "improvement_against_baseline":
                    reliability_intervals = []
                    for index in range(bins):
                        interval = {}
                        for name in ("mean_probability", "observed_win_frequency"):
                            values = [row[model]['reliability'][index][name] for row in sample_metrics
                                      if row[model]['reliability'][index][name] is not None]
                            interval[name] = [_quantile(values, .025), _quantile(values, .975)] if values else None
                        reliability_intervals.append(interval)
                    intervals[model]['reliability'] = reliability_intervals
        splits[split] = {"match_count": len(matches), "snapshot_count": sum(len(x["snapshots"]) for x in matches),
                         "metrics": metrics, "match_bootstrap_95_percent_intervals": intervals or None,
                         "interval_status": "estimated_from_match_groups" if intervals else "insufficient_independent_matches"}
    test = splits['test']
    reliability = test['metrics']['statistical']['reliability']
    supported_bins = [row for row in reliability if row['match_count'] >= 5]
    confidence = test['match_bootstrap_95_percent_intervals']
    gate = {"minimum_test_matches": test['match_count'] >= 20,
            "brier_improvement_lower_bound_above_zero": bool(confidence and confidence['improvement_against_baseline']['brier'][0] > 0),
            "log_loss_no_worse_than_baseline": test['metrics']['improvement_against_baseline']['log_loss'] >= 0,
            "reliability_gap_at_most_0_10_supported_bins": all(abs(row['mean_probability'] - row['observed_win_frequency']) <= .10 for row in supported_bins) if supported_bins else None}
    return {"schema_version": "1.0", "calibration_status": "not_fitted_or_qualified",
            "data_kind": manifest.get("data_kind", "declared_independent_matches_requires_audit"),
            "evaluation_only": True, "weighting": "each match has equal weight; points within match share that weight",
            "scope": "reviewed complete singles matches and confirmed initial scores in this manifest",
            "baseline": "same causal score/server with fixed neutral Beta(2,2) serve prior; no performance updates",
            "protocol": {"reliability_bins": bins, "bootstrap_samples": bootstrap_samples, "seed": seed,
                         "decision_thresholds": {"minimum_test_matches": 20, "minimum_bin_match_count": 5,
                             "maximum_supported_bin_reliability_gap": .10,
                             "brier_improvement_95_percent_lower_bound": ">0", "log_loss_delta_against_baseline": "<=0"},
                         "threshold_origin": "Proposed engineering gates frozen before independent data acquisition; synthetic tests cannot qualify them."},
            "preliminary_test_gates": gate,
            "qualification_status": "unqualified_pending_independent_calibration_and_data_review",
            "splits": splits, "matches": records}


def evaluate_file(manifest_path: Path, output_path: Path, **kwargs):
    raw = manifest_path.read_bytes()
    report = evaluate_manifest(json.loads(raw), **kwargs)
    report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    report["manifest_path"] = str(manifest_path.resolve())
    report['implementation_sha256'] = {str(path.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name('tennis.py'),
                     Path(__file__).resolve().parents[1] / 'scoring' / 'scoring_rules.py')}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
