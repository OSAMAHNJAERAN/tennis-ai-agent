"""Direct snapshot inputs stay causal even across unknown polling gaps."""

import copy
import json
from pathlib import Path

import pytest

from scripts.evaluate.evaluate_publisher_snapshots import (
    TapeRejected, add_calibrated, fit_calibration, forecast_match, metadata_reason,
    observed_state, parse_snapshots, summarize, terminal_identity,
)


PROTOCOL = json.loads(Path("data/prediction/livetennisapi_2026_06_protocol_v3.json").read_text())
METADATA = {"match_id": "m", "split": "development", "best_of": "BO3", "event_status": "",
            "player1_id": "a", "player2_id": "b", "round": "ATP Stuttgart - Final",
            "tournament": "Stuttgart", "scheduled_time_utc": "2026-06-10 12:00:00"}


def row(first=(6, 5), second=(4, 4), *, sets=(1, 0), points=("40", "15"), server="1", time=0, tiebreak=False):
    return {"sets_p1": str(sets[0]), "sets_p2": str(sets[1]), "games_p1": json.dumps(first), "games_p2": json.dumps(second),
            "points_p1": points[0], "points_p2": points[1], "server": server, "is_tiebreak": str(tiebreak),
            "timestamp_utc": f"2026-06-10 12:00:{time:02d}", "match_id": "m"}


def terminal(time=3):
    return row((6, 6), (4, 4), sets=(2, 0), points=("0", "0"), server="", time=time)


def test_unknown_polling_gap_preserves_direct_features_without_point_inference():
    tape = [row((6, 3), (4, 2), points=("0", "0")), row(time=2), terminal()]
    imported, counts = parse_snapshots(METADATA, tape, PROTOCOL)
    forecasts = forecast_match(imported)
    assert len(forecasts["snapshots"]) == 2
    assert all(r["past_observed_points"] == 0 for r in forecasts["snapshots"])
    assert all(r["probability"] == r["baseline_probability"] for r in forecasts["snapshots"])
    changed_target = copy.deepcopy(imported)
    changed_target["outcome"]["winner_id"] = 2
    other = forecast_match(changed_target)
    assert [r["probability"] for r in other["snapshots"]] == [r["probability"] for r in forecasts["snapshots"]]


def test_global_duplicate_removal_and_terminal_exclusion():
    tape = [row(), row(time=1), row(points=("40", "40"), time=2), row(time=3), terminal(4), terminal(5)]
    imported, counts = parse_snapshots(METADATA, tape, PROTOCOL)
    assert len(imported["snapshots"]) == 2
    assert counts["duplicate_identical_score_server_rows"] == 2
    assert counts["terminal_or_duplicate_terminal_rows"] == 2


def test_nonterminal_and_terminal_same_utc_reject_group_before_terminal_branch():
    with pytest.raises(TapeRejected, match="conflicting_scores_at_same_capture_time"):
        parse_snapshots(METADATA, [row(time=3), terminal(3)], PROTOCOL)


def test_same_utc_identical_state_and_terminal_cleanup_duplicates_remain_allowed():
    cleaned_terminal = {**terminal(3), "points_p1": "40", "points_p2": "15", "server": "2"}
    match, counts = parse_snapshots(METADATA, [row(), row(), terminal(3), cleaned_terminal], PROTOCOL)
    assert len(match["snapshots"]) == 1
    assert counts["duplicate_identical_score_server_rows"] == 1
    assert counts["terminal_or_duplicate_terminal_rows"] == 2


def test_player_join_excludes_doubles_even_when_event_name_omits_doubles():
    players = {"a": {"name": "One / Two"}, "b": {"name": "Three / Four"}}
    assert metadata_reason(METADATA, players, PROTOCOL) == "doubles_from_joined_players_or_event_metadata"
    singles = {"a": {"name": "One"}, "b": {"name": "Two"}}
    assert metadata_reason({**METADATA, "tournament": "French Open"}, singles, PROTOCOL) == "unknown_or_excluded_event_rules_family"
    assert metadata_reason({**METADATA, "best_of": ""}, singles, PROTOCOL) == "unsupported_or_unknown_match_format"


@pytest.mark.parametrize("invalid", [row(points=("AD", "15")), row(server=""), row((6, 6), (4, 6)),
                                    row((6, 6), (4, 6), points=("7", "2"), tiebreak=True)])
def test_ambiguous_or_unknown_current_state_rejected(invalid):
    with pytest.raises(TapeRejected):
        observed_state(invalid)


def test_known_third_set_tiebreak_uses_direct_server_with_documented_sequence():
    state = observed_state(row((6, 4, 6), (4, 6, 6), sets=(1, 1), points=("6", "5"), server="2", tiebreak=True))
    assert state.current_set == 3
    assert state.initial_server_id == 2
    assert state.tie_break_points_to_win == 7


@pytest.mark.parametrize("tape", [[row(), row((6, 4), (4, 4), time=1), terminal()],
                                [row(), row(points=("40", "30")), terminal()],
                                [row(), terminal(), row(time=4)]])
def test_reversed_or_conflicting_progress_rejects_whole_match(tape):
    with pytest.raises(TapeRejected):
        parse_snapshots(METADATA, tape, PROTOCOL)


def test_terminal_result_requires_legal_completed_arrays():
    assert terminal_identity(terminal())[0] == (2, 0)
    with pytest.raises(TapeRejected):
        terminal_identity(row((6, 5), (4, 4), sets=(2, 0)))


def records(count=20):
    return [{"match_id": str(i), "split": "development", "snapshots": [
        {"probability": .25 if i % 2 else .75, "baseline_probability": .25 if i % 2 else .75,
         "unconditional_probability": .5, "actual_winner_p1": int(i % 2 == 0)}]}
        for i in range(count)]


def test_calibration_is_fixed_development_only_and_cannot_use_test_groups():
    data = records()
    fitted = fit_calibration(data, PROTOCOL["calibration"])
    assert fitted["status"] == "fitted_development_only"
    assert fit_calibration(records(1), PROTOCOL["calibration"])["parameters"] is None
    with pytest.raises(ValueError, match="development"):
        fit_calibration([{**r, "split": "test"} for r in data], PROTOCOL["calibration"])
    add_calibrated(data, fitted, PROTOCOL["calibration"])
    metrics = summarize(data, bootstrap=30)
    assert metrics["improvement_against_neutral_score_only"]["raw_neutral_rules"] == {"brier": 0., "log_loss": 0.}
    assert metrics["paired_improvement_95_percent_intervals"]["raw_neutral_rules"]["brier"] == [0., 0.]


def test_match_weighting_does_not_reward_duplicate_snapshot_volume():
    data = records(2)
    first = summarize(data, bootstrap=20)
    data[0]["snapshots"] *= 100
    second = summarize(data, bootstrap=20)
    assert first["metrics"]["raw_neutral_rules"]["brier"] == pytest.approx(second["metrics"]["raw_neutral_rules"]["brier"])
