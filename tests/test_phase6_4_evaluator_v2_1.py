"""Evaluator v2.1 optimal-assignment and coverage-boundary regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from types import SimpleNamespace
from pathlib import Path

import pytest

from src.events.event_evaluator import (
    MATCHING_ALGORITHM,
    PHASE6_4_EVALUATOR_VERSION,
    canonical_one_to_one_matches,
    evaluate_covered_events,
    gt_coverage_boundary_status,
)
from scripts.reconstruct_phase6_4_evaluator_v2_1_artifacts import (
    STAGES,
    _conditional_player,
    _conditional_semantic,
    assert_causal_monotonicity,
    causal_stage_survival,
    first_failure_from_survival,
    stage_sets_from_analysis,
)


def _prediction(timestamp_s: float, **updates):
    return {"video_id": "video_A", "timestamp_s": timestamp_s, "frame": 0, **updates}


def _gt(timestamp_s: float, **updates):
    return {
        "video_id": "video_A",
        "timestamp_min_s": timestamp_s,
        "timestamp_max_s": timestamp_s,
        "frame_best": 0,
        **updates,
    }


def test_evaluator_version_and_matching_contract():
    assert PHASE6_4_EVALUATOR_VERSION == "2.1"
    assert "MAX_CARDINALITY_MIN_COST" in MATCHING_ALGORITHM


def test_adversarial_greedy_counterexample_returns_two_matches():
    # Greedy takes P1->G1 at zero cost, then P2 cannot match G2. The optimal
    # cardinality assignment is P1->G2 and P2->G1.
    predictions = [_prediction(1.00), _prediction(1.15)]
    ground_truth = [_gt(1.00), _gt(0.90)]
    matches = canonical_one_to_one_matches(predictions, ground_truth)
    assert {(p, g) for p, g, _ in matches} == {(0, 1), (1, 0)}
    assert len(matches) == 2


def test_cardinality_is_prioritized_over_lower_single_edge_error():
    matches = canonical_one_to_one_matches(
        [_prediction(1.00), _prediction(1.15)],
        [_gt(1.00), _gt(0.90)],
    )
    assert len(matches) == 2
    assert sum(error for _, _, error in matches) == pytest.approx(0.25)


def test_minimum_total_error_is_second_objective():
    matches = canonical_one_to_one_matches(
        [_prediction(1.00), _prediction(1.30)],
        [_gt(1.10), _gt(1.20)],
    )
    assert [(p, g) for p, g, _ in matches] == [(0, 0), (1, 1)]
    assert sum(error for _, _, error in matches) == pytest.approx(0.20)


def test_equal_cost_tie_is_deterministic():
    predictions = [_prediction(1.0), _prediction(1.0)]
    ground_truth = [_gt(1.0), _gt(1.0)]
    first = canonical_one_to_one_matches(predictions, ground_truth)
    for _ in range(10):
        assert canonical_one_to_one_matches(predictions, ground_truth) == first
    assert len(first) == 2


def _brute_force_matching_objective(predictions, ground_truth, require_event_type, require_player):
    def canonical_type(record):
        value = record.get("event_type", "")
        return "PLAYER_HIT" if value in {"PLAYER_1_HIT", "PLAYER_2_HIT"} else value

    edges = {}
    for prediction_index, prediction in enumerate(predictions):
        valid = []
        for gt_index, gt in enumerate(ground_truth):
            if prediction["video_id"] != gt["video_id"]:
                continue
            prediction_type = canonical_type(prediction)
            gt_type = canonical_type(gt)
            if require_event_type and prediction_type != gt_type:
                continue
            if (
                require_player
                and gt_type != "BOUNCE"
                and gt.get("player_id") is not None
                and prediction.get("player_id") != gt.get("player_id")
            ):
                continue
            timestamp = prediction["timestamp_s"]
            start, end = gt["timestamp_min_s"], gt["timestamp_max_s"]
            distance = start - timestamp if timestamp < start else (
                timestamp - end if timestamp > end else 0.0
            )
            if distance <= 0.2 + 1e-12:
                valid.append((gt_index, distance))
        edges[prediction_index] = valid

    best_cardinality = -1
    best_cost = float("inf")

    def search(prediction_index, used_gt, cardinality, cost):
        nonlocal best_cardinality, best_cost
        if prediction_index == len(predictions):
            if cardinality > best_cardinality or (
                cardinality == best_cardinality and cost < best_cost
            ):
                best_cardinality = cardinality
                best_cost = cost
            return
        search(prediction_index + 1, used_gt, cardinality, cost)
        for gt_index, distance in edges[prediction_index]:
            if gt_index not in used_gt:
                search(
                    prediction_index + 1,
                    used_gt | {gt_index},
                    cardinality + 1,
                    cost + distance,
                )

    search(0, set(), 0, 0.0)
    return best_cardinality, best_cost


def test_optimal_matcher_agrees_with_exhaustive_small_graph_oracle():
    rng = random.Random(64021)
    gt_types = ("SERVE_CONTACT", "PLAYER_HIT", "BOUNCE")
    for _ in range(100):
        prediction_count = rng.randint(0, 5)
        gt_count = rng.randint(0, 5)
        predictions = []
        for _prediction_index in range(prediction_count):
            event_type = rng.choice(gt_types)
            if event_type == "PLAYER_HIT":
                event_type = rng.choice(("PLAYER_HIT", "PLAYER_1_HIT", "PLAYER_2_HIT"))
            predictions.append(
                {
                    "video_id": rng.choice(("video_A", "video_B")),
                    "timestamp_s": rng.randint(0, 30) * 0.05,
                    "frame": 0,
                    "event_type": event_type,
                    "player_id": rng.choice((1, 2, None)),
                }
            )
        ground_truth = []
        for _gt_index in range(gt_count):
            center = rng.randint(0, 30) * 0.05
            half_width = rng.choice((0.0, 0.05))
            ground_truth.append(
                {
                    "video_id": rng.choice(("video_A", "video_B")),
                    "timestamp_min_s": max(0.0, center - half_width),
                    "timestamp_max_s": center + half_width,
                    "frame_best": 0,
                    "event_type": rng.choice(gt_types),
                    "player_id": rng.choice((1, 2, None)),
                }
            )
        require_event_type = rng.choice((False, True))
        require_player = rng.choice((False, True))
        oracle_cardinality, oracle_cost = _brute_force_matching_objective(
            predictions, ground_truth, require_event_type, require_player
        )
        matches = canonical_one_to_one_matches(
            predictions,
            ground_truth,
            require_event_type=require_event_type,
            require_player=require_player,
        )
        assert len(matches) == oracle_cardinality
        assert sum(error for _, _, error in matches) == pytest.approx(
            oracle_cost, abs=1e-12
        )
        assert canonical_one_to_one_matches(
            predictions,
            ground_truth,
            require_event_type=require_event_type,
            require_player=require_player,
        ) == matches


def test_duplicate_prediction_remains_false_positive():
    matches = canonical_one_to_one_matches(
        [_prediction(1.0), _prediction(1.0)], [_gt(1.0)]
    )
    assert len(matches) == 1


def test_optimal_matcher_preserves_same_video_type_player_and_bounce_rules():
    gt = [_gt(1.0, event_type="PLAYER_HIT", player_id=1)]
    assert not canonical_one_to_one_matches(
        [{**_prediction(1.0), "video_id": "video_B", "event_type": "PLAYER_HIT", "player_id": 1}],
        gt,
        require_event_type=True,
        require_player=True,
    )
    assert not canonical_one_to_one_matches(
        [_prediction(1.0, event_type="BOUNCE", player_id=1)],
        gt,
        require_event_type=True,
    )
    assert not canonical_one_to_one_matches(
        [_prediction(1.0, event_type="PLAYER_1_HIT", player_id=2)],
        gt,
        require_event_type=True,
        require_player=True,
    )
    assert canonical_one_to_one_matches(
        [_prediction(1.0, event_type="BOUNCE", player_id=None)],
        [_gt(1.0, event_type="BOUNCE", player_id=2)],
        require_event_type=True,
        require_player=True,
    )


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_native_fps_and_inclusive_200ms_boundary(fps):
    gt_frame = int(fps)
    matches = canonical_one_to_one_matches(
        [
            {
                "video_id": "video_A",
                "frame": gt_frame,
                "timestamp_s": gt_frame / fps + 0.2,
            }
        ],
        [{"video_id": "video_A", "frame_min": gt_frame, "frame_best": gt_frame, "frame_max": gt_frame}],
        fps=fps,
    )
    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(0.2)


def _coverage(start: int, end: int, *, fps: float = 10.0, frame_count: int = 20):
    return {
        "schema_version": "1.0",
        "coverage_version": "test",
        "evaluation_scope_id": "test",
        "annotation_scope": "PHYSICAL_EVENTS_EXHAUSTIVE",
        "coverage_basis": "RAW_VIDEO",
        "videos": {
            "video_A": {
                "video_id": "video_A",
                "fps": fps,
                "frame_count": frame_count,
                "fully_reviewed_intervals": [
                    {
                        "interval_id": "covered",
                        "start_frame": start,
                        "end_frame": end,
                        "start_timestamp_s": start / fps,
                        "end_timestamp_s": end / fps,
                        "physical_event_annotation_complete": True,
                    }
                ],
            }
        },
    }


def _manifest(*, fps: float = 10.0, frame_count: int = 20):
    return {"videos": {"video_A": {"fps": fps, "frame_count": frame_count}}}


def _frame_gt(start: int, end: int):
    return {
        "video_id": "video_A",
        "event_id": 1,
        "frame_min": start,
        "frame_best": start,
        "frame_max": end,
    }


def test_gt_interval_plus_tolerance_fully_inside_coverage_is_evaluable():
    audit = gt_coverage_boundary_status(
        "video_A", _frame_gt(5, 6), _coverage(2, 9), _manifest(), tolerance_s=0.2
    )
    assert audit["coverage_safe"] is True
    assert audit["left_margin_s"] == pytest.approx(0.3)
    assert audit["right_margin_s"] == pytest.approx(0.3)


def test_right_tolerance_crossing_coverage_is_boundary_censored():
    audit = gt_coverage_boundary_status(
        "video_A", _frame_gt(8, 9), _coverage(0, 9), _manifest(), tolerance_s=0.2
    )
    assert audit["coverage_safe"] is False
    assert audit["evaluation_status"] == "BOUNDARY_CENSORED"


def test_left_tolerance_crossing_coverage_is_boundary_censored():
    audit = gt_coverage_boundary_status(
        "video_A", _frame_gt(2, 3), _coverage(2, 19), _manifest(), tolerance_s=0.2
    )
    assert audit["coverage_safe"] is False


def test_physical_video_start_and_end_truncate_required_margin():
    coverage = _coverage(0, 9, frame_count=10)
    manifest = _manifest(frame_count=10)
    assert gt_coverage_boundary_status(
        "video_A", _frame_gt(0, 1), coverage, manifest, tolerance_s=0.2
    )["coverage_safe"]
    assert gt_coverage_boundary_status(
        "video_A", _frame_gt(8, 9), coverage, manifest, tolerance_s=0.2
    )["coverage_safe"]


def test_extending_reviewed_coverage_makes_gt_evaluable():
    gt = _frame_gt(8, 9)
    original = gt_coverage_boundary_status(
        "video_A", gt, _coverage(0, 9), _manifest(), tolerance_s=0.2
    )
    extended = gt_coverage_boundary_status(
        "video_A", gt, _coverage(0, 11), _manifest(), tolerance_s=0.2
    )
    assert original["coverage_safe"] is False
    assert extended["coverage_safe"] is True


def test_outside_scope_prediction_cannot_become_false_positive():
    result = evaluate_covered_events(
        {"video_A": [{"video_id": "video_A", "frame": 15}]},
        {"video_A": []},
        _manifest(),
        _coverage(0, 9),
        tolerance_s=0.2,
    )
    assert result["aggregate"]["false_positives"] == 0
    assert result["prediction_statuses"][0]["evaluation_status"] == "OUTSIDE_SCOPE"


def test_boundary_censoring_excludes_gt_and_temporally_eligible_prediction_symmetrically():
    result = evaluate_covered_events(
        {"video_A": [{"video_id": "video_A", "frame": 9}]},
        {"video_A": [_frame_gt(8, 9)]},
        _manifest(),
        _coverage(0, 9),
        tolerance_s=0.2,
    )
    assert result["aggregate"]["false_negatives"] == 0
    assert result["aggregate"]["false_positives"] == 0
    assert result["aggregate"]["boundary_censored_gt_count"] == 1
    assert result["prediction_statuses"][0]["evaluation_status"] == "BOUNDARY_CENSORED"


def test_coverage_definition_is_not_mutated_by_evaluation():
    coverage = _coverage(0, 9)
    before = copy.deepcopy(coverage)
    evaluate_covered_events(
        {"video_A": []}, {"video_A": []}, _manifest(), coverage
    )
    assert coverage == before


def _trace(**updates):
    trace = {
        "candidate_id": 1,
        "discovery_frame": 10,
        "discovery_timestamp_s": 1.0,
        "refined_frame": 11,
        "refined_timestamp_s": 1.1,
        "physical_event_type": "BALL_CONTACT_WITH_PLAYER",
        "candidate_event_type": "PLAYER_1_HIT",
        "selected_player": 1,
        "stage_pass": {
            "raw_candidate_generator": True,
            "physics_verification": True,
            "event_type_classification": True,
            "player_attribution": True,
            "temporal_suppression": False,
            "final_authoritative_event": False,
        },
        "final_event_emitted": False,
        "final_event_id": None,
        "rejection_reason": "DUPLICATE_VERIFIED_EVENT",
    }
    trace.update(updates)
    return trace


def _survival(trace=None, *, stage0=True, stage1=True, stage2=True, event_type="PLAYER_HIT"):
    return causal_stage_survival(
        stage_0_match=stage0,
        stage_1_match=stage1,
        stage_2_match=stage2,
        trace=trace,
        gt_event={"event_type": event_type, "player_id": 1},
    )


def test_stage3_false_forces_stage4_stage5_and_stage6_false():
    trace = _trace()
    trace["stage_pass"]["physics_verification"] = False
    trace["final_event_emitted"] = True
    survival = _survival(trace)
    assert survival["stage_3"] is False
    assert survival["stage_4"] is False
    assert survival["stage_5"] is False
    assert survival["stage_6"] is False


def test_stage2_false_forces_all_later_stages_false():
    survival = _survival(_trace(final_event_emitted=True), stage2=False)
    assert all(not survival[stage] for stage in ("stage_2", "stage_3", "stage_4", "stage_5", "stage_6"))


def test_causal_lineage_monotonicity_per_record_and_aggregate():
    records = [
        {"video_id": "video_A", "gt_event_id": 1, "causal_stage_survival": _survival(_trace())},
        {"video_id": "video_A", "gt_event_id": 2, "causal_stage_survival": _survival(None, stage2=False)},
    ]
    assert_causal_monotonicity(records)
    records[1]["causal_stage_survival"]["stage_4"] = True
    with pytest.raises(AssertionError, match="resurrection"):
        assert_causal_monotonicity(records)


def test_independent_capability_may_be_non_monotonic_but_is_separately_named():
    record = {
        "video_id": "video_A",
        "gt_event_id": 1,
        "causal_stage_survival": _survival(None, stage2=False),
        "independent_stage_match_capability": {"stage_3": False, "stage_4": True},
    }
    assert_causal_monotonicity([record])
    assert record["independent_stage_match_capability"]["stage_4"] is True


def _analysis(trace, *, emit=False):
    candidate = SimpleNamespace(
        frame_index=10,
        timestamp_s=1.0,
        score=0.8,
        discovery_frame_index=9,
        discovery_timestamp_s=0.9,
    )
    events = []
    if emit:
        events = [
            SimpleNamespace(
                event_id=1,
                frame_index=11,
                timestamp_s=1.1,
                event_type=SimpleNamespace(value="PLAYER_1_HIT"),
                player_id=1,
                confidence=0.7,
                trajectory_state="TRACKED",
                evidence={"candidate_id": 1},
            )
        ]
    return SimpleNamespace(candidates=[candidate], verification_traces=[trace], events=events)


def test_physical_verification_exists_before_semantics_and_is_not_final_output_proxy():
    trace = _trace(candidate_event_type="UNKNOWN_EVENT")
    trace["stage_pass"]["event_type_classification"] = False
    stages = stage_sets_from_analysis("video_A", _analysis(trace, emit=False))
    assert len(stages["stage_3"]) == 1
    assert stages["stage_4"] == []
    assert stages["stage_6"] == []


def test_stable_candidate_identity_persists_through_stage6():
    trace = _trace(final_event_emitted=True, final_event_id=1)
    trace["stage_pass"]["temporal_suppression"] = True
    trace["stage_pass"]["final_authoritative_event"] = True
    stages = stage_sets_from_analysis("video_A", _analysis(trace, emit=True))
    assert {stages[stage][0]["candidate_id"] for stage in ("stage_2", "stage_3", "stage_4", "stage_5", "stage_6")} == {1}


def test_stage6_suppression_reason_remains_linked_to_candidate_id():
    stages = stage_sets_from_analysis("video_A", _analysis(_trace(), emit=False))
    assert stages["stage_5"][0]["candidate_id"] == 1
    assert stages["stage_5"][0]["suppression_reason"] == "DUPLICATE_VERIFIED_EVENT"
    assert stages["stage_6"] == []


def test_first_failure_is_the_first_causal_failure():
    trace = _trace(final_event_emitted=True)
    trace["stage_pass"]["physics_verification"] = False
    survival = _survival(trace)
    assert first_failure_from_survival(survival) == "PHYSICAL_CONTACT_REJECTED"


def test_stage4_conditional_denominator_comes_only_from_stage3_matches():
    stage3 = {
        "video_A": [
            {"video_id": "video_A", "frame": 5, "candidate_id": 1, "event_type": "PLAYER_1_HIT"}
        ]
    }
    gt = {
        "video_A": [
            {"video_id": "video_A", "event_id": 1, "frame_min": 5, "frame_best": 5, "frame_max": 5, "event_type": "PLAYER_HIT", "player_id": 1},
            {"video_id": "video_A", "event_id": 2, "frame_min": 8, "frame_best": 8, "frame_max": 8, "event_type": "BOUNCE", "player_id": None},
        ]
    }
    stage3_result = {"matches": [{"video_id": "video_A", "prediction_index": 0, "gt_event_id": 1}]}
    conditional = _conditional_semantic(stage3_result, stage3, gt, _coverage(0, 19),)
    assert conditional["aggregate"]["conditional_support"] == 1


def test_stage5_conditional_denominator_excludes_bounce():
    conditional_semantic = {
        "records": [
            {"video_id": "video_A", "candidate_id": 1, "gt_event_id": 1, "correct": True},
            {"video_id": "video_A", "candidate_id": 2, "gt_event_id": 2, "correct": True},
        ]
    }
    stage3 = {
        "video_A": [
            {"video_id": "video_A", "frame": 5, "candidate_id": 1, "player_id": 1},
            {"video_id": "video_A", "frame": 8, "candidate_id": 2, "player_id": None},
        ]
    }
    gt = {
        "video_A": [
            {"event_id": 1, "event_type": "PLAYER_HIT", "player_id": 1},
            {"event_id": 2, "event_type": "BOUNCE", "player_id": None},
        ]
    }
    result = _conditional_player(conditional_semantic, stage3, _coverage(0, 19), gt)
    assert result["aggregate"]["conditional_support"] == 1
    assert result["aggregate"]["correct"] == 1
    assert result["bounce_excluded_from_denominator"] is True


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "artifacts" / "validation"
V2_1_ARTIFACTS = (
    "phase6_4_v2_1_final_output_physical_metrics.json",
    "phase6_4_v2_1_stage3_physical_verification_metrics.json",
    "phase6_4_v2_1_causal_stage_metrics.json",
    "phase6_4_v2_1_causal_event_lineage.json",
    "phase6_4_v2_1_independent_stage_capability.json",
    "phase6_4_v2_1_semantic_metrics.json",
    "phase6_4_v2_1_coverage_boundary_audit.json",
    "phase6_4_v2_1_evaluator_integrity.json",
)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_required_v2_1_artifacts_exist_and_are_diagnostic_only():
    for name in V2_1_ARTIFACTS:
        artifact = _json(VALIDATION / name)
        assert artifact["scientific_split"] == "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED"
        assert artifact["qualification_evidence"] is False
        assert artifact["coverage_review_status"] == "MODEL_ASSISTED_PROVISIONAL"
        assert artifact["metric_authority"] == "DIAGNOSTIC_ONLY"
        assert artifact["human_approved"] is False


def test_all_v2_1_artifacts_share_current_evaluator_and_input_hashes():
    expected = {
        "evaluator_sha256": _hash(ROOT / "src/events/event_evaluator.py"),
        "gt_sha256": _hash(ROOT / "data/benchmarks/cross_match_final_holdout/ground_truth_events.json"),
        "coverage_sha256": _hash(ROOT / "data/benchmarks/cross_match_final_holdout/annotation_coverage.json"),
        "video_manifest_sha256": _hash(ROOT / "data/benchmarks/cross_match_final_holdout/videos.json"),
        "config_sha256": _hash(ROOT / "configs/phase6_analytics/pipeline.yaml"),
    }
    for name in V2_1_ARTIFACTS:
        provenance = _json(VALIDATION / name)["evaluator_provenance"]
        assert provenance["evaluator_version"] == "2.1"
        assert provenance["matching_algorithm"] == MATCHING_ALGORITHM
        for key, value in expected.items():
            assert provenance[key] == value


def test_generated_causal_lineage_has_no_impossible_transition():
    records = _json(VALIDATION / "phase6_4_v2_1_causal_event_lineage.json")["records"]
    assert len(records) == 40
    assert_causal_monotonicity(records)
    metrics = _json(VALIDATION / "phase6_4_v2_1_causal_stage_metrics.json")
    assert metrics["impossible_transition_count"] == 0
    counts = [metrics["aggregate"][stage] for stage in STAGES]
    assert counts == sorted(counts, reverse=True)


def test_v2_0_impossible_lineage_is_retained_but_superseded():
    old = _json(VALIDATION / "phase6_4_corrected_event_lineage.json")
    assert old["scientific_status"] == "SUPERSEDED_BY_EVALUATOR_V2_1_CAUSAL_STAGE_SEMANTICS"
    assert any(
        not row["stage_survival"]["stage_3"] and row["stage_survival"]["stage_4"]
        for row in old["records"]
    )


def test_true_stage3_is_not_final_output_proxy():
    stage3 = _json(VALIDATION / "phase6_4_v2_1_stage3_physical_verification_metrics.json")
    final_output = _json(VALIDATION / "phase6_4_v2_1_final_output_physical_metrics.json")
    assert stage3["metric_name"] == "TRUE_STAGE3_PHYSICAL_VERIFICATION"
    assert final_output["metric_name"] == "FINAL_OUTPUT_PHYSICAL_MATCH"
    assert stage3["aggregate"]["covered_prediction_count"] == 69
    assert final_output["aggregate"]["covered_gt_count"] == 40
    assert sum(row["covered_prediction_count"] for row in final_output["per_video"].values()) == 46


def test_coverage_extension_resolves_all_boundary_censoring_provisionally():
    audit = _json(VALIDATION / "phase6_4_v2_1_coverage_boundary_audit.json")
    assert audit["coverage_safe_gt_count"] == 40
    assert audit["boundary_censored_gt_count"] == 0
    assert audit["per_video"]["video_08"]["coverage_extension"]["new_end_frame"] == 206
    assert audit["per_video"]["video_09"]["coverage_extension"]["new_end_frame"] == 244
    assert audit["per_video"]["video_10"]["coverage_extension"] is None


def test_metric_change_is_attributed_to_coverage_not_real_benchmark_greedy_collision():
    metric = _json(VALIDATION / "phase6_4_v2_1_final_output_physical_metrics.json")
    attribution = metric["change_attribution"]
    assert attribution["v2_0_greedy_old_coverage"] == {
        "true_positives": 20,
        "false_positives": 25,
        "false_negatives": 20,
    }
    assert attribution["coverage_boundary_delta"] == {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": -1,
    }
    assert attribution["optimal_assignment_delta"] == {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }


def test_production_regression_and_analytics_contract_pass():
    integrity = _json(VALIDATION / "phase6_4_v2_1_evaluator_integrity.json")
    regression = integrity["production_regression"]
    assert regression["production_prediction_count_before"] == 128
    assert regression["production_prediction_count_after"] == 128
    assert regression["changed_prediction_frames"] == []
    assert regression["changed_event_labels_or_players"] == []
    assert integrity["analytics_contract_schema_version"] == "1.0"
    assert integrity["production_gt_leakage_check"] == "PASS"
