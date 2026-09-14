"""Integrity tests for the GT-assisted Phase 6.4 oracle shot diagnostic."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_phase6_4_oracle_shots import (
    ALLOWED_GT_ASSISTED_FIELDS,
    ARTIFACT_TYPE,
    DIAGNOSTIC_LABELS,
    EVALUATION_ONLY_GT_FIELDS,
    classify_oracle_row,
    compute_oracle_metrics,
    join_oracle_annotations,
    sha256_file,
)
from src.shot_analysis.shot_types import ShotClassificationSource, ShotType


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPOSITORY_ROOT / "artifacts/validation/phase6_4_oracle_shot_diagnostic.json"


def _load_json(relative_path: str):
    with (REPOSITORY_ROOT / relative_path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _metric_row(actual: str, predicted: str):
    return {
        "classifier_output": {"shot_type": predicted},
        "evaluation_only": {"shot_type": actual},
    }


def test_oracle_gt_hit_join_is_exact_complete_and_unique():
    shots = _load_json(
        "data/benchmarks/cross_match_final_holdout/ground_truth_shots.json"
    )["shots"]
    events = _load_json(
        "data/benchmarks/cross_match_final_holdout/ground_truth_events.json"
    )["events"]

    joined = join_oracle_annotations(shots, events)

    assert len(joined) == 20
    assert len({
        (row["video_id"], row["event"]["frame_best"], row["event"]["player_id"])
        for row in joined
    }) == 20
    assert sum(row["event"]["event_type"] == "SERVE_CONTACT" for row in joined) == 3
    assert sum(row["event"]["event_type"] == "PLAYER_HIT" for row in joined) == 17


def test_oracle_classifier_receives_no_evaluation_labels():
    class SpyClassifier:
        def __init__(self):
            self.kwargs = None

        def classify_shot(self, **kwargs):
            self.kwargs = kwargs
            return (
                ShotType.UNKNOWN,
                0.5,
                ShotClassificationSource.ABSTENTION_UNKNOWN,
                {"event": 0.7},
                "test abstention",
            )

    joined = {
        "video_id": "video_test",
        "shot": {
            "shot_id": 7,
            "frame_index": 30,
            "player_id": 2,
            "shot_type": "FOREHAND",
            "direction": "CROSS_COURT",
            "court_side": "far",
            "annotation_confidence": 1.0,
        },
        "event": {
            "event_id": 11,
            "frame_best": 30,
            "player_id": 2,
            "event_type": "PLAYER_HIT",
        },
    }
    detection = {
        "frame_index": 30,
        "player_1": None,
        "player_2": {"bbox": [100.0, 50.0, 140.0, 170.0]},
    }
    trajectory = {
        "frame_index": 30,
        "timestamp_seconds": 1.0,
        "x_px": None,
        "y_px": None,
        "court_x_m": None,
        "court_y_m": None,
        "speed_kmh": None,
        "confidence": None,
        "state": "MISSING",
    }
    classifier = SpyClassifier()

    record = classify_oracle_row(joined, detection, trajectory, 30.0, classifier)

    assert set(classifier.kwargs) == {
        "event_type",
        "hit_frame",
        "player_id",
        "player_box",
        "ball_point",
    }
    assert classifier.kwargs["event_type"] == "PLAYER_2_HIT"
    assert classifier.kwargs["hit_frame"] == 30
    assert classifier.kwargs["player_id"] == 2
    assert record["oracle_input"]["event_type"] == "PLAYER_HIT"
    assert record["evaluation_only"]["shot_type"] == "FOREHAND"
    assert record["evaluation_only"]["passed_to_classifier"] is False
    assert record["predicted_feature_inputs"]["ball_point"]["position_px"] is None


def test_oracle_metric_math_is_direct_conditional_classification():
    records = [
        _metric_row("FOREHAND", "FOREHAND"),
        _metric_row("FOREHAND", "UNKNOWN"),
        _metric_row("BACKHAND", "FOREHAND"),
        _metric_row("SERVE", "SERVE"),
    ]

    metrics = compute_oracle_metrics(records)

    assert metrics["per_class"]["FOREHAND"] == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 2,
    }
    assert metrics["per_class"]["BACKHAND"]["false_negatives"] == 1
    assert metrics["per_class"]["SERVE"]["f1"] == 1.0
    assert metrics["unknown_abstentions"] == 1
    assert metrics["unknown_rate"] == 0.25
    assert metrics["coverage"] == 0.75
    assert metrics["evaluation_semantics"] == "EXACT_GT_HIT_ROW_CONDITIONAL_CLASSIFICATION"


def test_generated_oracle_artifact_is_explicitly_non_production_and_hashed():
    artifact = _load_json("artifacts/validation/phase6_4_oracle_shot_diagnostic.json")

    assert artifact["schema_version"] == "1.0"
    assert artifact["artifact_type"] == ARTIFACT_TYPE
    assert artifact["diagnostic_labels"] == list(DIAGNOSTIC_LABELS)
    assert artifact["scientific_split"] == "CROSS_MATCH_DIAGNOSTIC"
    assert artifact["qualification_evidence"] is False
    assert artifact["production_inference"] is False
    assert artifact["end_to_end_evidence"] is False
    assert artifact["oracle_boundary"]["gt_assisted_fields"] == list(ALLOWED_GT_ASSISTED_FIELDS)
    assert artifact["oracle_boundary"]["evaluation_only_gt_fields"] == list(
        EVALUATION_ONLY_GT_FIELDS
    )
    assert artifact["oracle_boundary"]["predicted_feature_policy"] == (
        "EXACT_FRAME_ONLY_NO_NEAREST_FILL"
    )
    assert artifact["dataset_integrity"]["one_to_one_gt_hit_join"] is True
    assert artifact["dataset_integrity"]["oracle_record_count"] == 20

    for provenance_key in (
        "video_manifest",
        "ground_truth_events",
        "ground_truth_shots",
        "active_pipeline_config",
        "classifier_implementation",
        "evaluator_implementation",
    ):
        provenance = artifact["input_provenance"][provenance_key]
        source_path = REPOSITORY_ROOT / provenance["path"]
        assert source_path.is_file()
        assert provenance["sha256"] == sha256_file(source_path)


def test_generated_oracle_metrics_and_missing_provenance_are_preserved():
    artifact = _load_json("artifacts/validation/phase6_4_oracle_shot_diagnostic.json")
    metrics = artifact["aggregate_metrics"]

    assert artifact["dataset_integrity"]["class_support"] == {
        "FOREHAND": 10,
        "BACKHAND": 7,
        "SERVE": 3,
    }
    assert metrics["per_class"]["FOREHAND"]["f1"] == 0.0
    assert metrics["per_class"]["BACKHAND"]["f1"] == 0.0
    assert metrics["per_class"]["SERVE"]["f1"] == 1.0
    assert metrics["macro_f1"] == pytest.approx(1.0 / 3.0)
    assert metrics["unknown_abstentions"] == 17
    assert metrics["unknown_rate"] == 0.85
    assert metrics["coverage"] == 0.15

    availability = artifact["aggregate_feature_availability"]
    assert availability["player_box_available"] == 20
    assert availability["ball_position_available"] == 13
    assert availability["ball_state_counts"]["MISSING"] == 7
    missing_records = [
        row
        for row in artifact["records"]
        if row["predicted_feature_inputs"]["ball_point"]["state"] == "MISSING"
    ]
    assert len(missing_records) == 7
    assert all(
        row["predicted_feature_inputs"]["ball_point"]["position_px"] is None
        and row["predicted_feature_inputs"]["ball_point"]["court_position_m"] is None
        for row in missing_records
    )
    assert all(
        set(row["oracle_input"]).isdisjoint(EVALUATION_ONLY_GT_FIELDS)
        for row in artifact["records"]
    )
