"""Explicit synthetic integrity fixtures; no checkpoint accuracy is measured."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from src.evaluation.yolo26_qualification import (QualificationError, SCHEMA, TARGETS,
                                               qualification_config_sha256, sha256_file,
                                               validate_qualification_receipt)


class SyntheticReceipt:
    """Retained toy files for checker tests, rejected by production by design."""

    def __init__(self, root):
        self.root = root
        self.config = yaml.safe_load(Path("configs/production/yolo26.yaml").read_text())
        self.manifest = json.loads(Path("configs/production/yolo26.artifacts.json").read_text())
        self.root.joinpath("tracker.yaml").write_text("tracker_type: botsort\nwith_reid: false\n")
        self.config["player_detection"]["tracker"] = "tracker.yaml"
        for role, record in self.manifest["models"].items():
            file = self.root / f"{role}.pt"
            file.write_bytes(f"SYNTHETIC NONMODEL FIXTURE {role}".encode())
            record.update(path=file.name, sha256=sha256_file(file), training_provenance="SYNTHETIC FIXTURE",
                          independent_accuracy_validated=True)
        self.config["ball_detection"]["model"] = "ball.pt"
        self.config["player_detection"]["model"] = "players.pt"
        self.config["court_detection"]["model_path"] = "court.pt"
        self.config["model_system"]["artifact_manifest"] = "artifacts.json"
        source_records = []
        sample_records = []
        for index, split in enumerate(("train", "val", "test", "test")):
            source_id = f"source{index}"
            source_records.append({"source_id": source_id, "match_id": f"match{index}", "session_id": f"session{index}",
                                   "original_source_video_id": f"original{index}", "sha256": f"{index + 1:064x}",
                                   "split": split, "rights_evidence": "Synthetic fixture only", "evaluation_allowed": True,
                                   "view": "elevated_end_court_singles", "camera_reviewed": True, "used_for_development": False,
                                   "owner": "synthetic", "surface": "hard", "url": "https://example.invalid/synthetic",
                                   "acquired_at": "2025-12-01T00:00:00Z", "fps": 60, "resolution": [1920, 1080], "duration_seconds": 120})
            for role in ("ball", "players", "court"):
                media = self.write(f"{source_id}_{role}.image", {"explicitly_synthetic": True, "source": source_id, "role": role})
                annotation = self.write(f"{source_id}_{role}.label", {"explicitly_synthetic": True, "annotation": source_id, "role": role})
                sample_records.append({"sample_id": f"{source_id}_{role}", "source_id": source_id, "split": split, "role": role,
                                       "frame_index": 0, "timestamp_seconds": 0,
                                       "media": media, "annotation": annotation, "near_duplicate_group": f"unique{index}_{role}",
                                       "annotation_provenance": "HUMAN_MANUAL", "reviewer_type": "HUMAN", "reviewer_ids": ["fixture_review"],
                                       "independent_human_reviewed": True})
        self.dataset = {"frozen": True, "frozen_at": "2026-01-01T00:00:00Z", "scope": "elevated_end_court_singles",
                        "duplicate_audit_complete": True, "duplicate_audit_method": "Explicitly synthetic fixture audit",
                        "sources": source_records, "samples": sample_records}
        model_hashes = {role: record["sha256"] for role, record in self.manifest["models"].items()}
        self.protocol = {"frozen_at": "2026-01-02T00:00:00Z", "scope": self.dataset["scope"], "code_revision": "synthetic",
                         "environment_lock": self.write("synthetic.lock", {"synthetic": True}),
                         "evaluator_implementation": self.write("synthetic_evaluator.py", {"synthetic": True}),
                         "model_sha256": model_hashes, "config_sha256": qualification_config_sha256(self.config, root=root),
                         "minimum_independent_test_matches": 2, "tasks": {}}
        self.results = {"tasks": {}}
        self.references = []
        self.review = {"reviewer_type": "HUMAN_EXTERNAL", "reviewer_id": "fixture_review", "reviewed_at": "2026-01-04T00:00:00Z",
                       "source_registry_complete": True, "no_test_development_reuse": True, "task_decisions": {}}
        for task, role in (("visible_ball", "ball"), ("player_identity", "players"), ("player_ground_position", "players"), ("calibration", "court")):
            self.add_task(task, role)
        self.receipt = {"schema_version": SCHEMA, "receipt_id": "SYNTHETIC-FIXTURE-NEVER-PRODUCTION",
                        "evidence_origin": "SYNTHETIC_SOFTWARE_FIXTURE", "evaluation_started_at": "2026-01-03T00:00:00Z",
                        "created_at": "2026-01-05T00:00:00Z", "developer_ids": ["fixture_developer"],
                        "model_sha256": model_hashes, "config_sha256": self.protocol["config_sha256"]}
        self.persist()

    def write(self, name, document):
        path = self.root / name
        path.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
        return {"path": name, "sha256": sha256_file(path)}

    def add_task(self, task, role):
        row = {"total": 120, "evaluable": 100, "measured": 96, "unknown": 4, "excluded": 20,
               "measurement_definition": f"Explicit synthetic {task} definition", "denominator_definition": "All frozen evaluable fixture observations",
               "exclusion_definition": "Retained unsupported fixture observations", "tolerance": {"definition": "Frozen synthetic reference tolerance"},
               "sample_ids": [f"source2_{role}", f"source3_{role}"], "metrics": {},
               "confidence_interval": {"grouping": "match", "method": "Synthetic fixture interval", "independent_matches": 2, "lower": 0, "upper": 1},
               "unknown_cases": self.write(f"{task}_unknown.json", {"explicitly_synthetic": True, "task": task, "unknown_count": 4,
                                          "cases": [{"case_id": f"{task}_{index}", "match_id": f"match{index}", "count": 2,
                                                     "reason": "Synthetic unknown", "source_interval": "Synthetic source frames 0-2"} for index in (2, 3)]}),
               "breakdowns": {key: {"evaluable": 50, "measured": 48, "unknown": 2, "metrics": {"synthetic_error": .1}} for key in ("near", "far", "blur")}}
        if task == "visible_ball":
            row.update(unit="original_image_px", counts={"tp": 96, "fp": 3, "fn": 4}, negative_duration_minutes=1, evaluated_duration_minutes=2,
                       ground_truth_states={"visible": 100, "absent": 20, "occluded": 0, "ambiguous": 0})
            row["metrics"].update(median_localization_error_px=1, p95_localization_error_px=3, false_alarms_per_minute=1.5)
        elif task == "player_identity":
            row.update(counts={"idtp": 96, "idfp": 3, "idfn": 4}, identity_switches=1, interruptions=2)
            row["metrics"]["hota"] = .96
        elif task == "player_ground_position":
            row.update(unit="m", absolute_errors=[.1] * 96)
        elif task == "calibration":
            row.update(unit="m", verification_points_excluded_from_fit=True, verification_lines_excluded_from_fit=True,
                       verification_regions=["near", "far"])
            row["metrics"].update(median_error_m=.12, p95_error_m=.3, valid_time_fraction=.96)
        elif task in {"strokes", "bounces"}:
            row.update(counts={"tp": 96, "fp": 3, "fn": 4}, tolerance={"matching": "one_to_one", "milliseconds": 80}, dead_ball_controls_reviewed=True)
        elif task == "player_speed":
            row.update(unit="m/s", absolute_errors=[.2] * 96)
            row["metrics"].update(bias_m_s=.1, p95_error_m_s=.2)
        elif task == "physical_ball_speed":
            row.update(unit="km/h", absolute_errors=[3.] * 96)
        elif task == "point_winner":
            row["counts"] = {"tp": 95, "fp": 1, "fn": 4}
        group_metrics = row["metrics"] or {"synthetic_error": .1}
        row["match_groups"] = [{"match_id": f"match{index}", "total": 60, "evaluable": 50, "measured": 48, "unknown": 2, "excluded": 10,
                                "metrics": deepcopy(group_metrics)} for index in (2, 3)]
        thresholds = {key: {"operator": op, "value": value} for key, (op, value) in TARGETS.get(task, {}).items()}
        if task == "calibration":
            thresholds = {"p95_error_m": {"operator": "<=", "value": .4}, "valid_time_fraction": {"operator": ">=", "value": .9}}
        self.protocol["tasks"][task] = {key: deepcopy(row[key]) for key in ("measurement_definition", "denominator_definition", "exclusion_definition", "tolerance")}
        self.protocol["tasks"][task].update(thresholds=thresholds, reference_id=f"reference_{task}")
        self.results["tasks"][task] = row
        reference = {"reference_id": f"reference_{task}", "independent_of_model": True, "reviewer_type": "HUMAN_EXTERNAL", "reviewer_id": "fixture_review",
                     "method": "Explicitly synthetic software fixture reference", "tasks": [task], "source_ids": ["source2", "source3"],
                     "measurement_definitions": {task: row["measurement_definition"]}, "synchronized": True}
        descriptor = self.write(f"reference_{task}.json", reference)
        self.references.append(descriptor)
        self.review["task_decisions"][task] = {"decision": "ACCEPTED", "reference_sha256": descriptor["sha256"],
                                               "rationale": "Synthetic fixture", "sample_size_rationale": "Synthetic fixture", "unknown_coverage_rationale": "Synthetic fixture"}

    def persist(self):
        evidence = {"dataset": self.write("dataset.json", self.dataset), "protocol": self.write("protocol.json", self.protocol),
                    "results": self.write("results.json", self.results), "references": self.references}
        for key in ("dataset", "protocol", "results"):
            self.review[f"{key}_sha256"] = evidence[key]["sha256"]
        evidence["review"] = self.write("review.json", self.review)
        self.receipt["evidence"] = evidence
        self.manifest["qualification_receipt"] = self.write("receipt.json", self.receipt)
        self.write("artifacts.json", self.manifest)

    def validate(self):
        self.persist()
        return validate_qualification_receipt(self.manifest, self.config, root=self.root, allow_synthetic=True)


@pytest.fixture
def fixture(tmp_path):
    return SyntheticReceipt(tmp_path)


def test_valid_synthetic_registry_recomputes_targets_and_is_never_production(fixture):
    result = fixture.validate()
    assert result["production_eligible"] is False
    assert result["qualified_tasks"]["visible_ball"]["metrics"]["recall"] == .96
    assert result["qualified_tasks"]["player_ground_position"]["metrics"]["p95_error_m"] == .1
    with pytest.raises(QualificationError, match="Synthetic/development"):
        validate_qualification_receipt(fixture.manifest, fixture.config, root=fixture.root)
    from src.detection.yolo26_bundle import validate_yolo26_config
    with pytest.raises(QualificationError, match="Synthetic/development"):
        validate_yolo26_config(fixture.config, check_assets=True, root=fixture.root)


def test_boolean_only_promotion_is_rejected_even_with_three_hashed_artifacts(fixture):
    fixture.manifest["qualification_receipt"] = None
    fixture.write("artifacts.json", fixture.manifest)
    from src.detection.yolo26_bundle import validate_yolo26_config
    with pytest.raises(QualificationError, match="independent tennis qualification receipt"):
        validate_yolo26_config(fixture.config, check_assets=True, root=fixture.root)


@pytest.mark.parametrize("name", ["receipt.json", "dataset.json", "protocol.json", "results.json", "review.json", "reference_calibration.json", "source2_ball.label"])
def test_any_retained_evidence_tamper_is_rejected(fixture, name):
    fixture.root.joinpath(name).write_text("tampered retained bytes")
    with pytest.raises(QualificationError, match="SHA256 mismatch"):
        validate_qualification_receipt(fixture.manifest, fixture.config, root=fixture.root, allow_synthetic=True)


def test_wrong_weight_or_changed_inference_and_tracker_settings_fail(fixture):
    fixture.receipt["model_sha256"]["ball"] = "f" * 64
    with pytest.raises(QualificationError, match="exact selected"):
        fixture.validate()
    fixture.receipt["model_sha256"]["ball"] = fixture.manifest["models"]["ball"]["sha256"]
    fixture.config["ball_detection"]["high_conf"] = .99
    with pytest.raises(QualificationError, match="Inference/tracker settings"):
        fixture.validate()


@pytest.mark.parametrize("section,settings", [("camera_intervals", {"max_source_gap_seconds": 99}),
                                            ("match_identity", {"max_colour_distance": 99})])
def test_camera_and_identity_operating_settings_are_bound_by_receipt(fixture, section, settings):
    fixture.config[section] = settings
    with pytest.raises(QualificationError, match="Inference/tracker settings"):
        fixture.validate()


@pytest.mark.parametrize("key", ["match_id", "session_id", "original_source_video_id", "sha256"])
def test_rehashed_group_leakage_still_fails(fixture, key):
    fixture.dataset["sources"][2][key] = fixture.dataset["sources"][0][key]
    with pytest.raises(QualificationError, match="Split leakage"):
        fixture.validate()


@pytest.mark.parametrize("mutation,match", [
    (lambda f: f.dataset["sources"][2].update(used_for_development=True), "used for development"),
    (lambda f: f.dataset["samples"][6].update(reviewer_type="AGENT_VISUAL_REVIEW"), "independent human"),
    (lambda f: f.dataset["samples"][6].update(annotation_provenance="MODEL_PREDICTION"), "final ground truth"),
    (lambda f: f.dataset["samples"][6].update(near_duplicate_group="unique0_ball"), "Sample split leakage"),
    (lambda f: f.dataset.update(frozen_at="2026-01-06T00:00:00Z"), "frozen before"),
    (lambda f: f.protocol.update(frozen_at="2026-01-06T00:00:00Z"), "frozen before"),
    (lambda f: f.review.update(reviewer_id="fixture_developer"), "external human reviewer"),
    (lambda f: f.protocol["tasks"]["calibration"].update(thresholds={}), "external predeclared"),
    (lambda f: f.results["tasks"]["calibration"].update(verification_lines_excluded_from_fit=False), "points AND lines"),
    (lambda f: f.protocol["tasks"]["visible_ball"]["thresholds"]["precision"].update(value=.8), "Cannot lower"),
])
def test_semantic_integrity_cannot_be_repaired_by_rehashing(fixture, mutation, match):
    mutation(fixture)
    with pytest.raises(QualificationError, match=match):
        fixture.validate()


@pytest.mark.parametrize("task,mutation", [
    ("visible_ball", lambda row: row["counts"].update(fp=20)),
    ("player_identity", lambda row: row["counts"].update(idfp=20)),
    ("player_ground_position", lambda row: row.update(absolute_errors=[.21] * 96)),
    ("player_ground_position", lambda row: row.update(absolute_errors=[.1] * 86 + [.6] * 10)),
    ("calibration", lambda row: row["metrics"].update(p95_error_m=.9)),
])
def test_numeric_failures_are_rejected_even_with_external_accepted_decision(fixture, task, mutation):
    mutation(fixture.results["tasks"][task])
    if task == "visible_ball":
        fixture.results["tasks"][task]["metrics"]["false_alarms_per_minute"] = 10
    with pytest.raises(QualificationError, match="Numerical qualification failed"):
        fixture.validate()


def test_missed_ball_frames_cannot_be_hidden_or_replaced_by_claimed_high_recall(fixture):
    row = fixture.results["tasks"]["visible_ball"]
    row["counts"]["fn"] = 1
    row["metrics"]["recall"] = .999
    with pytest.raises(QualificationError, match="include missed/unknown"):
        fixture.validate()


def test_empty_malformed_unknown_scope_and_original_units_fail(fixture):
    row = fixture.results["tasks"]["visible_ball"]
    row["sample_ids"].append("source0_ball")
    with pytest.raises(QualificationError, match="development/unregistered"):
        fixture.validate()
    row["sample_ids"].pop()
    row["unit"] = "resized_px"
    with pytest.raises(QualificationError, match="original source units"):
        fixture.validate()


@pytest.mark.parametrize("task,role", [("strokes", "ball"), ("bounces", "ball"), ("player_speed", "players"), ("physical_ball_speed", "ball"), ("point_winner", "ball")])
def test_optional_task_targets_are_separate_declared_capabilities(fixture, task, role):
    fixture.add_task(task, role)
    assert task in fixture.validate()["qualified_tasks"]


def test_retention_paths_duplicate_json_and_nonfinite_evidence_fail_closed(fixture):
    fixture.manifest["qualification_receipt"]["path"] = "../outside.json"
    with pytest.raises(QualificationError, match="escapes"):
        validate_qualification_receipt(fixture.manifest, fixture.config, root=fixture.root, allow_synthetic=True)
    path = fixture.root / "bad.json"
    for content, message in (("{\"schema_version\":1,\"schema_version\":2}", "Duplicate JSON"), ("{\"metric\":NaN}", "Nonfinite JSON")):
        path.write_text(content)
        fixture.manifest["qualification_receipt"] = {"path": "bad.json", "sha256": sha256_file(path)}
        with pytest.raises(QualificationError, match=message):
            validate_qualification_receipt(fixture.manifest, fixture.config, root=fixture.root, allow_synthetic=True)
