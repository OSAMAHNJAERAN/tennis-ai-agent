"""Integrity checks for the measured Phase 6.4 recovery artifacts."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "artifacts" / "validation"


def _load(name):
    with (VALIDATION / name).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_forensic_table_covers_every_diagnostic_gt_event_once():
    artifact = _load("phase6_4_event_forensic_table.json")
    identities = {
        (row["video_id"], row["gt_event_id"])
        for row in artifact["records"]
    }

    assert artifact["qualification_evidence"] is False
    assert artifact["record_count"] == 40
    assert len(identities) == 40
    assert all(row["gt_timestamp_provenance"] == "DERIVED_FRAME_BEST_DIVIDED_BY_FPS" for row in artifact["records"])


def test_false_negative_taxonomies_are_complete_and_separate():
    artifact = _load("phase6_4_false_negative_audit.json")

    assert artifact["pre_recovery_audit"]["false_negative_count"] == 36
    assert sum(artifact["pre_recovery_audit"]["false_negative_taxonomy"].values()) == 36
    assert artifact["post_recovery_false_negative_count"] == sum(artifact["post_recovery_taxonomy"].values())
    assert len(artifact["post_recovery_records"]) == artifact["post_recovery_false_negative_count"]


def test_gt_event_survival_is_cumulative_and_monotonic():
    artifact = _load("phase6_4_event_stage_metrics.json")
    expected = [
        "raw_candidate_generator",
        "physics_verification",
        "player_attribution",
        "event_type_classification",
        "temporal_suppression",
        "final_authoritative_events",
    ]
    rows = artifact["final_stage_metrics"]
    recalls = [row["recall"]["OVERALL"] for row in rows]

    assert [row["stage"] for row in rows] == expected
    assert recalls == sorted(recalls, reverse=True)
    assert recalls[0] == 0.625
    assert recalls[-1] >= 0.15


def test_recovery_result_is_an_explicit_failed_diagnostic_gate():
    artifact = _load("phase6_4_event_recovery_results.json")
    gate = artifact["readiness_gate"]
    final = artifact["final_metrics"]["event_detection"]

    assert artifact["scientific_split"] == "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED"
    assert artifact["qualification_evidence"] is False
    assert final["true_positives"] + final["false_negatives"] == 40
    assert gate["passed"] is False
    assert gate["ready_for_production_evaluator_freeze"] is False
    assert "video_10" in gate["primary_remaining_blocker"]


def test_recovery_input_hashes_match_current_sources_and_exclude_pristine_media():
    artifact = _load("phase6_4_event_recovery_results.json")
    provenance = artifact["input_provenance"]
    flat_records = [
        provenance["ground_truth_events"],
        provenance["video_manifest"],
        provenance["active_event_config"],
        provenance["event_detector_implementation"],
        provenance["recovery_evaluator"],
    ]
    for video in provenance["preserved_video_inputs"].values():
        flat_records.extend(video.values())

    for record in flat_records:
        source = ROOT / record["path"]
        assert source.is_file()
        assert _sha256(source) == record["sha256"]
    serialized = json.dumps(provenance)
    assert "video_11" not in serialized
    assert "video_12" not in serialized
    assert "video_13" not in serialized
