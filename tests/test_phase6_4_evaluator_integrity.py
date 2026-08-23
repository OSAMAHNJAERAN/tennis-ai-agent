"""Scientific-integrity regressions for the Phase 6.4 evaluator."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.events.event_evaluator import (
    canonical_one_to_one_matches,
    evaluate_covered_events,
    validate_annotation_coverage,
)


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "artifacts" / "validation"


def _pred(video_id="video_A", frame=100, **extra):
    return {"video_id": video_id, "frame": frame, **extra}


def _gt(video_id="video_A", frame=100, **extra):
    return {
        "video_id": video_id,
        "event_id": extra.pop("event_id", 1),
        "frame_best": frame,
        "frame_min": frame,
        "frame_max": frame,
        **extra,
    }


def _matches(predictions, gt, *, fps=30.0, **kwargs):
    return canonical_one_to_one_matches(predictions, gt, fps=fps, **kwargs)


def test_prediction_cannot_match_gt_from_another_video():
    assert _matches([_pred("video_A")], [_gt("video_B")]) == []


def test_identical_frames_across_videos_cannot_match():
    assert _matches([_pred("video_A", 55)], [_gt("video_B", 55)]) == []


def test_identical_timestamps_across_videos_cannot_match():
    prediction = _pred("video_A", 55, timestamp_s=1.0)
    gt = _gt("video_B", 30)
    assert _matches([prediction], [gt]) == []


def test_same_video_inside_tolerance_matches():
    assert len(_matches([_pred(frame=106)], [_gt(frame=100)], tolerance_s=0.2)) == 1


def test_same_video_outside_tolerance_fails():
    assert _matches([_pred(frame=107)], [_gt(frame=100)], tolerance_s=0.2) == []


def test_one_prediction_cannot_match_multiple_gt_events():
    assert len(_matches([_pred()], [_gt(event_id=1), _gt(event_id=2)])) == 1


def test_one_gt_cannot_match_multiple_predictions():
    assert len(_matches([_pred(frame=99), _pred(frame=101)], [_gt()])) == 1


def test_duplicate_prediction_remains_unmatched():
    matches = _matches([_pred(), _pred()], [_gt()])
    assert len(matches) == 1
    assert len({prediction_index for prediction_index, _, _ in matches}) == 1


def test_exact_event_type_requirement():
    prediction = _pred(event_type="BOUNCE")
    gt = _gt(event_type="PLAYER_HIT")
    assert _matches([prediction], [gt], require_event_type=True) == []
    assert len(_matches([prediction], [gt], require_event_type=False)) == 1


def test_player_requirement_for_player_contact():
    prediction = _pred(event_type="PLAYER_HIT", player_id=1)
    gt = _gt(event_type="PLAYER_HIT", player_id=2)
    assert _matches([prediction], [gt], require_player=True) == []


@pytest.mark.parametrize("prediction_type", ["PLAYER_1_HIT", "PLAYER_2_HIT"])
def test_numbered_player_hit_normalizes_to_player_hit(prediction_type):
    prediction = _pred(event_type=prediction_type)
    gt = _gt(event_type="PLAYER_HIT")
    assert len(_matches([prediction], [gt], require_event_type=True)) == 1


def test_bounce_does_not_require_player_attribution():
    prediction = _pred(event_type="BOUNCE", player_id=None)
    gt = _gt(event_type="BOUNCE", player_id=2)
    assert len(_matches([prediction], [gt], require_event_type=True, require_player=True)) == 1


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_native_fps_temporal_equivalence(fps):
    gt_frame = int(fps * 2)
    prediction = _pred(frame=gt_frame, timestamp_s=2.2)
    gt = _gt(frame=gt_frame)
    matches = _matches([prediction], [gt], fps=fps, tolerance_s=0.2)
    assert len(matches) == 1
    assert matches[0][2] == pytest.approx(0.2)


def test_gt_uncertainty_interval_matching():
    gt = _gt(frame=102)
    gt.update(frame_min=100, frame_max=104)
    matches = _matches([_pred(frame=103)], [gt], tolerance_s=0.0)
    assert matches[0][2] == 0.0


def test_200_ms_boundary_is_deterministic():
    gt = _gt(frame=30)
    assert len(_matches([_pred(frame=36)], [gt], tolerance_s=0.2)) == 1
    assert _matches([_pred(frame=37)], [gt], tolerance_s=0.2) == []


def test_missing_video_id_in_multi_video_mode_fails_closed():
    with pytest.raises(ValueError, match="Missing video_id"):
        _matches([_pred("video_A")], [{"frame_best": 100}])


@pytest.fixture
def video_manifest():
    return {
        "videos": {
            "video_A": {"fps": 30.0, "frame_count": 300},
            "video_B": {"fps": 60.0, "frame_count": 600},
        }
    }


@pytest.fixture
def coverage():
    return {
        "schema_version": "1.0",
        "coverage_version": "1.0",
        "evaluation_scope_id": "TEST_SCOPE",
        "annotation_scope": "PHYSICAL_EVENTS_EXHAUSTIVE",
        "coverage_basis": "RAW_VIDEO",
        "videos": {
            "video_A": {
                "video_id": "video_A",
                "fps": 30.0,
                "frame_count": 300,
                "fully_reviewed_intervals": [
                    {
                        "start_frame": 0,
                        "end_frame": 149,
                        "start_timestamp_s": 0.0,
                        "end_timestamp_s": 149 / 30.0,
                        "physical_event_annotation_complete": True,
                    }
                ],
            },
            "video_B": {
                "video_id": "video_B",
                "fps": 60.0,
                "frame_count": 600,
                "fully_reviewed_intervals": [
                    {
                        "start_frame": 0,
                        "end_frame": 299,
                        "start_timestamp_s": 0.0,
                        "end_timestamp_s": 299 / 60.0,
                        "physical_event_annotation_complete": True,
                    }
                ],
            },
        },
    }


def test_unmatched_prediction_inside_negative_coverage_is_fp(video_manifest, coverage):
    result = evaluate_covered_events(
        {"video_A": [_pred("video_A", 120)]}, {"video_A": []}, video_manifest, coverage
    )
    assert result["aggregate"]["false_positives"] == 1


def test_prediction_outside_coverage_is_not_fp(video_manifest, coverage):
    result = evaluate_covered_events(
        {"video_A": [_pred("video_A", 200)]}, {"video_A": []}, video_manifest, coverage
    )
    assert result["aggregate"]["false_positives"] == 0
    assert result["aggregate"]["outside_scope_prediction_count"] == 1
    assert result["prediction_statuses"][0]["evaluation_status"] == "OUTSIDE_SCOPE"


def test_gt_inside_coverage_matches_normally(video_manifest, coverage):
    result = evaluate_covered_events(
        {"video_A": [_pred("video_A", 100)]},
        {"video_A": [_gt("video_A", 100)]},
        video_manifest,
        coverage,
    )
    assert result["aggregate"]["true_positives"] == 1


def test_aggregate_counts_equal_sum_of_per_video(video_manifest, coverage):
    result = evaluate_covered_events(
        {
            "video_A": [_pred("video_A", 100), _pred("video_A", 120)],
            "video_B": [_pred("video_B", 200)],
        },
        {
            "video_A": [_gt("video_A", 100), _gt("video_A", 130, event_id=2)],
            "video_B": [_gt("video_B", 200)],
        },
        video_manifest,
        coverage,
    )
    for key in ("true_positives", "false_positives", "false_negatives"):
        assert result["aggregate"][key] == sum(
            metric[key] for metric in result["per_video"].values()
        )


def test_coverage_intervals_cannot_overlap(video_manifest, coverage):
    invalid = deepcopy(coverage)
    invalid["videos"]["video_A"]["fully_reviewed_intervals"].append(
        {
            "start_frame": 149,
            "end_frame": 200,
            "start_timestamp_s": 149 / 30.0,
            "end_timestamp_s": 200 / 30.0,
            "physical_event_annotation_complete": True,
        }
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_annotation_coverage(invalid, video_manifest)


def test_coverage_bounds_stay_inside_frame_count(video_manifest, coverage):
    invalid = deepcopy(coverage)
    invalid["videos"]["video_A"]["fully_reviewed_intervals"][0]["end_frame"] = 300
    with pytest.raises(ValueError, match="bounds"):
        validate_annotation_coverage(invalid, video_manifest)


def test_coverage_fps_and_timestamps_agree_with_media(video_manifest, coverage):
    invalid_fps = deepcopy(coverage)
    invalid_fps["videos"]["video_A"]["fps"] = 24.0
    with pytest.raises(ValueError, match="FPS"):
        validate_annotation_coverage(invalid_fps, video_manifest)
    invalid_timestamp = deepcopy(coverage)
    invalid_timestamp["videos"]["video_A"]["fully_reviewed_intervals"][0][
        "end_timestamp_s"
    ] = 99.0
    with pytest.raises(ValueError, match="timestamp"):
        validate_annotation_coverage(invalid_timestamp, video_manifest)


def test_prediction_output_cannot_define_coverage(video_manifest, coverage):
    invalid = deepcopy(coverage)
    invalid["coverage_basis"] = "MODEL_PREDICTIONS"
    with pytest.raises(ValueError, match="RAW_VIDEO"):
        validate_annotation_coverage(invalid, video_manifest)


def test_authoritative_coverage_document_validates_against_manifest():
    actual_coverage = json.loads(
        (ROOT / "data/benchmarks/cross_match_final_holdout/annotation_coverage.json").read_text(
            encoding="utf-8"
        )
    )
    actual_manifest = json.loads(
        (ROOT / "data/benchmarks/cross_match_final_holdout/videos.json").read_text(
            encoding="utf-8"
        )
    )
    validate_annotation_coverage(actual_coverage, actual_manifest)
    assert actual_coverage["review_provenance"]["prediction_inputs_used"] is False


def test_corrected_artifacts_share_evaluator_provenance_and_count_invariants():
    names = (
        "phase6_4_corrected_per_video_metrics.json",
        "phase6_4_corrected_aggregate_metrics.json",
        "phase6_4_corrected_stage_metrics.json",
        "phase6_4_corrected_event_lineage.json",
        "phase6_4_corrected_fp_manifest.json",
        "phase6_4_fp_human_review_manifest_v2.json",
    )
    artifacts = [json.loads((VALIDATION / name).read_text(encoding="utf-8")) for name in names]
    signatures = {
        (
            artifact["evaluator_provenance"]["phase6_4_evaluator_version"],
            artifact["evaluator_provenance"]["evaluator_sha256"],
            artifact["evaluator_provenance"]["gt_sha256"],
            artifact["evaluator_provenance"]["annotation_coverage_sha256"],
            artifact["evaluator_provenance"]["video_manifest_sha256"],
            artifact["evaluator_provenance"]["config_sha256"],
        )
        for artifact in artifacts
    }
    assert len(signatures) == 1
    aggregate = artifacts[1]
    assert all(aggregate["count_invariants"].values())


def test_corrected_fp_and_outside_scope_population_contract():
    manifest = json.loads(
        (VALIDATION / "phase6_4_corrected_fp_manifest.json").read_text(encoding="utf-8")
    )
    fps = [row for row in manifest["records"] if row["evaluation_status"] == "FP"]
    outside = [
        row for row in manifest["records"] if row["evaluation_status"] == "OUTSIDE_SCOPE"
    ]
    assert len(fps) == manifest["fp_count"]
    assert len(outside) == manifest["outside_scope_count"]
    assert all(row["inside_evaluation_coverage"] for row in fps)
    assert all(not row["inside_evaluation_coverage"] for row in outside)
    assert all(row["human_reviewed"] is False for row in manifest["records"])
    assert all(row["human_label"] is None for row in manifest["records"])


def test_corrected_lineage_has_required_same_video_proof_fields():
    lineage = json.loads(
        (VALIDATION / "phase6_4_corrected_event_lineage.json").read_text(encoding="utf-8")
    )
    required = {
        "video_id",
        "evaluation_scope_id",
        "inside_annotation_coverage",
        "gt_event_id",
        "prediction_id",
        "same_video_match",
        "timing_error_s",
        "first_failure_stage",
    }
    assert lineage["record_count"] == 40
    assert lineage["cross_video_match_count"] == 0
    assert all(required <= set(record) for record in lineage["records"])


def test_production_inference_modules_cannot_access_evaluation_metadata():
    production_paths = (
        ROOT / "src/events/event_detector.py",
        ROOT / "src/tracking/temporal_ball_tracker.py",
        ROOT / "src/pipeline/phase6_pipeline.py",
        ROOT / "src/shot_analysis/shot_classifier.py",
    )
    forbidden = ("annotation_coverage", "ground_truth_events", "human_label")
    for path in production_paths:
        content = path.read_text(encoding="utf-8").lower()
        assert all(token not in content for token in forbidden)
