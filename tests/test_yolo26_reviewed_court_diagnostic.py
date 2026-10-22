"""Pure standard-library scoring tests; never import or construct an ML model."""

import builtins
import copy
import importlib
import json
import math
from types import SimpleNamespace

import pytest

from scripts.evaluate import diagnose_yolo26_reviewed_court as diagnostic


def ground_truth():
    points = [[100. + index * 10, 100., 2] for index in range(14)]
    points[13] = None
    return {"bbox_xyxy_px": [0., 0., 1000., 600.], "keypoints_px": points}


def prediction(points=None, boxes=None, scores=None, classes=None):
    points = points if points is not None else [[100. + index * 10, 100., .9] for index in range(14)]
    boxes = boxes if boxes is not None else [[0., 0., 1000., 600.]]
    return diagnostic.raw_candidates(boxes, scores if scores is not None else [.9] * len(boxes),
                                     classes if classes is not None else [0.] * len(boxes),
                                     [copy.deepcopy(points) for _ in boxes])


def test_import_has_no_numpy_torch_ultralytics_or_image_runtime(monkeypatch):
    original = builtins.__import__

    def pure_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "torch", "ultralytics", "psutil", "cv2", "PIL"}:
            pytest.fail(f"Pure scoring module imported runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", pure_import)
    importlib.reload(diagnostic)


def test_highest_confidence_wrong_box_is_not_replaced_using_ground_truth():
    candidates = prediction(boxes=[[1100., 0., 1250., 100.], [0., 0., 1000., 600.]], scores=[.9, .8])
    result = diagnostic.score_image(ground_truth(), candidates)
    summary = diagnostic.summarize_points(result["points"])
    assert result["selected_returned_index"] == 0
    assert result["instance_missing_reason"] == "SELECTED_BOX_IOU_BELOW_0_5"
    assert result["unmatched_extra_instances"] == 2
    assert summary["visible_gt_denominator"] == 13
    assert summary["recall_at_20_px"] == 0
    assert summary["median_error_native_px"] is None


def test_confidence_ties_use_original_index_and_nonclass0_is_retained():
    candidates = prediction(boxes=[[0., 0., 1000., 600.]] * 3, scores=[.9, .9, 1.], classes=[0., 0., 1.])
    result = diagnostic.score_image(ground_truth(), candidates)
    assert result["selected_returned_index"] == 0
    assert result["returned_instances"] == 3
    assert result["unmatched_extra_instances"] == 2
    assert len(result["raw_predictions"][2]["keypoints_xy_confidence_px"]) == 14


def test_inclusive_iou_point_confidence_and_pixel_threshold_boundaries():
    points = [[100. + index * 10, 100., .25] for index in range(14)]
    points[0][0] += 10
    points[1][0] += 20
    result = diagnostic.score_image(ground_truth(), prediction(points, boxes=[[0., 0., 500., 600.]]))
    assert result["selected_bbox_iou"] == .5
    assert result["points"][0]["within_10_px"] is True
    assert result["points"][1]["within_20_px"] is True
    assert result["points"][1]["within_10_px"] is False


def test_point_confidence_below_half_is_retained_and_scored_at_protocol_quarter():
    points = [[100. + index * 10, 100., .3] for index in range(14)]
    result = diagnostic.score_image(ground_truth(), prediction(points))
    assert diagnostic.summarize_points(result["points"])["recall_at_10_px"] == 1
    assert result["raw_predictions"][0]["keypoints_xy_confidence_px"][0] == [100., 100., .3]


def test_all_visible_points_remain_denominator_for_lowconf_nonfinite_and_large_error():
    points = [[100. + index * 10, 100., .9] for index in range(14)]
    points[0][2] = .24999
    points[1][0] = math.nan
    points[2][0] += 100
    candidates = prediction(points)
    result = diagnostic.score_image(ground_truth(), candidates)
    summary = diagnostic.summarize_points(result["points"])
    assert summary["visible_gt_denominator"] == 13
    assert summary["unknown_gt_excluded"] == 1
    assert summary["finite_confident_matched_predictions"] == 11
    assert summary["within_20_px"] == 10
    assert summary["recall_at_20_px"] == pytest.approx(10 / 13)
    assert 100. in summary["errors_native_px"]
    assert summary["p95_error_native_px"] == pytest.approx(50.)
    assert candidates[0]["keypoints_xy_confidence_px"][1][0] is None
    assert "keypoint[1][0]" in candidates[0]["nonfinite_fields"]
    json.dumps(result, allow_nan=False)


def test_no_predictions_is_complete_miss_with_explicit_zero_coverage():
    result = diagnostic.score_image(ground_truth(), [])
    summary = diagnostic.summarize_points(result["points"])
    assert result["complete_instance_miss"] is True
    assert result["complete_visible_keypoint_miss"] is True
    assert summary["missing_reasons"] == {"NO_RETURNED_INSTANCE": 13}
    assert summary["coverage"] == 0
    assert summary["median_error_native_px"] is None
    assert summary["p95_error_native_px"] is None
    assert result["points"][13]["missing_or_exclusion_reason"] == "UNKNOWN_GT_EXCLUDED"


def test_missing_keypoint_data_does_not_become_zero_error():
    candidates = diagnostic.raw_candidates([[0., 0., 1000., 600.]], [.9], [0.], None)
    result = diagnostic.score_image(ground_truth(), candidates)
    assert result["selected_instance_matched"] is True
    assert diagnostic.summarize_points(result["points"])["missing_reasons"] == {"KEYPOINT_DATA_NOT_RETURNED": 13}


def test_extreme_finite_values_cannot_create_nonfinite_metric_json():
    points = [[100. + index * 10, 100., .9] for index in range(14)]
    points[0] = [1.7e308, 1.7e308, .9]
    result = diagnostic.score_image(ground_truth(), prediction(points))
    assert result["points"][0]["error_native_px"] is None
    json.dumps(result, allow_nan=False)


def test_even_finite_large_errors_have_finite_median_and_aggregate_json():
    target = ground_truth()
    target["keypoints_px"] = [[100., 100., 2], [110., 100., 2]] + [None] * 12
    points = [[1e308, 100., .9]] * 14
    assessment = diagnostic.score_image(target, prediction(points))
    summary = diagnostic.summarize_points(assessment["points"])
    assert summary["median_error_native_px"] == 1e308
    assert summary["p95_error_native_px"] == 1e308
    json.dumps(diagnostic.aggregate([{"context_group": "synthetic", "assessment": assessment}]), allow_nan=False)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_nonfinite_boxes_or_confidences_stay_raw_null_and_fail_visible_recall(bad):
    candidates = prediction(boxes=[[bad, 0., 1000., 600.]])
    result = diagnostic.score_image(ground_truth(), candidates)
    assert result["instance_missing_reason"] == "SELECTED_BOX_INVALID_OR_NONFINITE"
    assert result["raw_predictions"][0]["box_xyxy_px"][0] is None
    json.dumps(result, allow_nan=False)
    candidates = prediction(scores=[bad])
    assert diagnostic.score_image(ground_truth(), candidates)["selected_returned_index"] is None


def test_per_landmark_and_context_denominators_include_whole_image_misses():
    rows = [{"context_group": "context_A", "assessment": diagnostic.score_image(ground_truth(), prediction())},
            {"context_group": "context_B", "assessment": diagnostic.score_image(ground_truth(), [])}]
    result = diagnostic.aggregate(rows)
    assert result["all_visible_keypoints"]["visible_gt_denominator"] == 26
    assert result["all_visible_keypoints"]["recall_at_10_px"] == .5
    assert result["per_landmark"][0]["visible_gt_denominator"] == 2
    assert result["per_landmark"][13]["visible_gt_denominator"] == 0
    assert result["per_context"][1]["coverage"] == 0
    assert result["complete_instance_misses"] == 1


def test_guard_requires_each_unchanged_launch_floor():
    resources = {"free_disk_bytes": 1024 ** 3, "free_ram_bytes": 1536 * 1024 ** 2, "free_vram_bytes": 2 * 1024 ** 3}
    diagnostic.check_resources(resources)
    for field in resources:
        blocked = dict(resources)
        blocked[field] -= 1
        with pytest.raises(ValueError, match="guard requires"):
            diagnostic.check_resources(blocked)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, True, "2000000000"])
def test_guard_rejects_nonfinite_or_nonnumeric_resource_readings(bad):
    resources = {"free_disk_bytes": 1024 ** 3, "free_ram_bytes": 1536 * 1024 ** 2, "free_vram_bytes": 2 * 1024 ** 3}
    for field in resources:
        blocked = dict(resources, **{field: bad})
        with pytest.raises(ValueError, match="guard requires"):
            diagnostic.check_resources(blocked)


def test_protocol_cannot_change_threshold_split_or_image_count():
    protocol = diagnostic.read_json(diagnostic.PROTOCOL)
    diagnostic.validate_protocol(protocol)
    for field, value in [("split", "test"), ("expected_images", 19), ("keypoint_confidence_minimum", .2),
                         ("matched_instance_bbox_iou_minimum", .4), ("qualification_evidence", True)]:
        changed = copy.deepcopy(protocol)
        changed[field] = value
        with pytest.raises(ValueError, match="unchanged predeclared"):
            diagnostic.validate_protocol(changed)


def test_court_head_cannot_lose_a_semantic_slot():
    points = [[100., 100., .9]] * 13
    with pytest.raises(ValueError, match="fourteen"):
        prediction(points)


def test_model_validation_requires_the_actual_yolo26_court14_pose_head():
    class Pose26:
        kpt_shape = [14, 3]
        end2end = True

    model = SimpleNamespace(task="pose", names={0: "court"}, model=SimpleNamespace(
        yaml={"scale": "s", "yaml_file": "yolo26s-court-pose.yaml", "nc": 1, "kpt_shape": [14, 3]}, model=[Pose26()]))
    assert diagnostic.validate_model(model)["head_class"] == "Pose26"
    model.model.yaml["nc"] = 2
    with pytest.raises(ValueError, match="court Pose14"):
        diagnostic.validate_model(model)
    model.model.yaml["nc"] = 1
    model.model.model[0].kpt_shape = [17, 3]
    with pytest.raises(ValueError, match="court Pose14"):
        diagnostic.validate_model(model)


def monitored_receipt():
    return {"schema": "tennis.yolo26.controlled-court-execution.v1", "status": "COMPLETED_PRIVATE_TRAINED_UNQUALIFIED",
            "exit_code": 0, "source_freeze_verified": True, "changed_runtime_inputs": [], "added_runtime_sources": [],
            "trainer_status": "TRAINED_UNQUALIFIED", "metadata_sha256": "a" * 64, "runtime_input_count": 861,
            "qualification_evidence": False, "final_test_used": False}


def test_completed_root_monitor_requires_exact_unchanged_source_snapshots():
    snapshot = {f"synthetic/path{index}": {"sha256": "b" * 64, "bytes": 1} for index in range(861)}
    diagnostic.validate_completed_execution(monitored_receipt(), "a" * 64, snapshot, copy.deepcopy(snapshot))
    changed = copy.deepcopy(snapshot)
    changed["synthetic/path0"]["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="all861"):
        diagnostic.validate_completed_execution(monitored_receipt(), "a" * 64, snapshot, changed)


@pytest.mark.parametrize(("field", "value"), [
    ("status", "RUNNING_PRIVATE_UNQUALIFIED"), ("exit_code", 1), ("exit_code", False),
    ("source_freeze_verified", False), ("changed_runtime_inputs", ["changed.py"]),
    ("added_runtime_sources", ["added.py"]), ("trainer_status", "TRAINING_STARTED_UNQUALIFIED"),
    ("metadata_sha256", "wrong"), ("runtime_input_count", 860), ("qualification_evidence", True),
])
def test_root_monitor_cannot_be_replaced_by_success_boolean_or_unbound_metadata(field, value):
    receipt = monitored_receipt()
    receipt[field] = value
    snapshot = {f"synthetic/path{index}": {} for index in range(861)}
    with pytest.raises(ValueError, match="root monitoring"):
        diagnostic.validate_completed_execution(receipt, "a" * 64, snapshot, snapshot)


def initial20_metadata():
    return {"status": "TRAINED_UNQUALIFIED", "role": "court", "mode": "REVIEWED_STATIC_COURT_DEVELOPMENT",
            "training_script_sha256": diagnostic.TRAINER_SHA256, "extension_review": None,
            "initial_epoch_protocol": 20, "training_arguments": {"epochs": 20},
            "qualification_evidence": False, "final_test_used": False, "independent_human_reviewed": False,
            "data_audit": {"manifest_sha256": diagnostic.MANIFEST_SHA256, "split_image_counts": {"train": 100, "val": 20, "test": 0},
                           "static_image_timing_is_unknown": True, "final_test_used": False},
            "optimizer_update_receipts": [{"epoch": 20}]}


@pytest.mark.parametrize(("section", "field", "value"), [
    (None, "status", "TRAINING_STARTED_UNQUALIFIED"), ("training_arguments", "epochs", 19),
    ("training_arguments", "epochs", 21), (None, "extension_review", {"approved": True}),
    ("data_audit", "static_image_timing_is_unknown", False), (None, "qualification_evidence", True),
])
def test_initial20_metadata_gate_rejects_active_short_extended_or_qualified_runs(section, field, value):
    metadata = initial20_metadata()
    diagnostic.validate_initial20_metadata(metadata)
    (metadata[section] if section else metadata)[field] = value
    with pytest.raises(ValueError, match="completed unqualified initial20"):
        diagnostic.validate_initial20_metadata(metadata)


def test_initial20_metadata_needs_actual_epoch20_optimizer_receipt():
    metadata = initial20_metadata()
    metadata["optimizer_update_receipts"] = [{"epoch": 19}]
    with pytest.raises(ValueError, match="actually completed"):
        diagnostic.validate_initial20_metadata(metadata)


@pytest.mark.parametrize("checkpoints", [[], [{"path": str(diagnostic.CHECKPOINT), "sha256": "wrong"}],
                                        [{"path": str(diagnostic.CHECKPOINT.with_name("last.pt")), "sha256": "a" * 64}]])
def test_diagnostic_never_substitutes_unbound_best_or_last_checkpoint(checkpoints):
    with pytest.raises(ValueError, match="best.pt must be bound"):
        diagnostic.validate_best_checkpoint_binding({"checkpoints": checkpoints}, {"sha256": "a" * 64})


@pytest.mark.parametrize("epochs", [list(range(1, 20)), list(range(1, 22)), list(range(20, 0, -1)),
                                   list(range(1, 20)) + [19]])
def test_initial20_csv_rejects_short_extended_reordered_or_duplicate_epochs(tmp_path, epochs):
    path = tmp_path / "synthetic_results.csv"
    path.write_text("epoch,val/pose_loss\n" + "".join(f"{epoch},2.0\n" for epoch in epochs), encoding="utf-8")
    with pytest.raises(ValueError, match="chronological epochs1 through20"):
        diagnostic.validate_initial20_csv(path)
