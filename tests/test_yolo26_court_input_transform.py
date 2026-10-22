"""Pure transform arithmetic, forward association and paired scoring; no ML runtime."""

import builtins
import copy
import importlib
import json

import pytest

from scripts.evaluate import diagnose_yolo26_court_input_transform as diagnostic


def target():
    return {"bbox_xyxy_px": [0., 0., 1000., 600.],
            "keypoints_px": [[100. + index * 10, 100., 2] for index in range(13)] + [None]}


def assessment(*, shift=0., missing=False, low_confidence_index=None):
    points = [[100. + index * 10 + shift, 100., .9] for index in range(14)]
    if low_confidence_index is not None:
        points[low_confidence_index][2] = .24
    candidates = [] if missing else diagnostic.reference.raw_candidates(
        [[0., 0., 1000., 600.]], [.9], [0.], [points])
    return diagnostic.reference.score_image(target(), candidates)


def rows(*assessments):
    return [{"id": f"image{index}", "context_group": f"context{index}", "assessment": value}
            for index, value in enumerate(assessments)]


def test_import_and_helpers_do_not_load_image_or_ml_runtime(monkeypatch):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "numpy", "ultralytics", "cv2", "PIL", "psutil"}:
            pytest.fail(f"Preflight imported heavy runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    importlib.reload(diagnostic)
    assert diagnostic.calibration_roundtrip(diagnostic.letterbox_geometry([720, 1280], rect=True))["maximum_absolute_error_native_px"] == 0
    assert diagnostic.paired_report(rows(assessment()), rows(assessment()))["all_visible"]["both_available"] == 13


@pytest.mark.parametrize(("rect", "canvas", "padding"), [
    (True, [736, 1280], [0, 8, 0, 8]), (False, [1280, 1280], [0, 280, 0, 280]),
])
def test_native_1280x720_inputs_have_exact_predeclared_canvas_and_padding(rect, canvas, padding):
    geometry = diagnostic.letterbox_geometry([720, 1280], rect=rect)
    assert geometry["canvas_hw"] == canvas
    assert geometry["padding_ltrb"] == padding
    assert geometry["resized_wh"] == [1280, 720]
    assert geometry["scale_xy"] == [1., 1.]
    calibration = diagnostic.calibration_roundtrip(geometry)
    assert calibration["uses_predictions_or_labels"] is False
    for row in calibration["probes"]:
        assert row["recovered_native_xy"] == pytest.approx(row["native_xy"], abs=1e-9, rel=0)


def test_odd_padding_preserves_integer_content_origin_and_subpixel_coordinate_roundtrip():
    geometry = diagnostic.letterbox_geometry([719, 1280], rect=True)
    assert geometry["padding_ltrb"] == [0, 8, 0, 9]
    assert geometry["canvas_hw"] == [736, 1280]
    point = [123.25, 456.75]
    assert diagnostic.map_native(point, geometry) == [123.25, 464.75]
    assert diagnostic.map_back(diagnostic.map_native(point, geometry), geometry) == point


def test_resizing_math_uses_uniform_ratio_and_keeps_non_square_geometry():
    geometry = diagnostic.letterbox_geometry([1080, 1920], rect=False)
    assert geometry["scale_xy"] == [2 / 3, 2 / 3]
    assert geometry["resized_wh"] == [1280, 720]
    assert geometry["padding_ltrb"] == [0, 280, 0, 280]
    assert diagnostic.calibration_roundtrip(geometry)["maximum_absolute_error_native_px"] <= 1e-9


@pytest.mark.parametrize("changes", [
    {"native_hw": [0, 1280]}, {"native_hw": [720., 1280]}, {"native_hw": [True, 1280]},
    {"target_hw": [1280, 0]}, {"stride": 0}, {"stride": True}, {"rect": 1},
])
def test_invalid_geometry_is_rejected(changes):
    arguments = {"native_hw": [720, 1280], "rect": True, **changes}
    with pytest.raises(ValueError, match="dimensions/stride"):
        diagnostic.letterbox_geometry(**arguments)


def record_preprocess(trace, shape=(1, 3, 736, 1280)):
    trace.record_preprocess({"actual_network_shape_nchw": list(shape)})


def test_square_warmup_cannot_be_misattributed_as_rectangular_image_forward():
    trace = diagnostic.ForwardTrace()
    trace.begin_image("rect", "image0")
    trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=False)
    record_preprocess(trace)
    with trace.image_inference():
        trace.observe([1, 3, 736, 1280], "torch.float32", "cuda:0", done_warmup=True)
    observation = trace.end_image()
    assert observation["forward"]["id"] == "image0"
    assert observation["forward"]["phase"] == "ACTUAL_IMAGE_INFERENCE"
    warmup = observation["preceding_non_image_warmup_calls"][0]
    assert warmup["id"] is None
    assert warmup["triggering_invocation_image_id"] == "image0"
    assert "NOT_ASSUMED_ZERO" in warmup["input_values_policy"]


def test_same_square_shape_is_still_separated_by_actual_predictor_phase():
    trace = diagnostic.ForwardTrace()
    trace.begin_image("square", "image0")
    trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=False)
    record_preprocess(trace, [1, 3, 1280, 1280])
    with trace.image_inference():
        trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=True)
    assert len(trace.end_image()["preceding_non_image_warmup_calls"]) == 1
    trace.begin_image("square", "image1")
    record_preprocess(trace, [1, 3, 1280, 1280])
    with trace.image_inference():
        trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=True)
    assert trace.end_image()["preceding_non_image_warmup_calls"] == []


def test_unbound_and_unclassified_forwards_are_rejected():
    trace = diagnostic.ForwardTrace()
    with pytest.raises(ValueError, match="outside"):
        trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=False)
    trace.begin_image("rect", "image0")
    with pytest.raises(ValueError, match="Unclassified"):
        trace.observe([1, 3, 1280, 1280], "torch.float32", "cuda:0", done_warmup=True)
    with pytest.raises(ValueError, match="requires an observed"):
        with trace.image_inference():
            pass
    record_preprocess(trace)
    with pytest.raises(ValueError, match="exactly one observed"):
        record_preprocess(trace)
    with pytest.raises(ValueError, match="Exactly one actual"):
        trace.end_image()


@pytest.mark.parametrize(("shape", "dtype"), [([1, 3, 1280, 1280], "torch.float32"), ([1, 3, 736, 1280], "torch.float16")])
def test_actual_forward_must_match_recorded_shape_and_float32(shape, dtype):
    trace = diagnostic.ForwardTrace()
    trace.begin_image("rect", "image0")
    record_preprocess(trace)
    with trace.image_inference(), pytest.raises(ValueError, match="differs"):
        trace.observe(shape, dtype, "cuda:0", done_warmup=True)


def test_two_actual_forwards_for_one_image_are_not_silently_counted_as_one():
    trace = diagnostic.ForwardTrace()
    trace.begin_image("rect", "image0")
    record_preprocess(trace)
    with trace.image_inference():
        for _ in range(2):
            trace.observe([1, 3, 736, 1280], "torch.float32", "cuda:0", done_warmup=True)
    with pytest.raises(ValueError, match="Exactly one actual"):
        trace.end_image()


def test_paired_report_keeps_whole_image_miss_and_low_confidence_in_denominator():
    rect = rows(assessment(shift=30.), assessment(missing=True))
    square = rows(assessment(shift=5., low_confidence_index=0), assessment(shift=15.))
    report = diagnostic.paired_report(rect, square)
    summary = report["all_visible"]
    assert summary["visible_gt_denominator"] == 26
    assert summary["unknown_gt_excluded"] == 2
    assert summary["both_available"] == 12
    assert summary["rect_only_available"] == 1
    assert summary["square_only_available"] == 13
    assert summary["both_missing"] == 0
    assert summary["rect_recall_at_20_px"] == 0
    assert summary["square_recall_at_20_px"] == pytest.approx(25 / 26)
    assert summary["common_available_square_minus_rect_median_error_px"] == -25.
    assert summary["rect_missing_reasons"] == {"NO_RETURNED_INSTANCE": 13}
    assert report["per_landmark"][0]["rect_only_available"] == 1
    assert report["per_landmark"][13]["visible_gt_denominator"] == 0
    assert report["per_context"][1]["square_only_available"] == 13
    assert len(report["points"]) == 28
    json.dumps(report, allow_nan=False)


def test_two_missing_arms_have_null_error_delta_and_zero_recall():
    report = diagnostic.paired_report(rows(assessment(missing=True)), rows(assessment(missing=True)))
    assert report["all_visible"]["both_missing"] == 13
    assert report["all_visible"]["common_available_square_minus_rect_median_error_px"] is None
    assert report["all_visible"]["square_recall_at_10_px"] == 0
    assert report["all_visible"]["rect_recall_at_10_px"] == 0


@pytest.mark.parametrize("mutation", ["id", "context", "target", "length", "slots"])
def test_pairing_rejects_population_or_ground_truth_changes(mutation):
    first, second = rows(assessment()), rows(assessment())
    if mutation == "id":
        second[0]["id"] = "different"
    elif mutation == "context":
        second[0]["context_group"] = "different"
    elif mutation == "target":
        second[0]["assessment"]["points"][0]["gt_xy_visibility_px"][0] += 1
    elif mutation == "length":
        second.append(copy.deepcopy(second[0]))
    else:
        second[0]["assessment"]["points"].pop()
    with pytest.raises(ValueError):
        diagnostic.paired_report(first, second)


def test_exact_protocol_rejects_changed_bool_even_when_python_equality_accepts_int():
    candidate = {"rect": True, "threshold": .25, "arms": diagnostic.ARMS}
    diagnostic.validate_frozen_protocol(candidate, dict(reversed(list(candidate.items()))))
    changed = copy.deepcopy(candidate)
    changed["rect"] = 1
    assert changed == candidate
    with pytest.raises(ValueError, match="exact preflight protocol"):
        diagnostic.validate_frozen_protocol(candidate, changed)
    changed = copy.deepcopy(candidate)
    changed["arms"].reverse()
    with pytest.raises(ValueError, match="exact preflight protocol"):
        diagnostic.validate_frozen_protocol(candidate, changed)


def test_preflight_writes_protocol_without_importing_any_runtime(monkeypatch, tmp_path):
    original = builtins.__import__
    output = tmp_path / "fresh"
    protocol = {"synthetic_preflight": True}
    monkeypatch.setattr(diagnostic, "prepare_inputs", lambda: ([{}] * 20, {}, protocol))
    monkeypatch.setattr(diagnostic.reference, "local_path", lambda value: value)

    def guard(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "numpy", "ultralytics", "cv2", "PIL", "psutil"}:
            pytest.fail(f"Preflight imported heavy runtime: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    diagnostic.main(["--output", str(output)])
    result = json.loads((output / "diagnostic.json").read_text())
    assert result["status"] == "PREFLIGHT_PASSED_NOT_INFERRED"
    assert result["runtime_probe_performed"] is False
    assert result["resource_guard_pending"] is True
    assert result["arms"] == []
    assert json.loads((output / "protocol_candidate.json").read_text()) == protocol


@pytest.mark.parametrize("extra", [["--execute"], ["--protocol", "x.json"], ["--protocol-sha256", "a" * 64]])
def test_execution_needs_both_frozen_protocol_path_and_hash_before_preparing_inputs(monkeypatch, extra):
    monkeypatch.setattr(diagnostic, "prepare_inputs", lambda: pytest.fail("Inputs were opened before argument rejection"))
    with pytest.raises(SystemExit):
        diagnostic.main(["--output", "unused", *extra])


def test_reference_guard_is_reused_without_lowering_any_launch_floor():
    resources = {"free_ram_bytes": 1536 * 1024 ** 2, "free_vram_bytes": 2 * 1024 ** 3, "free_disk_bytes": 1024 ** 3}
    diagnostic.reference.check_resources(resources)
    for field in resources:
        with pytest.raises(ValueError, match="guard requires"):
            diagnostic.reference.check_resources(dict(resources, **{field: resources[field] - 1}))
