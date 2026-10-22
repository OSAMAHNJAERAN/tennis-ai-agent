"""Two-arm, hash-frozen court DEV input-transform diagnostic; no training.

The default command performs a standard-library-only provenance preflight and
writes a protocol candidate. Execution requires that exact protocol saved by the
root reviewer, with its SHA256 supplied explicitly. Both arms retain every DEV
image, prediction and native overlay; no arm or setting is chosen per image.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import util
from importlib.metadata import distribution, version
import json
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "scripts/evaluate/diagnose_yolo26_reviewed_court.py"
REFERENCE_SHA256 = "49b4bf18c7b377458442ce084595269ea158c5dc13066f0fb2abc6ae68e050a3"
PREVIOUS = ROOT / "outputs/yolo26_court_reviewed_development_diagnostic_v1/diagnostic.json"
PREVIOUS_SHA256 = "768fb2f4d016e0d704c2bb555e5927395f3a3c76b5453f8cc6fbcb7d98b09aec"
PROPOSAL = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/court_development_review_v1/proposed_next_diagnostic.json"
PROPOSAL_SHA256 = "51d540b0ceeed310509069cf31c06e9e0b1d8bbd150ec60fead395d990ca0679"
CHECKPOINT_SHA256 = "e47db3784adb39b82be6df7fca73ed0206f67f4f5d1bb029f0f862a34a4feefa"
ARMS = [{"name": "EXPLICIT_RECTANGULAR_BASELINE", "rect": True},
        {"name": "EXPLICIT_SQUARE_INPUT_MATCHING_TRAIN_CONFIGURATION", "rect": False}]
INSTALLED_TRANSFORM_SOURCES = {
    "ultralytics/engine/predictor.py": "423c11df2c1853aa97cf756b9b571367422b9749f9bbf42de03896ff6c57ed72",
    "ultralytics/data/augment.py": "37a32d0d0555db44bd84f149673d9a85d7be1529e793e4e204a5001b62201dae",
    "ultralytics/nn/autobackend.py": "c9ffbc4584f8fdd6f58b3a58f8f5b640dbc3c43bd1f244b2e9da76ca6faee063",
    "ultralytics/utils/ops.py": "96cf96b815d141dcb4e618e2d6b0a4552c2a982c3d89461de34d7d63abef32b9",
}


def _load_reference():
    if hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The original diagnostic/scoring source changed")
    spec = util.spec_from_file_location("court_diagnostic_frozen_reference", REFERENCE_PATH)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reference = _load_reference()
INFERENCE = dict(reference.INFERENCE, nms=False, verbose=False, save=False,
                 save_txt=False, save_crop=False, show=False, visualize=False, embed=None)


def letterbox_geometry(native_hw, target_hw=(1280, 1280), *, rect, stride=32):
    """Independent arithmetic for pinned LetterBox defaults, without image/ML imports."""
    if (len(native_hw) != 2 or len(target_hw) != 2 or type(rect) is not bool
            or type(stride) is not int or stride <= 0
            or any(type(value) is not int or value <= 0 for value in [*native_hw, *target_hw])):
        raise ValueError("Letterbox dimensions/stride must be positive integers and rect a boolean")
    height, width = native_hw
    ratio = min(target_hw[0] / height, target_hw[1] / width)
    resized_wh = [round(width * ratio), round(height * ratio)]
    dw, dh = target_hw[1] - resized_wh[0], target_hw[0] - resized_wh[1]
    if rect:
        dw, dh = dw % stride, dh % stride
    left, right = round(dw / 2 - .1), round(dw / 2 + .1)
    top, bottom = round(dh / 2 - .1), round(dh / 2 + .1)
    return {"native_hw": list(native_hw), "requested_hw": list(target_hw), "rect": rect,
            "stride": stride, "same_shapes": True, "auto_minimum_rectangle": rect,
            "scaleup": True, "scale_fill": False, "center": True, "padding_value_uint8": 114,
            "scale_xy": [ratio, ratio], "resized_wh": resized_wh,
            "padding_ltrb": [left, top, right, bottom],
            "canvas_hw": [resized_wh[1] + top + bottom, resized_wh[0] + left + right],
            "coordinate_definition": "CONTINUOUS_XY_NATIVE_TIMES_SCALE_PLUS_INTEGER_LEFT_TOP_PADDING"}


def map_native(point, geometry):
    return [point[index] * geometry["scale_xy"][index] + geometry["padding_ltrb"][index]
            for index in range(2)]


def map_back(point, geometry):
    return [(point[index] - geometry["padding_ltrb"][index]) / geometry["scale_xy"][index]
            for index in range(2)]


def calibration_roundtrip(geometry):
    height, width = geometry["native_hw"]
    probes = [[0., 0.], [float(width), 0.], [0., float(height)], [float(width), float(height)],
              [width / 2, height / 2], [width * .17325, height * .63875]]
    rows = [{"native_xy": point, "letterbox_xy": map_native(point, geometry),
             "recovered_native_xy": map_back(map_native(point, geometry), geometry)} for point in probes]
    maximum = max(abs(item["native_xy"][axis] - item["recovered_native_xy"][axis])
                  for item in rows for axis in range(2))
    if maximum > 1e-9:
        raise ValueError("Independent native/letterbox calibration roundtrip failed")
    return {"probes": rows, "maximum_absolute_error_native_px": maximum,
            "uses_predictions_or_labels": False, "tolerance_native_px": 1e-9}


class ForwardTrace:
    """Associate calls by actual predictor phase, never by tensor dimensions alone."""

    def __init__(self):
        self.events = []
        self.phase = "NON_IMAGE_LIBRARY_WARMUP"
        self.active = None
        self.preprocessed = None

    def begin_image(self, arm, image_id):
        if self.active is not None:
            raise ValueError("An image forward is already active")
        self.active = {"arm": arm, "id": image_id, "start_event": len(self.events)}
        self.preprocessed = None

    def record_preprocess(self, information):
        if self.active is None or self.preprocessed is not None:
            raise ValueError("Each image must have exactly one observed preprocessing call")
        self.preprocessed = information

    @contextmanager
    def image_inference(self):
        if self.active is None or self.preprocessed is None:
            raise ValueError("Image inference requires an observed image preprocessing call")
        self.phase = "ACTUAL_IMAGE_INFERENCE"
        try:
            yield
        finally:
            self.phase = "NON_IMAGE_LIBRARY_WARMUP"

    def observe(self, shape, dtype, device, *, done_warmup):
        if self.active is None:
            raise ValueError("Unexpected model forward outside a bound prediction invocation")
        if self.phase == "NON_IMAGE_LIBRARY_WARMUP" and (done_warmup or self.preprocessed is not None):
            raise ValueError("Unclassified non-image forward after library warmup")
        event = {"ordinal": len(self.events), "phase": self.phase, "arm": self.active["arm"],
                 "id": self.active["id"] if self.phase == "ACTUAL_IMAGE_INFERENCE" else None,
                 "triggering_invocation_image_id": self.active["id"], "shape_nchw": list(shape),
                 "dtype": dtype, "device": device, "predictor_done_warmup_at_forward": done_warmup}
        if self.phase == "ACTUAL_IMAGE_INFERENCE":
            if list(shape) != self.preprocessed["actual_network_shape_nchw"] or dtype != "torch.float32":
                raise ValueError("Observed image network forward differs from recorded preprocessing")
        else:
            event["input_values_policy"] = "PINNED_LIBRARY_TORCH_EMPTY_UNINITIALIZED_NOT_ASSUMED_ZERO"
        self.events.append(event)

    def end_image(self):
        events = self.events[self.active["start_event"]:]
        images = [event for event in events if event["phase"] == "ACTUAL_IMAGE_INFERENCE"]
        if self.preprocessed is None or len(images) != 1:
            raise ValueError("Exactly one actual image forward is required per DEV image")
        result = {"preprocessing": self.preprocessed, "forward": images[0],
                  "preceding_non_image_warmup_calls": [event for event in events if event not in images]}
        self.active = None
        return result


def validate_previous(previous, inputs):
    if (previous.get("status") != "COMPLETED_FIXED_PROTOCOL_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION"
            or previous.get("qualification_evidence") is not False or previous.get("final_test_used") is not False
            or previous.get("bindings", {}).get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
            or previous.get("inference_arguments") != dict(reference.INFERENCE, nms=False, verbose=False)
            or len(previous.get("rows", [])) != 20):
        raise ValueError("Require the unchanged completed original twenty-image DEV diagnostic")
    for old, current in zip(previous["rows"], inputs):
        if old["id"] != current["sample"]["id"] or old["context_group"] != current["source"]["context_group"]:
            raise ValueError("Paired comparison must retain original DEV order/context")
        for kind in ("image", "native_source_image", "label"):
            if old[kind] != current[kind]:
                raise ValueError("Paired comparison must retain exact original DEV image/label bytes")


def prepare_inputs():
    _, inputs, bindings = reference.prepare_inputs()
    bindings["original_diagnostic_source"] = reference.bind_file(REFERENCE_PATH, REFERENCE_SHA256)
    bindings["completed_original_diagnostic"] = reference.bind_file(PREVIOUS, PREVIOUS_SHA256)
    bindings["review_proposal"] = reference.bind_file(PROPOSAL, PROPOSAL_SHA256)
    bindings["paired_diagnostic_source"] = reference.bind_file(__file__)
    reference.bind_file(reference.CHECKPOINT, CHECKPOINT_SHA256)
    validate_previous(reference.read_json(PREVIOUS), inputs)
    proposal = reference.read_json(PROPOSAL)
    if proposal["arms"] != ARMS or proposal["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("The paired proposal changed")
    installed = distribution("ultralytics")
    transform_bindings = {}
    for relative, expected in INSTALLED_TRANSFORM_SOURCES.items():
        path = Path(installed.locate_file(relative)).resolve()
        if reference.sha256_file(path) != expected:
            raise ValueError("Installed transform/warmup implementation changed")
        transform_bindings[relative] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    bindings["installed_transform_sources"] = transform_bindings
    if version("ultralytics") != reference.PINNED_ULTRALYTICS:
        raise ValueError("Use pinned Ultralytics8.4.36")
    protocol = {"schema": "tennis.yolo26.court-paired-input-transform.v1",
                "qualification_evidence": False, "final_test_used": False, "independent_human_reviewed": False,
                "training_authorized": False, "checkpoint_sha256": CHECKPOINT_SHA256,
                "source_manifest_sha256": reference.MANIFEST_SHA256,
                "original_diagnostic_sha256": PREVIOUS_SHA256, "original_scoring_source_sha256": REFERENCE_SHA256,
                "paired_diagnostic_source_sha256": bindings["paired_diagnostic_source"]["sha256"],
                "review_proposal_sha256": PROPOSAL_SHA256,
                "arms": ARMS, "arm_order": "COMPLETE_RECTANGULAR_THEN_COMPLETE_SQUARE_MANIFEST_ORDER",
                "inference_arguments_except_rect": INFERENCE,
                "instance_selection": "HIGHEST_CLASS0_CONFIDENCE_THEN_STABLE_RETURNED_INDEX_NO_GT_SELECTION",
                "matched_bbox_iou_minimum": .5, "keypoint_confidence_minimum": .25,
                "visible_error_thresholds_px": [10, 20], "all_visible_denominator": 278, "unknown_gt_excluded": 2,
                "expected_images_per_arm": 20, "expected_native_overlays_per_arm": 20,
                "expected_actual_image_forwards_per_arm": 20, "expected_library_warmup_forwards_per_arm": 1,
                "launch_guard_after_cuda": {"free_ram_bytes": 1536 * 1024 ** 2,
                                            "free_vram_bytes": 2 * 1024 ** 3, "free_disk_bytes": 1024 ** 3},
                "expected_transform": {"native_hw": [720, 1280], "stride": 32, "dtype": "torch.float32",
                                       "rect_canvas_hw": [736, 1280], "square_canvas_hw": [1280, 1280],
                                       "rect_padding_ltrb": [0, 8, 0, 8], "square_padding_ltrb": [0, 280, 0, 280]},
                "installed_transform_sources": transform_bindings,
                "selected_dev_inputs": [{"id": item["sample"]["id"], "context_group": item["source"]["context_group"],
                                         "original_source_video_id": item["source"]["original_source_video_id"],
                                         "image": item["image"], "label": item["label"]} for item in inputs],
                "prohibited": ["TEST_DATA", "GT_CROPS", "PER_IMAGE_ARM_SELECTION", "THRESHOLD_SEARCH",
                               "COORDINATE_CORRECTION", "LABEL_CHANGES", "MODEL_PROMOTION", "TRAINING"],
                "interpretation": proposal["interpretation_rule"]}
    return inputs, bindings, protocol


def validate_frozen_protocol(candidate, supplied):
    if json.dumps(supplied, sort_keys=True, allow_nan=False) != json.dumps(candidate, sort_keys=True, allow_nan=False):
        raise ValueError("Require the exact preflight protocol frozen before inference")


def paired_points(rect_rows, square_rows):
    if len(rect_rows) != len(square_rows):
        raise ValueError("Both arms must contain the same images")
    points = []
    for first, second in zip(rect_rows, square_rows):
        if first["id"] != second["id"] or first["context_group"] != second["context_group"]:
            raise ValueError("Paired image ID/order/context changed")
        if len(first["assessment"]["points"]) != 14 or len(second["assessment"]["points"]) != 14:
            raise ValueError("Both arms must retain all fourteen semantic landmarks")
        for a, b in zip(first["assessment"]["points"], second["assessment"]["points"]):
            if (a["point_index"] != b["point_index"] or a["gt_xy_visibility_px"] != b["gt_xy_visibility_px"]
                    or a["visible_gt"] != b["visible_gt"]):
                raise ValueError("Paired landmark ground truth changed")
            errors = [a["error_native_px"], b["error_native_px"]]
            points.append({"id": first["id"], "context_group": first["context_group"],
                           "point_index": a["point_index"], "point_name": a["point_name"], "visible_gt": a["visible_gt"],
                           "rect_error_native_px": errors[0], "square_error_native_px": errors[1],
                           "square_minus_rect_error_native_px": errors[1] - errors[0] if all(error is not None for error in errors) else None,
                           "rect_missing_reason": a["missing_or_exclusion_reason"], "square_missing_reason": b["missing_or_exclusion_reason"],
                           "rect_within_10_px": a["within_10_px"], "square_within_10_px": b["within_10_px"],
                           "rect_within_20_px": a["within_20_px"], "square_within_20_px": b["within_20_px"]})
    return points


def summarize_pairs(points):
    visible = [point for point in points if point["visible_gt"]]
    patterns = Counter((point["rect_error_native_px"] is not None, point["square_error_native_px"] is not None) for point in visible)
    deltas = [point["square_minus_rect_error_native_px"] for point in visible if point["square_minus_rect_error_native_px"] is not None]
    result = {"visible_gt_denominator": len(visible), "unknown_gt_excluded": len(points) - len(visible),
              "both_available": patterns[True, True], "rect_only_available": patterns[True, False],
              "square_only_available": patterns[False, True], "both_missing": patterns[False, False],
              "common_available_square_minus_rect_median_error_px": reference.percentile(deltas, .5),
              "common_available_square_minus_rect_p95_error_px": reference.percentile(deltas, .95),
              "delta_error_population": "BOTH_ARMS_FINITE_CONFIDENT_BBOX_MATCHED_SAME_VISIBLE_TARGET_ONLY",
              "no_missing_error_imputation": True}
    for arm in ("rect", "square"):
        result[f"{arm}_missing_reasons"] = dict(Counter(point[f"{arm}_missing_reason"] for point in visible if point[f"{arm}_error_native_px"] is None))
        for threshold in (10, 20):
            count = sum(point[f"{arm}_within_{threshold}_px"] for point in visible)
            result[f"{arm}_within_{threshold}_px"] = count
            result[f"{arm}_recall_at_{threshold}_px"] = count / len(visible) if visible else None
    return result


def paired_report(rect_rows, square_rows):
    points = paired_points(rect_rows, square_rows)
    return {"all_visible": summarize_pairs(points), "points": points,
            "per_image": [{"id": row["id"], "context_group": row["context_group"],
                           **summarize_pairs([point for point in points if point["id"] == row["id"]])} for row in rect_rows],
            "per_landmark": [{"point_index": index, "point_name": name,
                              **summarize_pairs([point for point in points if point["point_index"] == index])}
                             for index, name in enumerate(reference.POINT_NAMES)],
            "per_context": [{"context_group": context, **summarize_pairs([point for point in points if point["context_group"] == context])}
                            for context in dict.fromkeys(row["context_group"] for row in rect_rows)]}


def runtime_predictor(trace):
    """Observe inherited preprocessing/inference; do not replace their operations."""
    import numpy as np
    import torch
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.utils import ops

    class ObservedPosePredictor(PosePredictor):
        def preprocess(self, images):
            if not isinstance(images, list) or len(images) != 1 or list(images[0].shape) != [720, 1280, 3]:
                raise ValueError("Only one unchanged native1280x720 color image may be preprocessed")
            if self.model.format != "pt" or int(self.model.stride) != 32:
                raise ValueError("The frozen transform requires PyTorch format and stride32")
            geometry = letterbox_geometry([720, 1280], tuple(self.imgsz), rect=self.args.rect, stride=int(self.model.stride))
            tensor = super().preprocess(images)
            if list(tensor.shape) != [1, 3, *geometry["canvas_hw"]] or str(tensor.dtype) != "torch.float32":
                raise ValueError("Actual preprocessing shape/dtype differs from the predeclared transform")
            if geometry["scale_xy"] != [1., 1.] or geometry["resized_wh"] != [1280, 720]:
                raise ValueError("These native stills must not be resized")
            observed = tensor.detach().cpu().numpy()
            left, top, right, bottom = geometry["padding_ltrb"]
            native_rgb = np.ascontiguousarray(images[0][..., ::-1].transpose((2, 0, 1)))
            expected = native_rgb.astype(np.float32) / np.float32(255.)
            if not np.allclose(observed[0, :, top:top + 720, left:left + 1280], expected, atol=1e-7, rtol=0):
                raise ValueError("The observed input canvas does not preserve the decoded native pixels")
            bands = {"left": observed[..., :left], "top": observed[..., :top, :],
                     "right": observed[..., observed.shape[-1] - right:] if right else observed[..., :0],
                     "bottom": observed[..., observed.shape[-2] - bottom:, :] if bottom else observed[..., :0, :]}
            for band in bands.values():
                if band.size and not np.allclose(band, np.float32(114.) / np.float32(255.), atol=1e-7, rtol=0):
                    raise ValueError("Observed letterbox padding pixels differ from the frozen value114")
            calibration = calibration_roundtrip(geometry)
            independently_mapped = torch.tensor([item["letterbox_xy"] for item in calibration["probes"]], dtype=torch.float64)
            inverse = ops.scale_coords(geometry["canvas_hw"], independently_mapped, (720, 1280)).tolist()
            error = max(abs(actual[axis] - row["native_xy"][axis]) for actual, row in zip(inverse, calibration["probes"]) for axis in range(2))
            if error > 1e-9:
                raise ValueError("Pinned runtime coordinate inverse fails independent calibration probes")
            calibration.update(pinned_scale_coords_recovered_native_xy=inverse, pinned_scale_coords_maximum_error_px=error)
            trace.record_preprocess({"actual_network_shape_nchw": list(tensor.shape), "actual_dtype": str(tensor.dtype),
                                     "device": str(tensor.device), "geometry": geometry, "calibration": calibration,
                                     "decoded_native_bgr_sha256": hashlib.sha256(images[0].tobytes()).hexdigest(),
                                     "actual_network_tensor_cpu_bytes_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                                     "native_pixels_preserved": True, "every_padding_pixel_verified": True,
                                     "pixel_comparison_absolute_tolerance": 1e-7,
                                     "coordinate_map_uses_predictions_or_labels": False})
            return tensor

        def inference(self, tensor, *args, **kwargs):
            with trace.image_inference():
                return super().inference(tensor, *args, **kwargs)

    return ObservedPosePredictor


def changed_bindings(bindings, inputs):
    changed = []
    for binding in bindings.values():
        entries = [binding] if "path" in binding else list(binding.values())
        changed.extend(item["path"] for item in entries if reference.sha256_file(item["path"]) != item["sha256"])
    for item in inputs:
        changed.extend(item[key]["path"] for key in ("image", "native_source_image", "label")
                       if reference.sha256_file(item[key]["path"]) != item[key]["sha256"])
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.protocol) != bool(args.protocol_sha256) or (args.execute and not args.protocol):
        parser.error("Execution requires an externally frozen --protocol and exact --protocol-sha256")
    output = reference.local_path(args.output)
    if output.exists():
        raise FileExistsError("Preserve all prior receipts; choose a fresh paired diagnostic output")
    inputs, bindings, protocol = prepare_inputs()
    if args.protocol:
        bindings["frozen_paired_protocol"] = reference.bind_file(args.protocol, args.protocol_sha256)
        validate_frozen_protocol(protocol, reference.read_json(bindings["frozen_paired_protocol"]["path"]))
    output.mkdir(parents=True)
    reference.write_json(output / "protocol_candidate.json", protocol)
    receipt = {"schema": "tennis.yolo26.court-paired-input-transform-result.v1",
               "status": "PREFLIGHT_PASSED_NOT_INFERRED", "created_utc": datetime.now(timezone.utc).isoformat(),
               "qualification_evidence": False, "final_test_used": False, "independent_human_reviewed": False,
               "training_performed": False, "production_promoted": False, "protocol": protocol, "bindings": bindings,
               "runtime_probe_performed": False, "resource_guard_pending": True, "arms": [], "forward_events": [],
               "keypoint_coordinate_policy": reference.read_json(PREVIOUS)["keypoint_coordinate_policy"],
               "warmup_policy": "OBSERVE_INHERITED_PREDICTOR_WARMUP_SEPARATELY; PINNED_TORCH_EMPTY_NOT_ZERO; LIBRARY_NMS_WARMUP_IS_NOT_IMAGE_POSTPROCESSING",
               "not_established": ["INDEPENDENT_QUALIFICATION", "GROUND_METRIC_GEOMETRY", "VIDEO_STABILITY", "NEGATIVE_VIDEO_FALSE_ALARM_RATE", "CAUSE_OF_LOCALIZATION_ERROR"]}
    path = output / "diagnostic.json"
    reference.write_json(path, receipt)
    if not args.execute:
        print(json.dumps({"status": receipt["status"], "receipt": str(path), "protocol_candidate": str(output / "protocol_candidate.json"),
                          "protocol_candidate_sha256": reference.sha256_file(output / "protocol_candidate.json"), "images_per_arm": len(inputs)}))
        return
    started = time.perf_counter()
    trace = ForwardTrace()
    try:
        import psutil
        import torch
        from ultralytics import YOLO

        available = torch.cuda.is_available()
        resources = {"free_vram_bytes": torch.cuda.mem_get_info()[0] if available else None,
                     "free_ram_bytes": psutil.virtual_memory().available, "free_disk_bytes": shutil.disk_usage(output).free,
                     "cuda": torch.version.cuda, "torch": torch.__version__, "ultralytics": version("ultralytics"),
                     "gpu": torch.cuda.get_device_name(0) if available else None}
        receipt.update(runtime_probe_performed=True, resources_after_cuda_initialization=resources)
        reference.write_json(path, receipt)
        reference.check_resources(resources)
        receipt["resource_guard_pending"] = False
        receipt["status"] = "PAIRED_DIAGNOSTIC_STARTED_UNQUALIFIED"
        model = YOLO(str(reference.CHECKPOINT), task="pose")
        receipt["model_config"] = reference.validate_model(model)
        predictor_class = runtime_predictor(trace)

        def observe_forward(_model, arguments):
            tensor = arguments[0]
            trace.observe(tensor.shape, str(tensor.dtype), str(tensor.device), done_warmup=model.predictor.done_warmup)

        hook = model.model.register_forward_pre_hook(observe_forward)
        torch.cuda.reset_peak_memory_stats()
        try:
            for arm in ARMS:
                model.predictor = None  # Fresh identical predictor/warmup for each arm; same checkpoint/model weights.
                arm_output = output / arm["name"].lower()
                arm_output.mkdir()
                arm_result = {**arm, "inference_arguments": dict(INFERENCE, rect=arm["rect"]), "rows": []}
                receipt["arms"].append(arm_result)
                for index, item in enumerate(inputs):
                    sample, source = item["sample"], item["source"]
                    trace.begin_image(arm["name"], sample["id"])
                    results = model.predict(item["image"]["path"], predictor=predictor_class, **arm_result["inference_arguments"])
                    if len(results) != 1 or list(results[0].orig_shape) != [720, 1280]:
                        raise ValueError("Each invocation must return one unchanged native-image result")
                    result = results[0]
                    observed = trace.end_image()
                    boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
                    confidences = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
                    classes = result.boxes.cls.detach().cpu().tolist() if result.boxes is not None else []
                    keypoints = result.keypoints.data.detach().cpu().tolist() if result.keypoints is not None else None
                    assessment = reference.score_image(sample, reference.raw_candidates(boxes, confidences, classes, keypoints))
                    overlay = reference.render_overlay(item["image"]["path"], assessment, arm_output / f"dev_{index + 1:02d}_{sample['id']}.png",
                                                       f"{arm['name']} DEV {index + 1}/20 {sample['id']} | {source['context_group']}")
                    arm_result["rows"].append({"id": sample["id"], "split": "val", "context_group": source["context_group"],
                                               "original_source_video_id": source["original_source_video_id"],
                                               "publisher_frame_ordinal": source["publisher_frame_ordinal"],
                                               "source_fps": None, "source_pts": None, "source_time_base": None,
                                               "image": item["image"], "native_source_image": item["native_source_image"], "label": item["label"],
                                               "observed_input": observed, "assessment": assessment, "overlay": overlay})
                    receipt["forward_events"] = trace.events
                    reference.write_json(path, receipt)
                arm_result["aggregates"] = reference.aggregate(arm_result["rows"])
                events = [event for event in trace.events if event["arm"] == arm["name"]]
                counts = Counter(event["phase"] for event in events)
                if counts != {"ACTUAL_IMAGE_INFERENCE": 20, "NON_IMAGE_LIBRARY_WARMUP": 1}:
                    raise ValueError("Each arm requires twenty bound image forwards and one separately observed library warmup")
                if arm_result["aggregates"]["all_visible_keypoints"]["visible_gt_denominator"] != 278:
                    raise ValueError("An arm lost a visible target from the fixed denominator")
        finally:
            hook.remove()
        receipt["paired"] = paired_report(receipt["arms"][0]["rows"], receipt["arms"][1]["rows"])
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        receipt["changed_inputs"] = changed_bindings(bindings, inputs)
        if receipt["changed_inputs"]:
            raise ValueError("Frozen diagnostic source/input bytes changed during inference")
        receipt.update(status="COMPLETED_PAIRED_DEV_INPUT_TRANSFORM_DIAGNOSTIC_NOT_QUALIFICATION",
                       completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                       input_hashes_unchanged=True)
        reference.write_json(path, receipt)
    except Exception as exc:
        receipt.update(status="PAIRED_DIAGNOSTIC_BLOCKED_OR_FAILED_NOT_QUALIFICATION", failure_type=type(exc).__name__,
                       failure_reason=str(exc), failed_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.perf_counter() - started, forward_events=trace.events)
        reference.write_json(path, receipt)
        raise
    print(json.dumps({"status": receipt["status"], "receipt": str(path), "paired_all_visible": receipt["paired"]["all_visible"]}, allow_nan=False))


if __name__ == "__main__":
    main()
