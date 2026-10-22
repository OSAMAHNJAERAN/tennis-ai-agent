"""Predeclared native-tile diagnostic for the reviewed ball-development wave.

Prepare before training, then consume that exact protocol and the registered
DEV-fitness best checkpoint. All twenty accepted DEV targets remain in the
denominator. Full-frame unmatched objects are unadjudicated proposals because
only the reviewed crops have complete visible-ball labels. No qualification,
threshold search, final-test use, training or checkpoint promotion occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/ball_reviewed_crop_preparation_v1"
PINS = {
    "native_crop_review_boards.json": "537d16626181ab74a7f332350d36bfd36c2efad9ee463aa8f3730924f04164cb",
    "derived_development_manifest.json": "b94a4e0d6cfc3504817b9223abc5e43a2252fd5d826ea95bf2dd14e129b41110",
    "crop_full_review_head_root_v1.json": "770c8e9486aeede27fcad5de271fdab60153c851fa7abda0bb6c428ee228e14b",
    "crop_full_review_tail_v1.json": "82c5ec2443d91619ab61e240f19011021195a1599bbf4e01dbfd211de0cbb2ff",
}
EXPORT = ROOT / "data/yolo26/ball_reviewed_development_v1"
GRID = [[x, y, x + 640, y + 640] for y in (0, 440) for x in (0, 512, 1024, 1280)]
INFERENCE = dict(imgsz=640, batch=1, conf=.01, classes=[0], device=0, end2end=True,
                 nms=False, iou=.7, max_det=300, half=False, rect=False, augment=False,
                 save=False, save_txt=False, save_crop=False, verbose=False)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(part)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def workspace_path(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Retain diagnostic inputs/outputs within the private workspace")
    return result


def binding(path, expected=None):
    path = workspace_path(path)
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen diagnostic input changed: {path}")
    return {"path": str(path), "sha256": actual}


def prepare_protocol(run_directory, frozen_at=None):
    """Read/hash only; no numerical, image or model imports and no file writes."""
    if version("ultralytics") != "8.4.36":
        raise ValueError("Require pinned Ultralytics8.4.36")
    bindings = {name: binding(PREP / name, expected) for name, expected in PINS.items()}
    for relative in ("scripts/data/diagnose_yolo26_native_tiles.py",
                     "scripts/evaluate/diagnose_yolo26_court_input_transform.py",
                     "scripts/evaluate/diagnose_yolo26_reviewed_court.py"):
        bindings[relative] = binding(ROOT / relative)
    bindings["diagnostic_source"] = binding(__file__)
    # Early software checks may inspect the proposal before export. A saved
    # executable protocol, however, must bind the completed reviewed export.
    if (EXPORT / "source_manifest.json").is_file():
        bindings["export_manifest"] = binding(EXPORT / "source_manifest.json")
    for relative in ("scripts/data/export_yolo26_reviewed_ball_wave.py", "scripts/train/train_yolo26_reviewed_ball.py"):
        if (ROOT / relative).is_file():
            bindings[relative] = binding(ROOT / relative)
    rendered = read(PREP / "native_crop_review_boards.json")
    derived = read(PREP / "derived_development_manifest.json")
    records = read(PREP / "crop_full_review_head_root_v1.json")["records"] + read(PREP / "crop_full_review_tail_v1.json")["records"]
    if len(records) != 85 or [row["proposal_id"] for row in records] != [row["proposal_id"] for row in rendered["crops"]]:
        raise ValueError("Require exactly the complete ordered85crop decisions")
    accepted = {row["proposal_id"] for row in records if row["decision"] == "ACCEPT_CROP_SINGLE_REVIEWED_BALL"}
    samples = []
    for row in derived["positive_observations"]:
        if row["proposal_id"] in accepted and row["derived_split"] == "development":
            samples.append({"id": row["proposal_id"], "source_id": row["source_id"], "clip": row["clip"],
                            "frame_index": row["frame_index"], "source_pts": row["source_pts"],
                            "source_time_base": row["source_time_base"], "component": row["derived_component_id"],
                            "native_frame": binding(row["native_frame"]["path"], row["native_frame"]["sha256"]),
                            "target_box": row["bbox_xyxy_native_px"], "visible_center": row["visible_center_xy_native_px"]})
    negatives = []
    for row in derived["effective_root_negative_observations"]:
        if row["effective_decision"] == "ACCEPT_VISIBLE_ABSENCE" and row["derived_split"] == "development":
            image = row["original_observation"]["actually_viewed_full_native_tile"]
            negatives.append({"id": row["proposal_id"], "source_id": row["source_id"], "frame_index": row["frame_index"],
                              "source_pts": row["source_pts"], "source_time_base": row["source_time_base"],
                              "crop": binding(image["path"], image["sha256"]),
                              "origin": row["original_observation"]["tile_origin_xy_px"]})
    if len(accepted) != 75 or len(samples) != 20 or len(negatives) != 1:
        raise ValueError("Require75accepted positives, all20DEVtargets and exactly1DEVabsence crop")
    return {"schema": "tennis.yolo26.reviewed-ball-native-tile-diagnostic.v1",
            "frozen_at_utc": frozen_at or datetime.now(timezone.utc).isoformat(),
            "training_run_directory": str(workspace_path(run_directory)), "bindings": bindings,
            "checkpoint_selection": "REGISTERED_ULTRALYTICS_DEV_FITNESS_BEST_ONLY",
            "inference": INFERENCE, "native_frame_wh": [1920, 1080], "fixed_native_tile_grid": GRID,
            "target_iou_minimum": .5, "dedup_iou_or_center": [.5, 2.],
            "positive_samples": samples, "negative_samples": negatives,
            "all_target_denominator": 20, "actual_image_forward_count": 161,
            "unmatched_full_frame_candidates": "UNADJUDICATED_OTHER_BALL_OR_FALSE_OBJECT_NOT_PRECISION_GROUND_TRUTH",
            "center_definition": "MIDPOINT_OF_AGENT_REVIEWED_VISIBLE_EXTENT_NOT_PHYSICAL_3D_CENTER",
            "qualification_evidence": False, "final_test_used": False,
            "independent_human_review": False, "training_authorized": False,
            "launch_guard_after_cuda": {"free_ram_bytes": 1024**3, "free_vram_bytes": 2 * 1024**3, "free_disk_bytes": 1024**3}}


def validate_checkpoint(protocol):
    run = workspace_path(protocol["training_run_directory"])
    metadata_path = run / "run_metadata.json"
    metadata = read(metadata_path)
    if (metadata.get("role") != "ball" or metadata.get("mode") != "REVIEWED_NATIVE_BALL_DEVELOPMENT"
            or metadata.get("status") not in {"TRAINED_UNQUALIFIED", "COMPLETED_PRIVATE_TRAINED_UNQUALIFIED"}
            or metadata.get("qualification_evidence") is not False or metadata.get("final_test_used") is not False):
        raise ValueError("Require the completed private reviewed-native-ball development run")
    if datetime.fromisoformat(metadata["started_utc"]) <= datetime.fromisoformat(protocol["frozen_at_utc"]):
        raise ValueError("Diagnostic protocol must have been frozen before this training run started")
    checkpoint = run / "training/weights/best.pt"
    matches = [row for row in metadata["checkpoints"] if Path(row["path"]).resolve() == checkpoint]
    if len(matches) != 1 or matches[0]["sha256"] != digest(checkpoint):
        raise ValueError("Only the registered DEV-fitness best checkpoint may be diagnosed")
    manifest = EXPORT / "source_manifest.json"
    expected_manifest = protocol["bindings"].get("export_manifest")
    if expected_manifest is None or expected_manifest != binding(manifest) or metadata["data_audit"]["manifest_sha256"] != digest(manifest):
        raise ValueError("Training/export data binding changed")
    from scripts.data.export_yolo26_reviewed_ball_wave import verify_export
    verify_export(EXPORT)
    return checkpoint, {"run_metadata": binding(metadata_path), "checkpoint": binding(checkpoint), "export_manifest": binding(manifest)}


def score_reviewed_target(scoring, candidates, sample):
    result = scoring.score_candidates(candidates, sample["target_box"], sample["visible_center"])
    for assessment in result["candidate_target_assessments"]:
        assessment["visible_extent_center_error_native_px"] = assessment.pop("publisher_center_error_native_px")
    return result


def observed_predictor(trace):
    import numpy as np
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    class NativePixelsPredictor(DetectionPredictor):
        def preprocess(self, images):
            if not isinstance(images, list) or len(images) != 1 or images[0].shape != (640, 640, 3):
                raise ValueError("Require one native640color input per invocation")
            tensor = super().preprocess(images)
            expected = np.ascontiguousarray(images[0][..., ::-1].transpose(2, 0, 1)).astype(np.float32) / np.float32(255.)
            actual = tensor.detach().cpu().numpy()
            if list(tensor.shape) != [1, 3, 640, 640] or str(tensor.dtype) != "torch.float32" or not np.allclose(actual[0], expected, atol=1e-7, rtol=0):
                raise ValueError("Actual input tensor must preserve all native pixels without scaling/cropping")
            trace.record_preprocess({"actual_network_shape_nchw": list(tensor.shape), "actual_dtype": str(tensor.dtype),
                                     "native_pixels_preserved": True, "decoded_native_bgr_sha256": hashlib.sha256(images[0].tobytes()).hexdigest(),
                                     "actual_network_tensor_cpu_bytes_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
            return tensor

        def inference(self, tensor, *args, **kwargs):
            with trace.image_inference():
                return super().inference(tensor, *args, **kwargs)
    return NativePixelsPredictor


def execute(protocol, output, protocol_binding):
    checkpoint, run_bindings = validate_checkpoint(protocol)
    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics import YOLO
    from scripts.data import diagnose_yolo26_native_tiles as scoring
    from scripts.evaluate.diagnose_yolo26_court_input_transform import ForwardTrace

    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="STARTED_PRIVATE_DEVELOPMENT_DIAGNOSTIC_UNQUALIFIED", protocol=protocol_binding,
                   run_bindings=run_bindings, qualification_evidence=False, final_test_used=False,
                   training_performed=False, checkpoint_promoted=False)
    receipt_path = output / "diagnostic.json"
    write(receipt_path, receipt)
    try:
        torch.cuda.init()
        resources = dict(free_ram_bytes=int(psutil.virtual_memory().available), free_vram_bytes=int(torch.cuda.mem_get_info(0)[0]),
                         free_disk_bytes=int(shutil.disk_usage(ROOT).free))
        receipt["resources_after_cuda"] = resources
        scoring.check_resources(resources)
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats(0)
        model = YOLO(str(checkpoint), task="detect")
        if (model.task != "detect" or model.model.yaml.get("scale") != "m" or not model.model.yaml.get("end2end")
                or "yolo26" not in str(model.model.yaml.get("yaml_file", "")).lower() or model.names != {0: "tennis-ball"}):
            raise ValueError("Require exactly the registered end-to-end YOLO26m ball specialist")
        trace = ForwardTrace()
        predictor = observed_predictor(trace)
        def observe(_, arguments):
            tensor = arguments[0]
            trace.observe(tensor.shape, str(tensor.dtype), str(tensor.device), done_warmup=model.predictor.done_warmup)
        hook = model.model.register_forward_pre_hook(observe)
        rows, negatives, started = [], [], time.perf_counter()
        (output / "overlays").mkdir()
        def infer(image, ident, region, width=1920, height=1080):
            trace.begin_image("FIXED_NATIVE640", ident)
            result = model.predict(image, predictor=predictor, **protocol["inference"])[0]
            observed = trace.end_image()
            boxes = result.boxes
            candidates = scoring.map_candidates(boxes.xyxy.detach().cpu().numpy(), boxes.conf.detach().cpu().numpy(), boxes.cls.detach().cpu().numpy(), region, ident, width, height) if boxes is not None else []
            return candidates, observed
        try:
            for sample in protocol["positive_samples"]:
                image = cv2.imread(sample["native_frame"]["path"])
                if image is None or image.shape != (1080, 1920, 3):
                    raise ValueError("Native source frame must decode1920x1080")
                raw, tiles = [], []
                for index, region in enumerate(protocol["fixed_native_tile_grid"]):
                    left, top, right, bottom = region
                    candidates, observed = infer(image[top:bottom, left:right], f"{sample['id']}:t{index}", region)
                    raw.extend(candidates)
                    tiles.append(dict(region=region, observed=observed))
                accepted, associations = scoring.deduplicate(raw)
                overlay = image.copy()
                for candidate in accepted:
                    x1, y1, x2, y2 = map(lambda value: int(round(value)), candidate["global_box_xyxy_px"])
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
                x1, y1, x2, y2 = map(int, sample["target_box"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(overlay, "GREEN: reviewed target; ORANGE: model proposals, other objects unadjudicated", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                overlay_path = output / "overlays" / (sample["id"] + ".png")
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise ValueError("Unable to retain native diagnostic overlay")
                rows.append(dict(sample=sample, tiles=tiles, raw_candidates=raw, deduplicated_candidates=accepted,
                                 dedup_associations=associations, overlay=binding(overlay_path), native_tiles=score_reviewed_target(scoring, accepted, sample)))
            for sample in protocol["negative_samples"]:
                image = cv2.imread(sample["crop"]["path"])
                candidates, observed = infer(image, sample["id"], [0, 0, 640, 640], 640, 640)
                accepted, associations = scoring.deduplicate(candidates)
                negatives.append(dict(sample=sample, raw_candidates=candidates, deduplicated_candidates=accepted, dedup_associations=associations, observed=observed))
        finally:
            hook.remove()
        actual = [event for event in trace.events if event["phase"] == "ACTUAL_IMAGE_INFERENCE"]
        if len(actual) != protocol["actual_image_forward_count"]:
            raise ValueError("Diagnostic did not execute every predeclared native input")
        for item in list(protocol["bindings"].values()) + list(run_bindings.values()) + [s["native_frame"] for s in protocol["positive_samples"]] + [s["crop"] for s in protocol["negative_samples"]]:
            binding(item["path"], item["sha256"])
        aggregate = scoring.aggregate(rows)
        matched_errors = [row["native_tiles"]["matched_candidate_center_error_native_px"] for row in rows if row["native_tiles"]["matched_candidate_center_error_native_px"] is not None]
        aggregate["conditional_matched_center_error_p95_native_px"] = float(np.percentile(matched_errors, 95)) if matched_errors else None
        aggregate["matched_center_error_sample_count"] = len(matched_errors)
        # This count has no complete full-frame negative/other-ball truth.
        aggregate["unmatched_full_frame_candidates_unadjudicated"] = aggregate.pop("unmatched_or_duplicate_candidates")
        receipt.update(status="COMPLETED_FIXED_NATIVE_TILE_DEVELOPMENT_DIAGNOSTIC_NOT_QUALIFICATION", rows=rows, negative_crop_rows=negatives,
                       aggregates=aggregate, negative_crop_returned_proposals=sum(len(row["deduplicated_candidates"]) for row in negatives),
                       full_frame_precision=None, negative_video_false_alarms_per_minute=None,
                       actual_image_forward_count=len(actual), forward_events=trace.events,
                       elapsed_seconds=time.perf_counter()-started, input_hashes_changed=[],
                       peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()), peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()))
    except Exception as error:
        receipt.update(status="FAILED_PRIVATE_DIAGNOSTIC_UNQUALIFIED", failure_type=type(error).__name__, failure_reason=str(error))
        raise
    finally:
        receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "sha256": digest(receipt_path), "aggregates": receipt.get("aggregates")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args(argv)
    output = workspace_path(args.output)
    if output.exists():
        raise ValueError("Preserve diagnostic evidence; require a fresh nonexistent output")
    if args.execute:
        if not args.protocol or not args.expected_protocol_sha256:
            parser.error("Execution requires the exact pre-training protocol and its separately recorded SHA256")
        frozen = read(args.protocol)
        candidate = prepare_protocol(args.run_directory, frozen["frozen_at_utc"])
        if frozen != candidate:
            raise ValueError("Prepared protocol, source, data or predeclared run directory changed")
        execute(frozen, output, binding(args.protocol, args.expected_protocol_sha256))
    else:
        if args.protocol or args.expected_protocol_sha256:
            parser.error("Protocol inputs apply only to explicit execution")
        protocol = prepare_protocol(args.run_directory)
        if "export_manifest" not in protocol["bindings"] or "scripts/train/train_yolo26_reviewed_ball.py" not in protocol["bindings"]:
            raise ValueError("Freeze the executable protocol only after the reviewed export and trainer are complete")
        output.mkdir(parents=True, exist_ok=False)
        write(output / "protocol.json", protocol)
        print(json.dumps({"status": "PREPARED_BEFORE_TRAINING_NO_MODEL_OR_IMAGE_RUNTIME", "protocol": str(output / "protocol.json"), "sha256": digest(output / "protocol.json"), "targets": 20, "native_inputs": 161}))


if __name__ == "__main__":
    main()
