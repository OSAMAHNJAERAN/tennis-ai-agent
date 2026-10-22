"""Frozen all26 DEV native court diagnostic; default never imports ML runtime.

The same protocol governs the prior court14 baseline and the one registered
court175 DEV-fitness best checkpoint. No threshold, geometry or checkpoint sweep.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from scripts.train import train_yolo26_court_repair as training
from scripts.evaluate import diagnose_yolo26_reviewed_court as scoring
from scripts.evaluate import diagnose_yolo26_court_input_transform as observed


ROOT = training.ROOT
COMPLETED = "COMPLETED_ALL26_NATIVE_COURT175_FIXED_DIAGNOSTIC_NOT_QUALIFICATION"


def prepare(protocol_path, protocol_sha256, checkpoint, checkpoint_sha256, *, phase):
    protocol_path = training.validate_binding({"path": str(Path(protocol_path).resolve()), "sha256": protocol_sha256})
    frozen = training.read_json(protocol_path)
    manifest = training.read_json(training.EXPORT / "source_manifest.json")
    training.verify_export(training.EXPORT, expected_manifest_sha256=training.MANIFEST_SHA)
    protocol = training.diagnostic_protocol(manifest)
    if (frozen.get("status") != "FROZEN_BEFORE_ANY_COURT175_GPU_DIAGNOSTIC_OR_TRAINING"
            or frozen.get("protocol") != protocol):
        raise ValueError("The exact all26 DEV protocol must be frozen before GPU activity")
    checkpoint = training.validate_binding({"path": str(Path(checkpoint).resolve()), "sha256": checkpoint_sha256})
    metadata_binding = None
    if phase == "baseline":
        training.audit_initialization()
        if checkpoint != training.INITIAL or checkpoint_sha256 != training.INITIAL_SHA:
            raise ValueError("Baseline must use the pinned prior court14 best checkpoint")
    elif phase == "post20":
        if checkpoint.parts[-3:] != ("training", "weights", "best.pt"):
            raise ValueError("Post20 diagnostic uses only registered engineering DEV-fitness best.pt")
        metadata_path = checkpoint.parents[2] / "run_metadata.json"
        metadata = training.read_json(metadata_path)
        if (metadata.get("status") != "FINISHED_SAME20_FINITE_REPAIR_UNQUALIFIED"
                or metadata.get("mode") != training.MODE or metadata.get("manifest_sha256") != training.MANIFEST_SHA
                or metadata.get("qualification_evidence") is not False or metadata.get("final_test_used") is not False
                or not any(row.get("path") == str(checkpoint) and row.get("sha256") == checkpoint_sha256
                           for row in metadata.get("checkpoints", []))):
            raise ValueError("Require the completed same20 court175 run and its exact registered best checkpoint")
        metadata_binding = training._binding(metadata_path)
    else:
        raise ValueError("Only the before-training baseline and completed post20 diagnostic are allowed")
    return {"protocol": protocol, "protocol_binding": training._binding(protocol_path),
            "checkpoint": training._binding(checkpoint), "phase": phase, "training_metadata": metadata_binding,
            "source_bindings": training.source_bindings(), "runtime_source_bindings": training.runtime_bindings(),
            "resources_before_gpu_runtime_guard": {"available_ram_bytes": 4608 * 1024**2,
                "available_disk_bytes": 1024**3}, "post_cuda_guard": {"available_ram_bytes": 1536 * 1024**2,
                "available_vram_bytes": 2 * 1024**3, "available_disk_bytes": 1024**3}}


def execute(plan, output):
    output = training.fresh_output(output)
    # Reconstruct source bytes and membership immediately before this invocation.
    if prepare(plan["protocol_binding"]["path"], plan["protocol_binding"]["sha256"],
               plan["checkpoint"]["path"], plan["checkpoint"]["sha256"], phase=plan["phase"]) != plan:
        raise ValueError("All26 diagnostic plan changed before execution")
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "NOT_STARTED_RESOURCE_ADMISSION_PENDING", "created_utc": datetime.now(timezone.utc).isoformat(),
        "qualification_evidence": False, "final_test_used": False, "independent_human_review": False,
        "production_promoted": False, "training_performed": False, "manifest_sha256": training.MANIFEST_SHA,
        "plan": plan, "rows": [], "runtime_probe_performed": False,
        "keypoint_coordinate_policy": "PINNED_RESULTS_DATA_NO_CONFIDENCE_MASK; NATIVE_COORDINATES_INCLUDE_LIBRARY_BOUND_CLIPPING",
        "not_established": ["INDEPENDENT_QUALIFICATION", "METRIC_GROUND_GEOMETRY", "VIDEO_STABILITY", "LINE_CALLS"]}
    path = output / "diagnostic.json"
    from scripts.train.train_yolo26_reviewed_ball import _write_metadata
    _write_metadata(path, receipt)
    started = time.perf_counter()
    try:
        receipt["resources_before_gpu_runtime"] = training._resources(output)
        training.before_gpu_runtime_guard(receipt["resources_before_gpu_runtime"])
        _write_metadata(path, receipt)
        import torch
        from ultralytics import YOLO
        from scripts.train.train_yolo26_tennis import check_resource_guard
        if not torch.cuda.is_available():
            raise ValueError("CUDA0 required for the frozen diagnostic; no runtime fallback")
        torch.cuda.set_device(0)
        torch.cuda.init()
        resources = training._resources(output, torch=torch)
        receipt.update(runtime_probe_performed=True, resources_after_cuda=resources)
        _write_metadata(path, receipt)
        check_resource_guard(resources, gpu_requested=True, role="court", image_size=1280, development=True)
        if torch.__version__ != training.VERSIONS["torch"] or torch.version.cuda != "12.6":
            raise ValueError("Pinned diagnostic Torch/CUDA changed")
        model = YOLO(plan["checkpoint"]["path"], task="pose")
        training.validate_model(model)
        trace = observed.ForwardTrace()
        predictor = observed.runtime_predictor(trace)
        def forward(_model, arguments):
            tensor = arguments[0]
            trace.observe(tensor.shape, str(tensor.dtype), str(tensor.device),
                          done_warmup=bool(getattr(model.predictor, "done_warmup", False)))
        hook = model.model.register_forward_pre_hook(forward)
        inference = dict(plan["protocol"]["inference"], nms=False, verbose=False, save=False,
                         save_txt=False, save_crop=False, show=False, visualize=False, embed=None)
        receipt.update(status="ALL26_DIAGNOSTIC_RUNNING_UNQUALIFIED", inference_arguments=inference)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            for index, sample in enumerate(plan["protocol"]["samples"]):
                trace.begin_image("FROZEN_RECT_NATIVE_BASELINE_POLICY", sample["id"])
                result = model.predict(sample["image"]["path"], predictor=predictor, **inference)[0]
                if list(result.orig_shape) != [720, 1280]:
                    raise ValueError("All26 predictions must retain native1280x720 coordinates")
                boxes = result.boxes
                candidates = scoring.raw_candidates(
                    [] if boxes is None else boxes.xyxy.detach().cpu().tolist(),
                    [] if boxes is None else boxes.conf.detach().cpu().tolist(),
                    [] if boxes is None else boxes.cls.detach().cpu().tolist(),
                    None if result.keypoints is None else result.keypoints.data.detach().cpu().tolist())
                assessment = scoring.score_image(sample, candidates)
                overlay = scoring.render_overlay(sample["image"]["path"], assessment,
                    output / f"dev_{index + 1:02d}_{sample['id']}.png",
                    f"DEV {index + 1}/26 {sample['id']} | phase={plan['phase']} | IoU={assessment['selected_bbox_iou']}")
                receipt["rows"].append({"id": sample["id"], "split": "val", "context_group": sample["context_group"],
                    "visual_event_family": sample["visual_event_family"], "image": sample["image"], "label": sample["label"],
                    "source_fps": None, "source_pts": None, "source_time_base": None,
                    "observed_preprocessing_and_forward": trace.end_image(), "assessment": assessment, "overlay": overlay})
                _write_metadata(path, receipt)
        finally:
            hook.remove()
        if len(receipt["rows"]) != 26:
            raise ValueError("Completed court175 diagnostics require all26 DEV rows")
        receipt.update(aggregates=scoring.aggregate(receipt["rows"]), forward_events=trace.events,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0), resources_at_end=training._resources(output, torch=torch))
        after = prepare(plan["protocol_binding"]["path"], plan["protocol_binding"]["sha256"],
                        plan["checkpoint"]["path"], plan["checkpoint"]["sha256"], phase=plan["phase"])
        if after != plan:
            raise ValueError("Frozen diagnostic inputs/source/runtime membership changed")
        receipt.update(status=COMPLETED, input_source_runtime_changes=[],
            completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        _write_metadata(path, receipt)
    except BaseException as error:
        receipt.update(status="DIAGNOSTIC_NOT_STARTED_OR_FAILED_PRESERVED_NO_RETRY", failure_type=type(error).__name__,
            failure_reason=str(error), failed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        _write_metadata(path, receipt)
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, default=training.INITIAL)
    parser.add_argument("--checkpoint-sha256", default=training.INITIAL_SHA)
    parser.add_argument("--phase", choices=("baseline", "post20"), default="baseline")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--save-plan", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.protocol, args.protocol_sha256, args.checkpoint, args.checkpoint_sha256, phase=args.phase)
    training.fresh_output(args.output)
    if args.save_plan:
        training.write_json(args.save_plan, plan)
    if args.execute:
        result = execute(plan, args.output)
        print(json.dumps({"status": result["status"], "images": len(result["rows"]), "output": str(args.output)}))
    else:
        print(json.dumps({"status": "PURE_FROZEN_ALL26_DIAGNOSTIC_PREFLIGHT_NOT_INFERRED", "images": 26,
                          "checkpoint_sha256": plan["checkpoint"]["sha256"], "phase": args.phase}))


if __name__ == "__main__":
    main()
