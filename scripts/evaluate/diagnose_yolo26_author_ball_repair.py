"""Fixed before/after20DEV native640 diagnostic for private ball101 repair.

The original twenty targets, one absence crop, confidence and tile protocol are
immutable. Default preparation uses metadata/hashes only. Runtime requires a
separate ROOT phase release; unmatched full-frame proposals remain unadjudicated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from scripts.data import export_yolo26_author_ball_repair_v2 as exported

ROOT = exported.ROOT
REFERENCE = ROOT / "artifacts/validation/yolo26_execution_2026-10-15/ball_reviewed_diagnostic_protocol_v1/protocol.json"
REFERENCE_SHA = "21a40651572f6666153b28fc6656502e2b12b8f9f0a27b856c13e64fda5d7abe"
LEGACY_SOURCE_SHA = "f8cedb210b160d6e21c4d028b045712ee4bbd5ad41b0de2e08275b6310446283"
MANIFEST_SHA = "fdd63635665ebd03426ab452496cf0b93ad52d462f9c32886e6ff7f3ab564c22"
INITIAL_SHA = "90a141489dc7198789043466d7ab3c0245b86cb0135122f7b78d62115a8a9fcc"
RELEASE = "ROOT_APPROVED_BALL101_FIXED_TWENTY_DEV_DIAGNOSTIC_MEMBER"


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": exported.parent.digest(path)}


def prepare_protocol(_run_directory=None):
    if exported.parent.digest(REFERENCE) != REFERENCE_SHA:
        raise ValueError("Original frozen20DEV protocol changed")
    legacy_path = ROOT / "scripts/evaluate/diagnose_yolo26_reviewed_ball.py"
    if exported.parent.digest(legacy_path) != LEGACY_SOURCE_SHA:
        raise ValueError("Original native preprocessing/scoring helper changed")
    from scripts.evaluate import diagnose_yolo26_reviewed_ball as legacy
    reference = exported.read_json(REFERENCE)
    if legacy.prepare_protocol(reference["training_run_directory"], reference["frozen_at_utc"]) != reference:
        raise ValueError("Original native target/proposal/crop evidence changed")
    result = exported.verify_export(expected_manifest_sha256=MANIFEST_SHA)
    dev = [row for row in result["samples"] if row["development_split"] == "val"]
    if len(dev) != 21 or any(not row["inherited_from_old79"] for row in dev):
        raise ValueError("Repair must leave every original DEV crop unchanged")
    keys = ("inference", "native_frame_wh", "fixed_native_tile_grid", "target_iou_minimum", "dedup_iou_or_center",
            "positive_samples", "negative_samples", "all_target_denominator", "actual_image_forward_count",
            "unmatched_full_frame_candidates", "center_definition")
    protocol = {key: deepcopy(reference[key]) for key in keys}
    protocol.update(schema="tennis.private-ball101-fixed-before-after20dev.v1",
        reference_protocol=binding(REFERENCE), legacy_native_helper=binding(legacy_path),
        dataset_manifest=binding(exported.OUTPUT / "source_manifest.json"),
        source=binding(__file__), checkpoint_comparison="PINNED_OLD_BEST_BEFORE_VS_REGISTERED_DEV_FITNESS_BEST_AFTER_APPROVED_STAGE",
        full_frame_precision=None, negative_video_false_alarms_per_minute=None,
        qualification_evidence=False, independent_human_review=False, final_test_used=False)
    if (protocol["all_target_denominator"] != 20 or protocol["actual_image_forward_count"] != 161
            or len(protocol["positive_samples"]) != 20 or len(protocol["negative_samples"]) != 1
            or protocol["inference"]["conf"] != .01):
        raise ValueError("Fixed target denominator or inference operating point changed")
    return protocol


def validate_member(protocol, phase, metadata_binding=None):
    from scripts.train import train_yolo26_author_ball_repair as trainer
    if prepare_protocol() != protocol:
        raise ValueError("Frozen source/data/native diagnostic protocol changed")
    if phase == "before":
        if exported.parent.digest(trainer.INITIAL) != INITIAL_SHA or metadata_binding is not None:
            raise ValueError("Before member must use only the exact prior trained best")
        return trainer.INITIAL, {"checkpoint": binding(trainer.INITIAL)}
    if phase != "after" or metadata_binding is None:
        raise ValueError("Use explicit before or registered completed-stage after member")
    path = trainer.validate_binding(metadata_binding)
    metadata = exported.read_json(path)
    if (metadata.get("mode") != trainer.MODE or metadata.get("status") not in {
            "STOPPED_AFTER_TWO_COMPLETED_EPOCHS_FOR_ROOT_REVIEW", "FINISHED_SAME_TWENTY_EPOCH_REPAIR_UNQUALIFIED"}
            or metadata.get("dataset", {}).get("manifest", {}).get("sha256") != MANIFEST_SHA
            or any(metadata.get(key) is not False for key in ("qualification_evidence", "final_test_used", "production_promoted"))):
        raise ValueError("After member requires an actual completed authorized private ball101 stage")
    checkpoint = path.parent / "training/weights/best.pt"
    registered = [row for row in metadata.get("checkpoints", []) if Path(row["path"]).resolve() == checkpoint]
    if len(registered) != 1 or registered[0] != binding(checkpoint):
        raise ValueError("After checkpoint must be the hash-registered DEV-fitness best")
    return checkpoint, {"checkpoint": binding(checkpoint), "metadata": binding(path)}


def validate_release(approval, protocol, phase):
    if (approval.get("status") != RELEASE or approval.get("reviewer_type") != "ROOT_AGENT"
            or approval.get("phase") != phase or approval.get("protocol_sha256") != exported.canonical_hash(protocol)
            or not isinstance(approval.get("review_note"), str) or not approval["review_note"].strip()
            or any(approval.get(key) is not False for key in ("qualification_evidence", "final_test_used", "public_release_cleared"))):
        raise ValueError("Separate ROOT release required for this exact fixed diagnostic member")


def validate_completed_receipt(receipt, protocol, phase):
    """Reject plausible completion labels without the complete fixed inputs."""
    rows, negatives = receipt.get("rows", []), receipt.get("negative_crop_rows", [])
    actual = [row for row in receipt.get("forward_events", []) if row.get("phase") == "ACTUAL_IMAGE_INFERENCE"]
    expected_ids = [f"{sample['id']}:t{index}" for sample in protocol["positive_samples"] for index in range(8)]
    expected_ids.extend(sample["id"] for sample in protocol["negative_samples"])
    if (receipt.get("status") != "COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED" or receipt.get("phase") != phase
            or receipt.get("protocol_sha256") != exported.canonical_hash(protocol)
            or receipt.get("all_target_denominator") != 20 or len(rows) != 20 or len(negatives) != 1
            or receipt.get("actual_image_forward_count") != 161 or [row.get("id") for row in actual] != expected_ids
            or any(row.get("shape_nchw") != [1, 3, 640, 640] or row.get("dtype") != "torch.float32" for row in actual)
            or any(receipt.get(key) is not False for key in ("qualification_evidence", "final_test_used", "training_performed"))
            or receipt.get("full_frame_precision") is not None or receipt.get("negative_video_false_alarms_per_minute") is not None):
        raise ValueError("Completed diagnostic lacks every fixed native input or makes unsupported precision claims")
    for row, sample in zip(rows, protocol["positive_samples"], strict=True):
        if row.get("sample") != sample or len(row.get("tiles", [])) != 8:
            raise ValueError("Fixed DEV target was changed, dropped or replaced")
        for tile, region in zip(row["tiles"], protocol["fixed_native_tile_grid"], strict=True):
            observed = tile.get("observed", {})
            if tile.get("region") != region or observed.get("preprocessing", {}).get("native_pixels_preserved") is not True:
                raise ValueError("Actual diagnostic preprocessing does not preserve fixed native pixels")
    if negatives[0].get("sample") != protocol["negative_samples"][0]:
        raise ValueError("Fixed reviewed absence crop changed")
    aggregate = receipt.get("aggregates", {})
    matched = aggregate.get("matched_targets_conf001_iou05")
    missed = aggregate.get("missed_targets_conf001_iou05")
    if (type(matched) is not int or type(missed) is not int or matched < 0 or missed < 0 or matched + missed != 20
            or aggregate.get("reviewed_targets") != 20):
        raise ValueError("Every DEV target including misses must remain in the denominator")
    empty_changes = {"changed": [], "added": [], "removed": []}
    if (receipt.get("input_hashes_changed") != [] or receipt.get("full_source_integrity_changes") != empty_changes
            or receipt.get("runtime_integrity_changes") != empty_changes
            or not receipt.get("fixed_input_bindings_before") or not receipt.get("full_source_bindings_before")
            or receipt.get("full_source_bindings_after") != receipt.get("full_source_bindings_before")
            or not receipt.get("runtime_source_bindings")
            or receipt.get("runtime_source_bindings_after") != receipt.get("runtime_source_bindings")):
        raise ValueError("Completed diagnostic lacks unchanged actual helper/input/runtime evidence")


def execute_member(protocol, phase, output, approval, frozen_utc):
    from scripts.train import train_yolo26_author_ball_repair as trainer
    validate_release(approval, protocol, phase)
    checkpoint, sources = validate_member(protocol, phase, approval.get("run_metadata"))
    if phase == "after" and datetime.fromisoformat(frozen_utc) >= datetime.fromisoformat(exported.read_json(sources["metadata"]["path"])["started_utc"]):
        raise ValueError("The before/after diagnostic must have been frozen before new training")
    output = trainer.fresh_output(output)
    before = trainer.resource_snapshot(output)
    trainer.resource_guard(before, "before_gpu")
    sources_before, runtime_before = trainer.frozen_runtime_sources()
    result = exported.verify_export(expected_manifest_sha256=MANIFEST_SHA)
    fixed_inputs = {}
    for path, sha256 in result["manifest"]["contract"]["input_bindings"].items():
        exported.bind(fixed_inputs, path, sha256)
    for sample in result["samples"]:
        exported.bind(fixed_inputs, sample["image_path"], sample["image_source_sha256"])
        exported.bind(fixed_inputs, sample["label_path"], sample["label_sha256"])
    for item in (list(sources.values()) + [protocol["reference_protocol"], protocol["legacy_native_helper"], protocol["dataset_manifest"], protocol["source"]]
                 + list(exported.read_json(REFERENCE)["bindings"].values())
                 + [row["native_frame"] for row in protocol["positive_samples"]]
                 + [row["crop"] for row in protocol["negative_samples"]]):
        exported.bind(fixed_inputs, item["path"], item["sha256"])
    for filename in ("export_receipt.json", "ball/data.yaml"):
        item = binding(exported.OUTPUT / filename)
        exported.bind(fixed_inputs, item["path"], item["sha256"])
    exported.check_bindings(fixed_inputs)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "FIXED_MEMBER_STARTED", "phase": phase, "protocol_sha256": exported.canonical_hash(protocol),
        "checkpoint": sources["checkpoint"], "source_bindings": sources, "resources_before_torch": before,
        "full_source_bindings_before": sources_before, "runtime_source_bindings": runtime_before,
        "fixed_input_bindings_before": fixed_inputs,
        "all_target_denominator": 20, "full_frame_precision": None, "negative_video_false_alarms_per_minute": None,
        "qualification_evidence": False, "final_test_used": False, "training_performed": False}
    rows, negatives, start, trace = [], [], time.perf_counter(), None
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        from scripts.data import diagnose_yolo26_native_tiles as scoring
        from scripts.evaluate.diagnose_yolo26_court_input_transform import ForwardTrace
        from scripts.evaluate.diagnose_yolo26_reviewed_ball import observed_predictor, score_reviewed_target
        if torch.__version__ != trainer.VERSIONS["torch"] or not torch.cuda.is_available():
            raise ValueError("Pinned CUDA0 diagnostic required; no CPU inference fallback")
        torch.cuda.set_device(0)
        torch.cuda.init()
        receipt["resources_after_cuda"] = trainer.resource_snapshot(output, torch)
        trainer.resource_guard(receipt["resources_after_cuda"], "after_cuda")
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats(0)
        model = YOLO(str(checkpoint), task="detect")
        trainer.validate_model(model)
        trace = ForwardTrace()
        predictor = observed_predictor(trace)
        def observe(_, arguments):
            tensor = arguments[0]
            trace.observe(tensor.shape, str(tensor.dtype), str(tensor.device), done_warmup=model.predictor.done_warmup)
        hook = model.model.register_forward_pre_hook(observe)
        def infer(image, ident, region, width=1920, height=1080):
            trace.begin_image("FIXED_NATIVE640", ident)
            result = model.predict(image, predictor=predictor, **protocol["inference"])[0]
            observed = trace.end_image()
            boxes = result.boxes
            candidates = scoring.map_candidates(boxes.xyxy.detach().cpu().numpy(), boxes.conf.detach().cpu().numpy(), boxes.cls.detach().cpu().numpy(), region, ident, width, height) if boxes is not None else []
            return candidates, observed
        (output / "overlays").mkdir()
        try:
            for sample in protocol["positive_samples"]:
                image = cv2.imread(sample["native_frame"]["path"])
                if image is None or image.shape != (1080, 1920, 3):
                    raise ValueError("Fixed DEV input must decode native1920x1080")
                raw, tiles = [], []
                for index, region in enumerate(protocol["fixed_native_tile_grid"]):
                    left, top, right, bottom = region
                    candidates, observed = infer(image[top:bottom, left:right], f"{sample['id']}:t{index}", region)
                    raw.extend(candidates)
                    tiles.append({"region": region, "observed": observed})
                accepted, associations = scoring.deduplicate(raw)
                overlay = image.copy()
                for candidate in accepted:
                    x1, y1, x2, y2 = [int(round(value)) for value in candidate["global_box_xyxy_px"]]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 1)
                x1, y1, x2, y2 = map(int, sample["target_box"])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
                path = output / "overlays" / (sample["id"] + ".png")
                if not cv2.imwrite(str(path), overlay):
                    raise ValueError("Unable to retain native diagnostic overlay")
                rows.append({"sample": sample, "tiles": tiles, "raw_candidates": raw,
                    "deduplicated_candidates": accepted, "dedup_associations": associations,
                    "overlay": binding(path), "native_tiles": score_reviewed_target(scoring, accepted, sample)})
            for sample in protocol["negative_samples"]:
                image = cv2.imread(sample["crop"]["path"])
                if image is None or image.shape != (640, 640, 3):
                    raise ValueError("Fixed absence crop must decode native640")
                candidates, observed = infer(image, sample["id"], [0, 0, 640, 640], 640, 640)
                accepted, associations = scoring.deduplicate(candidates)
                negatives.append({"sample": sample, "deduplicated_candidates": accepted,
                    "dedup_associations": associations, "observed": observed})
        finally:
            hook.remove()
        actual = [event for event in trace.events if event["phase"] == "ACTUAL_IMAGE_INFERENCE"]
        if len(actual) != 161 or len(rows) != 20 or len(negatives) != 1:
            raise ValueError("Fixed diagnostic omitted a target or actual image forward")
        validate_member(protocol, phase, approval.get("run_metadata"))
        aggregate = scoring.aggregate(rows)
        aggregate["unmatched_full_frame_candidates_unadjudicated"] = aggregate.pop("unmatched_or_duplicate_candidates")
        errors = [row["native_tiles"]["matched_candidate_center_error_native_px"] for row in rows
                  if row["native_tiles"]["matched_candidate_center_error_native_px"] is not None]
        aggregate["conditional_matched_center_error_p95_native_px"] = float(np.percentile(errors, 95)) if errors else None
        aggregate["matched_center_error_sample_count"] = len(errors)
        receipt.update(status="COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED", aggregates=aggregate,
            actual_image_forward_count=len(actual), forward_events=trace.events,
            negative_crop_returned_proposals=sum(len(row["deduplicated_candidates"]) for row in negatives),
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(0)), peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))
        receipt.update(rows=rows, negative_crop_rows=negatives)
    except BaseException as error:
        receipt.update(status="FAILED_FIXED_MEMBER_PRESERVED_NO_AUTOMATIC_RETRY", failure_type=type(error).__name__, failure_reason=str(error))
        raise
    finally:
        receipt["forward_events"] = [] if trace is None else list(trace.events)
        receipt["actual_image_forward_count"] = sum(event.get("phase") == "ACTUAL_IMAGE_INFERENCE" for event in receipt["forward_events"])
        final_error = None
        try:
            sources_after, runtime_after = trainer.frozen_runtime_sources()
            from scripts.train.train_yolo26_reviewed_ball import _binding_changes
            receipt["full_source_bindings_after"] = sources_after
            receipt["full_source_integrity_changes"] = _binding_changes(sources_before, sources_after)
            receipt["runtime_source_bindings_after"] = runtime_after
            receipt["runtime_integrity_changes"] = _binding_changes(runtime_before, runtime_after)
            receipt["input_hashes_changed"] = [path for path, sha256 in fixed_inputs.items()
                if not Path(path).is_file() or exported.parent.digest(path) != sha256]
            if (any(receipt["full_source_integrity_changes"].values())
                    or any(receipt["runtime_integrity_changes"].values()) or receipt["input_hashes_changed"]):
                final_error = ValueError("Actual diagnostic helper/source/runtime/model/data bytes changed")
        except Exception as error:
            receipt["end_integrity_observation_error"] = {"type": type(error).__name__, "message": str(error)}
            final_error = ValueError("Complete final diagnostic integrity could not be observed")
        actual_torch = sys.modules.get("torch")
        try:
            receipt["cuda_initialized"] = bool(actual_torch.cuda.is_initialized()) if actual_torch is not None else None
        except Exception as error:
            receipt["cuda_initialized"] = None
            receipt["cuda_observation_error"] = {"type": type(error).__name__, "message": str(error)}
        try:
            receipt["resources_at_end"] = trainer.resource_snapshot(output)
            receipt["owned_peak_working_set_bytes"] = receipt["resources_at_end"]["peak_rss"]
        except Exception as error:
            receipt["end_resource_observation_error"] = {"type": type(error).__name__, "message": str(error)}
            final_error = final_error or ValueError("Final diagnostic resources could not be observed")
        if final_error is None and receipt["status"] == "COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED":
            try:
                validate_completed_receipt(receipt, protocol, phase)
            except Exception as error:
                final_error = error
        raise_final_error = final_error is not None and receipt["status"] == "COMPLETED_BALL101_FIXED20DEV_MEMBER_UNQUALIFIED"
        if final_error is not None:
            receipt["end_integrity_failure"] = {"type": type(final_error).__name__, "message": str(final_error)}
        if raise_final_error:
            receipt.update(status="FAILED_FIXED_MEMBER_PRESERVED_NO_AUTOMATIC_RETRY", failure_type=type(final_error).__name__, failure_reason=str(final_error))
        receipt.update(rows=rows, negative_crop_rows=negatives, elapsed_seconds=time.perf_counter() - start,
            completed_utc=datetime.now(timezone.utc).isoformat())
        trainer.save(output / "diagnostic.json", receipt)
        if raise_final_error:
            raise final_error
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--phase", choices=("before", "after"))
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--approval-sha256")
    args = parser.parse_args(argv)
    from scripts.train import train_yolo26_author_ball_repair as trainer
    if args.command == "prepare":
        output = trainer.fresh_output(args.output)
        protocol = prepare_protocol()
        output.mkdir(parents=True, exist_ok=False)
        exported.write_new_json(output / "protocol.json", {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol})
        print(json.dumps({"status": "FIXED20DEV_PROTOCOL_PREPARED_NO_RUNTIME", "protocol_sha256": exported.canonical_hash(protocol)}, indent=2))
    else:
        if (not args.protocol or exported.parent.digest(args.protocol) != args.protocol_sha256
                or not args.approval or exported.parent.digest(args.approval) != args.approval_sha256 or not args.phase):
            raise ValueError("Require exact saved protocol and separate ROOT phase release")
        frozen = exported.read_json(args.protocol)
        execute_member(frozen["protocol"], args.phase, args.output, exported.read_json(args.approval), frozen["frozen_at_utc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
