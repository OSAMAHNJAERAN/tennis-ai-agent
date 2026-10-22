"""Execute the separately ROOT-approved private101 export; no models or video decode."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
import traceback

TASK_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(TASK_ROOT))

import psutil
from scripts.data import export_yolo26_author_ball_repair_v2 as repair


def main():
    output = Path(__file__).resolve().parent
    proposal_path = TASK_ROOT / "artifacts/validation/yolo26_execution_2026-10-16/ball101_export_preflight_v2/plan.json"
    expected_plan = "7c6fede7dc51f1a3d555d434276d440fd12786647bedc2e8dd1df8b46b62c135"
    expected_source = "b4067e9cd0068628c5c0cf464c86ac78ab17f61cb6e0569690f9bcf198120ac0"
    expected_proposal = "d24c2ed8a7249aa95a12cf91cfd0b83b33d05ee6b33f98f03577d36b92f332b0"
    old_five = [TASK_ROOT / path for path in (
        "scripts/train/train_yolo26_reviewed_ball.py",
        "scripts/train/train_yolo26_tennis.py",
        "scripts/data/export_yolo26_reviewed_ball_wave.py",
        "scripts/data/prepare_yolo26_reviewed_ball_wave.py",
        "scripts/data/prepare_yolo26_ball_review_wave.py",
    )]
    before = {str(path): repair.parent.digest(path) for path in old_five}
    started = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    stopped = threading.Event()
    samples, sampling_errors = [], []
    process = psutil.Process()

    def observe():
        while not stopped.is_set():
            try:
                memory = process.memory_info()
                samples.append({"elapsed_seconds": time.perf_counter() - start,
                    "available_ram_bytes": int(psutil.virtual_memory().available),
                    "owned_rss_bytes": int(memory.rss),
                    "owned_windows_peak_wset_bytes": int(getattr(memory, "peak_wset", 0))})
            except Exception as error:
                sampling_errors.append({"type": type(error).__name__, "message": str(error)})
            stopped.wait(0.25)

    thread = threading.Thread(target=observe, daemon=True)
    thread.start()
    status, error, export_receipt, integrity, approval = "FAILED", None, None, None, None
    admission = None
    try:
        if repair.parent.digest(proposal_path) != expected_plan or repair.parent.digest(repair.__file__) != expected_source:
            raise ValueError("ROOT-approved source or concrete plan changed")
        original = repair.read_json(proposal_path)
        if original["proposal_sha256"] != expected_proposal or original["root_final_approval"] is not None:
            raise ValueError("The frozen original proposal or its pending approval status changed")
        if repair.OUTPUT.exists():
            raise ValueError("ROOT-approved dataset destination is no longer fresh")
        approval = original["approval_template"]
        approval["review_note"] = (
            "ROOT approved actual fresh private101 export after reading the complete exporter, bounded stdlib PNG reader, "
            "source/closure/rejected-row/context reconstruction and native-source metadata function, and checking the final126-test receipt. "
            "Approval preserves original79 image and label bytes, order, all provenance and splits, including allold600/96 review exposure; "
            "all32 source frames/34 publisher proposals including ten rejected frames/twelve rejected proposals; all16 original/logical aliases "
            "and conservatively connected unseen compilations; and all22 actually reviewed complete640 crops with exact native RGB ROI evidence. "
            "Original PTS, nominal and average FPS, full pinned probe metadata and acquisition/publisher/grant attribution remain explicit. "
            "The author grant supports only private noncommercial academic development. Normal-speed timing and original-match/session independence "
            "remain unverified. This approval grants no independent human ground truth, qualification, TEST admission, public raw-media/examples, "
            "public weight release or commercial clearance. Execution must repeat all22 pixel comparisons and unchanged source hashes under "
            "the original1GiB available RAM/2GiB disk guards. No GPU or training is authorized by this export approval."
        )
        approval_path = output / "root_approval.json"
        repair.write_new_json(approval_path, approval)
        approval_hash = repair.parent.digest(approval_path)
        admission = repair.parent.guarded_resources()
        plan = repair.prepare_plan(repair.OUTPUT, approval_path, approval_hash)
        if plan["proposal_sha256"] != expected_proposal or plan["contract"] != original["contract"]:
            raise ValueError("Approval reconstruction changed the exact frozen concrete proposal")
        repair.write_new_json(output / "approved_plan.json", plan)
        export_receipt = repair.execute_export(plan)
        manifest_hash = repair.parent.digest(repair.OUTPUT / "source_manifest.json")
        integrity = repair.verify_export(repair.OUTPUT, expected_manifest_sha256=manifest_hash)
        repair.check_bindings(plan["contract"]["input_bindings"])
        if integrity["counts"] != repair.EXPECTED_COUNTS:
            raise ValueError("Terminal verified export counts differ")
        if before != {str(path): repair.parent.digest(path) for path in old_five}:
            raise ValueError("An immutable old dependency changed during export")
        status = "PASS_ACTUAL_PRIVATE101_EXPORT_AND_EXACT_INTEGRITY_VERIFICATION"
    except Exception as caught:
        error = {"type": type(caught).__name__, "message": str(caught), "traceback": traceback.format_exc()}
        status = "FAILED_BEFORE_DATASET_CREATED" if not repair.OUTPUT.exists() else "FAILED_PRESERVED_PARTIAL_DATASET"
    finally:
        stopped.set()
        thread.join()
    after = {str(path): repair.parent.digest(path) for path in old_five}
    heavy = [name for name in ("PIL", "numpy", "torch", "ultralytics", "cv2") if name in sys.modules]
    if heavy and status.startswith("PASS"):
        status = "FAILED_FORBIDDEN_RUNTIME_IMPORTED"
    files = [path for path in repair.OUTPUT.rglob("*") if path.is_file()] if repair.OUTPUT.exists() else []
    receipt = {
        "status": status, "started_utc": started, "ended_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start, "error": error,
        "exporter_sha256_before": expected_source, "exporter_sha256_after": repair.parent.digest(repair.__file__),
        "proposal_plan_sha256": expected_plan, "proposal_sha256": expected_proposal,
        "root_approval_path": str(output / "root_approval.json"),
        "root_approval_sha256": repair.parent.digest(output / "root_approval.json") if approval else None,
        "dataset_path": str(repair.OUTPUT), "counts": integrity["counts"] if integrity else None,
        "source_manifest_sha256": integrity["manifest_sha256"] if integrity else None,
        "export_receipt_sha256": integrity["receipt_sha256"] if integrity else None,
        "data_yaml_sha256": integrity["data_yaml_sha256"] if integrity else None,
        "actual_native_pixel_rois_rechecked": len(integrity["manifest"]["actual22_native_pixel_roi_results"]) if integrity else None,
        "export_files": len(files), "export_total_logical_bytes": sum(path.stat().st_size for path in files),
        "input_bindings_changed": [] if integrity else None,
        "old_five_before": before, "old_five_after": after, "old_five_unchanged": before == after,
        "launch_guard_resources": admission,
        "resource_guard_unchanged": "1GiB available RAM and 2GiB free disk",
        "resource_sample_count": len(samples), "sampling_errors": sampling_errors,
        "minimum_available_ram_bytes": min((row["available_ram_bytes"] for row in samples), default=None),
        "peak_owned_rss_bytes": max((row["owned_rss_bytes"] for row in samples), default=None),
        "windows_peak_wset_bytes": max((row["owned_windows_peak_wset_bytes"] for row in samples), default=None),
        "forbidden_runtime_modules_loaded": heavy, "models_used": [], "source_video_decode": False,
        "training_performed": False, "qualification_evidence": False, "independent_human_review": False,
    }
    repair.write_new_json(output / "resource_samples.json", samples)
    repair.write_new_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
