"""One guarded real-video compatibility run of the new caller-owned shot seam.

Passive wrappers record storage lifetime and return the original objects. No
model, evidence, configuration, threshold or authority result is changed.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/private_yolo26_shot_store_run1_2026-10-16"
CONFIG = "configs/evaluation/yolo26.private-reviewed20-match172-2026-10-16.yaml"
CONFIG_SHA = "6f90ce4b90a6a9d0323f20337deb55491f365b50b7945195c382700298fd0c57"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def child_main():
    sys.path.insert(0, str(ROOT))
    from scripts.evaluate import run_yolo26_candidate as runner
    observed = {"scope": "PASSIVE_REAL_THIRTY_FRAME_PRIVATE_STORAGE_COMPATIBILITY_NOT_QUALIFICATION",
                "started_utc": datetime.now(timezone.utc).isoformat(), "shot_link_calls": 0}
    original_instrument = runner.instrument_models

    def instrument(pipeline):
        stats = original_instrument(pipeline)
        observed["models"] = {role: {"python_object_id": id(owner.model.model), "path": owner.model_path}
            for role, owner in (("ball", pipeline.ball_detector), ("players", pipeline.player_detector),
                                ("court", pipeline.court_detector))}
        original_link = pipeline.shot_linker.link_shots

        def link(*args, **kwargs):
            result = original_link(*args, **kwargs)
            output, event_owner = kwargs.get("result_output"), kwargs.get("event_output")
            observed["shot_link_calls"] += 1
            observed["shot_storage"] = {
                "owner_type": type(output).__name__, "result_type": type(result).__name__,
                "result_is_exact_owner": result is output, "owner_open_at_link_return": output._connection is not None,
                "scratch_path": str(output.path), "scratch_exists_at_link_return": output.path.exists(),
                "shot_count": len(result), "input_authoritative_events_count": len(kwargs["events"]),
                "event_owner_supplied": event_owner is not None,
                "input_is_event_owner_alias": kwargs["events"] is event_owner.events if event_owner else None,
            }
            return result

        pipeline.shot_linker.link_shots = link
        return stats

    runner.instrument_models = instrument
    try:
        runner.main(["--config", CONFIG, "--source-id", "match172", "--output", str(OUTPUT),
                     "--frames", "30", "--device", "0", "--execute"])
    finally:
        observed["completed_utc"] = datetime.now(timezone.utc).isoformat()
        observed["distinct_model_objects"] = len({r["python_object_id"] for r in observed.get("models", {}).values()})
        storage = observed.get("shot_storage")
        if storage:
            storage["scratch_exists_after_pipeline_exit"] = Path(storage["scratch_path"]).exists()
        if OUTPUT.is_dir():
            write(OUTPUT / "shot_storage_observer.json", observed)


def main():
    assert sha(ROOT / CONFIG) == CONFIG_SHA
    if "--child" in sys.argv:
        child_main()
        return 0
    if OUTPUT.exists():
        raise FileExistsError("Keep prior attempts; fresh output required")
    before = sources()
    receipt = {"status": "RUNNING_ONE_OWNED_GUARDED_PRIVATE_ATTEMPT", "source_bindings_before": before,
               "started_utc": datetime.now(timezone.utc).isoformat(), "max_wall_seconds": 180,
               "observer_sha256": sha(Path(__file__)), "config_sha256": CONFIG_SHA,
               "qualification_evidence": False, "final_test_used": False, "automatic_retry": False}
    start = time.monotonic()
    with (BASE / "execution_console.log").open("x", encoding="utf-8") as console:
        child = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--child"], cwd=ROOT,
            stdout=console, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        receipt["owned_pid"] = child.pid
        write(BASE / "execution_start.json", receipt)
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], timeout=15,
                stdout=console, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            code = child.wait(timeout=15)
            receipt["owned_child_timeout_terminated"] = True
    after = sources()
    receipt.update(status="PROCESS_TERMINAL_CHECK_EVALUATION_AND_OBSERVER", exit_code=code,
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
        source_changes={"changed": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
                        "added": sorted(after.keys() - before.keys()), "removed": sorted(before.keys() - after.keys())},
        console_sha256=sha(BASE / "execution_console.log"))
    actual = OUTPUT / "evaluation_receipt.json"
    if actual.is_file():
        value = json.loads(actual.read_text(encoding="utf-8-sig"))
        receipt.update(evaluation_status=value["status"], evaluation_receipt_sha256=sha(actual))
    write(BASE / "execution_process_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_bindings_before"}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
