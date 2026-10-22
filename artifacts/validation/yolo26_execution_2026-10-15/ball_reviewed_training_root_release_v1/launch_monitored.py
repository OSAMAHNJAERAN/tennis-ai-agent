"""Run one exact approved private development job; monitor only its process tree."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
APPROVAL = BASE / "root_initial20_approval.json"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            value.update(part)
    return value.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    approval = json.loads(APPROVAL.read_text(encoding="utf-8"))
    if (approval["status"] != "ROOT_REVIEWED_INITIAL20_PRIVATE_BALL_DEVELOPMENT_APPROVED"
            or approval["qualification_evidence"] is not False or approval["final_test_used"] is not False
            or approval["maximum_wall_seconds"] != 1800):
        raise ValueError("Require the exact separately reviewed initial private development approval")
    for path, expected in approval["bindings"].items():
        if sha(path) != expected:
            raise ValueError(f"Approved input/source changed before launch: {path}")
    command = approval["argv"]
    if (command[0] != sys.executable or command[1:4] != ["-u", "-m", "scripts.train.train_yolo26_reviewed_ball"]
            or command[-1] != "--execute"):
        raise ValueError("Only the exact reviewed local trainer command may run")
    output = Path(approval["run_output"]).resolve()
    if output.exists() or not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Preserve old runs; require a fresh workspace output")
    for name in ("training_console.log", "resource_samples.json", "execution_receipt.json"):
        if (BASE / name).exists():
            raise FileExistsError("Preserve existing monitored execution evidence")
    samples, seen_epoch, failure, process = [], 0, None, None
    started = datetime.now(timezone.utc).isoformat()
    clock_start = time.perf_counter()
    with (BASE / "training_console.log").open("x", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            owned = psutil.Process(process.pid)
            print(json.dumps({"status": "OWNED_TRAINING_PROCESS_STARTED", "pid": process.pid, "output": str(output)}), flush=True)
            while process.poll() is None:
                elapsed = time.perf_counter()-clock_start
                if elapsed > approval["maximum_wall_seconds"]:
                    raise TimeoutError("Approved thirty-minute wall ceiling reached")
                tree = [owned]
                try:
                    tree += owned.children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
                rss = 0
                for member in tree:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                samples.append(dict(elapsed_seconds=elapsed, available_ram_bytes=int(psutil.virtual_memory().available),
                                    owned_tree_rss_bytes=int(rss)))
                metadata_path = output / "run_metadata.json"
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    updates = metadata.get("optimizer_update_receipts", [])
                    epoch = updates[-1]["epoch"] if updates else 0
                    if epoch > seen_epoch:
                        seen_epoch = epoch
                        print(json.dumps({"epoch_finished": epoch, "optimizer_steps": updates[-1]["optimizer_step_values"],
                                          "available_ram_bytes": samples[-1]["available_ram_bytes"]}), flush=True)
                except (OSError, json.JSONDecodeError):
                    pass  # A monitored read does not substitute for trainer status.
                time.sleep(2)
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error))
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    metadata_path = output / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    exit_code = process.returncode if process is not None else None
    changed = [path for path, expected in approval["bindings"].items() if sha(path) != expected]
    write(BASE / "resource_samples.json", samples)
    status = ("COMPLETED_PRIVATE_TRAINED_UNQUALIFIED" if exit_code == 0 and metadata is not None
              and metadata["status"] == "TRAINED_UNQUALIFIED" and failure is None and not changed
              else "FAILED_OR_GUARD_REFUSED_PRIVATE_TRAINING_UNQUALIFIED")
    receipt = dict(status=status, started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.perf_counter()-clock_start, exit_code=exit_code, failure=failure,
                   pid=process.pid if process is not None else None, argv=command,
                   root_approval_sha256=sha(APPROVAL), approved_bindings_changed=changed,
                   run_metadata_sha256=sha(metadata_path) if metadata_path.exists() else None,
                   training_console_sha256=sha(BASE / "training_console.log"),
                   resource_samples_sha256=sha(BASE / "resource_samples.json"), sampled_records=len(samples),
                   minimum_sampled_available_ram_bytes=min((row["available_ram_bytes"] for row in samples), default=None),
                   peak_sampled_owned_tree_rss_bytes=max((row["owned_tree_rss_bytes"] for row in samples), default=None),
                   qualification_evidence=False, final_test_used=False, production_promoted=False)
    write(BASE / "execution_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if status == "COMPLETED_PRIVATE_TRAINED_UNQUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
