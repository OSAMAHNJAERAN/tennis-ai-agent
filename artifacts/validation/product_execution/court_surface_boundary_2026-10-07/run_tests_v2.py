"""One exclusive, identity-monitored synthetic candidate test run."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
OUTPUT = UNIT / "tests_v2"
BASE = Path(r"C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe")
EXPECTED = {
    "candidate.py": "0e0a6420432ca3866326d0fe6c5fa69494110c2719e6faa6b7a658931ebccfd0",
    "test_candidate_v2.py": "0d8fd1760e0bf558454d33880a83eeddee5c77d50f7f20fcc1ed71f0a1bf98c9",
    "protocol.json": "6b6609c7743a5a5434f2a56951fdaf4f1b29febeb32b7d69782bd5ef2fc93421",
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": checksum}


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import psutil
    sys.path.insert(0, str(ROOT))
    from scripts.audit.measure_candidate_storage import OwnedProcessTree, stop_owned
    if OUTPUT.exists():
        raise FileExistsError("Preserve this consumed synthetic test attempt")
    if Path(sys.executable).resolve() != BASE.resolve():
        raise RuntimeError("Require ordinary BasePython usersite; no isolated -I mode")
    files = [UNIT / name for name in EXPECTED] + [
        UNIT / "run_tests_v2.py", ROOT / "pytest.ini", BASE,
        ROOT / "scripts/evaluate/probe_court_line_segments.py",
        ROOT / "scripts/evaluate/probe_court_surface_support.py",
        ROOT / "scripts/audit/measure_candidate_storage.py",
    ]
    before = {str(path): binding(path) for path in files}
    for name, checksum in EXPECTED.items():
        if before[str(UNIT / name)]["sha256"] != checksum:
            raise RuntimeError("Candidate, tests or protocol changed before launch")
    OUTPUT.mkdir()
    snapshot = OUTPUT / "sources"
    for path in files:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("interpreter") / path.name
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    command = [str(BASE), "-B", "-m", "pytest", str(UNIT / "test_candidate_v2.py"), "-q", "--tb=short",
               "--junitxml=" + str(OUTPUT / "junit.xml"), "--basetemp=" + str(OUTPUT / "tmp"),
               "-o", "junit_family=legacy"]
    guards = {"wallSeconds": 300, "ownedTreeRssBytes": 1_000_000_000,
              "minimumAvailableBeforeBytes": 750_000_000, "availableRamBytes": 400_000_000,
              "lowRamSamples": 5, "sampleSeconds": .25}
    record = {"status": "PREPARED", "passed": False, "sourceBefore": before,
              "command": command, "cwd": str(ROOT), "guards": guards, "startedAtUtc": stamp(),
              "scope": "Synthetic image and mask tests only; no source-video/model/production execution",
              "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
              "baseInterpreter": {"path": str(BASE), **binding(BASE)}}
    save(OUTPUT / "launch.json", record)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    process = owned = None
    began, low_ram = time.monotonic(), 0
    try:
        available = psutil.virtual_memory().available
        record["availableRamBeforeBytes"] = available
        if available < guards["minimumAvailableBeforeBytes"]:
            record["status"] = "NOT_STARTED_RESOURCE_GUARD"
            raise RuntimeError("Available RAM below frozen 750000000-byte preflight minimum")
        with (OUTPUT / "stdout.log").open("xb") as stdout, (OUTPUT / "stderr.log").open("xb") as stderr, (OUTPUT / "samples.jsonl").open("x", encoding="utf-8", newline="\n") as samples:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = OwnedProcessTree(process)
            record["launchedIdentity"] = {"pid": owned.root_identity[0], "createTimeUnixSeconds": owned.root_identity[1]}
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row["memoryBytes"]["rss"] for row in rows)
                sample = {"atUtc": stamp(), "elapsedSeconds": time.monotonic()-began,
                          "ownedProcesses": rows, "ownedTreeRssBytes": rss, "availableRamBytes": available}
                samples.write(json.dumps(sample, allow_nan=False) + "\n")
                samples.flush()
                low_ram = low_ram + 1 if available < guards["availableRamBytes"] else 0
                if rss > guards["ownedTreeRssBytes"] or low_ram >= guards["lowRamSamples"] or sample["elapsedSeconds"] > guards["wallSeconds"]:
                    raise RuntimeError("Predeclared synthetic test resource guard exceeded")
                time.sleep(guards["sampleSeconds"])
            record["exitCode"] = process.wait()
            if owned.wait_for_retirement():
                raise RuntimeError("Owned synthetic pytest descendants remain")
        suites = list(ET.parse(OUTPUT / "junit.xml").getroot().iter("testsuite"))
        record["suites"] = [dict(suite.attrib) for suite in suites]
        counts = {key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        counts["passed"] = counts["tests"]-counts["failures"]-counts["errors"]-counts["skipped"]
        record["counts"] = counts
        record["failureCases"] = [
            {"name": case.attrib["name"], "message": child.attrib.get("message", ""), "details": child.text}
            for case in ET.parse(OUTPUT / "junit.xml").getroot().iter("testcase")
            for child in case if child.tag in ("failure", "error")]
        sampled = any(row["sampleCount"] > 0 for row in owned.identities())
        record["ownedWorkloadSampled"] = sampled
        record["status"] = "PASSED" if record["exitCode"] == 0 and counts["tests"] > 0 and not counts["failures"] and not counts["errors"] and sampled else "TEST_FAILED"
    except BaseException as error:
        if record["status"] != "NOT_STARTED_RESOURCE_GUARD":
            record["status"] = "RUNNER_FAILED"
        record["error"] = repr(error)
        record["traceback"] = traceback.format_exc()
    finally:
        if process is not None and (process.poll() is None or (owned is not None and owned.live_identities())):
            try:
                record["stopActions"] = stop_owned(process, owned)
            except BaseException as error:
                record["cleanupError"] = repr(error)
        record["ownedIdentities"] = [] if owned is None else owned.identities()
        record["allOwnedIdentitiesRetired"] = (process is None or process.poll() is not None) and (owned is None or not owned.live_identities())
        record["sourceAfter"] = {str(path): binding(path) for path in files}
        record["sourcesUnchanged"] = before == record["sourceAfter"]
        record["elapsedSeconds"] = time.monotonic()-began
        record["completedAtUtc"] = stamp()
        record["passed"] = record["status"] == "PASSED" and record["sourcesUnchanged"] and record["allOwnedIdentitiesRetired"]
        record["outputs"] = {path.name: binding(path) for path in OUTPUT.iterdir() if path.is_file()}
        save(OUTPUT / "receipt.json", record)
    print(json.dumps({key: record.get(key) for key in ("status", "passed", "counts", "failureCases", "sourcesUnchanged", "allOwnedIdentitiesRetired", "elapsedSeconds")}, indent=2))
    return 0 if record["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
