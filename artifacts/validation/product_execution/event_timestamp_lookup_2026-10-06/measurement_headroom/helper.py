"""One-shot guarded actual event-feature comparison; prepare first, root reviews.

No inference/models/video/HTTP or production writes. Four sequential fresh clean
children at 600/18000 frames. Outputs retain every dataclasses.asdict field.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / "run"
FROZEN = UNIT.parent / "src/events/event_detector.py"
CURRENT = ROOT / "src/events/event_detector.py"
FIRST = UNIT.parent / "measurement"
HISTORY = UNIT / "first_measurement_bindings.json"
MONITOR = ROOT / "scripts/audit/measure_candidate_storage.py"
CLEAN = ROOT / "artifacts/validation/product_installation/clean_env/Scripts/python.exe"
BASE = Path(r"C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe")
PREFLIGHT = 750_000_000
RSS_LIMIT = 1_000_000_000
AVAILABLE_FLOOR = 400_000_000
LOW_SAMPLES = 5
WALL_LIMIT = 180
SAMPLE_INTERVAL = 0.25
EXPECTED_FROZEN = "078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552"
EXPECTED_CURRENT = "ccc00b450aa144d59a5fff13ede8a675749dc62fdc500143d3c114af715eff63"
PROJECT_SOURCES = (
    "src/__init__.py", "src/events/event_detector.py",
    "src/events/activity_state.py", "src/events/provenance_auditor.py",
    "src/tracking/__init__.py", "src/tracking/temporal_ball_tracker.py",
    "src/utils/__init__.py", "src/utils/bbox_utils.py",
)
FORBIDDEN_IMPORTS = ("torch", "ultralytics", "tensorflow", "transformers", "onnxruntime")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def resolved(pathname):
    path = Path(pathname)
    return path if path.is_absolute() else ROOT / path


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    save(temporary, value)
    temporary.rename(path)


def load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def source_bindings():
    paths = [ROOT / value for value in PROJECT_SOURCES]
    paths += [FROZEN, UNIT.parent / "baseline_receipt.json", MONITOR, CLEAN, BASE,
              Path(__file__).resolve(), UNIT / "protocol.json",
              UNIT / "prepared_bindings.json", UNIT / "preparation.json", HISTORY]
    return {name(path): binding(path) for path in paths}


def verify(values):
    for pathname, expected in values.items():
        if binding(resolved(pathname)) != expected:
            raise ValueError("Source/runtime binding changed: " + pathname)



def verify_first_history(expected):
    import psutil
    verify(expected)
    actual_names = {name(path) for path in FIRST.rglob("*") if path.is_file()}
    if actual_names != set(expected):
        raise ValueError("First consumed measurement file inventory changed")
    report = load(FIRST / "run/report.json")
    if report["passed"] or report["status"] != "FAILED" or report["comparisons"]:
        raise ValueError("First partial measurement terminal classification changed")
    if not report["sourcesUnchanged"] or not report["laterCasesHaltedAfterFirstFailure"]:
        raise ValueError("First measurement source/failure retention changed")
    if len(report["runs"]) != 2:
        raise ValueError("First partial case count changed")
    old, declined = report["runs"]
    if (old["mode"], old["frames"], old["passed"]) != ("old", 600, True):
        raise ValueError("First old case classification changed")
    if not old["actualWorkloadWasSampled"] or not old["allOwnedIdentitiesRetiredAtTerminal"] or old["remainingOwnedIdentitiesAtTerminal"]:
        raise ValueError("First old actual sampling/retirement incomplete")
    for identity in old["ownedProcessIdentitiesAndPeaks"]:
        try:
            if psutil.Process(identity["pid"]).create_time() == identity["createTimeUnixSeconds"]:
                raise ValueError("First measurement exact owned identity is still live")
        except psutil.NoSuchProcess:
            pass
    if (declined["mode"], declined["frames"], declined["status"], declined["passed"]) != ("new", 600, "NOT_STARTED", False):
        raise ValueError("First declined case classification changed")
    if declined["samples"] != 0 or declined.get("childStarted", False) or declined["availableRamImmediatelyBeforeLaunchBytes"] != 592830464:
        raise ValueError("First declined case child/preflight evidence changed")
    if {path.name for path in (FIRST / "run/600_new").iterdir()} != {"resource_receipt.json"}:
        raise ValueError("First NOT_STARTED case has unexpected child/work output")
    return {"fileCount": len(expected), "unchanged": True,
            "old600ActualAndAllOwnedIdentitiesRetiredNow": True,
            "new600NotStartedNoChild": True, "firstPartialMeasurementQualified": False}

def module_at(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def imported_bindings():
    """Bind actual ordinary imported Python/extensions, including dependencies.

    No asset inventory: model inputs are irrelevant to this actual method.
    Repeated physical files are represented once with their module names.
    """
    paths, modules = {}, {}
    for module_name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            continue
        path = Path(path).resolve()
        key = name(path)
        if key not in paths:
            paths[key] = binding(path)
        modules[module_name] = key
    return paths, modules


def fixture(count, point_type, states, stream):
    """Finite increasing 60-Hz synthetic clock and mixed explicit states/gaps."""
    trajectory, positions = [], []
    stream.write("[")
    state_cycle = (states.DETECTED, states.TRACKED, states.PREDICTED,
                   states.INTERPOLATED, states.OCCLUDED)
    for index in range(count):
        missing = index % 53 in (0, 1, 2)
        partial = not missing and index % 79 == 0
        state = states.MISSING if missing else state_cycle[index % len(state_cycle)]
        x = None if missing or partial else 550.0 + 230.0 * math.sin(index / 27.0)
        y = None if missing else 540.0 + 140.0 * math.sin(index / 19.0)
        confidence = None if index % 97 == 0 else (0.0 if index % 41 == 0 else 0.45 + (index % 50) / 100.0)
        point = point_type(frame_index=index, timestamp_seconds=index / 60.0,
                           x_px=x, y_px=y, confidence=confidence, state=state,
                           source="synthetic_features_only")
        position = (x, y) if x is not None and y is not None else None
        trajectory.append(point)
        positions.append(position)
        record = asdict(point)
        record["state"] = state.value
        if index:
            stream.write(", ")
        json.dump({"point": record, "featurePosition": position}, stream, allow_nan=True)
    stream.write("]")
    return trajectory, positions


def child(args):
    import psutil
    began = time.perf_counter()
    directory = Path(args.directory)
    report = {"startedAt": stamp(), "passed": False, "mode": args.mode,
              "frames": args.frames, "modelsImported": [], "inferenceRun": False,
              "instrumentation": "No method patch, query counter, profiler, model stub or lookup replacement; one perf_counter span around one actual _features call."}
    try:
        expected = load(RUN / "launch_source_bindings.json")
        verify(expected)
        verify_first_history(load(HISTORY))
        process = psutil.Process()
        identity = {"pid": process.pid, "createTimeUnixSeconds": process.create_time()}
        report["actualWorkloadIdentity"] = identity
        atomic_save(directory / "actual_identity.json", identity)
        deadline = time.perf_counter() + 10
        while not (directory / "actual_identity_sampled.json").exists():
            if time.perf_counter() > deadline:
                raise TimeoutError("Actual interpreter was not sampled/acknowledged within ten seconds")
            time.sleep(0.01)
        ack = load(directory / "actual_identity_sampled.json")
        if ack["identity"] != identity or ack["sampledByExactIdentity"] is not True:
            raise ValueError("Actual interpreter acknowledgment mismatch")
        report["parentActualIdentitySampledBeforeHeavyImports"] = True
        import_started = time.perf_counter()
        sys.path.insert(0, str(ROOT))
        selected = FROZEN if args.mode == "old" else CURRENT
        implementation = module_at("_measured_event", selected)
        from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
        report["ordinaryImportsSeconds"] = time.perf_counter() - import_started
        prohibited = [item for item in sys.modules if any(item == prefix or item.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS)]
        if prohibited:
            report["modelsImported"] = prohibited
            raise ValueError("Unexpected model/framework imports")
        actual_sources, imported_modules = imported_bindings()
        report["actualImportedSourceBindingsBefore"] = actual_sources
        report["actualImportedModules"] = imported_modules
        report["selectedImplementation"] = {"path": name(selected), **binding(selected)}
        report["actualInterpreter"] = {"sysExecutable": sys.executable, "osProcessExe": process.exe(),
                                       "osProcessExeBinding": binding(process.exe())}
        save(directory / "imported_sources.json", {"bindings": actual_sources, "modules": imported_modules,
                                                   "selectedImplementation": report["selectedImplementation"],
                                                   "actualInterpreter": report["actualInterpreter"]})
        detector = implementation.TennisEventDetector()
        input_started = time.perf_counter()
        with (directory / "fixture.json").open("x", encoding="utf-8", newline="\n") as stream:
            trajectory, positions = fixture(args.frames, TemporalBallPoint, BallState, stream)
        report["inputGenerationAndSerializationSeconds"] = time.perf_counter() - input_started
        save(directory / "settings.json", asdict(detector.settings))
        features_started = time.perf_counter()
        features = detector._features(trajectory, (1920, 1080), positions)
        report["actualFeaturesSeconds"] = time.perf_counter() - features_started
        if len(features) != args.frames:
            raise ValueError("Feature count mismatch")
        serialization_started = time.perf_counter()
        rows = 0
        with (directory / "features.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("[")
            for index, feature in enumerate(features):
                if feature.frame != index or feature.timestamp != index / 60.0:
                    raise ValueError("Actual feature frame/clock mismatch")
                if index:
                    stream.write(", ")
                json.dump(asdict(feature), stream, allow_nan=True)
                rows += 1
            stream.write("]")
        report["completeFeatureSerializationSeconds"] = time.perf_counter() - serialization_started
        report["serializationRows"] = rows
        verification_started = time.perf_counter()
        report["outputBindings"] = {file: binding(directory / file) for file in ("fixture.json", "settings.json", "features.json")}
        report["outputVerificationHashSeconds"] = time.perf_counter() - verification_started
        verify(expected)
        verify(actual_sources)
        report["allActualImportedSourcesUnchanged"] = True
        report["launchSourceBindingsUnchanged"] = True
        report["passed"] = True
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["totalChildElapsedSeconds"] = time.perf_counter() - began
        report["actualProcessMemoryBytes"] = psutil.Process().memory_info()._asdict()
        report["windowsPeakWorkingSetBytes"] = report["actualProcessMemoryBytes"].get("peak_wset")
        report["windowsPeakPagefileBytes"] = report["actualProcessMemoryBytes"].get("peak_pagefile")
        report["completedAt"] = stamp()
        save(directory / "child_receipt.json", report)
    return 0 if report["passed"] else 1


def monitor_case(mode, count, expected, monitor):
    import psutil
    verify(expected)
    verify_first_history(load(HISTORY))
    directory = RUN / f"{count}_{mode}"
    directory.mkdir(exist_ok=False)
    report = {"startedAt": stamp(), "mode": mode, "frames": count, "passed": False,
              "samples": 0, "inferenceChildStarted": False, "qualificationEvidence": False,
              "peakSampledOwnedTreeRssBytes": 0, "peakSampledOwnedTreePrivateBytes": 0,
              "minimumSampledAvailableRamBytes": None}
    process = owned = None
    began = time.perf_counter()
    try:
        available = psutil.virtual_memory().available
        report["availableRamImmediatelyBeforeLaunchBytes"] = available
        if available < PREFLIGHT:
            report["status"] = "NOT_STARTED"
            raise RuntimeError("Available RAM below unchanged 750 MB preflight; no launch")
        argv = [str(CLEAN), "-B", str(Path(__file__).resolve()), "--child", "--mode", mode,
                "--frames", str(count), "--directory", str(directory)]
        with (directory / "stdout.txt").open("x", encoding="utf-8") as log, \
                (directory / "resource_samples.jsonl").open("x", encoding="utf-8", newline="\n") as samples:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report["childStarted"] = True
            report["argv"] = argv
            owned = monitor.OwnedProcessTree(process)
            low = 0
            acknowledged = False
            while process.poll() is None:
                elapsed = time.perf_counter() - began
                available = psutil.virtual_memory().available
                rows = owned.sample()
                if not acknowledged and (directory / "actual_identity.json").exists():
                    identity = load(directory / "actual_identity.json")
                    if owned.observed_workload(identity):
                        atomic_save(directory / "actual_identity_sampled.json", {
                            "at": stamp(), "identity": identity, "sampledByExactIdentity": True})
                        acknowledged = True
                        report["actualWorkloadIdentity"] = identity
                        report["sampleBeforeAckCodeOrder"] = "owned.sample -> observed_workload exact identity -> atomic ack -> log-row timestamp"
                rss = sum(row["memoryBytes"]["rss"] for row in rows)
                private = [row["memoryBytes"].get("private") for row in rows]
                item = {"at": stamp(), "elapsedSeconds": elapsed, "processes": rows,
                        "availableRamBytes": available, "ownedTreeRssBytes": rss,
                        "ownedTreePrivateBytes": sum(value for value in private if value is not None),
                        "allPrivateValuesAvailable": all(value is not None for value in private)}
                samples.write(json.dumps(item, allow_nan=False) + "\n")
                samples.flush()
                report["samples"] += 1
                report["peakSampledOwnedTreeRssBytes"] = max(report["peakSampledOwnedTreeRssBytes"], rss)
                report["peakSampledOwnedTreePrivateBytes"] = max(report["peakSampledOwnedTreePrivateBytes"], item["ownedTreePrivateBytes"])
                previous = report["minimumSampledAvailableRamBytes"]
                report["minimumSampledAvailableRamBytes"] = available if previous is None else min(previous, available)
                low = low + 1 if available < AVAILABLE_FLOOR else 0
                reason = ("owned_tree_rss_above_1000000000" if rss > RSS_LIMIT else
                          "available_ram_below_400000000_for_5_samples" if low >= LOW_SAMPLES else
                          "child_wall_above_180_seconds" if elapsed > WALL_LIMIT else None)
                if reason:
                    report["safeguardStop"] = reason
                    raise RuntimeError(reason)
                time.sleep(SAMPLE_INTERVAL)
            report["exitCode"] = process.returncode
        remaining = owned.wait_for_retirement(timeout=2)
        report["remainingOwnedIdentitiesAfterNormalExit"] = remaining
        if remaining:
            raise RuntimeError("Owned identities remain after wrapper exit")
        receipt = load(directory / "child_receipt.json")
        report["childReceiptBinding"] = binding(directory / "child_receipt.json")
        if process.returncode != 0 or not receipt["passed"]:
            raise RuntimeError("Actual feature child failed; preserve output and halt later cases")
        if not owned.observed_workload(receipt["actualWorkloadIdentity"]):
            raise RuntimeError("Actual interpreter was never sampled by exact identity")
        verify(expected)
        verify(receipt["actualImportedSourceBindingsBefore"])
        for file, value in receipt["outputBindings"].items():
            if binding(directory / file) != value:
                raise ValueError("Child output changed: " + file)
        report["actualWorkloadWasSampled"] = True
        report["actualFeaturesSeconds"] = receipt["actualFeaturesSeconds"]
        report["completeFeatureSerializationSeconds"] = receipt["completeFeatureSerializationSeconds"]
        report["totalChildElapsedSeconds"] = receipt["totalChildElapsedSeconds"]
        report["actualOsPeakWorkingSetBytes"] = receipt["windowsPeakWorkingSetBytes"]
        report["actualOsPeakPagefileBytes"] = receipt["windowsPeakPagefileBytes"]
        report["passed"] = True
        report["status"] = "COMPLETED"
    except BaseException as error:
        report["status"] = report.get("status", "FAILED")
        report["error"] = type(error).__name__ + ": " + str(error)
        if process is not None:
            try:
                report["terminationActions"] = monitor.stop_owned(process, owned)
            except BaseException as cleanup:
                report["cleanupError"] = type(cleanup).__name__ + ": " + str(cleanup)
    finally:
        if (directory / "imported_sources.json").exists():
            try:
                imported = load(directory / "imported_sources.json")
                verify(imported["bindings"])
                report["actualImportedSourceBindingsUnchangedAtTerminal"] = True
                report["retainedImportedSourceRecordBinding"] = binding(directory / "imported_sources.json")
            except BaseException as source_error:
                report["actualImportedSourceVerificationError"] = repr(source_error)
                report["passed"] = False
        if owned is not None:
            report["ownedProcessIdentitiesAndPeaks"] = owned.identities()
            try:
                remaining = owned.wait_for_retirement(timeout=2)
                report["remainingOwnedIdentitiesAtTerminal"] = remaining
                report["allOwnedIdentitiesRetiredAtTerminal"] = not remaining
                if remaining:
                    report["passed"] = False
            except BaseException as cleanup:
                report["retirementCheckError"] = repr(cleanup)
                report["passed"] = False
        report["monitoredChildWallSeconds"] = time.perf_counter() - began
        report["completedAt"] = stamp()
        save(directory / "resource_receipt.json", report)
    return report


def exact_file(left, right):
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a, b = first.read(65536), second.read(65536)
            if a != b:
                return False
            if not a:
                return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("old", "new"))
    parser.add_argument("--frames", type=int, choices=(600, 18000))
    parser.add_argument("--directory")
    args = parser.parse_args()
    if args.child:
        if args.run or not args.mode or not args.frames or not args.directory:
            parser.error("Child requires declared mode/count/directory")
        directory = Path(args.directory).resolve()
        if directory != RUN / f"{args.frames}_{args.mode}" or not (RUN / "launch_marker.json").is_file():
            parser.error("Child directory/launch authorization mismatch")
        return child(args)
    if not args.run:
        parser.error("Preparation only; root must explicitly authorize --run")
    RUN.mkdir(exist_ok=False)
    save(RUN / "launch_marker.json", {"at": stamp(), "exclusive": True, "scope": "One-shot synthetic actual feature computation; no models or qualifications."})
    report = {"startedAt": stamp(), "passed": False, "runs": [], "comparisons": [],
              "inferenceRun": False, "modelsImported": [], "qualificationEvidence": False,
              "scope": "Uninstrumented actual _features on finite synthetic clocks; no complete event, video, pipeline, match-capacity or accuracy qualification."}
    expected = None
    ordinary_source_union = {}
    try:
        prepared = load(UNIT / "prepared_bindings.json")
        verify(prepared)
        report["firstMeasurementHistoryBefore"] = verify_first_history(load(HISTORY))
        expected = source_bindings()
        if expected[name(FROZEN)]["sha256"] != EXPECTED_FROZEN or expected[name(CURRENT)]["sha256"] != EXPECTED_CURRENT:
            raise ValueError("Frozen/current event source identity mismatch; do not rebind or retry")
        save(RUN / "launch_source_bindings.json", expected)
        report["sourceBindingsBefore"] = expected
        monitor = module_at("_owned_event_measurement_monitor", MONITOR)
        for count in (600, 18000):
            for mode in ("old", "new"):
                run = monitor_case(mode, count, expected, monitor)
                report["runs"].append(run)
                imported_record = RUN / f"{count}_{mode}" / "imported_sources.json"
                if imported_record.exists():
                    for pathname, value in load(imported_record)["bindings"].items():
                        if pathname in ordinary_source_union and ordinary_source_union[pathname] != value:
                            raise ValueError("Ordinary imported source changed between cases: " + pathname)
                        ordinary_source_union[pathname] = value
                if not run["passed"]:
                    raise RuntimeError("First failed/declined case; later cases are halted")
            comparisons = {}
            for file in ("fixture.json", "settings.json", "features.json"):
                old, new = RUN / f"{count}_old" / file, RUN / f"{count}_new" / file
                comparisons[file] = {"exactBytesEqual": exact_file(old, new),
                                     "old": binding(old), "new": binding(new)}
            comparison = {"frames": count, "outputs": comparisons,
                          "featureFieldExclusions": [], "passed": all(v["exactBytesEqual"] for v in comparisons.values())}
            report["comparisons"].append(comparison)
            save(RUN / f"comparison_{count}.json", comparison)
            if not comparison["passed"]:
                raise ValueError("Full feature/fixture/settings byte mismatch; halt later cases")
        report["passed"] = True
        report["status"] = "COMPLETED"
    except BaseException as error:
        report["status"] = "FAILED" if report["runs"] else "NOT_STARTED"
        report["error"] = type(error).__name__ + ": " + str(error)
        report["laterCasesHaltedAfterFirstFailure"] = True
    finally:
        try:
            report["firstMeasurementHistoryAfter"] = verify_first_history(load(HISTORY))
            report["firstMeasurementFilesUnchanged"] = True
        except BaseException as history_error:
            report["firstMeasurementHistoryError"] = repr(history_error)
            report["passed"] = False
        if ordinary_source_union:
            report["actualImportedSourceBindingsUnion"] = ordinary_source_union
            try:
                verify(ordinary_source_union)
                report["actualImportedSourcesUnchangedAcrossMeasuredCases"] = True
            except BaseException as error:
                report["actualImportedSourceVerificationError"] = repr(error)
                report["passed"] = False
        if expected is not None:
            try:
                verify(expected)
                report["sourceBindingsAfter"] = source_bindings()
                report["sourcesUnchanged"] = report["sourceBindingsAfter"] == expected
            except BaseException as error:
                report["sourceVerificationError"] = repr(error)
                report["passed"] = False
        report["completedAt"] = stamp()
        report["timingScope"] = "Actual feature span excludes imports/fixture/serialization/hash verification; child and monitor wall include additional work/handshake. No repeated timing trials or performance qualification threshold."
        report["memoryScope"] = "Actual interpreter Windows OS peak working set/pagefile are separate from sampled aggregate owned RSS/private; shared pages can be counted twice and brief sampled peaks missed. Host RAM guard includes unrelated processes."
        save(RUN / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("passed", "status", "error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
