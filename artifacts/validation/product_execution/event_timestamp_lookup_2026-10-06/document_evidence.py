"""Evidence-gated, one-shot event-lookup documentation publication.

PREPARED ONLY. The parent task must explicitly dispatch --apply. This helper
does not run tests, benchmarks, inference, imports of project modules, HTTP,
process discovery, server actions, or live SQLite queries. It only reads saved
evidence and atomically replaces the three reviewed documentation files.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RECEIPT = UNIT / "documentation_receipt.json"
STAGING = UNIT / "documentation_staging"
FROZEN = UNIT / "src/events/event_detector.py"
CURRENT = ROOT / "src/events/event_detector.py"
EXPECTED_FROZEN = "078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552"
EXPECTED_CURRENT = "ccc00b450aa144d59a5fff13ede8a675749dc62fdc500143d3c114af715eff63"
TRUSTED_RECEIPTS = {
    "baseline_receipt.json": "0e00b23d01676564b99427c5b4b204e9e31833402946951438b2151abeeb68dd",
    "red_receipt.json": "da48a3913fdc23512187e9c170df5cbdfdfe754a45f321cbeaf08d1b0d9af3a6",
    "green_run_2026-10-06/green_receipt.json": "3dbcd265a3c507b3cb150423fe3a62e8da36dd6d150d2fe6282e8b5a9fe2fb0d",
    "preserved_documents/receipt.json": "e223fc619037ac9fa12a381d5ecd8771a082a51a65fea70364549db1e788c87d",
    "backend/receipt.json": "0256606f10c25f78dea2216ac6bb7f11ef21e87fb49985955463c2026430f84f",
}
DOCUMENTS = (
    "docs/plans/TENNIS_EXECUTION_PROGRESS.md",
    "docs/deployment/PRODUCT_INSTALLATION.md",
    "docs/deployment/PRODUCT_RESTORE_MANIFEST.json",
)
GREEN = UNIT / "green_run_2026-10-06"
FIRST_MEASUREMENT = UNIT / "measurement"
MEASUREMENT = UNIT / "measurement_headroom"
MEASURE_RUN = MEASUREMENT / "run"
BACKEND = UNIT / "backend_final"
PROVENANCE = UNIT / "saved_diagnostic_replay_2026-10-06"
REPLAY_RECEIPT = PROVENANCE / "replay_receipt_v3.json"
REPLAY_RUN = PROVENANCE / "run_v3"
DECLINES = (
    "artifacts/validation/product_execution/result_storage_latency_worker_smoke_2026-10-06/resource_receipt.json",
    "artifacts/validation/product_execution/result_storage_latency_headroom_worker_smoke_2026-10-06/resource_receipt.json",
)
TRACKED = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def resolved(pathname):
    path = Path(pathname)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": sha}


def track(path, expected=None):
    path = Path(path).resolve()
    actual = binding(path)
    if isinstance(expected, str):
        require(actual["sha256"] == expected, "SHA-256 mismatch: " + name(path))
    elif expected is not None:
        require(actual == expected, "File binding mismatch: " + name(path))
    key = name(path)
    require(key not in TRACKED or TRACKED[key] == actual, "Evidence changed while reading: " + key)
    TRACKED[key] = actual
    return actual


def load(path, expected=None):
    track(path, expected)
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def verify_map(values):
    require(isinstance(values, dict) and values, "Missing nonempty source binding map")
    for pathname, expected in values.items():
        track(resolved(pathname), expected)


def exact(left, right):
    track(left)
    track(right)
    with Path(left).open("rb") as first, Path(right).open("rb") as second:
        while True:
            a, b = first.read(65536), second.read(65536)
            require(a == b, "Exact byte mismatch: " + name(left) + " / " + name(right))
            if not a:
                break


def xml_counts(path):
    track(path)
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    require(len(suites) == 1, "Require one complete JUnit suite: " + name(path))
    suite = suites[0]
    counts = {key: int(suite.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")}
    return counts, float(suite.get("time")), suite


def finite_number(value, label, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0), label)
    return value


def source_gate():
    track(FROZEN, EXPECTED_FROZEN)
    track(CURRENT, EXPECTED_CURRENT)
    track(Path(__file__))
    baseline = load(UNIT / "baseline_receipt.json", TRUSTED_RECEIPTS["baseline_receipt.json"])
    require(baseline["sha256"] == EXPECTED_FROZEN and baseline["inferenceRun"] is False,
            "Frozen baseline receipt does not identify the reviewed source")
    old_tree = ast.parse(FROZEN.read_text(encoding="utf-8"))
    new_tree = ast.parse(CURRENT.read_text(encoding="utf-8"))
    def method(tree, requested):
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "TennisEventDetector"]
        require(len(classes) == 1, "Event detector class missing or duplicated")
        methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
                   and node.name == requested]
        require(len(methods) == 1, "Event detector method missing or duplicated: " + requested)
        return ast.dump(methods[0], include_attributes=False)
    require(method(old_tree, "_nearest_index") == method(new_tree, "_nearest_index"),
            "General _nearest_index implementation changed")
    method(new_tree, "_nearest_ordered_index")


def preserved_gate():
    receipt = load(UNIT / "preserved_documents/receipt.json",
                   TRUSTED_RECEIPTS["preserved_documents/receipt.json"])
    require(receipt["beforeEventSourceChanges"] is True and receipt["qualificationEvidence"] is False,
            "Documents were not preserved before the event source revision")
    before = {}
    for pathname, expected in receipt["bindings"].items():
        original = ROOT / pathname
        preserved = UNIT / "preserved_documents" / pathname
        track(original, expected)
        track(preserved, expected)
        exact(original, preserved)
        if pathname in DOCUMENTS:
            before[pathname] = original.read_bytes()
    require(set(before) == set(DOCUMENTS), "Missing preserved documentation baseline")
    manifest = json.loads(before[DOCUMENTS[2]].decode("utf-8"))
    require(manifest["productionQualified"] is False, "Production qualification changed")
    require("eventTimestampLookupEngineering" not in manifest["validation"], "Unit already documented")
    # Earlier evidence records remain byte-bound and are never rebased to this revision.
    for entry in manifest.get("validationEvidence", []):
        if entry.get("sha256"):
            track(ROOT / entry["path"], entry["sha256"])
    declines = []
    for pathname in DECLINES:
        record = load(ROOT / pathname)
        require(record["status"] == "NOT_STARTED" and record["passed"] is False
                and record["inferenceChildStarted"] is False and record["samples"] == 0
                and record["productionStatePublished"] is False,
                "Historical RAM decline was changed or implies actual work")
        declines.append({"path": pathname, "availableRamBytes": record["minimumSampledAvailableRamBytes"],
                         "status": record["status"], "sha256": TRACKED[pathname]["sha256"]})
    return before, manifest, declines


def focused_gate():
    red = load(UNIT / "red_receipt.json", TRUSTED_RECEIPTS["red_receipt.json"])
    green = load(GREEN / "green_receipt.json", TRUSTED_RECEIPTS["green_run_2026-10-06/green_receipt.json"])
    for receipt, folder, xml, output in ((red, UNIT, "red.xml", "red_output.txt"),
                                         (green, GREEN, "green.xml", "green_output.txt")):
        counts, _, _ = xml_counts(folder / xml)
        require(counts == {key: receipt[key] for key in counts}, "Focused count receipt/JUnit mismatch")
        track(folder / xml, receipt["xmlSha256"])
        track(folder / output, receipt["outputSha256"])
        require(receipt["inferenceRun"] is False and receipt["modelsLoaded"] is False,
                "Focused evidence implies inference or model use")
    require(red["status"] == "RED" and red["returncode"] != 0 and red["tests"] == 33
            and red["failures"] == 1 and red["errors"] == 0 and red["skipped"] == 0
            and red["productionSourceUnchanged"] is True, "Missing genuine preserved RED")
    require(green["status"] == "GREEN" and green["returncode"] == 0 and green["tests"] == 38
            and all(green[key] == 0 for key in ("failures", "errors", "skipped"))
            and green["boundSourcesUnchangedDuringRun"] is True,
            "Focused GREEN did not pass unchanged")
    require(red["bindings"]["sourceSha256"] == EXPECTED_FROZEN
            and green["bindings"]["sourceSha256"] == EXPECTED_CURRENT
            and green["bindings"]["baselineSha256"] == EXPECTED_FROZEN,
            "Focused evidence source binding mismatch")
    track(UNIT / "red_test_snapshot.py", green["bindings"]["redTestSnapshotSha256"])
    track(ROOT / "tests/test_event_timestamp_lookup.py", green["bindings"]["testSha256"])
    track(GREEN / "run_green.py", green["bindings"]["runnerSha256"])
    track(UNIT / "run_red.py", red["bindings"]["runnerSha256"])
    require(green["bindings"]["redReceiptSha256"] == track(UNIT / "red_receipt.json")["sha256"],
            "GREEN does not bind this RED receipt")
    measurements = load(GREEN / "nearest_timestamp_measurements.json")
    require(measurements == green["nearestTimestampMeasurements"], "Focused measurements differ from receipt")
    counts, _, suite = xml_counts(GREEN / "green.xml")
    properties = {item.get("name"): item.get("value") for item in suite.findall(".//property")}
    require(json.loads(properties["nearestTimestampMeasurements"]) == measurements,
            "Focused JUnit measurements differ from receipt")
    red_rows = {row["frames"]: row for row in red["nearestTimestampMeasurements"]["measurements"]}
    rows = measurements["measurements"]
    require([row["frames"] for row in rows] == [128, 256, 512], "Unexpected instrumented read fixtures")
    for row in rows:
        count = row["frames"]
        require(row["exactFrozenFeatureBytes"] is True and row["nearestQueries"] == 4 * count
                and row["timestampReads"] <= row["timestampReadBudget"]
                and red_rows[count]["timestampReads"] == 4 * count * count,
                "Instrumented lookup evidence does not establish the intended regression")
        for suffix in ("current", "frozen"):
            track(GREEN / f"features_{count}_{suffix}.json",
                  {"bytes": row["featureBytes"], "sha256": row["featureSha256"]})
        exact(GREEN / f"features_{count}_current.json", GREEN / f"features_{count}_frozen.json")
    analyze_pairs = sorted(GREEN.glob("analyze_*_current.json"))
    require(len(analyze_pairs) == 4, "Require four complete analyze compatibility fixtures")
    for current in analyze_pairs:
        exact(current, current.with_name(current.name.replace("_current.json", "_frozen.json")))
    return red, green, rows, red_rows


def backend_gate():
    failed = load(UNIT / "backend/receipt.json", TRUSTED_RECEIPTS["backend/receipt.json"])
    failure_counts, _, failed_suite = xml_counts(UNIT / "backend/junit.xml")
    require(failed["passed"] is False and failed["exitCode"] != 0
            and failure_counts == failed["counts"]
            and failure_counts == {"tests": 1316, "failures": 1, "errors": 0, "skipped": 0}
            and failed["sourcesUnchanged"] is True, "Preserved initial backend failure differs")
    failures = [case for case in failed_suite.findall("testcase") if case.find("failure") is not None]
    require(len(failures) == 1 and failures[0].get("classname") == "tests.test_phase6_4_recovery_artifacts"
            and failures[0].get("name") == "test_recovery_input_hashes_match_current_sources_and_exclude_pristine_media",
            "Initial backend failure was not the recorded historical provenance mismatch")
    for leaf, expected in failed["artifacts"].items():
        track(UNIT / "backend" / leaf, expected)
    failed_bindings = load(UNIT / "backend/source_bindings.json")
    require(failed_bindings["src/events/event_detector.py"] == EXPECTED_CURRENT,
            "Initial backend failure does not bind this event revision")
    successful = load(BACKEND / "receipt.json")
    counts, elapsed, _ = xml_counts(BACKEND / "junit.xml")
    require(successful["passed"] is True and successful["exitCode"] == 0
            and counts == successful["counts"] and counts["tests"] > 0
            and all(counts[key] == 0 for key in ("failures", "errors", "skipped"))
            and successful["sourcesUnchanged"] is True, "Final complete backend suite did not pass")
    require(elapsed == successful["testElapsedSeconds"], "Backend elapsed receipt/JUnit mismatch")
    finite_number(elapsed, "Backend elapsed must be finite", positive=True)
    for leaf, expected in successful["artifacts"].items():
        track(BACKEND / leaf, expected)
    source_bindings = load(BACKEND / "source_bindings.json")
    require(source_bindings["src/events/event_detector.py"] == EXPECTED_CURRENT,
            "Successful backend suite binds another event revision")
    maintained_files = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py"),
                        ROOT / "pytest.ini", UNIT / "verify_backend_final.py"]
    require(set(source_bindings) == {name(path) for path in maintained_files if path.is_file()},
            "Final suite does not bind the complete maintained source/test inventory")
    require(successful["command"][1:4] == ["-m", "pytest", "tests"],
            "Final backend command does not run the complete maintained tests directory")
    verify_map(source_bindings)
    return failed, failed_bindings, successful


def provenance_gate(failed_bindings, successful):
    # This schema is intentionally explicit: no broad acceptance of a passing
    # flag can conceal a rebased historical report or a missing replay artifact.
    preserved = load(PROVENANCE / "preserved_integrity_test.json")
    require(preserved["historicalReportSha256"] == "e3d607adb5cb3bd5b35f7d9077e8ad28f48f094002fc37d097a69fbc86ae1b0e"
            and preserved["historicalArtifactsEdited"] is False and preserved["replayRun"] is False,
            "Original test/report preservation binding differs")
    original = ROOT / preserved["preservedTest"]
    test_path = preserved["test"]
    require(track(original, preserved["preservedTestSha256"])["sha256"] == failed_bindings[test_path],
            "Original provenance test is not the one from the failed suite")
    require(preserved["historicalEventDetectorSha256"] == EXPECTED_FROZEN
            and preserved["currentEventDetectorSha256"] == EXPECTED_CURRENT,
            "Historical/current event identities are not separate")
    track(ROOT / preserved["failedBackendReceipt"], preserved["failedBackendReceiptSha256"])
    report = load(ROOT / preserved["historicalReport"], preserved["historicalReportSha256"])
    require(report["qualification_evidence"] is False
            and report["scientific_split"] == "CROSS_MATCH_DIAGNOSTIC_ALREADY_CONSUMED",
            "Historical diagnostic scope changed")
    historical_sources = report["input_provenance"]
    require(historical_sources["event_detector_implementation"]["sha256"] == EXPECTED_FROZEN,
            "Historical diagnostic report event hash changed")
    for label, record in historical_sources.items():
        entries = [item for video in record.values() for item in video.values()] if label == "preserved_video_inputs" else [record]
        for item in entries:
            measured_path = FROZEN if label == "event_detector_implementation" else ROOT / item["path"]
            track(measured_path, {"bytes": item["size_bytes"], "sha256": item["sha256"]})
    corrected_counts, _, corrected_suite = xml_counts(UNIT / "provenance_green.xml")
    require(corrected_counts == {"tests": 5, "failures": 0, "errors": 0, "skipped": 0},
            "Scoped historical artifact integrity checks did not pass")
    require(any(case.get("name") == "test_recovery_input_hashes_match_measured_sources_and_exclude_pristine_media"
                for case in corrected_suite.findall("testcase")), "Corrected historical test was not run")
    v2_history = load(PROVENANCE / "v2_history_bindings.json")
    require(v2_history["consumedRevision"] == 2 and v2_history["failure"] == "KeyError: 'pid'"
            and v2_history["modelOrReplayStarted"] is False
            and v2_history["everyKnownOwnedIdentityRetired"] is True
            and v2_history["priorInputsAndReportsWritten"] is False,
            "Failed replay-v2 classification differs")
    for leaf, expected in v2_history["files"].items():
        track(PROVENANCE / leaf, expected)
    actual_v2_files = {path.relative_to(PROVENANCE).as_posix() for path in PROVENANCE.iterdir()
                       if path.is_file() and ("_v2" in path.name or path.name == "prepared_revision_2.json")}
    require((PROVENANCE / "run_v2").is_dir(), "Failed v2 run directory disappeared")
    actual_v2_files |= {path.relative_to(PROVENANCE).as_posix() for path in (PROVENANCE / "run_v2").rglob("*") if path.is_file()}
    actual_v2_dirs = ["run_v2"] + sorted(path.relative_to(PROVENANCE).as_posix()
                                        for path in (PROVENANCE / "run_v2").rglob("*") if path.is_dir())
    require(actual_v2_files == set(v2_history["files"]) and actual_v2_dirs == v2_history["directories"],
            "Failed-v2 evidence inventory changed")
    failed_v2 = load(PROVENANCE / "replay_receipt_v2.json")
    require(failed_v2["status"] == "FAILED" and failed_v2["passed"] is False
            and failed_v2["error"] == "KeyError: 'pid'" and failed_v2["samples"] == 2
            and failed_v2["remainingExactOwnedIdentitiesAtTerminal"] == []
            and failed_v2["allExactOwnedIdentitiesRetiredAtTerminal"] is True
            and failed_v2["sourceBindingsUnchanged"] is True and failed_v2["historicalArtifactsUnchanged"] is True,
            "Failed v2 receipt was changed or treated as replay evidence")
    identity_probe = load(PROVENANCE / "monitor_identity_contract_receipt.json")
    require(identity_probe["passed"] is True and identity_probe["actualProcessLaunched"] is False
            and identity_probe["actualProcessSampled"] is False and identity_probe["subprocessCalls"] == [],
            "Monitor method-contract probe implies unobserved live lifecycle work")
    receipt = load(REPLAY_RECEIPT)
    protocol = load(PROVENANCE / "protocol_v3.json")
    preparation = load(PROVENANCE / "prepared_revision_3.json")
    require(receipt["passed"] is True and receipt["status"] == "COMPLETED"
            and receipt["inferenceRun"] is False and receipt["modelsLoaded"] is False
            and receipt["qualificationEvidence"] is False
            and receipt["sourceBindingsUnchanged"] is True
            and receipt["historicalArtifactsUnchanged"] is True
            and receipt["preparationUnchanged"] is True
            and receipt["consumedV2HistoryUnchanged"] is True,
            "Historical provenance saved replay did not complete unchanged")
    require(receipt["sourceBindingsAfter"] == protocol["sourceBindingsBefore"] == preparation["sourceBindings"]
            and receipt["historicalArtifactsAfter"] == protocol["historicalArtifactsBefore"],
            "Replay source/history binding maps differ")
    verify_map(receipt["sourceBindingsAfter"])
    verify_map(receipt["historicalArtifactsAfter"])
    track(PROVENANCE / "prepared_revision_3.json", protocol["preparationSha256"])
    track(PROVENANCE / "helper_v3.py", preparation["scriptSha256"])
    require(protocol["consumedV2HistoryBefore"] == receipt["consumedV2HistoryAfter"]
            == {"files": v2_history["files"], "directories": v2_history["directories"]},
            "Consumed v2 inventory continuity failed")
    require(receipt["returncode"] == 0 and receipt["actualWorkloadSampledByExactIdentity"] is True
            and receipt["allExactOwnedIdentitiesRetiredAtTerminal"] is True
            and receipt["allOwnedIdentitiesRetiredAfterNormalExit"] is True
            and receipt["remainingExactOwnedIdentitiesAtTerminal"] == []
            and receipt["remainingOwnedIdentitiesAfterNormalExit"] == []
            and not any(key in receipt for key in ("safeguardStop", "error", "cleanupError", "retirementVerificationError")),
            "Saved replay guard/process retirement failed")
    child = load(PROVENANCE / "child_receipt_v3.json")
    require(child["passed"] is True and child["inferenceRun"] is False and child["modelsLoaded"] is False
            and child["historicalReportsWritten"] is False and child["networkAttempts"] == []
            and child["forbiddenInferenceImports"] == [] and child["comparisons"] == receipt["comparisons"],
            "Saved replay child had network/model work or incomplete comparisons")
    require(receipt["ordinaryImportedSourcesUnchangedAfterCalculation"] is True
            and receipt["ordinaryImportedBindingsVerifiedBeforeParentPass"] is True
            and receipt["ordinaryImportedModuleSourcesUnchangedAtTerminal"] is True
            and receipt["baseInterpreterUnchangedAtTerminal"] is True
            and child["ordinaryImportedSourcesUnchangedAfterCalculation"] is True
            and child["samplingAcknowledgementMatchesExactIdentity"] is True
            and receipt["samplingAcknowledgementMatchesExactIdentity"] is True,
            "Replay interpreter/import/acknowledgement continuity failed")
    for suffix in ("Before", "After"):
        imported = child["ordinaryImportedModuleBindings" + suffix]
        require(imported == receipt["ordinaryImportedModuleBindings" + suffix],
                "Replay child/parent ordinary import map differs")
        verify_map(imported)
    before_imports = child["ordinaryImportedModuleBindingsBefore"]
    after_imports = child["ordinaryImportedModuleBindingsAfter"]
    require(all(after_imports.get(pathname) == value for pathname, value in before_imports.items()),
            "An actual pre-analysis imported library changed during replay")
    require(load(REPLAY_RUN / "ordinary_import_bindings_before.json") == before_imports,
            "Saved pre-analysis library map differs")
    interpreter = protocol["interpreterBinding"]
    require(interpreter == preparation["baseInterpreterBinding"]
            == receipt["baseInterpreterBindingAtTerminal"]
            == receipt["baseInterpreterBindingAfterChildRetirement"]
            == receipt["actualProcessExecutableBefore"] == receipt["actualProcessExecutableAfterCalculation"]
            == child["actualProcessExecutableBefore"] == child["actualProcessExecutableImmediatelyBeforeCalculation"]
            == child["actualProcessExecutableAfterCalculation"], "Actual replay interpreter changed")
    track(interpreter["path"], {"bytes": interpreter["bytes"], "sha256": interpreter["sha256"]})
    actual_identity = child["actualWorkloadIdentity"]
    require(actual_identity == receipt["actualWorkloadIdentity"]
            == load(PROVENANCE / "actual_child_identity_v3.json"), "Replay actual identity receipt differs")
    acknowledgement = load(PROVENANCE / "actual_child_sampled_v3.json")
    require(acknowledgement["identity"] == actual_identity and acknowledgement["sampledByExactIdentity"] is True,
            "Replay acknowledgement did not bind the actual interpreter")
    variants = {row["variant"] for row in report["ablations"] if row["variant"] != "A_INTEGRATED_PRE_RECOVERY"}
    videos = set(historical_sources["preserved_video_inputs"])
    pairs = receipt["comparisons"]
    require(len(variants) == 7 and videos == {"video_08", "video_09", "video_10"}
            and len(pairs) == 21
            and {(pair["variant"], pair["video"]) for pair in pairs} == {(variant, video) for variant in variants for video in videos},
            "Replay does not cover all seven original variants and three consumed diagnostic inputs")
    for pair in pairs:
        require(pair["exactAnalysisBytes"] is True and pair["currentSha256"] == pair["frozenSha256"],
                "Saved diagnostic full analyze bytes differ")
        if pair["variant"] == "H_FINAL_INTEGRATED":
            require(pair["historicalHFilteredEventsEqual"] is True, "Frozen replay differs from historical final event records")
        prefix = f"{pair['variant']}_{pair['video']}"
        for mode, key in (("current", "currentSha256"), ("frozen", "frozenSha256")):
            track(REPLAY_RUN / f"{prefix}_{mode}.json", {"bytes": pair["bytes"], "sha256": pair[key]})
        exact(REPLAY_RUN / f"{prefix}_current.json", REPLAY_RUN / f"{prefix}_frozen.json")
    track(REPLAY_RUN / "variant_settings.json")
    track(PROVENANCE / "resource_samples_v3.jsonl")
    track(PROVENANCE / "child_stdout_v3.txt")
    receipt["verifiedCompletePairCount"] = len(pairs)
    return receipt


def measurement_gate():
    protocol = load(MEASUREMENT / "protocol.json")
    prepared = load(MEASUREMENT / "prepared_bindings.json")
    verify_map(prepared)
    first_bindings = load(MEASUREMENT / "first_measurement_bindings.json")
    verify_map(first_bindings)
    require(set(first_bindings) == {name(path) for path in FIRST_MEASUREMENT.rglob("*") if path.is_file()},
            "Original partial measurement file inventory changed")
    first = load(FIRST_MEASUREMENT / "run/report.json")
    require(first["passed"] is False and first["status"] == "FAILED" and first["comparisons"] == []
            and first["sourcesUnchanged"] is True and first["laterCasesHaltedAfterFirstFailure"] is True
            and len(first["runs"]) == 2, "Original guarded partial measurement classification changed")
    first_old, declined = first["runs"]
    require((first_old["frames"], first_old["mode"], first_old["passed"]) == (600, "old", True)
            and first_old["actualWorkloadWasSampled"] is True
            and first_old["allOwnedIdentitiesRetiredAtTerminal"] is True
            and first_old["remainingOwnedIdentitiesAtTerminal"] == [], "Original old600 process proof differs")
    require((declined["frames"], declined["mode"], declined["status"], declined["passed"]) == (600, "new", "NOT_STARTED", False)
            and declined["samples"] == 0 and declined.get("childStarted", False) is False
            and declined["availableRamImmediatelyBeforeLaunchBytes"] == 592830464,
            "Original new600 did not decline before launching at the unchanged guard")
    require({path.name for path in (FIRST_MEASUREMENT / "run/600_new").iterdir()} == {"resource_receipt.json"},
            "Original declined case acquired child output")
    guards = {"preflightAvailableRamBytes": 750000000,
              "maximumSampledOwnedTreeRssBytes": 1000000000,
              "minimumAvailableRamBytes": 400000000,
              "lowRamConsecutiveSamplesToStop": 5,
              "maximumChildWallSeconds": 180, "sampleIntervalSeconds": 0.25}
    require(all(protocol[key] == value for key, value in guards.items()), "Measurement guards changed")
    require(protocol["baselineSha256"] == EXPECTED_FROZEN and protocol["currentSha256"] == EXPECTED_CURRENT
            and protocol["productionWrites"] is False and protocol["inferenceRun"] is False
            and protocol["qualificationEvidence"] is False, "Measurement protocol scope/source mismatch")
    report = load(MEASURE_RUN / "report.json")
    require(report["passed"] is True and report["status"] == "COMPLETED"
            and report["inferenceRun"] is False and report["modelsImported"] == []
            and report["qualificationEvidence"] is False and report["sourcesUnchanged"] is True
            and report["actualImportedSourcesUnchangedAcrossMeasuredCases"] is True,
            "Four-case uninstrumented feature measurement has not passed")
    history_before, history_after = report["firstMeasurementHistoryBefore"], report["firstMeasurementHistoryAfter"]
    require(history_before == history_after and report["firstMeasurementFilesUnchanged"] is True
            and history_before["fileCount"] == len(first_bindings) == 18
            and history_before["unchanged"] is True
            and history_before["old600ActualAndAllOwnedIdentitiesRetiredNow"] is True
            and history_before["new600NotStartedNoChild"] is True
            and history_before["firstPartialMeasurementQualified"] is False,
            "Fresh measurement did not preserve the entire failed first attempt")
    require(report["sourceBindingsBefore"] == report["sourceBindingsAfter"], "Measurement source maps changed")
    verify_map(report["sourceBindingsBefore"])
    verify_map(report["actualImportedSourceBindingsUnion"])
    require(load(MEASURE_RUN / "launch_source_bindings.json") == report["sourceBindingsBefore"],
            "Measurement launch map mismatch")
    require(report["sourceBindingsBefore"][name(CURRENT)]["sha256"] == EXPECTED_CURRENT
            and report["sourceBindingsBefore"][name(FROZEN)]["sha256"] == EXPECTED_FROZEN,
            "Measurement old/current source identities mismatch")
    require([(row["frames"], row["mode"]) for row in report["runs"]]
            == [(600, "old"), (600, "new"), (18000, "old"), (18000, "new")],
            "Measurement order/count differs or a case was omitted")
    summaries = {}
    for row in report["runs"]:
        count, mode = row["frames"], row["mode"]
        case = MEASURE_RUN / f"{count}_{mode}"
        resource = load(case / "resource_receipt.json")
        require(resource == row, "Terminal run/resource receipt mismatch")
        child = load(case / "child_receipt.json", resource["childReceiptBinding"])
        require(resource["passed"] is True and resource["status"] == "COMPLETED" and resource["exitCode"] == 0
                and resource["actualWorkloadWasSampled"] is True
                and resource["allOwnedIdentitiesRetiredAtTerminal"] is True
                and resource["remainingOwnedIdentitiesAtTerminal"] == []
                and resource["remainingOwnedIdentitiesAfterNormalExit"] == []
                and resource["actualImportedSourceBindingsUnchangedAtTerminal"] is True
                and not any(key in resource for key in ("safeguardStop", "error", "cleanupError", "retirementCheckError")),
                "Feature process guard/sampling/retirement evidence failed")
        require(child["passed"] is True and child["modelsImported"] == [] and child["inferenceRun"] is False
                and child["serializationRows"] == count
                and child["allActualImportedSourcesUnchanged"] is True
                and child["launchSourceBindingsUnchanged"] is True
                and child["parentActualIdentitySampledBeforeHeavyImports"] is True
                and child["actualWorkloadIdentity"] == resource["actualWorkloadIdentity"],
                "Actual uninstrumented feature child is not bound/complete")
        selected = FROZEN if mode == "old" else CURRENT
        require(child["selectedImplementation"] == {"path": name(selected), **track(selected)},
                "Feature child selected another implementation")
        verify_map(child["actualImportedSourceBindingsBefore"])
        imported = load(case / "imported_sources.json", resource["retainedImportedSourceRecordBinding"])
        require(imported["bindings"] == child["actualImportedSourceBindingsBefore"]
                and imported["modules"] == child["actualImportedModules"]
                and imported["selectedImplementation"] == child["selectedImplementation"]
                and imported["actualInterpreter"] == child["actualInterpreter"],
                "Retained actual imported-source inventory differs")
        require(child["actualImportedModules"]["_measured_event"] == name(selected),
                "Actual imported event module differs from selected implementation")
        require(set(child["outputBindings"]) == {"fixture.json", "settings.json", "features.json"},
                "Missing complete feature/input/settings outputs")
        for leaf, value in child["outputBindings"].items():
            track(case / leaf, value)
        identity = child["actualWorkloadIdentity"]
        require(load(case / "actual_identity.json") == identity, "Actual process identity publication mismatch")
        ack = load(case / "actual_identity_sampled.json")
        require(ack["identity"] == identity and ack["sampledByExactIdentity"] is True,
                "Actual process was not acknowledged by exact identity")
        identities = resource["ownedProcessIdentitiesAndPeaks"]
        require(any(record["pid"] == identity["pid"] and
                    record["createTimeUnixSeconds"] == identity["createTimeUnixSeconds"] for record in identities),
                "Actual workload identity missing from remembered owned processes")
        samples = 0
        observed = False
        low = 0
        with (case / "resource_samples.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                samples += 1
                require(sample["ownedTreeRssBytes"] <= guards["maximumSampledOwnedTreeRssBytes"],
                        "A completed case exceeded the declared owned RSS guard")
                low = low + 1 if sample["availableRamBytes"] < guards["minimumAvailableRamBytes"] else 0
                require(low < guards["lowRamConsecutiveSamplesToStop"], "A completed case exceeded low RAM guard")
                observed |= any(record["pid"] == identity["pid"] and
                                record["createTimeUnixSeconds"] == identity["createTimeUnixSeconds"]
                                for record in sample["processes"])
        track(case / "resource_samples.jsonl")
        track(case / "stdout.txt")
        require(samples == resource["samples"] and observed, "Missing sampled actual workload evidence")
        require(resource["availableRamImmediatelyBeforeLaunchBytes"] >= guards["preflightAvailableRamBytes"],
                "Feature case preflight did not meet the unchanged RAM guard")
        feature_s = finite_number(child["actualFeaturesSeconds"], "Invalid actual feature time", positive=True)
        elapsed = finite_number(child["totalChildElapsedSeconds"], "Invalid actual child elapsed", positive=True)
        monitored = finite_number(resource["monitoredChildWallSeconds"], "Invalid monitored wall", positive=True)
        peak = finite_number(child["windowsPeakWorkingSetBytes"], "Missing Windows actual interpreter OS peak", positive=True)
        require(resource["actualFeaturesSeconds"] == feature_s and resource["totalChildElapsedSeconds"] == elapsed
                and resource["actualOsPeakWorkingSetBytes"] == peak,
                "Actual child timing/OS peak differs from terminal resource receipt")
        summaries[(count, mode)] = {"featuresSeconds": feature_s, "childElapsedSeconds": elapsed,
                                    "monitoredWallSeconds": monitored, "osPeakWorkingSetBytes": peak,
                                    "sampledOwnedTreeRssBytes": resource["peakSampledOwnedTreeRssBytes"],
                                    "featureBytes": child["outputBindings"]["features.json"]["bytes"]}
    require([row["frames"] for row in report["comparisons"]] == [600, 18000], "Missing paired feature comparisons")
    for comparison in report["comparisons"]:
        count = comparison["frames"]
        require(load(MEASURE_RUN / f"comparison_{count}.json") == comparison, "Saved comparison receipt differs")
        require(comparison["passed"] is True and comparison["featureFieldExclusions"] == []
                and set(comparison["outputs"]) == {"fixture.json", "settings.json", "features.json"},
                "Feature comparison omits fields or files")
        for leaf, values in comparison["outputs"].items():
            require(values["exactBytesEqual"] is True and values["old"] == values["new"],
                    "Feature comparison byte/hash mismatch")
            track(MEASURE_RUN / f"{count}_old" / leaf, values["old"])
            track(MEASURE_RUN / f"{count}_new" / leaf, values["new"])
            exact(MEASURE_RUN / f"{count}_old" / leaf, MEASURE_RUN / f"{count}_new" / leaf)
    return report, summaries, guards


def runtime_gate():
    preflight = load(UNIT / "runtime_preflight.json")
    runtime = load(UNIT / "runtime_verification.json")
    refusals = load(UNIT / "runtime_preflight_failures.json")
    require(refusals["noRootStopOccurred"] is True and refusals["subsequentRuntimeState"] == "ABSENT"
            and refusals["inferenceRun"] is False and len(refusals["failures"]) == 2,
            "Retained read-only runtime refusal/externally absent state differs")
    for leaf, expected in refusals["bindings"].items():
        track(UNIT / leaf, expected)
    require(preflight["runtime"]["state"] == "ABSENT"
            and preflight["runtime"]["processes"] == [] and preflight["runtime"]["ownedDescendants"] == []
            and runtime["previousRuntimeIdentitiesRetired"] == [],
            "Current runtime restoration must retain its actual ABSENT preflight scope")
    require(runtime["passed"] is True and runtime["savedJobsAndAnalysisFilesUnchanged"] is True
            and runtime["productStateChanged"] is False
            and runtime["runtimeStoppedOrStartedByThisVerifier"] is False,
            "Runtime verification incomplete or mutated product state")
    require(runtime["health"]["status"] == "ok" and runtime["health"]["worker_available"] is True,
            "Runtime health did not confirm the restored owned worker")
    require(preflight["productStateChanged"] is False and preflight["acceptedWorkCancelled"] is False
            and preflight["runtimeStoppedOrStarted"] is False,
            "Runtime preflight changed state or cancelled accepted work")
    require(runtime["preflightSha256"] == track(UNIT / "runtime_preflight.json")["sha256"]
            and runtime["scriptSha256"] == track(UNIT / "verify_runtime.py")["sha256"],
            "Runtime verifier/preflight binding mismatch")
    require(runtime["sourceBindings"] == preflight["sourceBindings"], "Runtime sources differ after idle reload")
    verify_map(runtime["sourceBindings"]["files"])
    complete_runtime_files = {name(path) for path in ROOT.joinpath("src").rglob("*.py")}
    complete_runtime_files |= {name(path) for path in ROOT.joinpath("frontend/dist").rglob("*") if path.is_file()}
    require(complete_runtime_files.issubset(runtime["sourceBindings"]["files"]),
            "Runtime verifier omitted a current source/frontend binding")
    track(runtime["sourceBindings"]["baseInterpreter"]["path"],
          runtime["sourceBindings"]["baseInterpreter"]["sha256"])
    require(runtime["sourceBindings"]["files"]["src/events/event_detector.py"]["sha256"] == EXPECTED_CURRENT,
            "Reloaded runtime binds another event revision")
    before = runtime["snapshotBefore"]
    require(before == runtime["snapshotAfter"] == preflight["snapshot"], "Saved jobs/files changed")
    require(runtime["savedJobs"] == len(before["jobs"]) == 10
            and runtime["savedAnalysisFiles"] == len(before["savedAnalysisFiles"]) == 319,
            "Unexpected current saved product inventory")
    require(not any(row["status"] in ("UPLOADED", "PROCESSING") for row in before["jobs"]),
            "Accepted work was active during idle reload")
    verify_map(before["savedAnalysisFiles"])
    actual_paths = {name(path) for path in (ROOT / "outputs/product/analyses").rglob("*") if path.is_file()}
    require(actual_paths == set(before["savedAnalysisFiles"]), "Saved analysis file inventory changed after verification")
    require(runtime["catalogEntries"] == 3 and len(runtime["exports"]) == 12
            and len(runtime["frontend"]["assets"]) == 7
            and all(item["status"] == 200 for item in runtime["exports"] + runtime["frontend"]["assets"]),
            "Existing demo exports/frontend compatibility check incomplete")
    for item in runtime["frontend"]["assets"]:
        track(ROOT / "frontend/dist" / item["url"].lstrip("/"),
              {"bytes": item["bytes"], "sha256": item["sha256"]})
    index = runtime["frontend"]["index"]
    track(ROOT / "frontend/dist/index.html", {"bytes": index["bytes"], "sha256": index["sha256"]})
    require(index["cacheControl"] == "no-cache" and runtime["runtime"]["state"] == "OWNED_IDLE_PAIR",
            "Current served frontend/runtime state is not the verified idle pair")
    old = preflight["runtime"]["processes"] + preflight["runtime"]["ownedDescendants"]
    retired = runtime["previousRuntimeIdentitiesRetired"]
    require(len(retired) == len(old) and all(item["retired"] is True for item in retired)
            and {(item["pid"], item["createTimeUnixSeconds"]) for item in retired}
            == {(item["pid"], item["createTimeUnixSeconds"]) for item in old},
            "Exact previous runtime identities were not retired")
    runtime["verifiedPreflightState"] = preflight["runtime"]["state"]
    return runtime


def link(path):
    return "../../" + name(path)


def build_documents(before, manifest, red, green, rows, red_rows, failed, backend,
                    provenance, measured, summaries, runtime, declines, at):
    read_lines = "\n".join(
        f"| {row['frames']:,} | {red_rows[row['frames']]['timestampReads']:,} | {row['timestampReads']:,} |"
        for row in rows)
    timing_lines = "\n".join(
        f"| {count:,} | {summaries[(count, 'old')]['featuresSeconds']:.6f} | "
        f"{summaries[(count, 'new')]['featuresSeconds']:.6f} | "
        f"{summaries[(count, 'old')]['osPeakWorkingSetBytes']:,} | "
        f"{summaries[(count, 'new')]['osPeakWorkingSetBytes']:,} |"
        for count in (600, 18000))
    replay_count = provenance["verifiedCompletePairCount"]
    backend_text = (f"The final complete backend suite passed **{backend['counts']['tests']:,} tests in "
                    f"{backend['testElapsedSeconds']:.3f} seconds**, with zero failures, errors or skips and "
                    f"unchanged bound sources. The first full run remains retained: "
                    f"{failed['counts']['tests']:,} tests, one failure, zero errors/skips, "
                    f"{failed['testElapsedSeconds']:.3f} seconds. Its failed provenance assertion incorrectly "
                    "required the current event source to equal the historical diagnostic source hash. "
                    "The scoped test correction verifies that historical report against its exact retained "
                    "pre-change event source, preserves the original test/report, and adds complete "
                    f"current-versus-frozen replay checks for {replay_count} saved diagnostic outputs.")
    limitations = ("No actual worker/inference run was performed on this event revision. The two earlier "
                   f"result-storage latency worker attempts remain historical NOT_STARTED RAM declines "
                   f"({declines[0]['availableRamBytes']:,} and {declines[1]['availableRamBytes']:,} bytes "
                   "available against their unchanged 2,500,000,000-byte preflight). Earlier actual worker "
                   "smokes and resident/disk result measurements retain their original revisions and scope. "
                   "Representative authorized recordings, independent labels and physical speed/3D references "
                   "remain unavailable; scientific accuracy and full-match capacity remain incomplete. "
                   "Pose, racket, selected-player, trajectory and browser arrays still grow with video length "
                   "(O(N)); dense continuity windows, candidate processing and _departure costs remain.")
    progress = f"""\n\n## Event timestamp lookup software evidence — 6 October 2026\n\nThe current `src/events/event_detector.py` revision (`{EXPECTED_CURRENT}`) replaces repeated full-timeline nearest timestamp scans in `_features` with bisect on its once-checked finite, strictly increasing timestamp list. Equal floating-point distance plateaus retain the earliest index chosen by the original `min`; the general `_nearest_index` is unchanged, nonfinite NaN/infinity inputs retain the old fallback, and existing duplicate/nonincreasing-order rejection remains.\n\n[The genuine RED]({link(UNIT / 'red_receipt.json')}) had {red['tests']} tests and one failure. [The focused GREEN]({link(GREEN / 'green_receipt.json')}) passed {green['tests']} tests with no failures/errors/skips. Every serialized feature field and all four complete analyze outputs match the frozen source byte for byte, including candidate/event/trace/settings/coordinate fields and numeric fallback cases. The deterministic read counter measures lookup work; these counts are separate from uninstrumented elapsed time and OS memory peaks.\n\n| Synthetic frames | Prior timestamp reads | Current timestamp reads |\n|---:|---:|---:|\n{read_lines}\n\n{backend_text} [Successful backend receipt]({link(BACKEND / 'receipt.json')}), [preserved failed run]({link(UNIT / 'backend/receipt.json')}), [historical provenance and replay]({link(PROVENANCE / 'replay_receipt.json')}).\n\n[The guarded synthetic feature measurement]({link(MEASURE_RUN / 'report.json')}) ran one old/current pair at each of 600 and 18,000 frames in four sequential fresh clean-environment children. Each child called the actual unmodified `_features` once without lookup instrumentation, models or inference. Full fixture, settings and complete feature JSON bytes match with no excluded fields. Feature-compute spans exclude imports, fixture construction, output serialization and hashing; child/monitored wall times include additional work and are separately retained in the receipts.\n\n| Synthetic frames | Prior `_features` seconds | Current `_features` seconds | Prior actual OS peak working set bytes | Current actual OS peak working set bytes |\n|---:|---:|---:|---:|---:|\n{timing_lines}\n\nActual workload PID/create-time identities were sampled before heavy imports, all remembered owned identities retired, and source/import/output hashes remained bound. Unchanged guards were 750 MB launch preflight, 1 GB sampled owned-tree RSS, less than 400 MB available RAM for five consecutive 0.25-second samples, and 180 seconds per child. Actual interpreter Windows OS peak working set is separate from sampled aggregate owned RSS/private commit; sampling can miss spikes and shared pages can be counted twice. One synthetic pair per count establishes neither repeatable throughput nor a full-video/event-pipeline performance threshold.\n\n[Fresh idle-reload compatibility]({link(UNIT / 'runtime_verification.json')}) preserves all {runtime['savedJobs']} saved jobs and {runtime['savedAnalysisFiles']} analysis files, verifies three existing demos, {len(runtime['exports'])} exact HTTP exports and {len(runtime['frontend']['assets'])} frontend assets, and retires the prior exact runtime identities. This is existing-result compatibility; it supplies no new worker/model/browser/scientific qualification.\n\n{limitations}\n"""
    install_paragraph = (
        "Current event timestamp lookup software update (6 October 2026): "
        f"the finite ordered `_features` lookup now uses bisect with earliest floating-distance ties; "
        f"the focused {green['tests']}-test check and final {backend['counts']['tests']:,}-test backend suite pass "
        f"with zero failures/errors/skips. The retained initial {failed['counts']['tests']:,}-test suite had "
        "one historical provenance mismatch, corrected against the exact retained prior source without "
        "rewriting its report. Two synthetic 600/18,000-frame old/current feature pairs preserve complete "
        "input/settings/feature bytes, with measured timings and memory recorded in the "
        f"[progress record](../plans/TENNIS_EXECUTION_PROGRESS.md) and "
        f"[measurement receipt]({link(MEASURE_RUN / 'report.json')}). Fresh idle reload preserves "
        f"{runtime['savedJobs']} jobs/{runtime['savedAnalysisFiles']} files and exact "
        f"{len(runtime['exports'])} exports/{len(runtime['frontend']['assets'])} assets. "
        "The successful preflight found ABSENT after the prior runtime exited externally; root stopped "
        "no prior process and restored the documented server. "
        "This paragraph supersedes older current-revision and 1,278-test labels below; those paragraphs "
        "retain their historical result-storage scope. "
        + limitations + "\n\n")
    old_progress = before[DOCUMENTS[0]]
    old_install = before[DOCUMENTS[1]]
    newline = b"\r\n" if b"\r\n" in old_install[:200] else b"\n"
    # New notes are inserted around untouched earlier prose. Removing these
    # exact insertions recovers every previous progress byte and prior metric.
    current_note = (
        f"Current event timestamp lookup unit (6 October): {backend['counts']['tests']:,} backend tests "
        f"pass in {backend['testElapsedSeconds']:.3f}s; focused RED {red['tests']}/1 failure and GREEN "
        f"{green['tests']}/0 failures preserve numeric fallback behavior and complete features/analyze. "
        f"The fresh [synthetic feature measurement]({link(MEASURE_RUN / 'report.json')}) preserves every "
        f"fixture/settings/feature byte: at 18,000 frames prior/current `_features` took "
        f"{summaries[(18000, 'old')]['featuresSeconds']:.6f}/{summaries[(18000, 'new')]['featuresSeconds']:.6f}s, "
        f"with actual interpreter OS peak working sets "
        f"{summaries[(18000, 'old')]['osPeakWorkingSetBytes']:,}/{summaries[(18000, 'new')]['osPeakWorkingSetBytes']:,} bytes. "
        f"The [saved-diagnostic replay]({link(REPLAY_RECEIPT)}) preserves all {replay_count} raw analyze "
        f"pairs and the three historical final-event lists. Finite nearest lookup improves; local dense "
        f"windows, candidate processing, analyze recomputation and O(N) arrays remain. No new actual "
        f"worker/model inference or accuracy/full-match qualification. Full details are appended below.\n\n")
    context_note = (
        "Historical context from the preceding result-storage latency revision follows unchanged. "
        "Its 'latest/current' test/source labels and nearest-event quadratic note refer to that earlier "
        "revision; the current event lookup unit above supersedes those labels and that finite nearest "
        "lookup limitation. Resident/disk timings, worker RAM declines and other remaining costs retain "
        "their original scope.\n\n")
    progress_newline = b"\r\n" if b"\r\n" in old_progress[:200] else b"\n"
    current_insert = current_note.replace("\n", progress_newline.decode()).encode("utf-8")
    context_insert = context_note.replace("\n", progress_newline.decode()).encode("utf-8")
    resume_anchor = b"## Resume here" + progress_newline + progress_newline
    history_anchor = b"Latest result-storage latency unit (6 October):"
    require(old_progress.count(resume_anchor) == 1 and old_progress.count(history_anchor) == 1,
            "Expected progress insertion anchors changed")
    progress_with_notes = old_progress.replace(resume_anchor, resume_anchor + current_insert, 1)
    progress_with_notes = progress_with_notes.replace(history_anchor, context_insert + history_anchor, 1)
    progress = progress.replace(str(PROVENANCE / 'replay_receipt.json'), str(REPLAY_RECEIPT))
    progress = progress.replace(link(PROVENANCE / 'replay_receipt.json'), link(REPLAY_RECEIPT))
    progress = progress.replace(
        ", and retires the prior exact runtime identities. This is existing-result compatibility;",
        ". The successful preflight found ABSENT after the prior pair had exited externally; root stopped "
        "no prior process and restored the documented server. The first strict preflight refused an "
        "additional server command; the subsequent duplicate-stop attempt found its PID already absent "
        "before any mutation. Both [no-action failures](" + link(UNIT / 'runtime_preflight_failures.json') +
        ") remain retained. This is existing-result compatibility;")
    progress += (
        "\nThe first [feature measurement attempt](" + link(FIRST_MEASUREMENT / "run/report.json") +
        ") remains FAILED/PARTIAL: only old 600 frames completed; current 600 was NOT_STARTED at "
        "592,830,464 available bytes below the unchanged 750,000,000-byte launch guard. Neither "
        "18,000-frame case ran and no pair comparison or speed/accuracy claim comes from that attempt. "
        "The fresh headroom attempt separately preserved all 18 first-attempt files and used identical "
        "guards. The [consumed replay-v2 failure](" + link(PROVENANCE / "replay_receipt_v2.json") +
        ") was KeyError 'pid' before acknowledgement/imports/numeric replay, with two samples and exact "
        "owned retirement. Its helper and all ten files plus empty run directory remain unchanged. "
        "The fresh v3 replay corrects that identity schema and passes the full declared comparisons; "
        "the method-only monitor probe supplies no live-process or numeric replay evidence.\n")
    title_end = old_install.find(newline)
    require(title_end >= 0 and old_install.startswith(b"# "), "Installation title structure changed")
    install_insert = install_paragraph.replace("\n", newline.decode()).encode("utf-8")
    installation = old_install[:title_end + len(newline)] + newline + install_insert + old_install[title_end + len(newline):]
    updated = deepcopy(manifest)
    updated["eventTimestampLookupEvidenceUpdatedAt"] = at
    updated["validation"]["backendEventTimestampLookupChecks"] = (
        f"PASSED_SOFTWARE_2026-10-06: {backend['counts']['tests']} complete backend tests in "
        f"{backend['testElapsedSeconds']:.3f}s; failures/errors/skipped 0, exact source bindings unchanged. "
        f"Focused RED {red['tests']} tests/1 failure; GREEN {green['tests']} tests/0 failures. "
        f"Original {failed['counts']['tests']}-test full-suite provenance failure is preserved; "
        "historical-source test correction and exact saved diagnostic current/frozen replay are bound.")
    updated["validation"]["eventTimestampLookupEngineering"] = (
        "PASSED_SYNTHETIC_SOFTWARE_ONLY_2026-10-06: checked finite ordered bisect lookup with earliest "
        "float-distance plateau, unchanged general nearest-index and legacy nonfinite/order behavior; "
        "complete focused feature and analyze outputs exact. "
        + "; ".join(f"{count} frames prior/current actual _features "
                    f"{summaries[(count, 'old')]['featuresSeconds']:.6f}/"
                    f"{summaries[(count, 'new')]['featuresSeconds']:.6f}s and actual interpreter OS peak "
                    f"{summaries[(count, 'old')]['osPeakWorkingSetBytes']}/"
                    f"{summaries[(count, 'new')]['osPeakWorkingSetBytes']} bytes" for count in (600, 18000))
        + ". Four fresh children, full fixture/settings/feature bytes exact, zero exclusions, exact "
        "workload sampling and retirement, unchanged source/import/guard bindings. Instrumented read "
        "counts are separate from uninstrumented time/OS peak. One pair/count, no throughput or "
        "accuracy/full-match qualification. " + limitations)
    updated["validation"]["eventTimestampLookupExistingResultCompatibility"] = (
        f"PASSED_IDLE_RELOAD_SOFTWARE_ONLY_2026-10-06: {runtime['savedJobs']} saved jobs/"
        f"{runtime['savedAnalysisFiles']} files unchanged, 3 demos/{len(runtime['exports'])} exact exports/"
        f"{len(runtime['frontend']['assets'])} exact assets. Prior owned identities retired; "
        "no inference, browser check or scientific qualification.")
    updated["validation"]["eventTimestampLookupExistingResultCompatibility"] = (
        updated["validation"]["eventTimestampLookupExistingResultCompatibility"].replace(
            "Prior owned identities retired;", "Preflight ABSENT after prior processes exited externally; root stopped no prior process, restored documented server; two no-action refusals retained;"))
    updated["validation"]["eventTimestampLookupRealWorkerSmoke"] = (
        "NOT_RUN_ON_EVENT_REVISION: no new actual worker/model inference. Prior two latency RAM "
        "declines remain historical; earlier actual worker evidence retains its original revision.")
    updated["validation"]["eventTimestampLookupRetainedFailedAttempts"] = (
        "FAILED_ATTEMPTS_RETAINED_WITHOUT_QUALIFICATION: first feature attempt completed old600 only, "
        "new600 NOT_STARTED at 592830464 available bytes against unchanged750000000 preflight; "
        "no18000cases/no paired performance inference; all18files unchanged in fresh headroom run. "
        "Replayv2 FAILED KeyError:'pid' before acknowledgement/imports/replay;2samples/exactownedretirement, "
        "all10files+emptyrun retained. Freshv3 fullreplay passes with canonical monitor identity schema.")
    evidence = (
        (UNIT / "red_receipt.json", False, "Genuine instrumented lookup RED; one expected failure; preserved."),
        (GREEN / "green_receipt.json", True, "38 focused numeric/read-count/complete feature and analyze software checks."),
        (UNIT / "backend/receipt.json", False, "Initial complete suite; historical provenance mismatch retained unchanged."),
        (BACKEND / "receipt.json", True, "Final complete maintained backend suite; zero failures/errors/skips; unchanged sources."),
        (REPLAY_RECEIPT, True, "Scoped historical-source provenance correction and 21 saved diagnostic exact analyze replay pairs."),
        (PROVENANCE / "replay_receipt_v2.json", False, "Consumed identity-schema failure before acknowledgement/imports/replay; retained unchanged."),
        (FIRST_MEASUREMENT / "run/report.json", False, "First partial feature measurement; current600 RAM preflight declined; no paired performance evidence."),
        (MEASURE_RUN / "report.json", True, "One actual uninstrumented synthetic feature pair/count at 600/18000; no models or qualification."),
        (UNIT / "runtime_verification.json", True, "Existing-result idle-reload compatibility; saved rows/files and exact exports/assets."),
        (UNIT / "runtime_preflight_failures.json", False, "Two refused/no-action runtime operations; prior identities externally absent before successful preflight."),
    )
    for path, passed, scope in evidence:
        updated["validationEvidence"].append({"path": name(path), "verifiedAt": at, "scope": scope,
                                              "passed": passed, "sha256": track(path)["sha256"],
                                              "qualificationEvidence": False})
    require(updated["productionQualified"] is False, "Qualification flag changed during staging")
    for key, old_value in manifest.items():
        if key == "validation":
            require(all(updated[key][entry] == value for entry, value in old_value.items()),
                    "Earlier validation record changed")
        elif key == "validationEvidence":
            require(updated[key][:len(old_value)] == old_value, "Earlier evidence records changed")
        else:
            require(updated[key] == old_value, "Earlier manifest field changed: " + key)
    output = {DOCUMENTS[0]: progress_with_notes + progress.encode("utf-8"),
              DOCUMENTS[1]: installation,
              DOCUMENTS[2]: (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")}
    require(progress_with_notes.replace(current_insert, b"", 1).replace(context_insert, b"", 1) == old_progress,
            "Earlier progress bytes changed")
    require(output[DOCUMENTS[1]].replace(newline + install_insert, b"", 1) == old_install,
            "Earlier installation bytes changed")
    return output


def unchanged_evidence(exclude_documents=False):
    for pathname, expected in TRACKED.items():
        if exclude_documents and pathname in DOCUMENTS:
            continue
        require(binding(resolved(pathname)) == expected, "Bound input changed before/after publication: " + pathname)


def exclusive_bytes(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def save_receipt(stream, value):
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def publish(before, output, summary):
    require(not RECEIPT.exists() and not STAGING.exists(), "Exclusive publication already reserved; no retry")
    unchanged_evidence()
    immutable = {path: value for path, value in TRACKED.items() if path not in DOCUMENTS}
    # The two disjoint maps cover every immutable input. Current documents use
    # their after-publication hashes rather than the replaced baseline hashes.
    source_bindings = {path: value for path, value in immutable.items()
                       if Path(path).suffix.lower() in (".py", ".pyd", ".dll", ".exe")
                       or path == "pytest.ini"}
    evidence_bindings = {path: value for path, value in immutable.items() if path not in source_bindings}
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        receipt = {"startedAt": stamp(), "status": "RESERVED", "passed": False,
                   "helperBinding": binding(Path(__file__)), "sourceSha256": EXPECTED_CURRENT,
                   "frozenSourceSha256": EXPECTED_FROZEN, "qualificationEvidence": False,
                   "productionQualified": False, "workloadsStarted": False, "modelsImported": False,
                   "serverActions": False, "productStateChanged": False,
                   "before": {path: binding(ROOT / path) for path in DOCUMENTS},
                   "verifiedInputBindings": deepcopy(immutable), "derivedEvidence": summary,
                   "sourceBindings": deepcopy(source_bindings), "evidenceBindings": deepcopy(evidence_bindings),
                   "currentDocumentHashes": {},
                   "rollback": {"attempted": False, "passed": None}}
        save_receipt(stream, receipt)
        replaced = []
        try:
            STAGING.mkdir(exist_ok=False)
            staged = {}
            rollback = {}
            for index, pathname in enumerate(DOCUMENTS):
                staged[pathname] = STAGING / f"{index}_after.bin"
                rollback[pathname] = STAGING / f"{index}_before.bin"
                exclusive_bytes(staged[pathname], output[pathname])
                exclusive_bytes(rollback[pathname], before[pathname])
                require((ROOT / pathname).read_bytes() == before[pathname], "Document baseline changed during staging")
            receipt["stagedAfter"] = {pathname: binding(path) for pathname, path in staged.items()}
            receipt["stagedBefore"] = {pathname: binding(path) for pathname, path in rollback.items()}
            save_receipt(stream, receipt)
            unchanged_evidence()
            for pathname in DOCUMENTS:
                require((ROOT / pathname).read_bytes() == before[pathname], "Document changed before atomic replacement")
                os.replace(staged[pathname], ROOT / pathname)
                replaced.append(pathname)
                require((ROOT / pathname).read_bytes() == output[pathname], "Atomic replacement readback differs")
            unchanged_evidence(exclude_documents=True)
            for pathname in DOCUMENTS:
                require((ROOT / pathname).read_bytes() == output[pathname], "Final document readback differs")
            require(json.loads((ROOT / DOCUMENTS[2]).read_text(encoding="utf-8"))["productionQualified"] is False,
                    "Published manifest changed qualification")
            receipt["after"] = {path: binding(ROOT / path) for path in DOCUMENTS}
            receipt["currentDocumentHashes"] = deepcopy(receipt["after"])
            require(receipt["after"] == receipt["stagedAfter"], "Final staged/published hashes differ")
            receipt["allEvidenceHashesUnchanged"] = True
            receipt["preservedBeforeBytesExact"] = True
            receipt["olderManifestRecordsUnchanged"] = True
            receipt["status"] = "COMPLETED"
            receipt["passed"] = True
        except BaseException as error:
            receipt["status"] = "FAILED"
            receipt["error"] = type(error).__name__ + ": " + str(error)
            receipt["replacedDocumentsBeforeFailure"] = replaced[:]
            receipt["rollback"]["attempted"] = bool(replaced)
            try:
                for pathname in reversed(replaced):
                    # Restore only a document still containing this helper's
                    # staged bytes; never overwrite an unrelated concurrent edit.
                    require((ROOT / pathname).read_bytes() == output[pathname],
                            "Concurrent document edit prevents safe rollback: " + pathname)
                    os.replace(rollback[pathname], ROOT / pathname)
                require(all((ROOT / path).read_bytes() == before[path] for path in DOCUMENTS),
                        "Rollback did not restore all exact before bytes")
                receipt["rollback"]["passed"] = True
            except BaseException as rollback_error:
                receipt["rollback"]["passed"] = False
                receipt["rollback"]["error"] = type(rollback_error).__name__ + ": " + str(rollback_error)
        finally:
            receipt["completedAt"] = stamp()
            save_receipt(stream, receipt)
    require(receipt["passed"], "Documentation publication failed; exclusive receipt and staged bytes retained")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="Read-only evidence validation; creates no files")
    mode.add_argument("--apply", action="store_true", help="Parent-authorized one-shot documentation publication")
    args = parser.parse_args()
    require(not RECEIPT.exists() and not STAGING.exists(), "Publication receipt/staging already exists; do not retry")
    source_gate()
    before, manifest, declines = preserved_gate()
    red, green, rows, red_rows = focused_gate()
    failed, failed_bindings, backend = backend_gate()
    provenance = provenance_gate(failed_bindings, backend)
    measured, summaries, guards = measurement_gate()
    runtime = runtime_gate()
    at = stamp()
    output = build_documents(before, manifest, red, green, rows, red_rows, failed, backend,
                             provenance, measured, summaries, runtime, declines, at)
    unchanged_evidence()
    summary = {"verifiedAt": at, "backendCounts": backend["counts"],
               "backendTestElapsedSeconds": backend["testElapsedSeconds"],
               "preservedFailedBackendCounts": failed["counts"],
               "focusedRedCounts": {key: red[key] for key in ("tests", "failures", "errors", "skipped")},
               "focusedGreenCounts": {key: green[key] for key in ("tests", "failures", "errors", "skipped")},
               "instrumentedReads": rows,
               "uninstrumentedActualFeatures": {f"{count}_{mode}": value for (count, mode), value in summaries.items()},
               "measurementGuards": guards,
               "savedJobs": runtime["savedJobs"], "savedAnalysisFiles": runtime["savedAnalysisFiles"],
               "exports": len(runtime["exports"]), "frontendAssets": len(runtime["frontend"]["assets"]),
               "historicalWorkerDeclines": declines, "actualWorkerInferenceOnEventRevision": False,
               "plannedAfter": {path: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                                for path, content in output.items()}}
    if args.validate_only:
        print(json.dumps({"passed": True, "mode": "READ_ONLY_VALIDATION", "filesWritten": 0,
                          "backendCounts": backend["counts"], "plannedAfter": summary["plannedAfter"]}, allow_nan=False))
        return 0
    receipt = publish(before, output, summary)
    print(json.dumps({"passed": receipt["passed"], "status": receipt["status"],
                      "receipt": name(RECEIPT), "after": receipt["after"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
