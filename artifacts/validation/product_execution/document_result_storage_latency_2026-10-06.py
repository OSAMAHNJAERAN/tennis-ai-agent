"""One-shot evidence-gated documentation update; root reviews before execution.

Imports no production code. No tests, inference, HTTP requests, installations,
server changes or benchmark reruns. An unused exclusive receipt is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
import xml.etree.ElementTree as ET

import psutil


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/validation/product_execution"
BASELINE = ART / "result_storage_latency_baseline_2026-10-06"
BENCH = ART / "result_storage_latency_measurement_2026-10-06"
BACKEND = ART / "result_storage_latency_backend_2026-10-06"
INITIAL_WORKER = ART / "result_storage_latency_worker_smoke_2026-10-06"
FRESH_WORKER = ART / "result_storage_latency_headroom_worker_smoke_2026-10-06"
RECEIPT = ART / "result_storage_latency_documentation_2026-10-06.json"
DOCS = (
    "docs/plans/TENNIS_EXECUTION_PROGRESS.md",
    "docs/deployment/PRODUCT_INSTALLATION.md",
    "docs/deployment/PRODUCT_RESTORE_MANIFEST.json",
)
EXPECTED_DOCS = {
    DOCS[0]: "0215a5dedf9777001aa8323b0ceeae50a7916430a1213d0c5b0e05667e09b4fc",
    DOCS[1]: "b0fda5dc0ed56a24ec57376a173bb470b6e79e35b0bc66c542d1d786098ce7b2",
    DOCS[2]: "e05c7933689fd305dd910f20ed6cbc54337cab45331bfeb3ecabbb078073781c",
}
BASELINE_RECEIPT_SHA = "ee8ebdb6e4723bf608a8eb0daf0684efd49de9c3af32d31aab64cac1e2607a85"
SOURCE_BINDINGS: dict[str, dict] = {}
EVIDENCE_BINDINGS: dict[str, dict] = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def bound(path, expected=None, *, source=False):
    path = Path(path)
    actual = binding(path)
    if expected is not None:
        if isinstance(expected, str):
            require(actual["sha256"] == expected, f"Hash mismatch: {path}")
        else:
            require(actual == expected, f"Binding mismatch: {path}")
    target = SOURCE_BINDINGS if source else EVIDENCE_BINDINGS
    name = rel(path) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
    require(name not in target or target[name] == actual, f"Changed during validation: {name}")
    target[name] = actual
    return actual


def load(path):
    bound(path)
    return json.loads(Path(path).read_bytes().decode("utf-8"),
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def verify_sources(values, base=ROOT):
    for name, expected in values.items():
        bound(base / name, expected, source=True)


def verify_artifacts(directory, receipt):
    for name, expected in receipt.get("artifacts", {}).items():
        bound(directory / name, expected)


def retired(identity):
    try:
        process = psutil.Process(identity["pid"])
        return process.create_time() != identity["createTimeUnixSeconds"]
    except psutil.NoSuchProcess:
        return True


def xml_counts(path):
    node = ET.parse(path).getroot()
    suites = [node] if node.tag == "testsuite" else list(node.findall("testsuite"))
    return {key: sum(int(s.attrib.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def check_backend_and_snapshot():
    backend = load(BACKEND / "receipt.json")
    require(backend["passed"] and backend["exitCode"] == 0 and backend["sourcesUnchanged"], "Backend not successful")
    expected = {"tests": 1278, "failures": 0, "errors": 0, "skipped": 0}
    require(backend["counts"] == expected == xml_counts(BACKEND / "junit.xml"), "Backend count mismatch")
    verify_artifacts(BACKEND, backend)
    verify_sources(load(BACKEND / "source_bindings.json"))
    phases = {}
    for phase, counts in (("red", {"tests": 3, "failures": 1, "errors": 0, "skipped": 0}),
                          ("green", {"tests": 4, "failures": 0, "errors": 0, "skipped": 0})):
        directory = ART / f"result_reader_snapshot_{phase}_2026-10-06"
        receipt = load(directory / "receipt.json")
        require(receipt["phase"] == phase and receipt["counts"] == counts == xml_counts(directory / "junit.xml"), "Snapshot phase/count mismatch")
        require(receipt["sourcesUnchanged"] and receipt["sourceBindingsBefore"] == receipt["sourceBindingsAfter"], "Snapshot source change")
        require(receipt["exitCode"] == (1 if phase == "red" else 0), "Snapshot exit mismatch")
        verify_artifacts(directory, receipt)
        if phase == "green":
            verify_sources(receipt["sourceBindingsAfter"])
        else:
            bound(BASELINE / "src/api/result_storage.py", receipt["sourceBindingsBefore"]["src/api/result_storage.py"])
            bound(directory / "test_source.py", receipt["sourceBindingsBefore"]["tests/test_result_reader_snapshot.py"])
        phases[phase] = receipt
    return backend, phases


def check_benchmark():
    report = load(BENCH / "run/report.json")
    audit = load(BENCH / "audit/terminal_audit.json")
    require(report["passed"] and report["status"] == "COMPLETED" and audit["passed"] and not audit["errors"], "Benchmark/audit failed")
    require(report["sourcesUnchanged"] and report["preparedBindingsUnchanged"], "Benchmark bindings changed")
    require(report["sourceBindingsBefore"] == report["sourceBindingsAfter"] == audit["sourceBindingsNow"], "Benchmark source snapshots differ")
    verify_sources(audit["sourceBindingsNow"])
    bound(BENCH / "prepared_source_bindings.json", audit["preparedBindings"])
    require(audit["retainedFileCount"] == len(audit["retainedFileBindings"]) == 124, "Retained file count mismatch")
    for name, value in audit["retainedFileBindings"].items():
        bound(BENCH / name, value)
    require(len(report["runs"]) == len(audit["runs"]) == 4, "Case count mismatch")
    require([(r["frames"], r["mode"]) for r in audit["runs"]] == [(600, "old"), (600, "new"), (18000, "old"), (18000, "new")], "Case order changed")
    for run, terminal in zip(report["runs"], audit["runs"], strict=True):
        require(run["passed"] and run["exitCode"] == 0 and run["actualWorkloadWasSampled"], "Case/actual sampling failed")
        require(run["allRememberedOwnedIdentitiesRetired"] and not run["remainingOwnedIdentities"], "Case ownership not retired")
        require(terminal["allOwnedIdentitiesRetiredNow"] and terminal["guardThresholdsNeverTriggered"] and terminal["modelsImported"] == [], "Case guard/model/retirement discrepancy")
        require(all(retired(identity) for identity in run["ownedIdentities"]), "Benchmark PID/createTime still live now")
    outputs = {"analysis/result.json", "analysis/summary.json", "read_result.json", "telemetry.csv", "report.html"}
    require([c["frames"] for c in audit["comparisons"]] == [600, 18000], "Comparison counts changed")
    for compare in audit["comparisons"]:
        require(compare["fiveExactByteComparisonsPassed"] and compare["sameStreamingInputsAndRecordedSemantics"] and not compare["volatileFieldsExcluded"], "Contract comparison failure")
        require(set(compare["outputs"]) == outputs and all(v["exactBytesEqualNow"] for v in compare["outputs"].values()), "Output byte comparison failure")
    require(audit["qualificationEvidence"] is False and audit["inferenceRun"] is False, "Synthetic evidence scope changed")
    return report, audit


def current_product_snapshot():
    database = ROOT / "outputs/product/analyses.sqlite3"
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        jobs = [dict(row) for row in connection.execute("SELECT * FROM analyses ORDER BY id")]
    # Logical saved-analysis tree only; SQLite volatile sidecars are not called unchanged here.
    files = {rel(path): binding(path)["sha256"] for path in sorted((ROOT / "outputs/product/analyses").rglob("*")) if path.is_file()}
    return {"jobs": jobs, "savedAnalysisFiles": files}


def check_runtime():
    runtime = load(ART / "result_storage_latency_runtime_verification_2026-10-06.json")
    preflight = load(ART / "result_storage_latency_runtime_preflight_2026-10-06.json")
    require(runtime["passed"] and runtime["savedJobsAndAnalysisFilesUnchanged"], "Runtime verification failed")
    require(runtime["snapshotBefore"] == runtime["snapshotAfter"] == preflight["snapshot"] == current_product_snapshot(), "Saved product rows/files changed")
    require(runtime["savedJobs"] == 10 and runtime["savedAnalysisFiles"] == 319 and runtime["catalogEntries"] == 3, "Runtime scope/count changed")
    require(runtime["productStateChanged"] is False and runtime["runtimeStoppedOrStartedByThisVerifier"] is False, "Runtime verifier mutated product")
    verify_sources(runtime["sourceBindings"]["files"])
    interpreter = runtime["sourceBindings"]["baseInterpreter"]
    bound(Path(interpreter["path"]), interpreter["sha256"], source=True)
    bound(ART / "result_storage_latency_runtime_preflight_2026-10-06.json", runtime["preflightSha256"])
    require(all(i["retired"] and retired(i) for i in runtime["previousRuntimeIdentitiesRetired"]), "Previous runtime identity still live")
    require(len(runtime["exports"]) == 12 and all(item["status"] == 200 for item in runtime["exports"]), "Runtime export checks incomplete")
    require(len(runtime["frontend"]["assets"]) == 7 and all(item["status"] == 200 for item in runtime["frontend"]["assets"]), "HTTP frontend asset checks incomplete")
    bound(ART / "verify_result_storage_latency_runtime_2026-10-06_initial.py")
    bound(ART / "verify_result_storage_latency_runtime_2026-10-06_wrapper_only.py")
    return runtime


def check_worker():
    initial = load(INITIAL_WORKER / "resource_receipt.json")
    require(initial["status"] == "NOT_STARTED" and initial["passed"] is False and initial["inferenceChildStarted"] is False, "Initial worker preflight history changed")
    require(initial["samples"] == 0 and initial["productionStatePublished"] is False and initial["monitoredChildWallSeconds"] is None, "Initial attempt did work")
    preparation = load(INITIAL_WORKER / "preparation.json")
    require(preparation["preflightAvailableRamBytes"] == 2500000000 and initial["minimumSampledAvailableRamBytes"] < 2500000000, "Initial preflight reason/guard changed")
    require(initial["qualificationEvidence"] is False and initial["productionQualified"] is False, "Worker qualification changed")
    bound(INITIAL_WORKER / "protocol.json")
    bound(ROOT / preparation["script"], preparation["scriptSha256"], source=True)
    path = FRESH_WORKER / "resource_receipt.json"
    if not path.exists():
        return {"status": "NOT_STARTED", "pending": True, "unit": rel(INITIAL_WORKER), "initial": initial,
                "statement": f"The current-revision 30-frame worker smoke is NOT_STARTED/pending: available RAM {initial['minimumSampledAvailableRamBytes']:,} bytes was below the unchanged 2,500,000,000-byte preflight. No child/model/work output was launched. Earlier 30-frame worker receipts remain historical evidence."}
    resource = load(path)
    fresh_preparation = load(FRESH_WORKER / "preparation.json")
    protocol = load(FRESH_WORKER / "protocol.json")
    require(fresh_preparation["preflightAvailableRamBytes"] == protocol["preflightAvailableRamBytes"] == 2500000000,
            "Fresh preflight was relaxed")
    require(fresh_preparation["maximumSampledOwnedTreeRssBytes"] == protocol["maximumSampledOwnedTreeRssBytes"] == 3000000000
            and fresh_preparation["minimumAvailableRamBytes"] == protocol["minimumAvailableRamBytes"] == 400000000
            and protocol["lowRamConsecutiveSamplesToStop"] == 5 and protocol["maximumChildWallSeconds"] == 300,
            "Fresh owned-process safeguards changed")
    bound(ROOT / fresh_preparation["script"], fresh_preparation["scriptSha256"], source=True)
    if resource["status"] == "NOT_STARTED":
        require(resource["passed"] is False and resource["inferenceChildStarted"] is False
                and resource["samples"] == 0 and resource["productionStatePublished"] is False
                and resource["monitoredChildWallSeconds"] is None
                and resource["minimumSampledAvailableRamBytes"] < 2500000000,
                "Fresh pending attempt did work or violated its preflight")
        require(resource["qualificationEvidence"] is False and resource["productionQualified"] is False,
                "Fresh pending qualification changed")
        require(not (INITIAL_WORKER / "run").exists() and not (FRESH_WORKER / "run").exists(),
                "NOT_STARTED attempt contains a run directory")
        return {"status": "NOT_STARTED", "pending": True, "unit": rel(FRESH_WORKER),
                "initial": initial, "resource": resource,
                "statement": f"The current-revision 30-frame worker check remains NOT_STARTED/pending. Both exclusive attempts declined before launching: available RAM was {initial['minimumSampledAvailableRamBytes']:,} and {resource['minimumSampledAvailableRamBytes']:,} bytes, below the unchanged 2,500,000,000-byte preflight. Both retained receipts have zero children/samples and no run directory; no model/work output was produced. No automatic retry or guard relaxation occurred. Earlier actual 30-frame worker evidence remains historical."}
    require(resource["passed"] and resource["status"] == "COMPLETED" and resource["inferenceChildStarted"], "Fresh worker evidence exists but is not a terminal pass")
    require(resource["qualificationEvidence"] is False and resource["productionQualified"] is False and resource["productionStatePublished"] is False, "Fresh worker scope changed")
    child = load(FRESH_WORKER / "child_receipt.json")
    comparison = load(FRESH_WORKER / "saved_result_comparison.json")
    require(child["passed"] and child["workerInvoked"] and child["actualPipelineRunCompletedByWorker"] and child["networkAttempts"] == [], "Fresh actual worker run incomplete")
    require(comparison["passed"] and comparison["volatileRuntimeFieldsRemoved"] == [] and comparison["comparisonUsesSameNewRawBytes"], "Fresh worker compatibility incomplete")
    require(set(comparison["comparisons"]) == {"json", "csv", "html"}
            and all(item["exactBytes"] for item in comparison["comparisons"].values()), "Fresh worker output bytes differ")
    for label in ("sourceBindings", "historicalArtifacts", "productionJobStates", "liveProductFileTree", "originHistoricalArtifacts"):
        require(resource[label + "Unchanged"] and resource[label + "After"] == protocol[label + "Before"], "Fresh worker binding/state changed: " + label)
    require(len(resource["productionJobStatesAfter"]["jobs"]) == 10, "Fresh worker product job count changed")
    verify_sources(resource["sourceBindingsAfter"])
    require(resource["actualWorkloadSampledByExactIdentity"] and resource["allExactOwnedIdentitiesRetiredAtTerminal"] and not resource["remainingExactOwnedIdentitiesAtTerminal"], "Fresh worker sampling/retirement incomplete")
    require(all(retired(identity) for identity in resource["ownedProcessIdentitiesAndPeaks"]), "Fresh worker owned identity still live")
    output = Path(child["outputDirectory"])
    if not output.is_absolute():
        output = ROOT / output
    require(output.resolve().is_relative_to(FRESH_WORKER.resolve()), "Fresh isolated output escaped its unit")
    saved = load(output / "result.json")
    require(saved["summary"]["frames"] == 30 and saved["summary"]["fps"] == 60, "Fresh worker clock/frame count changed")
    require(saved["ballMetrics"] is None and saved["world3d"] is None, "Fresh worker physical ball metrics/3D became available")
    require(saved["pipelineMetrics"]["raw"]["authoritative_events_enabled"] is False, "Fresh worker event authority changed")
    return {"status": "COMPLETED", "pending": False, "unit": rel(FRESH_WORKER), "initial": initial, "resource": resource,
            "statement": "A fresh current-revision isolated 30-frame/native-60-FPS actual worker smoke passed strict JSON/CSV/HTML compatibility, unchanged ten live jobs and complete live file tree, exact owned-process retirement, unavailable physical ball speed/3D and false event authority. The first NOT_STARTED attempt remains preserved; its 2.5 GB preflight was not relaxed. This is short compatibility evidence only."}


def markdown_link(path, base="../../"):
    return base + rel(path)


def build_documents(before, backend, phases, audit, runtime, worker):
    compare = next(c for c in audit["comparisons"] if c["frames"] == 18000)
    runs = {r["mode"]: r for r in audit["runs"] if r["frames"] == 18000}
    old, new = compare["baselineRawStagesSeconds"], compare["currentRawStagesSeconds"]
    metrics = (f"At 18,000 synthetic frames, old/new construction was {old['constructionSeconds']:.4f}/{new['constructionSeconds']:.4f} s, "
               f"publication {old['publicationSeconds']:.4f}/{new['publicationSeconds']:.4f} s, complete JSON read {old['strictReadJsonSeconds']:.4f}/{new['strictReadJsonSeconds']:.4f} s, "
               f"CSV {old['csvExportSeconds']:.4f}/{new['csvExportSeconds']:.4f} s and HTML {old['htmlExportSeconds']:.4f}/{new['htmlExportSeconds']:.4f} s. "
               f"Production-stage sums were {compare['baselineProductionStageTotalSeconds']:.4f}/{compare['currentProductionStageTotalSeconds']:.4f} s; "
               f"child elapsed {compare['baselineChildElapsedSeconds']:.4f}/{compare['currentChildElapsedSeconds']:.4f} s; parent-monitored wall {runs['old']['monitorWallSeconds']:.4f}/{runs['new']['monitorWallSeconds']:.4f} s. "
               f"Actual workload-interpreter OS peak working sets were {compare['baselineActualOsPeakWorkingSetBytes']:,}/{compare['currentActualOsPeakWorkingSetBytes']:,} bytes; "
               f"published databases were {compare['baselineDatabaseBytes']:,}/{compare['currentDatabaseBytes']:,} bytes.")
    limitations = ("This improves the previous disk-backed implementation. The earlier resident-result baseline remains faster overall; these local synthetic timings are not full-match or accuracy qualification. "
                   "Pose, racket, selected-player and trajectory pipeline arrays and browser result arrays still grow with video length (O(N)); nearest-event mapping can be quadratic. "
                   "Representative authorized recordings and independent speed/3D reference data remain unavailable, so scientific accuracy and long-match capacity remain unqualified.")
    section = ("Latest result-storage latency unit (6 October): normalization reuses each decoded frame locally, and readonly generation readers now hold one SQLite transaction from root validation through close. "
               "That transaction keeps one coherent snapshot across external atomic commits ([SQLite transaction documentation](https://www.sqlite.org/lang_transaction.html)). "
               f"[All {backend['counts']['tests']:,} backend tests]({markdown_link(BACKEND/'receipt.json')}) passed in {backend['testElapsedSeconds']:.3f} s with unchanged sources. "
               f"The [snapshot regression]({markdown_link(ART/'result_reader_snapshot_red_2026-10-06/receipt.json')}) first ran RED (3 tests, 1 failure); "
               f"[GREEN]({markdown_link(ART/'result_reader_snapshot_green_2026-10-06/receipt.json')}) ran 4 tests with no failures after the transaction change. "
               "The original RED storage source and test copy remain preserved; reader failure, root-validation failure and response iterator/send/disconnect cleanup coverage remains in the full suite.\n\n"
               f"[The frozen prior-disk/current-disk comparison]({markdown_link(BENCH/'run/report.json')}) passed all four fresh 600/18,000-frame children and "
               f"[ten complete byte comparisons]({markdown_link(BENCH/'audit/terminal_audit.json')}) for saved result, summary, read JSON, CSV and HTML, with no excluded fields. "
               "Frame clocks, gap/calibration/UNKNOWN-person abstention, null physical ball speed/3D and false authority retain the contract. All actual interpreters were sampled by PID/createTime before their heavy work; all owned identities were retired and checked again by the terminal audit. "
               + metrics + " Hash verification/full database reads belong to read costs; child elapsed and parent monitor wall also include other work and monitoring. OS process peaks and sampled aggregate tree RSS are distinct measurements.\n\n"
               + worker["statement"] + f" See [the current worker receipt]({markdown_link(ROOT/worker['unit']/'resource_receipt.json')}) and "
               f"[the preserved first preflight]({markdown_link(INITIAL_WORKER/'resource_receipt.json')}). "
               f"[Current idle-reload HTTP verification]({markdown_link(ART/'result_storage_latency_runtime_verification_2026-10-06.json')}) passed 12 result/export responses across three catalog entries and seven exact frontend assets; "
               f"all {runtime['savedJobs']} saved jobs and {runtime['savedAnalysisFiles']} saved analysis files stayed unchanged. This is HTTP compatibility evidence; no new browser or host-restoration check is claimed.\n\n"
               + limitations + "\n\n")
    progress = before[DOCS[0]].decode("utf-8")
    marker = "Latest verification and resume notes:"
    require(progress.count(marker) == 1 and progress.count("Latest result-storage unit (6 October):") == 1, "Progress insertion anchors changed")
    progress = progress.replace("Latest result-storage unit (6 October):", "Previous result-storage unit (6 October):", 1)
    index = progress.index(marker) + len(marker)
    while index < len(progress) and progress[index] in "\r\n":
        index += 1
    progress = progress[:index] + section + progress[index:]
    install = before[DOCS[1]].decode("utf-8")
    marker = "## Evidence and remaining checks"
    require(install.count(marker) == 1 and install.count("The result-storage update passes all 1,272") == 1, "Installation anchors changed")
    install = install.replace("The result-storage update passes all 1,272", "The previous result-storage unit passes all 1,272", 1)
    paragraph = (f"The current result-storage latency revision passed {backend['counts']['tests']:,} backend tests and four fresh synthetic prior-disk/current-disk children with ten exact saved/read/summary/CSV/HTML byte checks. "
                 + metrics + " " + worker["statement"] + " Current HTTP compatibility preserved ten saved jobs and 319 analysis files; seven frontend assets matched their exact bytes. "
                 + limitations + " See [current measured progress](../plans/TENNIS_EXECUTION_PROGRESS.md) and "
                 f"[the independent terminal audit]({markdown_link(BENCH/'audit/terminal_audit.json')}). Earlier result-storage, worker and restoration receipts below remain historical.\n\n")
    index = install.index(marker) + len(marker)
    while index < len(install) and install[index] in "\r\n":
        index += 1
    install = install[:index] + paragraph + install[index:]
    backup = "Back up the entire hidden `.result` directory"
    original_install = before[DOCS[1]].decode("utf-8")
    backup_paragraph = original_install[original_install.index(backup):].split("\n\n", 1)[0]
    require(backup_paragraph in install, "Entire .result backup/gate wording changed")
    manifest = json.loads(before[DOCS[2]].decode("utf-8"))
    original = json.loads(before[DOCS[2]].decode("utf-8"))
    require(manifest["productionQualified"] is False, "Manifest qualification changed")
    added = {
        "backendResultStorageLatencyChecks": f"PASSED: {backend['counts']['tests']} backend tests, failures/errors/skipped 0, {backend['testElapsedSeconds']:.3f} s; unchanged source bindings. Snapshot RED 3 tests/1 failure, GREEN 4 tests/0 failures; preserved originals and cleanup coverage.",
        "resultStorageLatencyEngineering": "PASSED: four fresh prior-disk/current-disk synthetic 600/18000 children; ten exact byte comparisons, no exclusions, all actual identities sampled and owned processes retired. " + metrics + " " + limitations,
        "resultStorageLatencyRealWorkerSmoke": worker["status"] + ("/PENDING: " if worker["pending"] else ": ") + worker["statement"],
    }
    require(not set(added).intersection(manifest["validation"]), "Latency validation fields already exist")
    manifest["validation"].update(added)
    now = stamp()
    evidence = [(BACKEND/"receipt.json", True, "Current 1278-test backend software regressions; unchanged sources."),
                (BENCH/"run/report.json", True, "Four paired synthetic cases; no models, full-match or accuracy qualification."),
                (BENCH/"audit/terminal_audit.json", True, "Independent ten-output byte/binding/process terminal audit."),
                (ART/"result_reader_snapshot_red_2026-10-06/receipt.json", False, "Historical mechanistic RED coherence regression, 3 tests/1 failure; retained unchanged."),
                (ART/"result_reader_snapshot_green_2026-10-06/receipt.json", True, "Current snapshot and normal/failure cleanup regressions; 4 tests/0 failures."),
                (INITIAL_WORKER/"resource_receipt.json", False, "Preserved NOT_STARTED current-revision first RAM preflight; no child/models/work writes."),
                (ART/"result_storage_latency_runtime_verification_2026-10-06.json", True, "Current idle-reload HTTP compatibility, saved rows/files and exact frontend assets; no browser/host inference.")]
    if (FRESH_WORKER/"resource_receipt.json").exists():
        evidence.append((FRESH_WORKER/"resource_receipt.json", not worker["pending"],
                         "Current second guarded attempt: NOT_STARTED/pending, no inference; unchanged guards."
                         if worker["pending"] else "Current isolated actual 30-frame worker compatibility; no accuracy/full-match qualification."))
    for path, passed, scope in evidence:
        manifest.setdefault("validationEvidence", []).append({"path": rel(path), "verifiedAt": now,
                                                             "scope": scope, "passed": passed, "sha256": bound(path)["sha256"]})
    manifest["evidenceUpdatedAt"] = now
    require(manifest["persistentState"] == original["persistentState"] and manifest["unresolvedReleaseRequirements"] == original["unresolvedReleaseRequirements"], "Existing backup/qualification requirements changed")
    require(all(manifest["validation"][k] == v for k, v in original["validation"].items()), "Historical validation overwritten")
    return {DOCS[0]: progress.encode("utf-8"), DOCS[1]: install.encode("utf-8"),
            DOCS[2]: (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")}, metrics


def stable_bindings():
    for name, expected in SOURCE_BINDINGS.items():
        require(binding(Path(name) if Path(name).is_absolute() else ROOT/name) == expected, "Source changed before/after documentation: " + name)
    for name, expected in EVIDENCE_BINDINGS.items():
        require(binding(Path(name) if Path(name).is_absolute() else ROOT/name) == expected, "Evidence changed before/after documentation: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True, help="Requires root review and final dispatch; writes only three documents and one exclusive receipt")
    parser.parse_args()
    result = {"startedAt": stamp(), "passed": False, "productionCodeImported": False,
              "testsOrWorkloadsLaunched": False, "qualificationEvidence": False, "productionQualified": False}
    before = {}
    committed = []
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as receipt_stream:
        try:
            bound(Path(__file__), source=True)
            bound(BASELINE/"receipt.json", BASELINE_RECEIPT_SHA)
            preservation = load(BASELINE/"receipt.json")
            require(len(preservation["bindings"]) == 17 and preservation["beforeLatencyChanges"], "Baseline preservation mismatch")
            for name, value in preservation["bindings"].items():
                bound(BASELINE/name, value)
            for name in DOCS:
                before[name] = (ROOT/name).read_bytes()
                require(hashlib.sha256(before[name]).hexdigest() == EXPECTED_DOCS[name] == preservation["bindings"][name]["sha256"], "Current document differs from preserved pre-change document: " + name)
            backend, phases = check_backend_and_snapshot()
            report, audit = check_benchmark()
            runtime = check_runtime()
            worker = check_worker()
            replacements, metrics = build_documents(before, backend, phases, audit, runtime, worker)
            stable_bindings()
            require(current_product_snapshot() == runtime["snapshotAfter"], "Product rows/files changed during document preparation")
            for name in DOCS:
                require((ROOT/name).read_bytes() == before[name], "Document changed before commit: " + name)
            staged = {}
            for name, content in replacements.items():
                temporary = (ROOT/name).with_name((ROOT/name).name + ".latency." + uuid.uuid4().hex + ".tmp")
                with temporary.open("xb") as stream:
                    stream.write(content)
                staged[name] = temporary
            for name in DOCS:
                staged[name].replace(ROOT/name)
                committed.append(name)
            stable_bindings()
            result.update({"passed": True, "previousDocumentHashes": EXPECTED_DOCS,
                           "currentDocumentHashes": {name: binding(ROOT/name)["sha256"] for name in DOCS},
                           "sourceBindings": SOURCE_BINDINGS, "evidenceBindings": EVIDENCE_BINDINGS,
                           "workerStatus": worker["status"], "currentWorkerPending": worker["pending"],
                           "initialWorkerNotStartedPreserved": True, "runtimePassed": True,
                           "historicalBaseline17Preserved": True, "backendCounts": backend["counts"],
                           "observed18000MetricsText": metrics, "syntheticByteComparisonCount": 10,
                           "allOwnedBenchmarkIdentitiesRetiredNow": True,
                           "existingQualificationAndBackupPolicyPreserved": True})
        except BaseException as error:
            result["error"] = type(error).__name__ + ": " + str(error)
            for name in reversed(committed):
                try:
                    temporary = (ROOT/name).with_name((ROOT/name).name + ".rollback." + uuid.uuid4().hex + ".tmp")
                    with temporary.open("xb") as stream:
                        stream.write(before[name])
                    temporary.replace(ROOT/name)
                except BaseException as rollback:
                    result.setdefault("rollbackErrors", []).append(name + ": " + repr(rollback))
            result["currentDocumentHashes"] = {name: binding(ROOT/name)["sha256"] for name in DOCS if (ROOT/name).exists()}
        finally:
            result["completedAt"] = stamp()
            json.dump(result, receipt_stream, indent=2, ensure_ascii=False, allow_nan=False)
            receipt_stream.write("\n")
    print(json.dumps({key: result.get(key) for key in ("passed", "workerStatus", "currentWorkerPending", "error")}, ensure_ascii=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
