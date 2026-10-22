"""Prepared-only frontend entry evidence publication; root must dispatch.

Only saved evidence and documents are read. No project imports, subprocess,
HTTP/SQLite/runtime actions, models, inference or work submission. --apply is
one-shot, backups are exclusive, and document replacement supports rollback.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
EVENT = UNIT.parent / "event_timestamp_lookup_2026-10-06"
EVENT_SEAL = EVENT / "final_verification.json"
EVENT_SEAL_SHA256 = "2402cc994a64210518f82754a0b97d5d7ae8d655e2b512be15eaa6ef7a318f16"
RESTORE = ROOT / "artifacts/validation/product_installation/frontend_source_restore_entry_recovery_2026-10-06"
RECEIPT = UNIT / "documentation_receipt.json"
BACKUP = UNIT / "documentation_backup"
STAGING = UNIT / "documentation_staging"
DOCUMENTS = ("docs/plans/TENNIS_EXECUTION_PROGRESS.md", "docs/deployment/PRODUCT_INSTALLATION.md",
             "docs/deployment/PRODUCT_RESTORE_MANIFEST.json")
ROOT_FILES = ("package.json", "pnpm-lock.yaml", "index.html", "vite.config.ts", "eslint.config.js",
              "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json")
SOURCE_DIRS = ("src", "public", "scripts")
SHARED_FILES = ("theme.css", "variables.css")
TRACKED = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def resolve(pathname):
    path = Path(pathname)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def binding(path):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": sha}


def track(path, expected=None):
    actual = binding(path)
    if isinstance(expected, str):
        require(actual["sha256"] == expected, "SHA mismatch: " + name(path))
    elif expected is not None:
        require(actual == expected, "Binding mismatch: " + name(path))
    pathname = name(path)
    require(pathname not in TRACKED or TRACKED[pathname] == actual, "Evidence changed while reading: " + pathname)
    TRACKED[pathname] = actual
    return actual


def load(path, expected=None):
    track(path, expected)
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def verify_map(values):
    require(isinstance(values, dict) and values, "Missing binding map")
    for pathname, expected in values.items():
        track(resolve(pathname), expected)


def exact(left, right):
    track(left)
    track(right)
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            first, second = a.read(65536), b.read(65536)
            require(first == second, "Exact bytes differ: " + name(left) + " / " + name(right))
            if not first:
                return


def inventory(folder):
    return {path.relative_to(folder).as_posix(): path for path in folder.rglob("*") if path.is_file()}


def frontend_inputs(folder):
    paths = [folder / leaf for leaf in ROOT_FILES]
    if (folder / ".npmrc").exists():
        paths.append(folder / ".npmrc")
    for directory in SOURCE_DIRS:
        paths.extend(path for path in (folder / directory).rglob("*") if path.is_file())
    require(all(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()) for path in paths),
            "Frontend source input escapes reviewed directory")
    return {path.relative_to(folder).as_posix(): path for path in paths}


def previous_gate():
    seal = load(EVENT_SEAL, EVENT_SEAL_SHA256)
    require(seal["passed"] is True and seal["productionQualified"] is False
            and seal["accuracyQualified"] is False and seal["fullMatchQualified"] is False
            and seal["physicalSpeedAnd3DQualified"] is False, "Prior event seal scope changed")
    before = {}
    for pathname in DOCUMENTS:
        track(ROOT / pathname, seal["bindings"][pathname])
        before[pathname] = (ROOT / pathname).read_bytes()
    # Preserve previously sealed receipts and frozen sources. Selected frontend
    # files have deliberately changed; the event seal is historical for those.
    for pathname, expected in seal["bindings"].items():
        if pathname in DOCUMENTS or pathname.startswith("frontend/"):
            continue
        track(resolve(pathname), expected)
    backend = load(EVENT / "backend_final/receipt.json")
    require(backend["passed"] is True and backend["exitCode"] == 0 and backend["sourcesUnchanged"] is True
            and backend["counts"]["tests"] == seal["backendTestsPassed"]
            and all(backend["counts"][key] == 0 for key in ("failures", "errors", "skipped")),
            "Unchanged prior backend suite is not applicable")
    backend_bindings = load(EVENT / "backend_final/source_bindings.json")
    verify_map(backend_bindings)
    require({name(path) for folder in (ROOT / "src", ROOT / "tests") for path in folder.rglob("*.py")}
            .issubset(backend_bindings), "Backend source/test inventory changed since independent full suite")
    manifest = json.loads(before[DOCUMENTS[2]].decode("utf-8"))
    require(manifest["productionQualified"] is False and "frontendEntryRecovery" not in manifest["validation"],
            "Current manifest qualification changed or unit already documented")
    for item in manifest["assets"]:
        if item["path"] in ("frontend/package.json", "frontend/pnpm-lock.yaml"):
            track(ROOT / item["path"], {"bytes": item["bytes"], "sha256": item["sha256"]})
    return seal, before, manifest, backend


def focused_gate():
    red = load(UNIT / "browser_layout_RED.json")
    require(red["status"] == "RED" and red["initialActualAnalyzedVideoVisible"] is False
            and red["persistentUploadHeaderVisibleAfterScroll"] is False
            and red["inferenceRun"] is False and red["qualificationEvidence"] is False
            and red["browserForcedAutoplayOrGPUFailureTested"] is False
            and red["eventUnitSealSha256"] == EVENT_SEAL_SHA256, "Actual browser RED changed")
    preservation = load(UNIT / "homepage_history/preservation.json")
    implementation = load(UNIT / "homepage_history/implementation.json")
    track(ROOT / implementation["preservationReceipt"], implementation["preservationReceiptSha256"])
    for pathname, value in preservation["files"].items():
        track(ROOT / value["preservedPath"], {"bytes": value["bytes"], "sha256": value["preservedSha256"]})
    verify_map(implementation["currentSources"])
    focused = implementation["focusedTests"]
    xml_path = ROOT / focused["path"]
    track(xml_path, focused["sha256"])
    suites = ET.parse(xml_path).getroot()
    suite = suites if suites.tag == "testsuite" else suites.find("testsuite")
    require(suite is not None and int(suite.get("tests")) == focused["tests"] > 0
            and all(int(suite.get(key, 0)) == 0 for key in ("failures", "errors", "skipped")),
            "Homepage focused checks did not pass")
    # Initial proof is retained against its own frozen snapshots, even if a
    # later edge-case correction adds tests and changes current Upload source.
    historical = []
    for phase in ("red", "green"):
        folder = UNIT / "upload_history" / phase
        receipt = load(folder / "receipt.json")
        result = load(folder / "vitest.json")
        require(receipt["total"] == result["numTotalTests"] and receipt["passed"] == result["numPassedTests"]
                and receipt["failed"] == result["numFailedTests"], "Retained upload RED/GREEN summary differs")
        if phase == "red":
            require(receipt["status"] == "GENUINE_RED" and receipt["failed"] > 0 and receipt["sourceUnchangedFromBefore"] is True,
                    "Upload initial RED is not genuine")
        else:
            require(receipt["status"] == "GREEN" and receipt["failed"] == 0 and receipt["passed"] == receipt["total"],
                    "Upload initial GREEN did not pass")
        for record in receipt["files"]:
            pathname = record["path"]
            if pathname.startswith("frontend/"):
                track(folder / "source" / pathname, {"bytes": record["bytes"], "sha256": record["sha256"]})
            else:
                track(ROOT / pathname, {"bytes": record["bytes"], "sha256": record["sha256"]})
        historical.append({"phase": phase, "total": receipt["total"], "passed": receipt["passed"], "failed": receipt["failed"]})
    final = {}
    for phase in ("last_good_red", "last_good_green"):
        folder = UNIT / "upload_history" / phase
        receipt = load(folder / "receipt.json")
        result = load(folder / "vitest.json")
        require(receipt["total"] == result["numTotalTests"] and receipt["passed"] == result["numPassedTests"]
                and receipt["failed"] == result["numFailedTests"], "Final known-exclusion test counts differ")
        is_green = phase.endswith("green")
        require(receipt["status"] == ("GREEN" if is_green else "GENUINE_RED")
                and receipt["exitCode"] == (0 if is_green else 1)
                and (receipt["passed"] == receipt["total"] and receipt["failed"] == 0 if is_green else receipt["failed"] == 1),
                "Final known-exclusion RED/GREEN failed to demonstrate the regression")
        for record in receipt["files"]:
            pathname = record["path"]
            expected = {"bytes": record["bytes"], "sha256": record["sha256"]}
            track(folder / "source" / pathname if pathname.startswith("frontend/") else ROOT / pathname, expected)
            if is_green and pathname.startswith("frontend/"):
                track(ROOT / pathname, expected)
        final[phase] = {"total": receipt["total"], "passed": receipt["passed"], "failed": receipt["failed"]}
    exact(UNIT / "upload_history/last_good_red/source/frontend/src/pages/Upload.test.tsx",
          UNIT / "upload_history/last_good_green/source/frontend/src/pages/Upload.test.tsx")
    for path in (UNIT / "upload_history").rglob("*"):
        if path.is_file():
            track(path)
    return {"homepageFocusedPassed": focused["tests"], "initialUploadRegression": historical,
            "lastKnownExclusionRegression": final}


def restore_gate():
    report = load(RESTORE / "receipt.json")
    require(report["passed"] is True and report["freshTarget"] is True and report["offlineInstall"] is True
            and report["frozenLockfile"] is True and report["qualificationEvidence"] is False
            and report["accuracyValidated"] is False and report["inputDifferences"] == []
            and report["boundSelectedInputsUnchanged"] is True and report["boundCopiedInputsUnchanged"] is True,
            "Complete current frontend restore has not passed unchanged")
    target = ROOT / report["target"]
    require(target.resolve() == (RESTORE / "frontend").resolve(), "Unexpected frontend restore target")
    require(report["copiedFrontendFiles"] == len(report["sourceBindings"])
            and report["copiedFiles"] == len(report["sourceBindings"]) + len(report["sharedBindings"])
            and set(report["sharedBindings"]) == set(SHARED_FILES), "Frontend copied counts/scope differ")
    for folder in (ROOT / "frontend", target):
        paths = frontend_inputs(folder)
        require(set(paths) == set(report["sourceBindings"]), "Current/copied full frontend input inventory differs")
        for pathname, digest in report["sourceBindings"].items():
            track(paths[pathname], digest)
    for pathname in report["sourceBindings"]:
        exact(ROOT / "frontend" / pathname, target / pathname)
    for pathname, digest in report["sharedBindings"].items():
        track(ROOT / pathname, digest)
        track(RESTORE / pathname, digest)
        exact(ROOT / pathname, RESTORE / pathname)
    track(RESTORE / "verifier.py", report["verifierSha256"])
    track(ROOT / "scripts/setup/verify_frontend_source_restore.py", report["verifierSha256"])
    stages = report["stages"]
    require([item["name"] for item in stages] == ["node_version", "pnpm_version", "install", "lint", "build", "tests"],
            "Incomplete frontend stage list")
    for item in stages:
        require(item["passed"] is True and item["returncode"] == 0 and item.get("timedOut", False) is False,
                "Frontend install/lint/build/full tests did not pass")
        track(ROOT / item["stdout"])
        track(ROOT / item["stderr"])
    require(stages[2]["command"][1:] == ["install", "--frozen-lockfile", "--offline"], "Install flags changed")
    summary = report["testSummary"]
    results = load(RESTORE / "tests.json", summary["receiptSha256"])
    expected = {str((target / pathname).resolve()).casefold() for pathname in report["sourceBindings"] if re.search(r"\.test\.tsx?$", pathname)}
    actual = {str(Path(item["name"]).resolve()).casefold() for item in results["testResults"]}
    require(summary["success"] is True and summary["allCopiedTestFilesExecuted"] is True
            and summary["total"] == summary["passed"] == results["numTotalTests"] == results["numPassedTests"] > 0
            and summary["failed"] == results["numFailedTests"] == 0 and results["success"] is True
            and summary["expectedFiles"] == summary["executedFiles"] == len(actual)
            and expected == actual and results.get("numPendingTests", 0) == 0
            and results.get("numTodoTests", 0) == 0 and results.get("numFailedTestSuites", 0) == 0
            and results.get("numPendingTestSuites", 0) == 0, "Complete actual frontend test set did not all pass")
    assertions = [assertion for item in results["testResults"] for assertion in item["assertionResults"]]
    require(len(assertions) == summary["total"] and all(item["status"] == "passed" for item in assertions),
            "Actual test assertion summary differs")
    require("index.html" in report["buildBindings"], "Full isolated build has no index")
    require(set(inventory(target / "dist")) == set(report["buildBindings"]), "Isolated production file inventory differs")
    for pathname, digest in report["buildBindings"].items():
        track(target / "dist" / pathname, digest)
    return report


def publication_gate(restore, seal):
    receipt = load(UNIT / "publish_receipt.json")
    track(UNIT / "publish_frontend.py", receipt["helperSha256"])
    require(receipt["passed"] is True and receipt["frontendRestore"]["path"] == name(RESTORE / "receipt.json")
            and receipt["frontendRestore"]["sha256"] == track(RESTORE / "receipt.json")["sha256"],
            "Current build publication does not bind the fresh frontend restore")
    require(receipt["checks"] and all(value is True for value in receipt["checks"].values()), "Publication checks did not all pass")
    require(receipt["snapshotBefore"] == receipt["snapshotAfter"]
            and receipt["runtimeBefore"] == receipt["runtimeAfter"], "Build publication changed runtime/product state")
    expected_inputs = {"frontend/" + pathname: digest for pathname, digest in restore["sourceBindings"].items()}
    expected_inputs.update(restore["sharedBindings"])
    require(receipt["frontendInputs"] == expected_inputs, "Publication frontend input binding map differs")
    verify_map(receipt["frontendInputs"])
    verify_map(receipt["backendBindings"])
    old = ROOT / receipt["previousBuildPath"]
    require(old.resolve().is_relative_to(UNIT) and old.is_dir(), "Previous build was not preserved within this unit")
    require(set(inventory(old)) == set(receipt["buildBefore"]), "Preserved earlier build inventory changed")
    for pathname, digest in receipt["buildBefore"].items():
        track(old / pathname, digest)
        require(seal["bindings"]["frontend/dist/" + pathname]["sha256"] == digest,
                "Preserved build differs from the previously sealed event frontend")
    require(receipt["buildAfter"] == restore["buildBindings"], "Published build differs from isolated restore")
    require(set(inventory(ROOT / "frontend/dist")) == set(receipt["buildAfter"]), "Current published build inventory differs")
    for pathname, digest in receipt["buildAfter"].items():
        track(ROOT / "frontend/dist" / pathname, digest)
        exact(ROOT / "frontend/dist" / pathname, RESTORE / "frontend/dist" / pathname)
    return receipt


def browser_gate(publication):
    receipt = load(UNIT / "browser_layout_GREEN.json")
    require(receipt["passed"] is True and receipt["publishReceiptSha256"] == track(UNIT / "publish_receipt.json")["sha256"],
            "Current browser GREEN is not bound to the published build")
    verify_map(receipt["sourceBindings"])
    require(receipt["publishedBuildBindings"] == publication["buildAfter"], "Browser observed another build")
    for key in ("desktopInitial", "desktopScrolled", "mobileInitial", "mobileScrolled", "mobileNavigation", "desktopUpload"):
        require(isinstance(receipt[key], dict) and receipt[key], "Missing actual browser observation: " + key)
    expected_assertions = {"desktopThreeDistinctAnalyzedClipsVisible", "desktopPreviewsDecodedAndPlaying",
                           "desktopUploadHeaderPersistent", "mobileFeaturedPreviewVisible",
                           "mobileUploadHeaderPersistent", "mobileMenuWorks", "uploadSavedHistoryVisible", "creditsPreserved"}
    require(expected_assertions == set(receipt.get("layoutAssertions", {}))
            and all(value is True for value in receipt["layoutAssertions"].values()),
            "Missing explicit actual desktop/mobile/header/history layout assertions")
    desktop = receipt["desktopInitial"]
    require(desktop["viewport"] == {"width": 1280, "height": 720} and desktop["scrollY"] == 0
            and len(desktop["videos"]) == 3 and len({video["source"] for video in desktop["videos"]}) == 3,
            "Actual initial desktop observation differs")
    for video in desktop["videos"]:
        require(video["top"] >= desktop["header"]["bottom"] and video["bottom"] <= desktop["viewport"]["height"]
                and video["x"] >= 0 and video["x"] + video["width"] <= desktop["viewport"]["width"]
                and video["readyState"] == 4 and video["videoWidth"] > 0 and video["videoHeight"] > 0
                and video["paused"] is False and video["muted"] is True,
                "Actual desktop media was not fully visible, decoded and playing")
    for key in ("desktopScrolled", "mobileScrolled"):
        observation = receipt[key]
        require(observation["scrollY"] > 0 and observation["header"]["position"] == "sticky"
                and abs(observation["header"]["top"]) < .01 and observation["uploadLink"]["top"] >= 0
                and observation["uploadLink"]["bottom"] <= observation["viewport"]["height"],
                "Actual scrolled upload header was not persistent")
    mobile = receipt["mobileInitial"]
    featured = mobile["videos"][0]
    require(mobile["viewport"] == {"width": 390, "height": 844} and mobile["scrollY"] == 0
            and featured["top"] >= mobile["header"]["bottom"] and featured["bottom"] <= 844
            and mobile["documentWidth"] <= 390 and featured["readyState"] == 4 and featured["paused"] is False,
            "Actual mobile featured preview was not visible and playing")
    menu = receipt["mobileNavigation"]
    require(menu["opened"]["menuExpanded"] == "true" and menu["opened"]["navVisible"] is True
            and menu["afterLink"]["url"].endswith("/examples") and menu["afterLink"]["menuExpanded"] == "false"
            and menu["afterLink"]["navVisible"] is False, "Mobile menu/link observation differs")
    upload = receipt["desktopUpload"]
    require(upload["url"].endswith("/upload") and upload["filePicker"] is True and upload["submitDisabled"] is True
            and len(upload["historyRows"]) == 7
            and any(link["text"] == "View" for row in upload["historyRows"] for link in row["links"])
            and any("Details" in row["buttons"] for row in upload["historyRows"]), "Saved history observation differs")
    require(len(desktop["credits"]) == 3 and all(len(credit["links"]) == 2 and "Annotated version:" in credit["text"]
            for credit in desktop["credits"]), "Actual media credits missing")
    for pathname, expected in receipt["screenshots"].items():
        track(UNIT / pathname, expected)
    require(receipt["viewportOverrideReset"] is True and receipt["browserCatalogFailureInjected"] is False,
            "Browser verification scope/reset differs")
    require(receipt.get("inferenceRun") is False and receipt.get("qualificationEvidence") is False
            and receipt.get("browserForcedAutoplayOrGPUFailureTested") is False,
            "Browser evidence scope implies unsupported scientific/GPU/autoplay qualification")
    return receipt


def runtime_gate(publication):
    receipt = load(UNIT / "runtime_verification.json")
    preflight = load(UNIT / "runtime_preflight.json", receipt["preflightSha256"])
    track(UNIT / "verify_runtime.py", receipt["scriptSha256"])
    require(receipt["passed"] is True and receipt["runtimeRetainedWithoutRestart"] is True
            and receipt["savedJobsAndAnalysisFilesUnchanged"] is True and receipt["productStateChanged"] is False
            and receipt["snapshotBefore"] == receipt["snapshotAfter"] == publication["snapshotAfter"],
            "Current existing-result runtime check changed state or restarted runtime")
    require(preflight["snapshot"] == receipt["snapshotBefore"]
            and preflight["sourceBindings"] == receipt["sourceBindings"]
            and preflight["runtime"] == receipt["runtime"] == publication["runtimeAfter"]
            and preflight["productStateChanged"] is False and preflight["acceptedWorkCancelled"] is False
            and preflight["runtimeStoppedOrStarted"] is False
            and receipt["runtimeStoppedOrStartedByThisVerifier"] is False,
            "Retained exact runtime/preflight/source continuity differs")
    verify_map(receipt["sourceBindings"]["files"])
    track(receipt["sourceBindings"]["baseInterpreter"]["path"], receipt["sourceBindings"]["baseInterpreter"]["sha256"])
    required_sources = {name(path) for path in (ROOT / "src").rglob("*.py")}
    required_sources |= {name(path) for path in (ROOT / "frontend/dist").rglob("*") if path.is_file()}
    require(required_sources.issubset(receipt["sourceBindings"]["files"]), "Runtime omitted a current source/build file")
    require(receipt["savedJobs"] == len(receipt["snapshotBefore"]["jobs"]) == 10
            and receipt["savedAnalysisFiles"] == len(receipt["snapshotBefore"]["savedAnalysisFiles"]) == 319
            and receipt["catalogEntries"] == 3 and len(receipt["exports"]) == 12
            and len(receipt["frontend"]["assets"]) == 7, "Existing-result runtime inventory differs")
    verify_map(receipt["snapshotAfter"]["savedAnalysisFiles"])
    require({name(path) for path in (ROOT / "outputs/product/analyses").rglob("*") if path.is_file()}
            == set(receipt["snapshotAfter"]["savedAnalysisFiles"]), "Saved analysis inventory changed")
    require(receipt["health"]["status"] == "ok" and receipt["health"]["worker_available"] is True
            and all(item["status"] == 200 for item in receipt["exports"] + receipt["frontend"]["assets"]),
            "HTTP existing-result/frontend compatibility failed")
    return receipt


def build_documents(before, manifest, backend, focused, restore, browser, runtime, at):
    counts = restore["testSummary"]
    note = (f"Current frontend entry recovery (6 October): the upload/navigation header stays visible while "
            f"scrolling, three real analyzed Commons previews appear in the first desktop viewport, and saved "
            f"Upload history remains visible while the public demo catalog is loading or fails. Known public "
            f"demo exclusions remain retained across failed or malformed later responses. Genuine browser "
            f"and component RED evidence is preserved. The fresh [frozen offline frontend restore]"
            f"(../../{name(RESTORE / 'receipt.json')}) passes lint/build and all {counts['passed']} tests in "
            f"{counts['executedFiles']} copied test files, binds {restore['copiedFiles']} current inputs, and "
            f"publishes {len(restore['buildBindings'])} exact isolated build files with the earlier build "
            f"preserved. [Actual browser observations](../../{name(UNIT / 'browser_layout_GREEN.json')}) "
            f"cover the desktop first screen, scrolled persistent header, mobile featured preview and navigation, "
            f"and saved history. They do not require all three stacked mobile clips on the first screen. "
            f"[Retained runtime checks](../../{name(UNIT / 'runtime_verification.json')}) preserve "
            f"{runtime['savedJobs']} jobs/{runtime['savedAnalysisFiles']} files, {len(runtime['exports'])} "
            f"exact existing demo exports and {len(runtime['frontend']['assets'])} index assets without restart. "
            f"The event seal is historical for frontend source/build metadata after this repair; its independent "
            f"{backend['counts']['tests']:,}-test backend suite remains applicable to unchanged bound sources. "
            f"Package/lock asset hashes remain the validated event-dependency values. The fresh clean-host "
            f"result for this document revision is recorded separately in the [host check]"
            f"(../../{name(UNIT / 'verify_installation.json')}); earlier checks are historical. "
            f"No backend algorithm, threshold, selected model or scientific label changes; no model inference, "
            f"full-match, physics or accuracy qualification, forced-autoplay or GPU-failure measurement.\n\n")
    output = {}
    for pathname in DOCUMENTS[:2]:
        old = before[pathname]
        newline = b"\r\n" if b"\r\n" in old[:200] else b"\n"
        insertion = note.replace("\n", newline.decode()).encode("utf-8")
        anchor = b"## Resume here" + newline + newline if pathname == DOCUMENTS[0] else old.split(newline, 1)[0] + newline + newline
        require(old.count(anchor) == 1, "Document insertion anchor changed")
        updated = old.replace(anchor, anchor + insertion, 1)
        require(updated.replace(insertion, b"", 1) == old, "Earlier document bytes changed")
        output[pathname] = updated
    updated = deepcopy(manifest)
    updated["frontendEntryEvidenceUpdatedAt"] = at
    updated["validation"]["frontendEntryRecovery"] = (
        f"PASSED_SOFTWARE_BROWSER_SCOPE_2026-10-06: sticky upload/header, three actual analyzed clips in the "
        f"first desktop viewport, saved Upload history independent of catalog loading/failure and retained "
        f"known exclusions. Fresh frozen offline restore: {restore['copiedFiles']} inputs, "
        f"{restore['packageStoreDirectories']} cached packages, lint/build and all {counts['passed']} tests "
        f"across {counts['executedFiles']} files; {len(restore['buildBindings'])} exact published build files, "
        f"previous build preserved. Actual desktop/mobile/header/history observations; 10 saved jobs, "
        f"319 files, 12 exports and 7 index assets unchanged; runtime retained without restart. "
        f"Earlier event seal is historical for frontend metadata only; independent backend "
        f"{backend['counts']['tests']} tests still pass on unchanged bound sources. The fresh clean-host "
        f"result for this document revision is recorded separately at {name(UNIT / 'verify_installation.json')}; "
        f"earlier checks are historical. No GPU, forced-autoplay, model inference, scientific "
        f"accuracy or full-match qualification.")
    updated["validation"]["frontendEntryCleanHostStatus"] = (
        "FRESH_CHECK_RECORDED_SEPARATELY: " + name(UNIT / "verify_installation.json") +
        "; this metadata does not assert a passing outcome before that separate receipt exists; earlier checks remain historical.")
    for path, passed, scope in (
        (UNIT / "browser_layout_RED.json", False, "Actual prior first-screen and scrolled-header RED; retained unchanged."),
        (UNIT / "browser_layout_GREEN.json", True, "Actual published desktop/mobile/header/history observations; no forced-autoplay/GPU/scientific qualification."),
        (RESTORE / "receipt.json", True, "Complete copied frontend frozen offline restore/lint/build/tests; dynamic actual counts."),
        (UNIT / "publish_receipt.json", True, "Exact isolated production publication, preserved prior build, unchanged backend/runtime/product state."),
        (UNIT / "runtime_verification.json", True, "Existing-result/export/frontend compatibility; 10 rows and 319 files retained without restart."),
    ):
        updated["validationEvidence"].append({"path": name(path), "verifiedAt": at, "passed": passed,
                                              "scope": scope, "sha256": track(path)["sha256"], "qualificationEvidence": False})
    for key, value in manifest.items():
        if key == "validation":
            require(all(updated[key][entry] == old for entry, old in value.items()), "Earlier validation changed")
        elif key == "validationEvidence":
            require(updated[key][:len(value)] == value, "Earlier evidence records changed")
        else:
            require(updated[key] == value, "Earlier manifest field changed: " + key)
    require(updated["productionQualified"] is False and updated["assets"] == manifest["assets"], "Asset or qualification records changed")
    output[DOCUMENTS[2]] = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return output


def unchanged(exclude_documents=False):
    for pathname, expected in TRACKED.items():
        if exclude_documents and pathname in DOCUMENTS:
            continue
        require(binding(resolve(pathname)) == expected, "Bound input changed: " + pathname)


def exclusive_bytes(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def save_receipt(stream, value):
    stream.seek(0)
    stream.truncate()
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())


def input_maps():
    values = {path: value for path, value in TRACKED.items() if path not in DOCUMENTS}
    sources = {path: value for path, value in values.items() if Path(path).suffix.lower() in
               (".py", ".pyd", ".exe", ".dll", ".ts", ".tsx", ".js", ".mjs", ".css", ".html")}
    evidence = {path: value for path, value in values.items() if path not in sources}
    return sources, evidence


def publish(before, output, summary):
    require(not any(path.exists() for path in (RECEIPT, BACKUP, STAGING)), "Exclusive publication already reserved; no retry")
    unchanged()
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        receipt = {"startedAt": stamp(), "passed": False, "status": "RESERVED", "helperBinding": binding(Path(__file__)),
                   "before": {path: binding(ROOT / path) for path in DOCUMENTS}, "sourceBindings": {}, "evidenceBindings": {},
                   "currentDocumentHashes": {}, "derivedEvidence": summary, "productionQualified": False,
                   "qualificationEvidence": False, "workloadsStarted": False, "modelsImported": False,
                   "serverActions": False, "productStateChanged": False,
                   "freshCleanHostStatus": "RECORDED_SEPARATELY_NO_PASS_ASSERTION",
                   "freshCleanHostReceiptPath": name(UNIT / "verify_installation.json"),
                   "rollback": {"attempted": False, "passed": None}}
        save_receipt(stream, receipt)
        replaced = []
        try:
            BACKUP.mkdir(exist_ok=False)
            STAGING.mkdir(exist_ok=False)
            backup_records = {}
            for source in [ROOT / path for path in DOCUMENTS] + [EVENT_SEAL]:
                target = BACKUP / source.relative_to(ROOT)
                exclusive_bytes(target, source.read_bytes())
                exact(source, target)
                backup_records[name(source)] = {"path": name(target), **track(target)}
            receipt["preservedBeforeFiles"] = backup_records
            receipt["beforeDocumentBackupBindings"] = {path: backup_records[path] for path in DOCUMENTS}
            receipt["historicalEventSealPreservedAs"] = backup_records[name(EVENT_SEAL)]
            staged, rollback = {}, {}
            for index, pathname in enumerate(DOCUMENTS):
                staged[pathname] = STAGING / f"{index}_after.bin"
                rollback[pathname] = STAGING / f"{index}_before.bin"
                exclusive_bytes(staged[pathname], output[pathname])
                exclusive_bytes(rollback[pathname], before[pathname])
            receipt["stagedAfter"] = {path: binding(staged[path]) for path in DOCUMENTS}
            receipt["sourceBindings"], receipt["evidenceBindings"] = input_maps()
            save_receipt(stream, receipt)
            unchanged()
            for pathname in DOCUMENTS:
                require((ROOT / pathname).read_bytes() == before[pathname], "Document changed before atomic replace")
                os.replace(staged[pathname], ROOT / pathname)
                replaced.append(pathname)
                require((ROOT / pathname).read_bytes() == output[pathname], "Document replacement readback differs")
            unchanged(exclude_documents=True)
            require(all((ROOT / path).read_bytes() == output[path] for path in DOCUMENTS), "Final readback differs")
            receipt["currentDocumentHashes"] = {path: binding(ROOT / path) for path in DOCUMENTS}
            require(receipt["currentDocumentHashes"] == receipt["stagedAfter"], "Final/staged hashes differ")
            receipt["after"] = deepcopy(receipt["currentDocumentHashes"])
            receipt["olderManifestRecordsUnchanged"] = True
            receipt["assetRecordsUnchanged"] = True
            receipt["immutableEvidenceUnchanged"] = True
            receipt["passed"] = True
            receipt["status"] = "COMPLETED"
        except BaseException as error:
            receipt["status"] = "FAILED"
            receipt["error"] = type(error).__name__ + ": " + str(error)
            receipt["rollback"]["attempted"] = bool(replaced)
            try:
                for pathname in reversed(replaced):
                    require((ROOT / pathname).read_bytes() == output[pathname], "Concurrent edit prevents rollback")
                    os.replace(rollback[pathname], ROOT / pathname)
                require(all((ROOT / path).read_bytes() == before[path] for path in DOCUMENTS), "Exact rollback failed")
                receipt["rollback"]["passed"] = True
            except BaseException as rollback_error:
                receipt["rollback"]["passed"] = False
                receipt["rollback"]["error"] = type(rollback_error).__name__ + ": " + str(rollback_error)
        finally:
            receipt["completedAt"] = stamp()
            save_receipt(stream, receipt)
    require(receipt["passed"], "Publication failed; exclusive receipt/backups retained")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    require(not any(path.exists() for path in (RECEIPT, BACKUP, STAGING)), "Publication already reserved; no retry")
    track(Path(__file__))
    seal, before, manifest, backend = previous_gate()
    focused = focused_gate()
    restore = restore_gate()
    publication = publication_gate(restore, seal)
    browser = browser_gate(publication)
    runtime = runtime_gate(publication)
    output = build_documents(before, manifest, backend, focused, restore, browser, runtime, stamp())
    unchanged()
    summary = {"frontendTestSummary": restore["testSummary"], "frontendInputFiles": restore["copiedFiles"],
               "productionBuildFiles": len(restore["buildBindings"]), "previousBackendCounts": backend["counts"],
               "focused": focused, "savedJobs": runtime["savedJobs"], "savedAnalysisFiles": runtime["savedAnalysisFiles"],
               "exactExports": len(runtime["exports"]), "exactIndexAssets": len(runtime["frontend"]["assets"]),
               "freshCleanHostStatus": "RECORDED_SEPARATELY_NO_PASS_ASSERTION",
               "freshCleanHostReceiptPath": name(UNIT / "verify_installation.json"),
               "plannedAfter": {path: {"bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest()} for path, value in output.items()}}
    if args.validate_only:
        print(json.dumps({"passed": True, "mode": "READ_ONLY_VALIDATION", "filesWritten": 0, "derivedEvidence": summary}, allow_nan=False))
        return 0
    receipt = publish(before, output, summary)
    print(json.dumps({"passed": True, "receipt": name(RECEIPT), "currentDocumentHashes": receipt["currentDocumentHashes"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
