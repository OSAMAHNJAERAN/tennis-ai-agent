"""Prepare-only scoped frontend installation metadata correction.

Root must dispatch --validate-only, then explicitly --apply after review. No
project imports, subprocess, model/workload, HTTP, SQLite or server actions.
Only saved evidence is read and the three reviewed documents are replaced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RESTORE = ROOT / "artifacts/validation/product_installation/frontend_source_restore_event_dependencies_2026-10-06"
FIRST_DOCUMENTATION = UNIT / "documentation_receipt.json"
FIRST_HOST_RECEIPT = UNIT / "verify_installation.json"
FIRST_HOST_WRAPPER = UNIT / "verify_installation.py"
FIRST_HOST_LOG = UNIT / "verify_installation.log"
FIRST_HOST_REPORT = ROOT / "artifacts/validation/product_installation/event_timestamp_lookup_host_verification_2026-10-06.json"
RECEIPT = UNIT / "installation_correction_receipt.json"
BACKUP = UNIT / "installation_correction_backup"
STAGING = UNIT / "installation_correction_staging"
DOCUMENTS = (
    "docs/plans/TENNIS_EXECUTION_PROGRESS.md",
    "docs/deployment/PRODUCT_INSTALLATION.md",
    "docs/deployment/PRODUCT_RESTORE_MANIFEST.json",
)
ASSETS = ("frontend/package.json", "frontend/pnpm-lock.yaml")
FRONTEND_ROOT_FILES = ("package.json", "pnpm-lock.yaml", "index.html", "vite.config.ts",
                       "eslint.config.js", "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json")
FRONTEND_DIRS = ("src", "public", "scripts")
SHARED_FILES = ("theme.css", "variables.css")
FIRST_DOCUMENTATION_SHA256 = "b693af89f924e581525712fe483172a2aa11140d1075067a862335946b606506"
FIRST_HOST_RECEIPT_SHA256 = "a2d981973141d12d08d589ae89857c6a948585b9398e472251e68fbb5c74a360"
FRONTEND_RESTORE_RECEIPT_SHA256 = "6eb76a2beb7ff6bedc744a0df642c8f0a1528abb4693bf552e3c78e0091c46ba"
RESTORE_VERIFICATION = UNIT / "frontend_dependency_restore_verification.json"
RESTORE_VERIFICATION_SHA256 = "3c1cb4b92af4ecab8a497dee1ce766076a53752f81af7cee6a8d6ae947e63b94"
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
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def track(path, expected=None):
    path = Path(path).resolve()
    actual = binding(path)
    if isinstance(expected, str):
        require(actual["sha256"] == expected, "SHA-256 mismatch: " + name(path))
    elif expected is not None:
        require(actual == expected, "File binding mismatch: " + name(path))
    key = name(path)
    require(key not in TRACKED or TRACKED[key] == actual, "Bound evidence changed while reading: " + key)
    TRACKED[key] = actual
    return actual


def load(path, expected=None):
    track(path, expected)
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def exact(left, right):
    track(left)
    track(right)
    with Path(left).open("rb") as first, Path(right).open("rb") as second:
        while True:
            a, b = first.read(65536), second.read(65536)
            require(a == b, "Exact bytes differ: " + name(left) + " / " + name(right))
            if not a:
                break


def frontend_inputs(folder):
    paths = [folder / leaf for leaf in FRONTEND_ROOT_FILES]
    if (folder / ".npmrc").exists():
        paths.append(folder / ".npmrc")
    for directory in FRONTEND_DIRS:
        paths += [path for path in (folder / directory).rglob("*") if path.is_file()]
    for path in paths:
        require(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()),
                "Frontend input escapes reviewed directory: " + str(path))
    return {path.relative_to(folder).as_posix(): path for path in paths}


def inventory(folder):
    return {path.relative_to(folder).as_posix(): path for path in folder.rglob("*") if path.is_file()}


def first_documentation_gate():
    first = load(FIRST_DOCUMENTATION, FIRST_DOCUMENTATION_SHA256)
    require(first["passed"] is True and first["status"] == "COMPLETED"
            and first["productionQualified"] is False and first["qualificationEvidence"] is False
            and first["allEvidenceHashesUnchanged"] is True, "First event documentation did not complete")
    require(first["currentDocumentHashes"] == first["after"]
            and set(first["currentDocumentHashes"]) == set(DOCUMENTS),
            "First event after-document hashes are incomplete")
    track(UNIT / "document_evidence.py", first["helperBinding"])
    before = {}
    for pathname in DOCUMENTS:
        track(ROOT / pathname, first["currentDocumentHashes"][pathname])
        before[pathname] = (ROOT / pathname).read_bytes()
    manifest = json.loads(before[DOCUMENTS[2]].decode("utf-8"))
    require(manifest["productionQualified"] is False, "Manifest qualification flag changed")
    require("frontendEventDependencyRestoration" not in manifest["validation"], "Correction already documented")
    return first, before, manifest


def failed_host_gate(first, manifest):
    wrapper = load(FIRST_HOST_RECEIPT, FIRST_HOST_RECEIPT_SHA256)
    require(wrapper["passed"] is False and wrapper["exitCode"] != 0
            and wrapper["inputsUnchanged"] is True and wrapper["inferenceRun"] is False
            and wrapper["checkCount"] == 75, "First event host failure classification changed")
    require(wrapper["report"] == name(FIRST_HOST_REPORT)
            and wrapper["inputs"][DOCUMENTS[2]] == first["currentDocumentHashes"][DOCUMENTS[2]]["sha256"],
            "First host failure is not bound to the published first event manifest")
    for pathname, digest in wrapper["inputs"].items():
        track(ROOT / pathname, digest)
    report = load(FIRST_HOST_REPORT, wrapper["reportSha256"])
    track(FIRST_HOST_LOG, wrapper["logSha256"])
    track(FIRST_HOST_WRAPPER, wrapper["inputs"][name(FIRST_HOST_WRAPPER)])
    require(report["checksPassed"] is False and report["manifestSha256"] == wrapper["inputs"][DOCUMENTS[2]]
            and report["modelsImported"] == [] and report["inferenceRun"] is False
            and report["productionQualified"] is False and len(report["checks"]) == 75,
            "First host report scope or count changed")
    checks = report["checks"]
    require(len({item["name"] for item in checks}) == 75, "Duplicate first host checks")
    failed = [item for item in checks if item["passed"] is False]
    require({item["name"] for item in failed} == {"asset:" + pathname for pathname in ASSETS}
            and len(failed) == 2 and sum(item["passed"] is True for item in checks) == 73,
            "First host failure is not exactly the two frontend dependency asset hashes")
    current_assets = {item["path"]: item for item in manifest["assets"] if item["path"] in ASSETS}
    require(set(current_assets) == set(ASSETS)
            and sum(item["path"] in ASSETS for item in manifest["assets"]) == 2,
            "Two existing frontend assets missing or duplicated")
    for item in failed:
        pathname = item["name"].removeprefix("asset:")
        require(item["expectedSha256"] == current_assets[pathname]["sha256"]
                and item["observedSha256"] == track(ROOT / pathname)["sha256"],
                "First host mismatch differs from reviewed original/current frontend asset")
    return wrapper, report, current_assets


def frontend_restore_gate():
    verification = load(RESTORE_VERIFICATION, RESTORE_VERIFICATION_SHA256)
    require(verification["passed"] is True and verification["frontendInputsExact"] == 127
            and verification["productionFilesExact"] == 47
            and verification["selectedAndCopiedInputsUnchanged"] is True
            and verification["selectedAndCopiedBuildsExact"] is True
            and verification["inferenceRun"] is False and verification["productionQualified"] is False
            and verification["processMemoryOrLifecycleMeasured"] is False
            and verification["browserBehaviorQualified"] is False,
            "Frontend independent source/build readback has not passed in its actual scope")
    require(verification["bindings"], "Missing independent frontend binding map")
    for pathname, expected in verification["bindings"].items():
        track(ROOT / pathname, expected)
    report = load(RESTORE / "receipt.json", FRONTEND_RESTORE_RECEIPT_SHA256)
    require(report["passed"] is True and report["completedAt"]
            and report["freshTarget"] is True and report["offlineInstall"] is True
            and report["frozenLockfile"] is True and report["qualificationEvidence"] is False
            and report["accuracyValidated"] is False and report["boundSelectedInputsUnchanged"] is True
            and report["boundCopiedInputsUnchanged"] is True and report["inputDifferences"] == [],
            "Fresh complete frontend restoration has not passed unchanged")
    target = ROOT / report["target"]
    require(target.resolve() == (RESTORE / "frontend").resolve(), "Frontend restore target differs")
    require(report["copiedFrontendFiles"] == len(report["sourceBindings"]) == 125
            and report["copiedFiles"] == 127 and set(report["sharedBindings"]) == set(SHARED_FILES),
            "Fresh frontend input count/scope differs")
    for folder in (ROOT / "frontend", target):
        paths = frontend_inputs(folder)
        require(set(paths) == set(report["sourceBindings"]), "Selected/copied frontend input inventory differs")
        for pathname, digest in report["sourceBindings"].items():
            track(paths[pathname], digest)
    for pathname in report["sourceBindings"]:
        exact(ROOT / "frontend" / pathname, target / pathname)
    for pathname, digest in report["sharedBindings"].items():
        track(ROOT / pathname, digest)
        track(RESTORE / pathname, digest)
        exact(ROOT / pathname, RESTORE / pathname)
    track(RESTORE / "verifier.py", report["verifierSha256"])
    require(report["packageStoreDirectories"] == 377, "Frozen offline package-store count differs")
    require(report["pnpmVersion"] == report["expectedPnpmVersion"]
            and report["nodeVersion"].startswith("v")
            and int(report["nodeVersion"][1:].split(".")[0]) >= 22, "Recorded frontend runtime/pnpm mismatch")
    stages = report["stages"]
    require([item["name"] for item in stages] == ["node_version", "pnpm_version", "install", "lint", "build", "tests"],
            "Fresh frontend install/lint/build/full-test stage list differs")
    for item in stages:
        require(item["returncode"] == 0 and item["passed"] is True and item.get("timedOut", False) is False
                and isinstance(item["elapsedSeconds"], (int, float)) and math.isfinite(item["elapsedSeconds"])
                and item["elapsedSeconds"] >= 0, "A frontend validation stage did not complete successfully")
        track(ROOT / item["stdout"])
        track(ROOT / item["stderr"])
    install = stages[2]["command"]
    require(install[1:] == ["install", "--frozen-lockfile", "--offline"], "Install did not retain frozen offline flags")
    summary = report["testSummary"]
    require(verification["frontendTestsPassed"] == summary["passed"]
            and verification["frontendTestFiles"] == summary["executedFiles"],
            "Independent/full-restore test summaries differ")
    results = load(RESTORE / "tests.json", summary["receiptSha256"])
    expected_names = {str((target / pathname).resolve()).casefold() for pathname in report["sourceBindings"]
                      if re.search(r"\.test\.tsx?$", pathname)}
    actual_names = {str(Path(item["name"]).resolve()).casefold() for item in results["testResults"]}
    require(summary["success"] is True and summary["allCopiedTestFilesExecuted"] is True
            and summary["total"] == summary["passed"] == results["numTotalTests"] == results["numPassedTests"]
            and summary["total"] > 0 and summary["failed"] == results["numFailedTests"] == 0
            and results["success"] is True and expected_names == actual_names
            and summary["expectedFiles"] == summary["executedFiles"] == len(actual_names),
            "Full copied frontend test summary/inventory differs")
    require(results.get("numPendingTests", 0) == 0 and results.get("numTodoTests", 0) == 0
            and results.get("numFailedTestSuites", 0) == 0 and results.get("numPendingTestSuites", 0) == 0,
            "Frontend tests contain skips/pending/todo/errors")
    assertions = [assertion for item in results["testResults"] for assertion in item["assertionResults"]]
    require(len(assertions) == summary["total"] and all(item["status"] == "passed" for item in assertions)
            and all(item["status"] == "passed" for item in results["testResults"]),
            "Actual frontend test assertions did not all pass")
    require(set(report["buildBindings"]) and len(report["buildBindings"]) == 47
            and "index.html" in report["buildBindings"], "Complete frozen frontend build inventory missing")
    for folder in (target / "dist", ROOT / "frontend/dist"):
        paths = inventory(folder)
        require(set(paths) == set(report["buildBindings"]), "Current/frozen production build inventory differs")
        for pathname, digest in report["buildBindings"].items():
            track(paths[pathname], digest)
    for pathname in report["buildBindings"]:
        exact(target / "dist" / pathname, ROOT / "frontend/dist" / pathname)
    return report


def build_documents(before, manifest, failed, restore, old_assets, at):
    updated = deepcopy(manifest)
    new_assets = {}
    for item in updated["assets"]:
        if item["path"] in ASSETS:
            value = track(ROOT / item["path"])
            require(value["sha256"] == restore["sourceBindings"][Path(item["path"]).relative_to("frontend").as_posix()],
                    "Replacement asset does not match validated frontend restore")
            item["bytes"], item["sha256"] = value["bytes"], value["sha256"]
            new_assets[item["path"]] = deepcopy(item)
    require(set(new_assets) == set(ASSETS), "Only the two frontend assets may be corrected")
    updated["frontendEventDependencyEvidenceUpdatedAt"] = at
    summary = restore["testSummary"]
    updated["validation"]["frontendEventDependencyRestoration"] = (
        f"PASSED_SOFTWARE_ONLY_2026-10-06: fresh frozen offline restore of "
        f"{restore['copiedFrontendFiles']} current frontend and {len(restore['sharedBindings'])} shared inputs; "
        f"{restore['packageStoreDirectories']} cached package-store directories; lint/build and all "
        f"{summary['passed']} tests across {summary['executedFiles']} complete copied test files passed. "
        f"All {len(restore['buildBindings'])} isolated production files match current dist byte for byte; "
        "source/copy/shared/build hashes and complete inventories rechecked. This binds the concurrent "
        "frontend revision, not the earlier124-input/37-build revision. No inference/model promotion, "
        "new scientific accuracy, browser behavior, full-match capacity or deployment qualification.")
    updated["validation"]["eventFrontendDependencyHostCorrection"] = (
        "FIRST_HOST_CHECK_FAILED_TWO_ASSET_HASHES_FRESH_RECHECK_RECORDED_SEPARATELY: retained first event host report "
        "has exactly75checks/73passed; frontend/package.json and frontend/pnpm-lock.yaml expected hashes "
        "were stale after the concurrent frontend revision. Only these2assets now match the completely "
        "validated fresh frontend restore; original records/documents/host wrapper-report-log are preserved. "
        "No new75checkPASS is claimed until a separate fresh host receipt exists.")
    for path, passed, scope in (
        (RESTORE / "receipt.json", True, "Complete current frontend frozen offline restore, copied source/test inventory and 47 exact build files; software only."),
        (RESTORE_VERIFICATION, True, "Independent complete current/copied frontend input and build readback; 165 tests across23 files; no lifecycle/browser/scientific qualification."),
        (FIRST_HOST_REPORT, False, "First event host check: exactly two stale frontend dependency asset hashes, other73checks passed; retained unchanged."),
        (FIRST_HOST_RECEIPT, False, "Consumed first event host wrapper/manifest binding; no inference; fresh corrected-manifest host verification pending."),
    ):
        updated["validationEvidence"].append({"path": name(path), "verifiedAt": at, "passed": passed,
                                              "scope": scope, "sha256": track(path)["sha256"],
                                              "qualificationEvidence": False})
    for key, old_value in manifest.items():
        if key == "assets":
            require(len(updated[key]) == len(old_value), "Asset inventory changed")
            for old, new in zip(old_value, updated[key]):
                if old["path"] in ASSETS:
                    require({k: v for k, v in new.items() if k not in ("bytes", "sha256")}
                            == {k: v for k, v in old.items() if k not in ("bytes", "sha256")},
                            "Frontend asset fields beyond bytes/hash changed")
                else:
                    require(new == old, "Another manifest asset changed")
        elif key == "validation":
            require(all(updated[key][entry] == value for entry, value in old_value.items()), "Earlier validation changed")
        elif key == "validationEvidence":
            require(updated[key][:len(old_value)] == old_value, "Earlier evidence records changed")
        else:
            require(updated[key] == old_value, "Earlier manifest field changed: " + key)
    require(updated["productionQualified"] is False, "Qualification changed")
    note = (
        f"Current frontend dependency restoration correction (6 October): a concurrent frontend revision "
        f"changed the package manifest/lockfile and additional source inputs while the event unit was "
        f"being validated. The fresh [frozen offline restore](../../{name(RESTORE / 'receipt.json')}) "
        f"preserves all {restore['copiedFiles']} current inputs ({restore['copiedFrontendFiles']} frontend "
        f"plus {len(restore['sharedBindings'])} shared), restores {restore['packageStoreDirectories']} cached "
        f"packages, passes lint/build and all {summary['passed']} tests across {summary['executedFiles']} "
        f"complete copied test files, and reproduces all {len(restore['buildBindings'])} current production "
        f"files byte for byte. Earlier 124-input/37-build restoration records remain historical. The "
        f"[first event host check](../../{name(FIRST_HOST_REPORT)}) failed exactly the two stale dependency "
        f"asset hashes while the other 73 of 75 checks passed; its receipt, wrapper, logs and the first "
        f"event documents remain preserved. Only the manifest's package/lock asset bytes and hashes "
        f"are corrected from the validated restore. The [separate host recheck](../../{name(UNIT / 'verify_installation_final.json')}) "
        f"records the corrected manifest's result. This supplies "
        f"software restoration evidence only; no inference, model promotion, scientific accuracy, "
        f"new browser behavior or full-match qualification.\n\n")
    outputs = {}
    for pathname in DOCUMENTS[:2]:
        old = before[pathname]
        newline = b"\r\n" if b"\r\n" in old[:200] else b"\n"
        insert = note.replace("\n", newline.decode()).encode("utf-8")
        anchor = b"## Resume here" + newline + newline if pathname == DOCUMENTS[0] else old.split(newline, 1)[0] + newline + newline
        require(old.count(anchor) == 1, "Current documentation insertion anchor changed")
        new = old.replace(anchor, anchor + insert, 1)
        require(new.replace(insert, b"", 1) == old, "Earlier document bytes changed")
        outputs[pathname] = new
    outputs[DOCUMENTS[2]] = (json.dumps(updated, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    return outputs, new_assets


def unchanged(exclude_documents=False):
    for pathname, expected in TRACKED.items():
        if exclude_documents and pathname in DOCUMENTS:
            continue
        require(binding(resolved(pathname)) == expected, "Bound input changed before/after correction: " + pathname)


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
    immutable = {path: value for path, value in TRACKED.items() if path not in DOCUMENTS}
    sources = {path: value for path, value in immutable.items()
               if Path(path).suffix.lower() in (".py", ".pyd", ".dll", ".exe", ".ts", ".tsx", ".js", ".mjs", ".css", ".html")}
    evidence = {path: value for path, value in immutable.items() if path not in sources}
    require(not set(sources).intersection(evidence), "Immutable binding maps overlap")
    return sources, evidence


def publish(before, output, old_assets, new_assets, summary):
    require(not any(path.exists() for path in (RECEIPT, BACKUP, STAGING)), "Exclusive correction already reserved; no retry")
    unchanged()
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        receipt = {"startedAt": stamp(), "status": "RESERVED", "passed": False,
                   "helperBinding": binding(Path(__file__)), "before": {path: binding(ROOT / path) for path in DOCUMENTS},
                   "originalFrontendAssetRecords": old_assets, "correctedFrontendAssetRecords": new_assets,
                   "firstDocumentationReceiptBinding": binding(FIRST_DOCUMENTATION),
                   "sourceBindings": {}, "evidenceBindings": {}, "currentDocumentHashes": {},
                   "derivedEvidence": summary, "productionQualified": False, "qualificationEvidence": False,
                   "modelsImported": False, "workloadsStarted": False, "serverActions": False,
                   "productStateChanged": False, "freshHostReverification": "PENDING_NOT_CLAIMED",
                   "rollback": {"attempted": False, "passed": None}}
        save_receipt(stream, receipt)
        replaced = []
        try:
            BACKUP.mkdir(exist_ok=False)
            STAGING.mkdir(exist_ok=False)
            copies = [ROOT / path for path in DOCUMENTS] + [FIRST_DOCUMENTATION, FIRST_HOST_RECEIPT,
                       FIRST_HOST_WRAPPER, FIRST_HOST_LOG, FIRST_HOST_REPORT]
            backup_records = {}
            for source in copies:
                destination = BACKUP / source.relative_to(ROOT)
                exclusive_bytes(destination, source.read_bytes())
                exact(source, destination)
                backup_records[name(source)] = {"path": name(destination), **track(destination)}
            receipt["preservedBeforeFiles"] = backup_records
            receipt["firstDocumentationAfterPreservedAs"] = {path: backup_records[path] for path in DOCUMENTS}
            receipt["beforeBackupBindings"] = {value["path"]: {"bytes": value["bytes"], "sha256": value["sha256"]}
                                               for value in backup_records.values()}
            staged = {}
            rollback = {}
            for index, pathname in enumerate(DOCUMENTS):
                staged[pathname] = STAGING / f"{index}_after.bin"
                rollback[pathname] = STAGING / f"{index}_before.bin"
                exclusive_bytes(staged[pathname], output[pathname])
                exclusive_bytes(rollback[pathname], before[pathname])
                require((ROOT / pathname).read_bytes() == before[pathname], "Document baseline changed before correction")
            receipt["stagedAfter"] = {path: binding(staged[path]) for path in DOCUMENTS}
            receipt["sourceBindings"], receipt["evidenceBindings"] = input_maps()
            save_receipt(stream, receipt)
            unchanged()
            for pathname in DOCUMENTS:
                require((ROOT / pathname).read_bytes() == before[pathname], "Document changed before atomic correction")
                os.replace(staged[pathname], ROOT / pathname)
                replaced.append(pathname)
                require((ROOT / pathname).read_bytes() == output[pathname], "Correction replacement readback differs")
            unchanged(exclude_documents=True)
            require(all((ROOT / path).read_bytes() == output[path] for path in DOCUMENTS), "Final correction readback differs")
            receipt["currentDocumentHashes"] = {path: binding(ROOT / path) for path in DOCUMENTS}
            require(receipt["currentDocumentHashes"] == receipt["stagedAfter"], "Final/staged correction hashes differ")
            receipt["after"] = deepcopy(receipt["currentDocumentHashes"])
            receipt["onlyTwoAssetBindingsChanged"] = True
            receipt["olderManifestRecordsUnchanged"] = True
            receipt["firstDocumentationAndHostEvidencePreservedExact"] = True
            receipt["immutableEvidenceUnchanged"] = True
            receipt["passed"] = True
            receipt["status"] = "COMPLETED"
        except BaseException as error:
            receipt["status"] = "FAILED"
            receipt["error"] = type(error).__name__ + ": " + str(error)
            receipt["replacedDocumentsBeforeFailure"] = replaced[:]
            receipt["rollback"]["attempted"] = bool(replaced)
            try:
                for pathname in reversed(replaced):
                    require((ROOT / pathname).read_bytes() == output[pathname], "Concurrent edit prevents safe rollback")
                    os.replace(rollback[pathname], ROOT / pathname)
                require(all((ROOT / path).read_bytes() == before[path] for path in DOCUMENTS), "Exact correction rollback failed")
                receipt["rollback"]["passed"] = True
            except BaseException as rollback_error:
                receipt["rollback"]["passed"] = False
                receipt["rollback"]["error"] = type(rollback_error).__name__ + ": " + str(rollback_error)
        finally:
            receipt["completedAt"] = stamp()
            save_receipt(stream, receipt)
    require(receipt["passed"], "Correction failed; exclusive receipt/backups/staging retained")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true", help="Read-only prerequisites; creates no files")
    mode.add_argument("--apply", action="store_true", help="Root-authorized one-shot scoped correction")
    args = parser.parse_args()
    require(not any(path.exists() for path in (RECEIPT, BACKUP, STAGING)), "Correction already reserved; no retry")
    track(Path(__file__))
    first, before, manifest = first_documentation_gate()
    failed, host_report, old_assets = failed_host_gate(first, manifest)
    restored = frontend_restore_gate()
    output, new_assets = build_documents(before, manifest, failed, restored, old_assets, stamp())
    unchanged()
    summary = {"frontendInputs": restored["copiedFrontendFiles"], "sharedInputs": len(restored["sharedBindings"]),
               "totalCopiedInputs": restored["copiedFiles"], "frozenPackageDirectories": restored["packageStoreDirectories"],
               "completeFrontendTestSummary": restored["testSummary"], "productionBuildFilesExact": len(restored["buildBindings"]),
               "firstHostChecks": {"total": len(host_report["checks"]), "passed": 73, "failed": 2,
                                   "failedNames": [item["name"] for item in host_report["checks"] if item["passed"] is False]},
               "freshHostReverification": "PENDING_NOT_CLAIMED",
               "plannedAfter": {path: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                                for path, value in output.items()}}
    if args.validate_only:
        print(json.dumps({"passed": True, "mode": "READ_ONLY_VALIDATION", "filesWritten": 0,
                          "derivedEvidence": summary}, allow_nan=False))
        return 0
    receipt = publish(before, output, old_assets, new_assets, summary)
    print(json.dumps({"passed": True, "receipt": name(RECEIPT), "currentDocumentHashes": receipt["currentDocumentHashes"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
