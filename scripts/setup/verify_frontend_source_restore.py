"""Restore the current frontend in a fresh directory and verify its software checks.

Uses the existing pnpm cache with a frozen lockfile. Never replaces the selected
frontend, its dependencies, prior receipts, product storage, or model assets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "frontend"
OUTPUT_BASE = ROOT / "artifacts/validation/product_installation"
ROOT_FILES = (
    "package.json", "pnpm-lock.yaml", "index.html", "vite.config.ts",
    "eslint.config.js", "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json",
)
SOURCE_DIRS = ("src", "public", "scripts")
SHARED_FILES = ("theme.css", "variables.css")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frontend_inputs(folder: Path) -> list[Path]:
    inputs = [folder / name for name in ROOT_FILES]
    if (folder / ".npmrc").exists():
        inputs.append(folder / ".npmrc")
    for directory in SOURCE_DIRS:
        inputs.extend(path for path in (folder / directory).rglob("*") if path.is_file())
    return sorted(inputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="frontend_source_restore_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not re.fullmatch(r"frontend_source_restore_[A-Za-z0-9_-]+", args.name):
        parser.error("Use a frontend_source_restore_ name with letters, digits, underscores or hyphens.")
    base = OUTPUT_BASE.resolve()
    run_dir = (base / args.name).resolve()
    if not base.is_relative_to(ROOT) or run_dir.parent != base:
        raise ValueError("The fresh restore directory must remain in the installation evidence directory.")
    run_dir.mkdir(exist_ok=False)
    target = run_dir / "frontend"
    target.mkdir()
    receipt_path = run_dir / "receipt.json"
    report = {
        "startedAt": utc_now(), "scope": "Current frontend source and frozen offline dependency restoration",
        "passed": False, "qualificationEvidence": False, "accuracyValidated": False,
        "freshTarget": True, "target": str(target.relative_to(ROOT)),
        "offlineInstall": True, "frozenLockfile": True, "stages": [],
        "sourceBindings": {}, "sharedBindings": {}, "copiedFiles": 0,
        "boundSelectedInputsUnchanged": None,
        "inputScope": "Explicit frontend build files, src/public/scripts, and root theme.css/variables.css; excludes selected node_modules and dist",
        "invocationScope": "Commands run in the copied frontend. Model pipeline and product API/storage are outside this verifier; whole-workspace integrity is not monitored.",
        "verifierSha256": digest(Path(__file__)),
    }

    def save() -> None:
        receipt_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run_stage(name: str, command: list[str], timeout: int) -> dict:
        stage = {"name": name, "command": command, "startedAt": utc_now()}
        report["stages"].append(stage)
        save()
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        started = time.monotonic()
        stdout_path, stderr_path = run_dir / f"{name}.stdout.txt", run_dir / f"{name}.stderr.txt"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=target, stdout=stdout, stderr=stderr)
            stage["pid"] = process.pid
            save()
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Limit cleanup to the process tree launched by this stage.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=15)
                else:
                    process.kill()
                process.wait(timeout=15)
                stage["timedOut"] = True
                raise
            finally:
                stage["elapsedSeconds"] = round(time.monotonic() - started, 3)
                stage["stdout"] = str(stdout_path.relative_to(ROOT))
                stage["stderr"] = str(stderr_path.relative_to(ROOT))
                save()
        stage["returncode"] = returncode
        stage["passed"] = returncode == 0
        save()
        print(json.dumps({"stage": name, "returncode": returncode, "elapsedSeconds": stage["elapsedSeconds"]}), flush=True)
        if returncode:
            raise RuntimeError(f"{name} failed; retained output is in {run_dir}.")
        return stage

    try:
        shutil.copy2(Path(__file__), run_dir / "verifier.py")
        if digest(run_dir / "verifier.py") != report["verifierSha256"]:
            raise RuntimeError("The retained verifier snapshot differs from the executed source.")
        inputs = frontend_inputs(SOURCE)
        for source in inputs:
            if source.is_symlink() or not source.resolve().is_relative_to(SOURCE.resolve()):
                raise ValueError(f"Source input escapes the selected frontend: {source}")
            relative = source.relative_to(SOURCE)
            report["sourceBindings"][relative.as_posix()] = digest(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if digest(destination) != report["sourceBindings"][relative.as_posix()]:
                raise RuntimeError(f"Copied source differs: {relative}")
        for name in SHARED_FILES:
            source = ROOT / name
            if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
                raise ValueError(f"Shared stylesheet escapes the repository: {source}")
            report["sharedBindings"][name] = digest(source)
            shutil.copy2(source, run_dir / name)
            if digest(run_dir / name) != report["sharedBindings"][name]:
                raise RuntimeError(f"Copied shared stylesheet differs: {name}")
        report["copiedFrontendFiles"] = len(inputs)
        report["copiedFiles"] = len(inputs) + len(SHARED_FILES)
        save()
        pnpm, node = shutil.which("pnpm"), shutil.which("node")
        if not pnpm or not node:
            raise RuntimeError("Node.js and pnpm must be available on PATH.")
        run_stage("node_version", [node, "--version"], 30)
        run_stage("pnpm_version", [pnpm, "--version"], 60)
        node_version = (run_dir / "node_version.stdout.txt").read_text(encoding="utf-8").strip()
        pnpm_version = (run_dir / "pnpm_version.stdout.txt").read_text(encoding="utf-8").strip()
        expected_pnpm = json.loads((target / "package.json").read_text(encoding="utf-8"))["packageManager"].split("@", 1)[1]
        report.update(nodeVersion=node_version, pnpmVersion=pnpm_version, expectedPnpmVersion=expected_pnpm)
        if not re.fullmatch(r"v\d+\.\d+\.\d+", node_version) or int(node_version[1:].split(".")[0]) < 22:
            raise RuntimeError("The current frontend requires Node.js 22 or newer.")
        if pnpm_version != expected_pnpm:
            raise RuntimeError("The effective pnpm version does not match the copied packageManager pin.")
        if (target / "node_modules").exists() or (target / "dist").exists():
            raise RuntimeError("The dependency/build target must be fresh before installation.")
        run_stage("install", [pnpm, "install", "--frozen-lockfile", "--offline"], 300)
        store = target / "node_modules/.pnpm"
        report["packageStoreDirectories"] = sum(path.is_dir() and path.name != "node_modules" for path in store.iterdir()) if store.is_dir() else 0
        if not report["packageStoreDirectories"]:
            raise RuntimeError("Frozen install did not create the isolated package store.")
        run_stage("lint", [pnpm, "lint"], 180)
        run_stage("build", [pnpm, "build"], 300)
        results_path = run_dir / "tests.json"
        run_stage("tests", [pnpm, "exec", "vitest", "run", "--reporter=json", f"--outputFile={results_path}"], 900)
        results = json.loads(results_path.read_text(encoding="utf-8"))
        expected_test_files = {str((target / name).resolve()).casefold() for name in report["sourceBindings"] if re.search(r"\.test\.tsx?$", name)}
        actual_test_files = {str(Path(result["name"]).resolve()).casefold() for result in results["testResults"]}
        report["testSummary"] = {
            "success": results["success"], "total": results["numTotalTests"],
            "passed": results["numPassedTests"], "failed": results["numFailedTests"],
            "expectedFiles": len(expected_test_files), "executedFiles": len(actual_test_files),
            "allCopiedTestFilesExecuted": expected_test_files == actual_test_files,
            "receiptSha256": digest(results_path),
        }
        if not (results["success"] and results["numTotalTests"] > 0 and results["numPassedTests"] == results["numTotalTests"] and results["numFailedTests"] == 0 and expected_test_files == actual_test_files):
            raise RuntimeError("The complete copied frontend test set did not pass.")
        report["buildBindings"] = {path.relative_to(target / "dist").as_posix(): digest(path) for path in sorted((target / "dist").rglob("*")) if path.is_file()}
        if "index.html" not in report["buildBindings"]:
            raise RuntimeError("The isolated production build has no index.html.")
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        differences = []
        for source_root, copied_root, bindings in ((SOURCE, target, report["sourceBindings"]), (ROOT, run_dir, report["sharedBindings"])):
            for relative, expected in bindings.items():
                for location, path in (("selected", source_root / relative), ("isolated", copied_root / relative)):
                    if not path.is_file() or digest(path) != expected:
                        differences.append({"location": location, "path": relative, "kind": "changed-or-missing"})
        # Re-enumeration also catches newly added source files, not just mutations
        # of files that happened to exist when the initial copy was made.
        expected_names = set(report["sourceBindings"])
        for location, folder in (("selected", SOURCE), ("isolated", target)):
            actual_names = {path.relative_to(folder).as_posix() for path in frontend_inputs(folder)}
            for name in sorted(actual_names - expected_names):
                differences.append({"location": location, "path": name, "kind": "added"})
            for name in sorted(expected_names - actual_names):
                differences.append({"location": location, "path": name, "kind": "removed"})
        report["inputDifferences"] = differences
        report["boundSelectedInputsUnchanged"] = not any(item["location"] == "selected" for item in differences)
        report["boundCopiedInputsUnchanged"] = not any(item["location"] == "isolated" for item in differences)
        report["passed"] = report["passed"] and not differences
        report["completedAt"] = utc_now()
        save()
        print(json.dumps({"passed": report["passed"], "receipt": str(receipt_path.relative_to(ROOT)), "error": report.get("error"), "testSummary": report.get("testSummary")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
