"""Bounded read-only terminal receipt/identity audit; writes one owned receipt."""

import hashlib
import json
from pathlib import Path

import psutil

OWN = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retired_identity(record):
    try:
        current = psutil.Process(record["pid"])
        current_time = current.create_time()
        if current_time != record["createTimeUnixSeconds"]:
            return {"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                    "retired": True, "status": "PID now has a different creation time"}
        return {"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                "retired": not current.is_running(), "status": "still running" if current.is_running() else "retired"}
    except psutil.NoSuchProcess:
        return {"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                "retired": True, "status": "NoSuchProcess"}


def main():
    protocol = json.loads((OWN / "protocol.json").read_text(encoding="utf-8"))
    report = json.loads((OWN / "report.json").read_text(encoding="utf-8"))
    source_bindings = {relative: digest(ROOT / relative) == expected
                       for relative, expected in protocol["sourceFiles"].items()}
    summary = {"scope": protocol["scope"], "benchmarkSourceSha256": protocol["sourceFiles"]["scripts/audit/measure_candidate_storage.py"],
               "benchmarkTerminalPassed": report["passed"], "sourcesMatchFrozenProtocol": source_bindings,
               "allFourComparisonsPassed": len(report["comparisons"]) == 4 and all(row["passed"] for row in report["comparisons"]),
               "runs": [], "terminalOwnedIdentityRetirementOnly": True,
               "limitations": ["This supplemental audit checks exact remembered identities after the whole run; it does not prove every remembered descendant was checked at each successful run transition.",
                               "The frozen monitor does not put Popen fallback in finally if descendant termination raises; the controlled successful forced-stop proof does not cover that exceptional branch.",
                               "Memory/time and equivalence evidence is synthetic candidate-only; no full-pipeline or accuracy qualification."]}
    for run in report["runs"]:
        child = run["childReceipt"]
        directory = OWN / f'{run["kind"]}_{run["mode"]}_{run["frames"]}'
        stored_run = json.loads((directory / "monitor_receipt.json").read_text(encoding="utf-8"))
        stored_child = json.loads((directory / "child_receipt.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "child_identity.json").read_text(encoding="utf-8"))
        count, actual_samples, peak_rss, peak_private, actual_peak_rss = 0, 0, 0, 0, 0
        available_min, sums_valid = None, True
        with (directory / "resource_samples.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                sample = json.loads(line)
                rows = sample["processes"]
                rss = sum(row["memoryBytes"]["rss"] for row in rows)
                private = sum(row["memoryBytes"].get("private") or 0 for row in rows)
                sums_valid = sums_valid and rss == sample["ownedTreeRssBytes"] and private == sample["ownedTreePrivateBytes"]
                peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
                available_min = sample["availableRamBytes"] if available_min is None else min(available_min, sample["availableRamBytes"])
                for row in rows:
                    if row["pid"] == identity["pid"] and row["createTimeUnixSeconds"] == identity["createTimeUnixSeconds"]:
                        actual_samples += 1
                        actual_peak_rss = max(actual_peak_rss, row["memoryBytes"]["rss"])
                count += 1
        retirement = [retired_identity(row) for row in run["ownedProcessIdentities"]]
        database = child.get("databaseBeforeCleanup")
        cleanup = None
        if database is not None:
            owned_path = Path(database["ownedPath"])
            if owned_path.resolve().parent != directory.resolve():
                raise ValueError("Owned scratch database receipt escaped its run directory")
            cleanup = {"logicalDatabaseBytesBeforeCleanup": database["logicalDatabaseBytes"],
                       "allExactOwnedFilesAbsent": all(not Path(str(owned_path) + suffix).exists()
                                                      for suffix in ("", "-journal", "-wal", "-shm")),
                       "settings": database["settings"]}
        row = {"kind": run["kind"], "mode": run["mode"], "frames": run["frames"],
               "actualWorkloadPid": identity["pid"], "actualWorkloadCreateTimeUnixSeconds": identity["createTimeUnixSeconds"],
               "actualWorkloadSampledByExactIdentity": run["actualWorkloadSampledByExactIdentity"] and actual_samples > 0,
               "actualWorkloadSamples": actual_samples, "actualPeakSampledRssBytes": actual_peak_rss,
               "windowsPeakWorkingSetBytes": child["windowsPeakWorkingSetBytes"],
               "windowsPeakPrivateCommitBytes": child["windowsPeakPrivateCommitBytes"],
               "peakSampledOwnedTreeRssBytes": peak_rss, "peakSampledOwnedTreePrivateBytes": peak_private,
               "minimumSampledAvailableRamBytes": available_min, "totalChildElapsedSeconds": child["totalChildElapsedSeconds"],
               "totalMonitoredWallSeconds": run["totalMonitoredWallSeconds"],
               "sampleSumsAndReceiptPeaksMatch": sums_valid and count == run["samples"] and peak_rss == run["peakSampledOwnedTreeRssBytes"]
                                              and peak_private == run["peakSampledOwnedTreePrivateBytes"] and available_min == run["minimumSampledAvailableRamBytes"],
               "duplicateStoredReceiptsMatch": stored_run == run and stored_child == child,
               "actualIdentityMatchesTerminalReceipt": identity["pid"] == child["pid"] and identity == run["actualWorkloadIdentity"],
               "allRememberedOwnedIdentitiesRetiredAtTerminalAudit": all(item["retired"] for item in retirement),
               "ownedIdentityRetirement": retirement, "databaseCleanup": cleanup}
        row["passed"] = (row["actualWorkloadSampledByExactIdentity"] and row["sampleSumsAndReceiptPeaksMatch"]
                         and row["duplicateStoredReceiptsMatch"] and row["actualIdentityMatchesTerminalReceipt"]
                         and row["allRememberedOwnedIdentitiesRetiredAtTerminalAudit"]
                         and (cleanup is None or cleanup["allExactOwnedFilesAbsent"]))
        summary["runs"].append(row)
    summary["allEightActualWorkloadsSampledByExactIdentity"] = len(summary["runs"]) == 8 and all(row["actualWorkloadSampledByExactIdentity"] for row in summary["runs"])
    summary["allRememberedOwnedIdentitiesRetiredAtTerminalAudit"] = all(row["allRememberedOwnedIdentitiesRetiredAtTerminalAudit"] for row in summary["runs"])
    summary["passed"] = (summary["benchmarkTerminalPassed"] and summary["allFourComparisonsPassed"]
                         and summary["allEightActualWorkloadsSampledByExactIdentity"] and all(source_bindings.values())
                         and all(row["passed"] for row in summary["runs"]))
    with (OWN / "terminal_receipt_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
