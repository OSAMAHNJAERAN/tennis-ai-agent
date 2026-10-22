"""Isolated synthetic candidate storage and genuine tracker lifetime measurements.

This script runs no models or video inference. Candidate-only measurements and
deterministic tracker equivalence do not qualify full-pipeline memory or accuracy.
Each child is a fresh clean interpreter; only one child runs at a time.
"""

import argparse
from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import weakref

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "artifacts/validation/product_installation/clean_env/Scripts/python.exe"
BASELINE = ROOT / "artifacts/validation/product_execution/candidate_spool_baseline_2026-10-05"
OLD_TRACKER = BASELINE / "src/tracking/temporal_ball_tracker.py"
FRAME_COUNTS = (600, 18000, 432000)
TRACKER_FRAMES = 5000
SAMPLE_INTERVAL = 0.25
MAXIMUM_RSS = 1_250_000_000
MINIMUM_AVAILABLE_RAM = 400_000_000
LOW_RAM_SAMPLES = 5
MAXIMUM_WALL_SECONDS = 900
sys.path.insert(0, str(ROOT))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_bytes(value):
    # Scratch candidate JSON deliberately preserves NaN/Infinity just as the
    # sequence does. Tracker outputs must instead be finite or null.
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n").encode("utf-8")


def normalize(value):
    if isinstance(value, Enum):
        return normalize(value.value)
    if is_dataclass(value):
        return {field.name: normalize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def candidate_frame(index):
    """Lazy deterministic payload with motion, static distractors and real gaps."""
    from src.tracking.temporal_ball_tracker import BallObservation
    from src.utils.bbox_utils import BBox

    # Initial and long gaps exercise missing/backward and reacquisition paths.
    if index < 11 or index % 97 == 0 or 410 <= index % 613 <= 422:
        return []
    precise = 0.12345678901234566
    moving_x = 200.0 + (index % 173) * 8.125 + precise
    moving_y = 250.0 + (index % 71) * 2.625 + precise
    confidence = 0.027000000000000003 if 35 <= index % 43 <= 42 else 0.8700000000000001
    result = []
    coordinates = ((moving_x, moving_y, confidence),
                   (640.0000000000001, 480.00000000000006, 0.7400000000000001),
                   (1750.0 + (index % 19) * 11.125, 620.0 + (index % 17) * 3.375, 0.012000000000000002))
    for offset, (x, y, confidence) in enumerate(coordinates):
        if offset == 2:
            if index % 211 == 0:
                x = -5.000000000000001
            elif index % 223 == 0:
                x = float("nan")
            elif index % 227 == 0:
                x = float("inf")
            elif index % 229 == 0:
                x = -0.0
        box = None
        if (index + offset) % 3:
            box = BBox(x - 2.7500000000000004, y - 3.1250000000000004,
                       x + 2.8750000000000004, y + 3.2500000000000004,
                       confidence=confidence, class_id=offset,
                       track_id=None if index % 5 else index * 3 + offset)
        result.append(BallObservation(x, y, confidence, box))
    return result


def candidate_frames(count):
    for index in range(count):
        yield candidate_frame(index)


def replacement_indices(count):
    # The index set is bounded, independent of match length.
    return sorted({0, 1, count // 2, count - 1, *range(0, count, max(1, count // 113))})


def replace_sparse(sequence):
    indices = replacement_indices(len(sequence))
    for index in indices:
        replacement = candidate_frame(index + 7)
        replacement.reverse()
        sequence[index] = replacement
    return indices


@contextmanager
def candidate_storage(mode, directory):
    from src.utils.ball_candidate_sequence import BallCandidateSequence

    if mode == "old":
        yield []
    else:
        with BallCandidateSequence(directory) as sequence:
            yield sequence


def database_snapshot(sequence):
    if not hasattr(sequence, "path") or sequence.path is None:
        return None
    # Diagnostics only: querying page counts uses this connection's view, so
    # pending final records are included without changing transaction cadence.
    connection = sequence._require_open()
    page_count = connection.execute("PRAGMA page_count").fetchone()[0]
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    settings = {name: connection.execute("PRAGMA " + name).fetchone()[0]
                for name in ("cache_size", "temp_store", "mmap_size", "journal_mode")}
    return {"ownedPath": str(sequence.path), "pageCount": page_count, "pageSizeBytes": page_size,
            "logicalDatabaseBytes": page_count * page_size, "settings": settings,
            "filesBeforeSuccessfulCleanup": {
                sequence.path.name + suffix: Path(str(sequence.path) + suffix).stat().st_size
                for suffix in ("", "-journal", "-wal", "-shm")
                if Path(str(sequence.path) + suffix).is_file()},
            "logicalBytesIncludePendingConnectionPages": True}


def export_frames(sequence, destination):
    hashed, count, candidates = hashlib.sha256(), 0, 0
    with destination.open("xb") as stream:
        for index, frame in enumerate(sequence):
            encoded = canonical_bytes({"frame_index": index, "candidates": normalize(frame)})
            stream.write(encoded)
            hashed.update(encoded)
            count += 1
            candidates += len(frame)
    return {"frames": count, "candidates": candidates, "canonicalSha256": hashed.hexdigest(),
            "outputFile": destination.name, "outputBytes": destination.stat().st_size,
            "outputSha256": digest(destination)}


def read_backwards_and_neighbors(sequence, destination):
    reverse_digest, count = hashlib.sha256(), 0
    for index in range(len(sequence) - 1, -1, -1):
        reverse_digest.update(canonical_bytes({"frame_index": index, "candidates": normalize(sequence[index])}))
        count += 1
    indices = replacement_indices(len(sequence))
    neighbor_digest = hashlib.sha256()
    with destination.open("xb") as stream:
        for index in indices:
            record = {"index": index,
                      "previous": None if index == 0 else normalize(sequence[index - 1]),
                      "current": normalize(sequence[index]),
                      "next": None if index + 1 == len(sequence) else normalize(sequence[index + 1])}
            encoded = canonical_bytes(record)
            neighbor_digest.update(encoded)
            stream.write(encoded)
        # These bounded probes cover Python negative and stepped-slice semantics.
        probes = {"negativeLast": normalize(sequence[-1]), "negativeFirst": normalize(sequence[-len(sequence)]),
                  "reverseTail": normalize(sequence[-1:-7:-2]),
                  "forwardHead": normalize(sequence[:7:2]), "emptySlice": normalize(sequence[3:3])}
        encoded = canonical_bytes({"sliceProbes": probes})
        neighbor_digest.update(encoded)
        stream.write(encoded)
    return {"reverseFrames": count, "reverseCanonicalSha256": reverse_digest.hexdigest(),
            "neighborProbeCount": len(indices), "neighborCanonicalSha256": neighbor_digest.hexdigest(),
            "neighborOutputFile": destination.name, "neighborOutputBytes": destination.stat().st_size,
            "neighborOutputSha256": digest(destination)}


def stage(name):
    print(json.dumps({"at": stamp(), "stage": name}), flush=True)


def storage_child(args, report):
    with candidate_storage(args.mode, args.directory) as sequence:
        stage("append_lazy_producer")
        began = time.perf_counter()
        sequence.extend(candidate_frames(args.frames))
        report["appendElapsedSeconds"] = time.perf_counter() - began
        if len(sequence) != args.frames:
            raise ValueError("Candidate sequence length differs from declared frames")
        stage("sparse_indexed_replacements")
        began = time.perf_counter()
        indices = replace_sparse(sequence)
        report["replacementElapsedSeconds"] = time.perf_counter() - began
        report["replacementIndices"] = indices
        stage("forward_full_record_jsonl_export")
        began = time.perf_counter()
        report["export"] = export_frames(sequence, args.directory / "candidates.jsonl")
        report["exportElapsedSeconds"] = time.perf_counter() - began
        stage("reverse_neighbor_and_slice_reads")
        began = time.perf_counter()
        report["reads"] = read_backwards_and_neighbors(sequence, args.directory / "neighbors.jsonl")
        report["reverseAndNeighborElapsedSeconds"] = time.perf_counter() - began
        report["databaseBeforeCleanup"] = database_snapshot(sequence)
    if report["databaseBeforeCleanup"] is not None:
        owned = Path(report["databaseBeforeCleanup"]["ownedPath"])
        remaining = [str(owned) + suffix for suffix in ("", "-journal", "-wal", "-shm")
                     if Path(str(owned) + suffix).exists()]
        report["successfulOwnedDatabaseCleanup"] = not remaining
        report["remainingOwnedDatabaseFiles"] = remaining
        if remaining:
            raise ValueError("Successful owned candidate context left database files")
    report["passed"] = report["export"]["frames"] == args.frames and report["reads"]["reverseFrames"] == args.frames


class CandidateLifetime:
    """Counters only; finalizers never retain the candidate or a frame list."""

    def __init__(self):
        self.created = self.live = self.peak_live = self.finalized = 0

    def observe(self, candidate):
        self.created += 1
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        weakref.finalize(candidate, self.released)

    def released(self):
        self.live -= 1
        self.finalized += 1

    def snapshot(self):
        return {"created": self.created, "live": self.live, "peakLive": self.peak_live,
                "finalized": self.finalized}


class ObservedCandidateSequence(Sequence):
    def __init__(self, source, lifetime):
        self.source, self.lifetime = source, lifetime

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        frame = self.source[index]
        for candidate in frame:
            self.lifetime.observe(candidate)
        return frame

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


TRACKER_SETTINGS = {
    "high_conf_thresh": 0.08, "low_conf_thresh": 0.01, "max_prediction_gap": 6,
    "max_interpolation_gap": 6, "max_valid_speed_px_per_frame": 90.0,
    "base_gating_radius_px": 45.0, "enable_multi_candidate_association": True,
    "enable_adaptive_gate": True, "enable_short_gap_reacquisition": True,
    "enable_camera_motion_compensation": False, "enable_scale_normalization": True,
    "enable_frame_bounds_filter": True,
}


def tracker_child(args, report):
    from src.utils.ball_candidate_sequence import BallCandidateSequence

    if args.mode == "old":
        spec = importlib.util.spec_from_file_location("candidate_capacity_frozen_tracker", OLD_TRACKER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        from src.tracking import temporal_ball_tracker as module
    report["trackerModulePath"] = str(Path(module.__file__).resolve())
    report["trackerModuleSha256"] = digest(Path(module.__file__).resolve())
    report["trackerSettings"] = TRACKER_SETTINGS
    report["fps"] = 60.0
    report["frameSize"] = [1920, 1080]
    report["lifetimeScope"] = "BallObservation snapshots decoded during the actual tracker call only; weak finalizers add overhead, so these memory/timing values are separate from the storage measurements."
    with BallCandidateSequence(args.directory) as sequence:
        stage("prepare_identical_disk_backed_tracker_input")
        sequence.extend(candidate_frames(args.frames))
        report["replacementIndices"] = replace_sparse(sequence)
        # Observe reads only. Preparation objects are already released; no
        # weak-reference registry or full-frame cache is kept by this wrapper.
        gc.collect()
        lifetime = CandidateLifetime()
        observed = ObservedCandidateSequence(sequence, lifetime)
        report["lifetimeBeforeTracker"] = lifetime.snapshot()
        stage("actual_frozen_tracker" if args.mode == "old" else "actual_current_tracker")
        began = time.perf_counter()
        points = module.TemporalBallTracker(**TRACKER_SETTINGS).track_video_candidates(
            observed, fps=report["fps"], frame_size=tuple(report["frameSize"]))
        report["trackerElapsedSeconds"] = time.perf_counter() - began
        report["lifetimeImmediatelyAfterTracker"] = lifetime.snapshot()
        gc.collect()
        report["lifetimeAfterGarbageCollection"] = lifetime.snapshot()
        report["temporalPointFields"] = [field.name for field in fields(module.TemporalBallPoint)]
        state_counts, source_counts, hashed = Counter(), Counter(), hashlib.sha256()
        destination = args.directory / "temporal_points.jsonl"
        with destination.open("xb") as stream:
            for index, point in enumerate(points):
                if point.frame_index != index:
                    raise ValueError("Tracker point frame order differs")
                record = normalize(point)
                # Reject nonfinite output; missing values must retain nulls.
                encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                hashed.update(encoded)
                stream.write(encoded)
                state_counts[record["state"]] += 1
                source_counts[record["source"]] += 1
        report["points"] = {"frames": len(points), "outputFile": destination.name,
                            "canonicalSha256": hashed.hexdigest(), "outputSha256": digest(destination),
                            "outputBytes": destination.stat().st_size, "stateCounts": dict(state_counts),
                            "sourceCounts": dict(source_counts)}
        report["databaseBeforeCleanup"] = database_snapshot(sequence)
        if lifetime.live != 0:
            raise ValueError("Candidate objects remain live after actual tracker return and garbage collection")
    owned = Path(report["databaseBeforeCleanup"]["ownedPath"])
    report["successfulOwnedDatabaseCleanup"] = all(not Path(str(owned) + suffix).exists()
                                                   for suffix in ("", "-journal", "-wal", "-shm"))
    report["passed"] = report["points"]["frames"] == args.frames and report["successfulOwnedDatabaseCleanup"]


def child(args):
    import psutil

    began = time.perf_counter()
    process = psutil.Process()
    save(args.directory / "child_identity.json", {
        "pid": process.pid, "parentPid": process.ppid(), "createTimeUnixSeconds": process.create_time(),
        "interpreter": sys.executable, "writtenAt": stamp(),
    })
    report = {"startedAt": stamp(), "passed": False, "kind": args.kind, "mode": args.mode,
              "frames": args.frames, "pid": os.getpid(), "interpreter": sys.executable,
              "pythonVersion": sys.version, "memoryBeforeBytes": process.memory_info()._asdict(),
              "scope": "Synthetic candidate-only systems measurement; no models, video inference, accuracy or full-match qualification."}
    try:
        if args.kind == "storage":
            storage_child(args, report)
        else:
            tracker_child(args, report)
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["completedAt"] = stamp()
        report["totalChildElapsedSeconds"] = time.perf_counter() - began
        memory = process.memory_info()._asdict()
        report["memoryAfterBytes"] = memory
        report["windowsPeakWorkingSetBytes"] = memory.get("peak_wset")
        report["windowsPeakPrivateCommitBytes"] = memory.get("peak_pagefile")
        report["peakPrivateCommitMeaning"] = "Windows PROCESS_MEMORY_COUNTERS PeakPagefileUsage (peak process commit); not disk paging-file occupancy. Includes interpreter/imports and this whole child."
        report["retainedOwnedFilesBytesAtReceipt"] = sum(path.stat().st_size for path in args.directory.iterdir()
                                                        if path.is_file())
        save(args.directory / "child_receipt.json", report)
        print(json.dumps({key: report.get(key) for key in ("passed", "kind", "mode", "frames", "totalChildElapsedSeconds", "windowsPeakWorkingSetBytes", "windowsPeakPrivateCommitBytes", "error")}), flush=True)


def source_bindings():
    paths = [Path(__file__).resolve(), ROOT / "src/utils/ball_candidate_sequence.py",
             ROOT / "src/utils/bbox_utils.py", ROOT / "src/tracking/temporal_ball_tracker.py",
             OLD_TRACKER, BASELINE / "receipt.json"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}


def verify_bindings(expected):
    actual = source_bindings()
    if actual != expected:
        raise ValueError("A source bound to this candidate-only measurement changed; retained receipts identify the frozen sources")


class OwnedProcessTree:
    """Remember only descendants discovered through exact owned identities.

    Windows virtual-environment python.exe is a redirector, so its workload
    interpreter must be sampled separately. Retaining identities also lets us
    stop a previously observed descendant if its parent has already exited.
    """

    def __init__(self, process):
        import psutil

        self._psutil = psutil
        root = psutil.Process(process.pid)
        self.root_identity = (root.pid, root.create_time())
        self._owned = {}
        self._remember(root, parent_pid=None, depth=0)

    def _remember(self, process, parent_pid, depth):
        identity = (process.pid, process.create_time())
        if identity not in self._owned:
            self._owned[identity] = {"process": process, "pid": process.pid,
                                     "createTimeUnixSeconds": identity[1], "parentPid": parent_pid,
                                     "depth": depth, "firstObservedAt": stamp(), "sampleCount": 0,
                                     "peakSampledRssBytes": 0, "peakSampledPrivateBytes": 0}
        return identity

    def _matches(self, record):
        process = record["process"]
        try:
            return process.is_running() and process.create_time() == record["createTimeUnixSeconds"]
        except self._psutil.NoSuchProcess:
            return False

    def discover(self):
        # Start from remembered live owners only, never a global PID scan.
        pending = list(self._owned)
        visited = set()
        while pending:
            identity = pending.pop()
            if identity in visited:
                continue
            visited.add(identity)
            record = self._owned[identity]
            if not self._matches(record):
                continue
            try:
                children = record["process"].children()
            except self._psutil.NoSuchProcess:
                continue
            for child in children:
                try:
                    # psutil validates its Process identity before signalling;
                    # record creation time too so receipts cannot confuse a
                    # recycled PID with an owned descendant.
                    child_identity = self._remember(child, record["pid"], record["depth"] + 1)
                except self._psutil.NoSuchProcess:
                    continue
                pending.append(child_identity)

    def sample(self):
        self.discover()
        sampled = []
        for record in self._owned.values():
            if not self._matches(record):
                continue
            try:
                memory = record["process"].memory_info()._asdict()
            except self._psutil.NoSuchProcess:
                continue
            record["sampleCount"] += 1
            record["peakSampledRssBytes"] = max(record["peakSampledRssBytes"], memory["rss"])
            if memory.get("private") is not None:
                record["peakSampledPrivateBytes"] = max(record["peakSampledPrivateBytes"], memory["private"])
            sampled.append({"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                            "parentPid": record["parentPid"], "depth": record["depth"], "memoryBytes": memory})
        return sampled

    def identities(self):
        return [{key: value for key, value in record.items() if key != "process"}
                for record in sorted(self._owned.values(), key=lambda row: (row["depth"], row["pid"]))]

    def observed_workload(self, identity):
        key = (identity["pid"], identity["createTimeUnixSeconds"])
        record = self._owned.get(key)
        return record is not None and record["sampleCount"] > 0

    def terminate(self):
        self.discover()
        actions, signalled = [], []
        records = sorted(self._owned.values(), key=lambda row: row["depth"], reverse=True)
        for record in records:
            if not self._matches(record):
                continue
            process = record["process"]
            try:
                process.terminate()
                signalled.append(process)
                actions.append({"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                                "depth": record["depth"], "signal": "terminate", "at": stamp()})
            except self._psutil.NoSuchProcess:
                pass
        _, alive = self._psutil.wait_procs(signalled, timeout=5)
        for process in alive:
            identity = (process.pid, process.create_time())
            record = self._owned.get(identity)
            if record is None or not self._matches(record):
                continue
            try:
                process.kill()
                actions.append({"pid": record["pid"], "createTimeUnixSeconds": record["createTimeUnixSeconds"],
                                "depth": record["depth"], "signal": "kill", "at": stamp()})
            except self._psutil.NoSuchProcess:
                pass
        if alive:
            self._psutil.wait_procs(alive, timeout=5)
        remaining = [row for row in records if self._matches(row)]
        if remaining:
            raise RuntimeError("Could not stop every exact owned candidate workload identity")
        return actions


def stop_owned(process, owned):
    actions = [] if owned is None else owned.terminate()
    # Popen holds the original redirector's exact Windows handle, providing
    # a fallback if discovery failed before an OwnedProcessTree was created.
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    return actions


def monitor_child(directory, kind, mode, frames, bindings):
    import psutil

    verify_bindings(bindings)
    directory.mkdir(exist_ok=False)
    argv = [str(CLEAN), str(Path(__file__).resolve()), "--directory", str(directory),
            "--child", "--kind", kind, "--mode", mode, "--frames", str(frames)]
    began = time.perf_counter()
    report = {"startedAt": stamp(), "kind": kind, "mode": mode, "frames": frames,
              "passed": False, "samples": 0, "peakSampledOwnedTreeRssBytes": 0,
              "peakSampledOwnedTreePrivateBytes": 0, "minimumSampledAvailableRamBytes": None,
              "scope": "Exact owned redirector and descendant identities. Aggregate RSS can count shared pages twice and sampled values can miss short peaks. Host available RAM includes unrelated activity and a low-RAM stop is not proof of a worker leak."}
    process = None
    owned = None
    try:
        with (directory / "stdout.txt").open("x", encoding="utf-8") as log, (directory / "resource_samples.jsonl").open("x", encoding="utf-8", newline="\n") as samples:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report["pid"] = process.pid
            save(directory / "launch.json", {"startedAt": stamp(), "pid": process.pid, "argv": argv,
                                              "sourceFiles": bindings})
            owned = OwnedProcessTree(process)
            report["processCreationTimeUnixSeconds"] = owned.root_identity[1]
            low_ram = 0
            while process.poll() is None:
                elapsed = time.perf_counter() - began
                available = psutil.virtual_memory().available
                processes = owned.sample()
                tree_rss = sum(row["memoryBytes"]["rss"] for row in processes)
                private_values = [row["memoryBytes"].get("private") for row in processes]
                tree_private = sum(value for value in private_values if value is not None)
                sample = {"at": stamp(), "elapsedSeconds": elapsed, "pid": process.pid,
                          "availableRamBytes": available, "freeDiskBytes": shutil.disk_usage(ROOT).free,
                          "processes": processes, "ownedTreeRssBytes": tree_rss,
                          "ownedTreePrivateBytes": tree_private,
                          "allPrivateValuesAvailable": all(value is not None for value in private_values)}
                samples.write(json.dumps(sample, allow_nan=False) + "\n")
                samples.flush()
                report["samples"] += 1
                previous_minimum = report["minimumSampledAvailableRamBytes"]
                report["minimumSampledAvailableRamBytes"] = available if previous_minimum is None else min(previous_minimum, available)
                report["peakSampledOwnedTreeRssBytes"] = max(report["peakSampledOwnedTreeRssBytes"], tree_rss)
                report["peakSampledOwnedTreePrivateBytes"] = max(report["peakSampledOwnedTreePrivateBytes"], tree_private)
                low_ram = low_ram + 1 if available < MINIMUM_AVAILABLE_RAM else 0
                reason = None
                if tree_rss > MAXIMUM_RSS:
                    reason = "aggregate_owned_tree_rss_exceeded_1250000000_bytes"
                elif low_ram >= LOW_RAM_SAMPLES:
                    reason = "host_available_ram_below_400000000_bytes_for_5_consecutive_samples"
                elif elapsed > MAXIMUM_WALL_SECONDS:
                    reason = "owned_child_wall_time_exceeded_900_seconds"
                if reason is not None:
                    report["safeguardStop"] = reason
                    raise RuntimeError("Predeclared candidate benchmark safeguard: " + reason)
                time.sleep(SAMPLE_INTERVAL)
            report["returncode"] = process.returncode
        receipt_path = directory / "child_receipt.json"
        if receipt_path.is_file():
            report["childReceipt"] = json.loads(receipt_path.read_text(encoding="utf-8"))
        identity_path = directory / "child_identity.json"
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            report["actualWorkloadIdentity"] = identity
            report["actualWorkloadSampledByExactIdentity"] = owned.observed_workload(identity)
        if process.returncode != 0 or not report.get("childReceipt", {}).get("passed"):
            raise RuntimeError("Owned candidate benchmark child failed; inspect " + str(directory / "stdout.txt"))
        if not report.get("actualWorkloadSampledByExactIdentity"):
            raise RuntimeError("Actual workload interpreter was not sampled by exact owned identity")
        if report["actualWorkloadIdentity"]["pid"] != report["childReceipt"]["pid"]:
            raise ValueError("Child identity and terminal receipt PID differ")
        verify_bindings(bindings)
        report["passed"] = True
        return report
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        if process is not None:
            try:
                report["ownedTerminationActions"] = stop_owned(process, owned)
            except BaseException as stop_error:
                error.add_note("Owned termination also failed: " + str(stop_error))
                report["ownedTerminationError"] = type(stop_error).__name__ + ": " + str(stop_error)
            report["returncode"] = process.returncode
        raise
    finally:
        report["completedAt"] = stamp()
        if owned is not None:
            report["ownedProcessIdentities"] = owned.identities()
        report["totalMonitoredWallSeconds"] = time.perf_counter() - began
        report["ownedFilesLogicalBytes"] = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
        save(directory / "monitor_receipt.json", report)


def exact_files(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            a, b = first.read(1024 * 1024), second.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def validate_jsonl(path, frames, kind, expected_fields=None):
    hashed, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for line in stream:
            record = json.loads(line)
            if record["frame_index"] != count:
                raise ValueError("Exported frame order differs")
            if kind == "tracker" and set(record) != set(expected_fields):
                raise ValueError("A TemporalBallPoint field was omitted")
            if kind == "storage" and set(record) != {"frame_index", "candidates"}:
                raise ValueError("Candidate record keys differ")
            encoded = canonical_bytes(record)
            if encoded != line:
                raise ValueError("Exported float/null values did not survive canonical JSON parsing")
            hashed.update(encoded)
            count += 1
    if count != frames:
        raise ValueError("Exported row count differs")
    return {"frames": count, "parsedCanonicalSha256": hashed.hexdigest(), "bytes": path.stat().st_size}


def compare_pair(directory, kind, frames, pair):
    receipts = [row["childReceipt"] for row in pair]
    paths = [directory / f"{kind}_{mode}_{frames}" for mode in ("old", "new")]
    key, filename = ("export", "candidates.jsonl") if kind == "storage" else ("points", "temporal_points.jsonl")
    parsed = [validate_jsonl(path / filename, frames, kind, receipts[index].get("temporalPointFields"))
              for index, path in enumerate(paths)]
    identical = exact_files(paths[0] / filename, paths[1] / filename)
    identical = identical and parsed[0] == parsed[1]
    identical = identical and all(row[key]["outputSha256"] == row[key]["canonicalSha256"] == parsed[index]["parsedCanonicalSha256"]
                                  for index, row in enumerate(receipts))
    comparison = {"kind": kind, "frames": frames, "exactOutputBytesAndParsedValues": identical, "verification": parsed}
    if kind == "storage":
        reads = [row["reads"] for row in receipts]
        comparison["reverseAndNeighborRecordsIdentical"] = reads[0] == reads[1] and exact_files(paths[0] / "neighbors.jsonl", paths[1] / "neighbors.jsonl")
        identical = identical and comparison["reverseAndNeighborRecordsIdentical"]
    else:
        lifetimes = [row["lifetimeAfterGarbageCollection"] for row in receipts]
        comparison["candidateLifetimes"] = lifetimes
        comparison["allCandidatesReleasedAfterTracker"] = all(row["live"] == 0 and row["created"] == row["finalized"] for row in lifetimes)
        comparison["currentPeakLiveCandidatesLower"] = lifetimes[1]["peakLive"] < lifetimes[0]["peakLive"]
        comparison["temporalPointFields"] = receipts[0]["temporalPointFields"]
        identical = identical and receipts[0]["temporalPointFields"] == receipts[1]["temporalPointFields"]
        identical = identical and comparison["allCandidatesReleasedAfterTracker"] and comparison["currentPeakLiveCandidatesLower"]
    comparison["passed"] = identical
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--kind", choices=("storage", "tracker"))
    parser.add_argument("--mode", choices=("old", "new"))
    parser.add_argument("--frames", type=int)
    args = parser.parse_args()
    args.directory = (ROOT / args.directory).resolve()
    allowed = (ROOT / "artifacts/validation/product_execution").resolve()
    if not args.directory.is_relative_to(allowed) or args.directory == allowed:
        raise ValueError("Use a fresh owned directory under product_execution")
    if args.child:
        if not args.directory.is_dir() or args.mode is None or args.kind is None:
            raise ValueError("Child requires an existing exclusively created run directory and kind/mode")
        if args.frames not in (FRAME_COUNTS if args.kind == "storage" else (TRACKER_FRAMES,)):
            raise ValueError("Child frame count differs from protocol")
        child(args)
        return
    import psutil

    if os.name != "nt" or not CLEAN.is_file():
        raise RuntimeError("This protocol requires the installed clean Windows interpreter")
    bindings = source_bindings()
    baseline_receipt = json.loads((BASELINE / "receipt.json").read_text(encoding="utf-8"))
    if digest(OLD_TRACKER) != baseline_receipt["sources"]["src/tracking/temporal_ball_tracker.py"]:
        raise ValueError("Frozen tracker differs from its preservation receipt")
    protocol = {"declaredAt": stamp(), "scope": "Synthetic candidate-only storage measurement plus actual frozen/current Kalman tracker equivalence and candidate lifetime, not inference, accuracy or full-pipeline memory qualification.",
                "sourceFiles": bindings, "cleanInterpreter": str(CLEAN), "cleanInterpreterSha256": digest(CLEAN),
                "launcherInterpreter": sys.executable, "launcherPythonVersion": sys.version,
                "launcherPid": os.getpid(), "launcherProcessCreationTimeUnixSeconds": psutil.Process().create_time(),
                "host": {"hostname": platform.node(), "platform": platform.platform(), "processor": platform.processor(),
                         "logicalCpuCount": psutil.cpu_count(), "physicalCpuCount": psutil.cpu_count(logical=False),
                         "psutilVersion": psutil.__version__, "ramBeforeBytes": psutil.virtual_memory()._asdict(),
                         "diskBeforeBytes": shutil.disk_usage(ROOT)._asdict()},
                "storageFrames": list(FRAME_COUNTS), "trackerFrames": TRACKER_FRAMES,
                "candidateCountPolicy": "Three independently constructed BallObservations per nonempty frame; initial, periodic and 13-frame gaps; moving/static/distractor coordinates, exact Python floats, optional BBox and track IDs, rejected coordinates and preserved nonfinite candidate coordinates.",
                "storageOperations": ["lazy append of all frames", "bounded sparse indexed replacement", "forward full-record JSONL export and incremental digest", "full reverse record digest", "bounded previous/current/next and negative/slice reads"],
                "trackerInput": "Both actual tracker modules read the same type of disk-backed source through a weak-finalizer counter wrapper. Only tracking reads are observed; no registry retains candidates.",
                "trackerSettings": TRACKER_SETTINGS, "sampleIntervalSeconds": SAMPLE_INTERVAL,
                "maximumOwnedChildRssBytes": MAXIMUM_RSS, "minimumAvailableRamBytes": MINIMUM_AVAILABLE_RAM,
                "lowRamConsecutiveSamplesToStop": LOW_RAM_SAMPLES, "maximumPerChildWallSeconds": MAXIMUM_WALL_SECONDS,
                "ownedProcessPolicy": "Track only the exact launched redirector and its discovered descendants by PID/createTime. Stream every per-process memory sample and aggregate owned-tree RSS/private; require the actual interpreter's early identity to have been sampled.",
                "terminationPolicy": "Stop only exact remembered owned descendants, deepest first, then the redirector; recheck identity before signalling. Preserve every artifact; no retries, unrelated PID scan or external cleanup.",
                "retention": "Retain every forward output, neighbor output, tracker point output, child log, streamed resource sample and receipt. A successful BallCandidateSequence context removes only its exclusively owned scratch database/sidecars; logical DB size is recorded beforehand. Exceptions retain owned partial databases.",
                "limitations": ["Synthetic candidates are not authorized match recordings or accuracy evidence.",
                                "Two-hour frame count is a scale reference, not a representative full-match workload.",
                                "Interpreter imports count toward process peaks; host background activity is uncontrolled.",
                                "Windows peak process commit does not mean resident RAM or paging-file disk use.",
                                "Weak finalizers add overhead and lifetime runs are separate from uninstrumented storage measurements.",
                                "The pipeline still retains player, calibration, trajectory and result records; these runs do not establish total pipeline bounds."]}
    args.directory.mkdir(parents=True, exist_ok=False)
    save(args.directory / "protocol.json", protocol)
    report = {"startedAt": stamp(), "passed": False, "scope": protocol["scope"], "runs": [], "comparisons": []}
    try:
        for kind, counts in (("storage", FRAME_COUNTS), ("tracker", (TRACKER_FRAMES,))):
            for frames in counts:
                for mode in ("old", "new"):
                    run_directory = args.directory / f"{kind}_{mode}_{frames}"
                    print(json.dumps({"at": stamp(), "launching": run_directory.name}), flush=True)
                    run = monitor_child(run_directory, kind, mode, frames, bindings)
                    report["runs"].append(run)
                    child_report = run["childReceipt"]
                    print(json.dumps({key: child_report.get(key) for key in ("passed", "kind", "mode", "frames", "totalChildElapsedSeconds", "windowsPeakWorkingSetBytes", "windowsPeakPrivateCommitBytes")}), flush=True)
                comparison = compare_pair(args.directory, kind, frames, report["runs"][-2:])
                report["comparisons"].append(comparison)
                save(args.directory / f"{kind}_{frames}_comparison.json", comparison)
                print(json.dumps({"comparison": kind, "frames": frames, "passed": comparison["passed"]}), flush=True)
                if not comparison["passed"]:
                    raise ValueError("Paired candidate benchmark outputs/lifetime conditions differ; comparison receipt retained")
        verify_bindings(bindings)
        report["sourceFilesUnchanged"] = True
        report["passed"] = True
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["completedAt"] = stamp()
        report["diskAfterBytes"] = shutil.disk_usage(ROOT)._asdict()
        report["retainedOwnedFilesLogicalBytes"] = sum(path.stat().st_size for path in args.directory.rglob("*") if path.is_file())
        save(args.directory / "report.json", report)
    print(json.dumps({"passed": report["passed"], "directory": str(args.directory)}), flush=True)


if __name__ == "__main__":
    main()
