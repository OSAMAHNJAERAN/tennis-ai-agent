"""Prepared one-shot synthetic pose/racket storage comparison; root dispatches.

Four sequential fresh children use complete immutable JSONL inputs. Only the
frozen sanitizer definition is compiled, never the model-bearing pipeline.
This is storage/export engineering evidence, not inference or qualification.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
from typing import Any
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
UNIT = HERE.parent
RUN = HERE / "run"
FROZEN = UNIT / "before/src/pipeline/phase6_pipeline.py"
PIPELINE = ROOT / "src/pipeline/phase6_pipeline.py"
IDENTITY = ROOT / "src/utils/identity_frame_sequence.py"
MONITOR = ROOT / "scripts/audit/measure_candidate_storage.py"
CLEAN = ROOT / "artifacts/validation/product_installation/clean_env/Scripts/python.exe"
BASE = Path(r"C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe")
FROZEN_SHA256 = "93b1bcf44a56ed70d76264ae7ab8cd2e8423616821632da09904d71ddc3c75ec"
MONITOR_SHA256 = "8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39"
COUNTS = (600, 18000)
PREFLIGHT = 750_000_000
RSS_LIMIT = 1_000_000_000
AVAILABLE_FLOOR = 400_000_000
LOW_SAMPLES = 5
WALL_LIMIT = 180
SAMPLE_INTERVAL = 0.25
FPS = 60000 / 1001
JOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
RACKET_KEYS = ("state", "position_px", "bbox_xyxy", "confidence", "velocity_px_per_s",
               "motion_direction_deg", "contact_event")
PROJECT_SOURCES = ("src/__init__.py", "src/utils/__init__.py", "src/utils/identity_frame_sequence.py",
                   "src/pipeline/phase6_pipeline.py", "src/tracking/player_motion_tracking.py",
                   "src/tracking/racket_tracking.py", "tests/test_identity_frame_sequence.py",
                   "tests/test_pipeline_pose_storage.py")
FORBIDDEN_IMPORTS = ("torch", "ultralytics", "tensorflow", "transformers", "onnxruntime",
                     "cv2", "src.pipeline", "src.detection", "src.models")
OUTPUT_NAMES = ("observations.jsonl", "render_audit.json", "player_motion.json", "racket_tracking.json")


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
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": path.stat().st_size, "sha256": digest}


def load(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def handshake(path, value):
    temporary = Path(str(path) + ".tmp")
    save(temporary, value)
    temporary.rename(path)


def verify(values):
    for pathname, expected in values.items():
        require(binding(resolve(pathname)) == expected, "Binding changed: " + pathname)


def source_bindings():
    paths = [ROOT / pathname for pathname in PROJECT_SOURCES]
    paths += [FROZEN, UNIT / "before.json", MONITOR, CLEAN, BASE,
              HERE / "helper.py", HERE / "protocol.json", HERE / "test_helper_contract.py",
              HERE / "README.md", HERE / "prepared_bindings.json", HERE / "preparation.json"]
    for phase, leaves in (("storage_focused", ("receipt.json", "junit.xml", "output.log", "preparation.json")),
                           ("pipeline_green", ("receipt.json", "junit.xml", "run_focused_green.py",
                                               "launch.json", "samples.jsonl", "stdout.log", "stderr.log"))):
        paths.extend(UNIT / phase / leaf for leaf in leaves)
        for subfolder in ("source", "sources"):
            folder = UNIT / phase / subfolder
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    paths.extend(UNIT / "backend_full" / leaf for leaf in
                 ("receipt.json", "verify_backend_full.py", "launch.json", "source_bindings_before.json",
                  "source_bindings_after.json", "stdout.log", "stderr.log", "samples.jsonl", "junit.xml"))
    return {name(path): binding(path) for path in paths}


def suite_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    require(bool(suites), "Missing actual JUnit suites")
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and all(counts[key] == 0 for key in ("failures", "errors", "skipped")),
            "Prerequisite actual software suite did not entirely pass")
    return counts


def focused_evidence_gate():
    storage = load(UNIT / "storage_focused/receipt.json")
    require(storage["exitCode"] == 0 and storage["sourceUnchanged"] is True
            and storage["sourceBefore"] == storage["sourceAfter"], "Storage focused GREEN did not pass unchanged")
    require(binding(UNIT / "storage_focused/junit.xml")["sha256"] == storage["junitSha256"]
            and binding(UNIT / "storage_focused/output.log")["sha256"] == storage["outputSha256"],
            "Storage focused result/log bindings differ")
    for pathname, value in storage["sourceBefore"].items():
        selected = FROZEN if pathname == "src/pipeline/phase6_pipeline.py" else resolve(pathname)
        require(binding(selected) == value, "Storage focused input binding differs: " + pathname)
    pipeline = load(UNIT / "pipeline_green/receipt.json")
    require(pipeline["status"] == "GREEN" and pipeline["exitCode"] == 0
            and pipeline["sourceBindingsUnchanged"] is True and pipeline["sourceBefore"] == pipeline["sourceAfter"]
            and pipeline["allOwnedIdentitiesRetired"] is True and not pipeline["failureCases"],
            "Actual pipeline integration GREEN did not pass unchanged/retired")
    verify(pipeline["sourceBefore"])
    return {"storage": suite_counts(UNIT / "storage_focused/junit.xml"),
            "pipeline": suite_counts(UNIT / "pipeline_green/junit.xml"),
            "scope": "Prior passed class and actual integration software checks; synthetic measurements do not invoke the pipeline or promote accuracy."}


def backend_evidence_gate():
    folder = UNIT / "backend_full"
    receipt = load(folder / "receipt.json")
    require(receipt["passed"] is True and receipt["status"] == "PASSED" and receipt["exitCode"] == 0
            and receipt["sourcesUnchanged"] is True and receipt["allOwnedIdentitiesRetired"] is True
            and receipt["baseInterpreterAfterMatchesBefore"] is True, "Fresh complete backend suite prerequisite failed")
    require(receipt["sourceBeforePath"] == name(folder / "source_bindings_before.json")
            and receipt["sourceAfterPath"] == name(folder / "source_bindings_after.json"), "Unexpected full backend source paths")
    before, after = load(folder / "source_bindings_before.json"), load(folder / "source_bindings_after.json")
    require(before == after, "Full backend source bindings differed")
    verify(before["files"])
    current_inventory = {name(path) for directory in ("src", "tests", "configs")
                         for path in (ROOT / directory).rglob("*") if path.is_file()
                         and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")}
    current_inventory.update(name(ROOT / leaf) for leaf in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini",
                             ".coveragerc", "requirements.txt", "requirements-product.txt") if (ROOT / leaf).is_file())
    current_inventory.update((name(MONITOR), name(folder / "verify_backend_full.py")))
    require(current_inventory == set(before["files"]), "Current full backend input inventory changed after its passed suite")
    for item in (before["baseInterpreter"], before["actualHelperInterpreter"]):
        require(binding(item["path"]) == {key: item[key] for key in ("bytes", "sha256")}, "Backend interpreter binding differs")
    actual = suite_counts(folder / "junit.xml")
    require(all(receipt["counts"][key] == value for key, value in actual.items())
            and receipt["counts"]["passed"] == actual["tests"], "Actual full backend counts differ or contain skips")
    for pathname, value in receipt["artifacts"].items():
        require(binding(folder / pathname) == value, "Retained backend artifact binding differs: " + pathname)
    for item in receipt["actualProcessExecutables"]:
        require(binding(item["path"]) == {key: item[key] for key in ("bytes", "sha256")}, "Actual backend executable bytes changed")
    require(bool(receipt["actualProcessExecutables"]), "Backend actual sampled interpreter proof missing")
    return {"actualCounts": receipt["counts"], "testElapsedSeconds": receipt["testElapsedSeconds"],
            "receiptBinding": binding(folder / "receipt.json"), "sourcesUnchanged": True,
            "scope": "Independent complete software regressions; no scientific or real inference qualification."}, before["files"]


def module_at(alias, path):
    specification = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[alias] = module
    specification.loader.exec_module(module)
    return module


def ordinary_imports():
    paths, modules = {}, {}
    for module_name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path or not Path(path).is_file():
            continue
        pathname = name(path)
        if pathname not in paths:
            paths[pathname] = binding(path)
        modules[module_name] = pathname
    return paths, modules


def reject_unsupported_imports():
    prohibited = [item for item in sys.modules if any(item == prefix or item.startswith(prefix + ".")
                  for prefix in FORBIDDEN_IMPORTS)]
    require(not prohibited, "Unsupported model/media/pipeline import: " + repr(prohibited))


def import_network_guard(event, arguments):
    if event == "import" and arguments and isinstance(arguments[0], str):
        module_name = arguments[0]
        require(not any(module_name == prefix or module_name.startswith(prefix + ".")
                        for prefix in FORBIDDEN_IMPORTS), "Forbidden import: " + module_name)
    require(event not in ("socket.connect", "socket.connect_ex", "socket.bind", "socket.getaddrinfo"),
            "Synthetic storage child attempted a network action")


def sanitizer_node(path):
    tree = ast.parse(Path(path).read_bytes(), filename=str(path))
    definitions = [item for item in tree.body if isinstance(item, ast.FunctionDef)
                   and item.name == "sanitize_json_value"]
    require(len(definitions) == 1, "Expected one frozen sanitizer definition")
    node = definitions[0]
    require(not node.decorator_list, "Unexpected sanitizer decorator")
    return node


def frozen_sanitizer(numpy):
    require(binding(FROZEN)["sha256"] == FROZEN_SHA256, "Frozen pipeline source changed")
    frozen, current = sanitizer_node(FROZEN), sanitizer_node(PIPELINE)
    representation = ast.dump(frozen, include_attributes=False)
    require(representation == ast.dump(current, include_attributes=False), "Production sanitizer changed")
    namespace = {"np": numpy, "Any": Any}
    selected = ast.Module(body=[frozen], type_ignores=[])
    exec(compile(selected, str(FROZEN) + "::sanitize_json_value_only", "exec"), namespace)
    return namespace["sanitize_json_value"], hashlib.sha256(representation.encode()).hexdigest()


def metadata():
    # Ground motion is deliberately unmeasured in this pose/racket storage unit.
    # Full schema fields remain present; no unrelated per-frame motion array.
    def unknown_motion():
        return {"observed_distance_m": None, "mean_speed_kmh": None, "measured_duration_s": 0.0,
                "accepted_intervals": 0, "measurement_scope": "OBSERVED_CONTIGUOUS_FOOT_PROXY_INTERVALS_ONLY",
                "distance_estimator": "INTEGRAL_OF_BACKWARD_WINDOW_LEAST_SQUARES_SPEED",
                "velocity_window_seconds": .15, "raw_observed_polyline_distance_m": None,
                "uncertainty_scope": "SHORT_WINDOW_FIT_RESIDUAL_IS_NOT_INDEPENDENT_SENSOR_ERROR; STATIONARY_NOISE_CAN_REMAIN",
                "samples": []}
    pose = {"schema_version": "2.0", "pose_enabled": True, "joint_names": list(JOINT_NAMES),
            "coordinate_system": "SOURCE_IMAGE_PIXELS",
            "ground_position_source": "CALIBRATED_BBOX_BOTTOM_CENTER_PROXY", "camera_registration_enabled": False,
            "ground_projection_scope": "STATIC_INITIAL_COURT_PLANE", "foot_contact_validated": False,
            "pose_accuracy_validated": False, "identity_scope": "NEAR_FAR_ROLE_WITHIN_STATIC_SINGLES_CAMERA_SEGMENT",
            "player_1": unknown_motion(), "player_2": unknown_motion()}
    racket = {"schema_version": "1.0", "enabled": True, "coordinate_system": "SOURCE_IMAGE_PIXELS",
              "accuracy_validated": False, "motion_semantics": "BBOX_CENTER_TRANSLATION_NOT_SWING_OR_CONTACT_CLASSIFICATION"}
    return {"playerMotion": pose, "racketTracking": racket, "fps": FPS,
            "fixtureScope": "Complete native pose/racket observations only; ground motion is unknown/empty; no media/inference/accuracy evidence."}


def generated_frame(index):
    order = (2, 1) if index % 2 else (1, 2)
    pose, racket = [], []
    for identity in order:
        joints = {}
        for joint_index, joint_name in enumerate(JOINT_NAMES):
            x = -0.0 if joint_index == 0 else math.nextafter(float(80 + identity * 140 + joint_index), math.inf)
            y = math.nextafter(float(150 + joint_index), -math.inf) + index / FPS
            joints[joint_name] = {"position_px": [x, y], "confidence": .9000000000000001}
        pose.append([identity, joints])
        if (index + identity) % 5 == 1:
            observation = {"state": "MISSING", "position_px": None, "bbox_xyxy": None,
                           "confidence": None, "velocity_px_per_s": None, "motion_direction_deg": None,
                           "contact_event": None}
        else:
            left = math.nextafter(float(180 + identity * 50), math.inf) + index / FPS
            box = [left, 160.0, left + 25.0, 185.0]
            velocity = [-0.0, index / FPS] if index else None
            observation = {"state": "DETECTED", "position_px": [left + 12.5, 172.5], "bbox_xyxy": box,
                           "confidence": .8750000000000001, "velocity_px_per_s": velocity,
                           "motion_direction_deg": 90.0 if index else None, "contact_event": None}
        racket.append([identity, observation])
    return {"frame_index": index, "timestamp_seconds": index / FPS, "identity_order": list(order),
            "pose": pose, "racket": racket}


def native_node(value):
    """Independent lossless native-value check, including ordered integer keys."""
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", value]
    if type(value) is float:
        return ["float64", struct.pack("!d", value).hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is list:
        return ["list", [native_node(item) for item in value]]
    if type(value) is tuple:
        return ["tuple", [native_node(item) for item in value]]
    if type(value) is dict:
        return ["dict", [[native_node(key), native_node(item)] for key, item in value.items()]]
    raise TypeError("Synthetic fixture contains an unexpected type: " + type(value).__name__)


def record_bytes(index, timestamp, pose, racket):
    return (json.dumps(native_node({"frame_index": index, "timestamp_seconds": timestamp,
                                   "pose": pose, "racket": racket}), separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def validate_frame(row, expected_index):
    require(row["frame_index"] == expected_index and row["timestamp_seconds"] == expected_index / FPS,
            "Fixture frame/clock mismatch")
    expected_order = [2, 1] if expected_index % 2 else [1, 2]
    require(row["identity_order"] == expected_order, "Fixture identity order differs")
    pose, racket = dict(row["pose"]), dict(row["racket"])
    require(list(pose) == list(racket) == expected_order and all(type(key) is int for key in pose),
            "Native ordered integer identities changed")
    for identity in expected_order:
        require(list(pose[identity]) == list(JOINT_NAMES), "Incomplete dense 17-joint pose")
        for joint in pose[identity].values():
            require(list(joint) == ["position_px", "confidence"] and len(joint["position_px"]) == 2
                    and all(type(value) is float for value in joint["position_px"])
                    and type(joint["confidence"]) is float, "Pose native value schema differs")
        observation = racket[identity]
        require(tuple(observation) == RACKET_KEYS, "Incomplete racket observation schema")
        if observation["state"] == "MISSING":
            require(all(observation[key] is None for key in RACKET_KEYS[1:]), "MISSING racket has inferred values")
        else:
            require(observation["state"] == "DETECTED" and len(observation["position_px"]) == 2
                    and len(observation["bbox_xyxy"]) == 4 and observation["contact_event"] is None,
                    "Detected racket native schema differs")
    return pose, racket


def generate_fixture(count):
    fixture_path = RUN / f"fixture_{count}.jsonl"
    began = time.perf_counter()
    digest = hashlib.sha256()
    with fixture_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index in range(count):
            row = generated_frame(index)
            pose, racket = validate_frame(row, index)
            digest.update(record_bytes(index, row["timestamp_seconds"], pose, racket))
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            del row, pose, racket
    receipt = {"passed": True, "frames": count, "completePoseIdentities": count * 2,
               "completePoseJoints": count * 2 * len(JOINT_NAMES), "fullRacketIdentities": count * 2,
               "fixture": {"path": name(fixture_path), **binding(fixture_path)},
               "expectedNativeObservationSha256": digest.hexdigest(), "metadata": metadata(),
               "generationElapsedSeconds": time.perf_counter() - began,
               "streamedConstruction": True, "largestRetainedFixtureFrameCount": 1,
               "inputFileCountPerPair": 1, "modelsImported": False, "inferenceRun": False,
               "qualificationEvidence": False}
    save(RUN / f"fixture_{count}_receipt.json", receipt)
    return receipt


def iter_fixture(path, count):
    seen = 0
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        for index, line in enumerate(stream):
            require(index < count and line.endswith("\n"), "Fixture has unexpected rows/truncated line")
            row = json.loads(line)
            pose, racket = validate_frame(row, index)
            yield index, row["timestamp_seconds"], pose, racket
            seen += 1
    require(seen == count, "Fixture row count differs")


def render_like_reads(poses, rackets, count):
    audit = {"indexedFrames": 0, "densePoseIdentities": 0, "poseJoints": 0,
             "detectedRackets": 0, "missingRackets": 0, "positionSum": 0.0,
             "identityColorSum": 0, "boundedNeighborReads": 0}
    for index in range(count):
        pose, racket = poses[index], rackets[index]
        expected_order = [2, 1] if index % 2 else [1, 2]
        require(list(pose) == list(racket) == expected_order and all(type(key) is int for key in pose),
                "Indexed render identities/order changed")
        audit["indexedFrames"] += 1
        for identity, joints in pose.items():
            require(tuple(joints) == JOINT_NAMES, "Indexed dense joints changed")
            audit["densePoseIdentities"] += 1
            audit["identityColorSum"] += 17 if identity == 1 else 31
            for joint in joints.values():
                audit["poseJoints"] += 1
                audit["positionSum"] += joint["position_px"][0] + joint["position_px"][1]
            item = racket[identity]
            if item["state"] == "MISSING":
                require(all(item[key] is None for key in RACKET_KEYS[1:]), "Missing render racket inferred")
                audit["missingRackets"] += 1
            else:
                audit["detectedRackets"] += 1
                audit["positionSum"] += sum(item["position_px"]) + sum(item["bbox_xyxy"])
    # The same small bounded access checks exercise negative/slice/neighbor
    # reads; no full slice or hidden registry of prior observations is built.
    for position in (0, count // 2, count - 1):
        for index in range(max(0, position - 1), min(count, position + 2)):
            require(list(poses[index]) == list(rackets[index]), "Neighbor identities differ")
            audit["boundedNeighborReads"] += 1
    require(native_node(poses[-1]) == native_node(poses[count - 1]), "Negative pose read differs")
    require(native_node(rackets[-1]) == native_node(rackets[count - 1]), "Negative racket read differs")
    require(native_node(poses[0:2]) == native_node([poses[0], poses[1]]), "Bounded pose slice differs")
    require(native_node(rackets[0:2]) == native_node([rackets[0], rackets[1]]), "Bounded racket slice differs")
    return audit


def database_bytes(sequences, directory):
    files = {}
    for sequence in sequences:
        path = sequence.path
        require(path.resolve().parent == directory.resolve(), "Owned synthetic database escaped scratch directory")
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                files[name(candidate)] = candidate.stat().st_size
    return {"filesBeforeCleanup": files, "databaseAndSidecarLogicalBytesBeforeCleanup": sum(files.values()),
            "databaseMainLogicalBytesBeforeCleanup": sum(value for key, value in files.items() if key.endswith(".sqlite3")),
            "scope": "Actual logical file lengths just before context cleanup, including pending-transaction sidecars; not allocated blocks or total result disk."}


def child(args):
    import psutil
    began = time.perf_counter()
    directory = Path(args.directory)
    report = {"startedAt": stamp(), "mode": args.mode, "frames": args.frames, "passed": False,
              "inferenceRun": False, "modelsImported": False, "qualificationEvidence": False,
              "actualRenderOrMediaRun": False, "instrumentation": "No profiler, query patch, weak-finalizer tracker or method replacement; separate perf_counter spans and full byte fidelity outputs."}
    source_before = ordinary_before = None
    scratch = directory / "scratch"
    try:
        source_before = load(RUN / "launch_source_bindings.json")
        verify(source_before)
        fixture = load(RUN / f"fixture_{args.frames}_receipt.json")
        require(fixture["passed"] is True and fixture["frames"] == args.frames, "Wrong declared complete fixture")
        fixture_path = resolve(fixture["fixture"]["path"])
        require(binding(fixture_path) == {key: fixture["fixture"][key] for key in ("bytes", "sha256")},
                "Immutable complete fixture changed before workload")
        process = psutil.Process()
        identity = {"pid": process.pid, "createTimeUnixSeconds": process.create_time()}
        report["actualWorkloadIdentity"] = identity
        report["actualInterpreterBeforeImports"] = {"sysExecutable": sys.executable, "osProcessExe": process.exe(),
                                                   "osProcessExeBinding": binding(process.exe())}
        require(Path(process.exe()).resolve() == BASE.resolve(), "Actual interpreter differs from prepared base Python")
        require(report["actualInterpreterBeforeImports"]["osProcessExeBinding"] == source_before[name(BASE)],
                "Actual interpreter bytes differ from launch binding")
        handshake(directory / "actual_identity.json", identity)
        deadline = time.perf_counter() + 10
        while not (directory / "actual_identity_sampled.json").exists():
            require(time.perf_counter() <= deadline, "Actual interpreter sampling acknowledgement timed out")
            time.sleep(.01)
        acknowledgement = load(directory / "actual_identity_sampled.json")
        require(acknowledgement["identity"] == identity and acknowledgement["sampledByExactIdentity"] is True,
                "Actual interpreter exact sampled acknowledgement differs")
        report["parentActualIdentitySampledBeforeHeavyImports"] = True
        sys.addaudithook(import_network_guard)
        import_began = time.perf_counter()
        import numpy as np
        # Both cases import the same ordinary implementation/dependencies so
        # import-related OS peaks are comparable; old does not instantiate it.
        sys.path.insert(0, str(ROOT))
        from src.utils.identity_frame_sequence import IdentityFrameSequence, write_identity_frame_json
        sanitize, sanitizer_sha = frozen_sanitizer(np)
        reject_unsupported_imports()
        report["ordinaryImportsAndSanitizerExtractionSeconds"] = time.perf_counter() - import_began
        report["frozenSanitizerAstSha256"] = sanitizer_sha
        report["wholeFrozenPipelineImported"] = False
        ordinary_before, modules = ordinary_imports()
        report["actualImportedSourceBindingsBefore"] = ordinary_before
        report["actualImportedModules"] = modules
        save(directory / "imported_sources.json", {"bindings": ordinary_before, "modules": modules,
             "actualInterpreter": report["actualInterpreterBeforeImports"], "frozenSanitizerAstSha256": sanitizer_sha,
             "wholeFrozenPipelineImported": False})
        scratch.mkdir(exist_ok=False)
        with ExitStack() as stack:
            sequences = []
            if args.mode == "old":
                poses, rackets = [], []
            else:
                poses = stack.enter_context(IdentityFrameSequence(scratch))
                rackets = stack.enter_context(IdentityFrameSequence(scratch))
                sequences = [poses, rackets]
            construction_began = time.perf_counter()
            for index, timestamp, pose, racket in iter_fixture(fixture_path, args.frames):
                poses.append(pose)
                rackets.append(racket)
            report["completeObservationConstructionAndAppendSeconds"] = time.perf_counter() - construction_began
            require(len(poses) == len(rackets) == args.frames, "Incomplete constructed frame stores")
            report["constructedPoseFrames"] = len(poses)
            report["constructedRacketFrames"] = len(rackets)
            report["retainedObservationContainer"] = "two resident lists" if args.mode == "old" else "two owned disk sequences with bounded append/iteration"
            report["observationContainerRetainsAllFrames"] = args.mode == "old"
            report["newWorkloadMaterializedFullFrameArray"] = False if args.mode == "new" else None
            reads_began = time.perf_counter()
            render_audit = render_like_reads(poses, rackets, args.frames)
            report["indexedRenderLikeReadsSeconds"] = time.perf_counter() - reads_began
            save(directory / "render_audit.json", render_audit)
            export_metadata = fixture["metadata"]
            for file, key, frames, values in (("player_motion.json", "pose_frames", poses, export_metadata["playerMotion"]),
                                             ("racket_tracking.json", "frames", rackets, export_metadata["racketTracking"])):
                export_began = time.perf_counter()
                if args.mode == "old":
                    with (directory / file).open("w", encoding="utf-8") as stream:
                        json.dump(sanitize({**values, key: frames}), stream, indent=2, allow_nan=False)
                else:
                    write_identity_frame_json(directory / file, values, key, frames, sanitize)
                report["playerMotionSerializationSeconds" if key == "pose_frames" else "racketTrackingSerializationSeconds"] = time.perf_counter() - export_began
            # Separate full native verification preserves every pre-sanitizer
            # value/order/type/IEEE bit. It is not part of the read/export spans.
            fidelity_began = time.perf_counter()
            digest = hashlib.sha256()
            rows = 0
            with (directory / "observations.jsonl").open("xb") as stream:
                for index in range(args.frames):
                    encoded = record_bytes(index, index / FPS, poses[index], rackets[index])
                    stream.write(encoded)
                    digest.update(encoded)
                    rows += 1
            report["completeNativeFidelitySerializationSeconds"] = time.perf_counter() - fidelity_began
            report["completeNativeFidelityRows"] = rows
            report["completeNativeObservationSha256"] = digest.hexdigest()
            require(digest.hexdigest() == fixture["expectedNativeObservationSha256"],
                    "Complete native observations differ from immutable input, including IEEE bits/types/order")
            report["storageDisk"] = database_bytes(sequences, scratch) if sequences else {
                "filesBeforeCleanup": {}, "databaseAndSidecarLogicalBytesBeforeCleanup": 0,
                "databaseMainLogicalBytesBeforeCleanup": 0, "scope": "Old pose/racket observations are resident lists; no observation database."}
            cleanup_began = time.perf_counter()
        report["sequenceContextCleanupSeconds"] = time.perf_counter() - cleanup_began
        report["ownedScratchCleanupPassed"] = not any(scratch.iterdir())
        require(report["ownedScratchCleanupPassed"], "Successful owned database cleanup left files")
        hashes_began = time.perf_counter()
        report["outputBindings"] = {file: binding(directory / file) for file in OUTPUT_NAMES}
        report["outputHashVerificationSeconds"] = time.perf_counter() - hashes_began
        report["fixtureBindingBefore"] = {key: fixture["fixture"][key] for key in ("bytes", "sha256")}
        report["fixtureBindingAfter"] = binding(fixture_path)
        require(report["fixtureBindingAfter"] == report["fixtureBindingBefore"], "Fixture changed during child")
        verify(source_before)
        verify(ordinary_before)
        reject_unsupported_imports()
        report["actualImportedSourceBindingsAfter"], report["actualImportedModulesAfter"] = ordinary_imports()
        verify(report["actualImportedSourceBindingsAfter"])
        require(all(report["actualImportedSourceBindingsAfter"].get(key) == value for key, value in ordinary_before.items()),
                "Pre-imported ordinary source binding changed")
        report["allActualImportedSourcesUnchanged"] = True
        report["launchSourceBindingsUnchanged"] = True
        report["passed"] = True
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report["totalChildElapsedSeconds"] = time.perf_counter() - began
        report["totalChildElapsedScope"] = "Child function span after psutil identity-support import; parent observed process lifetime includes interpreter startup and all module imports."
        report["actualProcessMemoryBytes"] = psutil.Process().memory_info()._asdict()
        report["windowsPeakWorkingSetBytes"] = report["actualProcessMemoryBytes"].get("peak_wset")
        report["windowsPeakProcessCommitBytes"] = report["actualProcessMemoryBytes"].get("peak_pagefile")
        report["partialScratchFilesAtChildTerminal"] = {name(path): path.stat().st_size for path in scratch.rglob("*") if path.is_file()} if scratch.exists() else {}
        report["completedAt"] = stamp()
        save(directory / "child_receipt.json", report)
    return 0 if report["passed"] else 1


def monitor_case(mode, count, expected, monitor):
    import psutil
    verify(expected)
    directory = RUN / f"{count}_{mode}"
    directory.mkdir(exist_ok=False)
    report = {"startedAt": stamp(), "mode": mode, "frames": count, "passed": False, "samples": 0,
              "childStarted": False, "inferenceRun": False, "qualificationEvidence": False,
              "peakSampledOwnedTreeRssBytes": 0, "peakSampledOwnedTreePrivateBytes": 0,
              "minimumSampledAvailableRamBytes": None}
    process = owned = None
    began = time.perf_counter()
    try:
        available = psutil.virtual_memory().available
        report["availableRamImmediatelyBeforeLaunchBytes"] = available
        if available < PREFLIGHT:
            report["status"] = "NOT_STARTED"
            raise RuntimeError("Available RAM below unchanged 750 MB preflight; no child launch")
        arguments = [str(CLEAN), "-B", str(HERE / "helper.py"), "--child", "--mode", mode,
                     "--frames", str(count), "--directory", str(directory)]
        with (directory / "stdout.txt").open("x", encoding="utf-8") as log, \
                (directory / "resource_samples.jsonl").open("x", encoding="utf-8", newline="\n") as samples:
            launch_began = time.perf_counter()
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report["childStarted"] = True
            report["argv"] = arguments
            owned = monitor.OwnedProcessTree(process)
            report["redirectorIdentity"] = {"pid": owned.root_identity[0], "createTimeUnixSeconds": owned.root_identity[1]}
            low, acknowledged = 0, False
            while process.poll() is None:
                elapsed = time.perf_counter() - began
                available = psutil.virtual_memory().available
                rows = owned.sample()
                if not acknowledged and (directory / "actual_identity.json").exists():
                    identity = load(directory / "actual_identity.json")
                    require(set(identity) == {"pid", "createTimeUnixSeconds"}, "Noncanonical actual identity schema")
                    if owned.observed_workload(identity):
                        handshake(directory / "actual_identity_sampled.json", {
                            "at": stamp(), "identity": identity, "sampledByExactIdentity": True})
                        acknowledged = True
                        report["actualWorkloadIdentity"] = identity
                        report["sampleBeforeAckCodeOrder"] = "owned.sample -> observed_workload exact PID/createTime -> atomic ack"
                rss = sum(row["memoryBytes"]["rss"] for row in rows)
                private = [row["memoryBytes"].get("private") for row in rows]
                item = {"at": stamp(), "elapsedSeconds": elapsed, "processes": rows,
                        "availableRamBytes": available, "ownedTreeRssBytes": rss,
                        "ownedTreePrivateBytes": sum(value for value in private if value is not None),
                        "allPrivateValuesAvailable": all(value is not None for value in private),
                        "freeDiskBytes": shutil.disk_usage(ROOT).free}
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
            report["processLaunchToObservedExitSeconds"] = time.perf_counter() - launch_began
        require(not owned.wait_for_retirement(timeout=2), "Owned identities remain after child exit")
        receipt = load(directory / "child_receipt.json")
        report["childReceiptBinding"] = binding(directory / "child_receipt.json")
        require(process.returncode == 0 and receipt["passed"] is True, "Child failed; preserve all artifacts and halt later cases")
        require(receipt["actualWorkloadIdentity"] == report["actualWorkloadIdentity"]
                and owned.observed_workload(receipt["actualWorkloadIdentity"])
                and receipt["parentActualIdentitySampledBeforeHeavyImports"] is True,
                "Actual interpreter was not sampled/acknowledged by exact identity")
        require(receipt["windowsPeakWorkingSetBytes"] is not None and receipt["windowsPeakProcessCommitBytes"] is not None,
                "Actual Windows process peak counters unavailable")
        verify(expected)
        verify(receipt["actualImportedSourceBindingsBefore"])
        verify(receipt["actualImportedSourceBindingsAfter"])
        for file, value in receipt["outputBindings"].items():
            require(binding(directory / file) == value, "Output binding changed: " + file)
        report["actualWorkloadWasSampled"] = True
        report["actualOsPeakWorkingSetBytes"] = receipt["windowsPeakWorkingSetBytes"]
        report["actualOsPeakProcessCommitBytes"] = receipt["windowsPeakProcessCommitBytes"]
        report["totalChildElapsedSeconds"] = receipt["totalChildElapsedSeconds"]
        report["passed"] = True
        report["status"] = "COMPLETED"
    except BaseException as error:
        report["status"] = report.get("status", "FAILED")
        report["error"] = type(error).__name__ + ": " + str(error)
        if process is not None:
            try:
                report["terminationActions"] = monitor.stop_owned(process, owned)
            except BaseException as cleanup:
                report["cleanupError"] = repr(cleanup)
    finally:
        imported = directory / "imported_sources.json"
        if imported.exists():
            try:
                verify(load(imported)["bindings"])
                report["retainedImportedSourceRecordBinding"] = binding(imported)
                report["actualImportedSourceBindingsUnchangedAtTerminal"] = True
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
            except BaseException as retirement:
                report["retirementCheckError"] = repr(retirement)
                report["passed"] = False
        report["partialScratchFilesAtParentTerminal"] = {name(path): path.stat().st_size for path in (directory / "scratch").rglob("*") if path.is_file()}
        report["monitoredChildWallSeconds"] = time.perf_counter() - began
        report["completedAt"] = stamp()
        if not report["passed"] and report.get("status") == "COMPLETED":
            report["status"] = "FAILED"
        save(directory / "resource_receipt.json", report)
    return report


def exact_file(left, right):
    with Path(left).open("rb") as first, Path(right).open("rb") as second:
        while True:
            a, b = first.read(65536), second.read(65536)
            if a != b:
                return False
            if not a:
                return True


def compare_pair(count, fixture):
    values = {}
    for file in OUTPUT_NAMES:
        old, new = RUN / f"{count}_old" / file, RUN / f"{count}_new" / file
        values[file] = {"exactBytesEqual": exact_file(old, new), "old": binding(old), "new": binding(new)}
    for mode in ("old", "new"):
        child_receipt = load(RUN / f"{count}_{mode}/child_receipt.json")
        require(child_receipt["fixtureBindingBefore"] == child_receipt["fixtureBindingAfter"] ==
                {key: fixture["fixture"][key] for key in ("bytes", "sha256")}, "Paired cases did not use the same complete input")
        require(child_receipt["completeNativeFidelityRows"] == count and child_receipt["completeNativeObservationSha256"] ==
                fixture["expectedNativeObservationSha256"], "Full native fidelity proof incomplete")
    comparison = {"frames": count, "passed": all(item["exactBytesEqual"] for item in values.values()),
                  "outputs": values, "fieldExclusions": [], "metadata": fixture["metadata"],
                  "scope": "Full native ordered observations and exact unmodified pose/racket JSON bytes; no sampling/field filtering, rendering/inference or scientific qualification."}
    save(RUN / f"comparison_{count}.json", comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("old", "new"))
    parser.add_argument("--frames", type=int, choices=COUNTS)
    parser.add_argument("--directory")
    args = parser.parse_args()
    if args.child:
        require(args.mode and args.frames and args.directory, "Declared child arguments missing")
        directory = Path(args.directory).resolve()
        require(directory == RUN / f"{args.frames}_{args.mode}" and (RUN / "launch_marker.json").is_file(),
                "Child launch directory/authorization differs")
        return child(args)
    require(not args.mode and not args.frames and not args.directory, "Parent accepts only --run")
    RUN.mkdir(exist_ok=False)
    save(RUN / "launch_marker.json", {"at": stamp(), "exclusive": True, "qualificationEvidence": False})
    report = {"startedAt": stamp(), "passed": False, "runs": [], "comparisons": [], "fixtures": [],
              "status": "NOT_STARTED", "inferenceRun": False, "modelsImported": False, "qualificationEvidence": False,
              "actualRenderOrMediaRun": False, "scope": "Synthetic native pose/racket storage/export only; no actual pipeline or media run, accuracy, physics, full-match or model qualification."}
    expected, input_bindings = None, {}
    ordinary_union = {}
    try:
        verify(load(HERE / "prepared_bindings.json"))
        require(binding(FROZEN)["sha256"] == FROZEN_SHA256 and binding(MONITOR)["sha256"] == MONITOR_SHA256,
                "Frozen implementation or owned monitor changed")
        expected = source_bindings()
        backend_summary, backend_sources = backend_evidence_gate()
        for pathname, value in backend_sources.items():
            require(pathname not in expected or expected[pathname] == value, "Backend/current launch source binding conflict")
            expected[pathname] = value
        save(RUN / "launch_source_bindings.json", expected)
        report["sourceBindingsBefore"] = expected
        report["softwarePrerequisites"] = focused_evidence_gate()
        report["freshBackendPrerequisite"] = backend_summary
        monitor = module_at("_pose_racket_owned_monitor", MONITOR)
        for count in COUNTS:
            fixture = generate_fixture(count)
            report["fixtures"].append(fixture)
            for path in (RUN / f"fixture_{count}.jsonl", RUN / f"fixture_{count}_receipt.json"):
                input_bindings[name(path)] = binding(path)
            save(RUN / f"immutable_inputs_{count}.json", input_bindings)
            for selected in ("old", "new"):
                verify(expected)
                verify(input_bindings)
                print(json.dumps({"at": stamp(), "launching": f"{count}_{selected}"}), flush=True)
                resource = monitor_case(selected, count, expected, monitor)
                report["runs"].append(resource)
                child_path = RUN / f"{count}_{selected}/child_receipt.json"
                if child_path.exists():
                    child_report = load(child_path)
                    for field in ("actualImportedSourceBindingsBefore", "actualImportedSourceBindingsAfter"):
                        for pathname, value in child_report.get(field, {}).items():
                            require(pathname not in ordinary_union or ordinary_union[pathname] == value,
                                    "Ordinary imported source changed across cases: " + pathname)
                            ordinary_union[pathname] = value
                require(resource["passed"] is True, "First failed/declined case; preserve partial evidence and halt all later cases")
            comparison = compare_pair(count, fixture)
            report["comparisons"].append(comparison)
            require(comparison["passed"], "Full output byte difference; no later cases or retry")
        report["passed"] = True
        report["status"] = "COMPLETED"
    except BaseException as error:
        report["status"] = "FAILED_PARTIAL" if report["runs"] else "NOT_STARTED"
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        report["laterCasesHaltedAfterFirstFailure"] = True
    finally:
        report["inputBindingsBefore"] = input_bindings
        try:
            verify(input_bindings)
            report["inputBindingsAfter"] = {pathname: binding(resolve(pathname)) for pathname in input_bindings}
            report["allCompleteFixtureInputsUnchanged"] = report["inputBindingsAfter"] == input_bindings
            if expected is not None:
                verify(expected)
                report["sourceBindingsAfter"] = {pathname: binding(resolve(pathname)) for pathname in expected}
                report["sourcesUnchanged"] = report["sourceBindingsAfter"] == expected
            verify(ordinary_union)
            if report["runs"]:
                terminal_backend, terminal_sources = backend_evidence_gate()
                require(terminal_backend == report["freshBackendPrerequisite"], "Backend prerequisite evidence changed during measurements")
                require(all(expected[pathname] == value for pathname, value in terminal_sources.items()),
                        "Full backend source values changed during measurements")
                report["freshBackendPrerequisiteUnchangedAtTerminal"] = True
            report["actualImportedSourceBindingsUnion"] = ordinary_union
            report["actualImportedSourcesUnchangedAcrossMeasuredCases"] = True
        except BaseException as verification_error:
            report["terminalBindingError"] = repr(verification_error)
            report["passed"] = False
        if not report["passed"] and report["status"] == "COMPLETED":
            report["status"] = "FAILED_PARTIAL"
        report["completedAt"] = stamp()
        report["timingScope"] = "Construction includes complete fixture parsing/native observation construction/append; indexed read and each native JSON export are separate uninstrumented spans. Full typed observation verification, hashes, imports and fixture generation are separate. One old/new case per count, no repetitions or performance acceptance threshold."
        report["memoryScope"] = "OS actual interpreter peak working set and peak process commit are separate from sampled aggregate owned RSS/private. Windows peak_pagefile means process commit, not disk file usage. Peaks include imports/export/fidelity work; aggregate RSS can count shared pages twice and samples can miss brief peaks. Host guards include unrelated activity."
        report["retainedFileBindingsAtTerminal"] = {name(path): binding(path) for path in RUN.rglob("*") if path.is_file()}
        save(RUN / "report.json", report)
    print(json.dumps({key: report.get(key) for key in ("passed", "status", "error")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
