"""Retained software-only classifier checks, with model import tripwires."""

from pathlib import Path
import builtins
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
os.environ["TENNIS_STORAGE_ROOT"] = str(HERE / "isolated_storage")
for name in ("TEMP", "TMP", "TMPDIR"):
    os.environ[name] = str(HERE / "tmp")
(HERE / "tmp").mkdir(exist_ok=True)
PINS = [
    "src/shot_analysis/shot_classifier.py",
    "tests/test_shot_classifier_lazy_import.py",
    "tests/test_phase6_shot_analytics.py",
    "src/shot_analysis/pose_feature_extractor.py",
    "src/shot_analysis/shot_types.py",
    "src/shot_analysis/shot_linker.py",
    "src/shot_analysis/shot_direction.py",
    "src/shot_analysis/court_zones.py",
    "src/tracking/temporal_ball_tracker.py",
    "src/utils/bbox_utils.py",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINS}


before = hashes()
for name in PINS[:2]:
    destination = HERE / "source_snapshot" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, destination)

attempts = []
original_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "ultralytics"}:
        attempts.append(name)
        raise AssertionError("Classifier checks attempted a model runtime import: " + name)
    return original_import(name, *args, **kwargs)


started = time.perf_counter()
buffer = io.StringIO()
builtins.__import__ = guarded
try:
    with redirect_stdout(buffer), redirect_stderr(buffer):
        import pytest
        code = pytest.main([
            "tests/test_shot_classifier_lazy_import.py",
            "tests/test_phase6_shot_analytics.py",
            "--noconftest", "-p", "no:cacheprovider", "-q",
            "--basetemp", str(HERE / "pytest_tmp"),
            "--junitxml", str(HERE / "junit.xml"),
        ])
finally:
    builtins.__import__ = original_import

elapsed = time.perf_counter() - started
after = hashes()
junit = ET.parse(HERE / "junit.xml").getroot()
counts = {
    name: sum(int(suite.get(name, 0)) for suite in junit.iter("testsuite"))
    for name in ("tests", "failures", "errors", "skipped")
}
(HERE / "pytest.txt").write_text(buffer.getvalue(), encoding="utf-8")
receipt = {
    "scope": "SOFTWARE_CLASSIFIER_IMPORT_AND_BRANCHES_ONLY_NO_MODEL_INFERENCE",
    "exit_code": int(code),
    "junit": counts,
    "elapsed_seconds": elapsed,
    "source_before": before,
    "source_after": after,
    "source_pins_stable": before == after,
    "model_import_attempts": attempts,
    "imported": {name: name in sys.modules for name in ("torch", "ultralytics", "numpy", "pandas", "cv2")},
    "extractor_imported": "src.shot_analysis.pose_feature_extractor" in sys.modules,
    "fresh_process_tripwire": "New test invokes a fresh interpreter and checks model import attempts plus loaded modules.",
    "vision_accuracy_claim": False,
}
(HERE / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(json.dumps(receipt, indent=2))
print(buffer.getvalue()[-12000:])
raise SystemExit(int(code) if before == after and not attempts else 2)
