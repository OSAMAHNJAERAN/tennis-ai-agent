"""One-shot isolated identity-frame regressions; no models or inference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent / "storage_focused"
SOURCES = ("src/utils/identity_frame_sequence.py", "src/pipeline/phase6_pipeline.py",
           "tests/test_identity_frame_sequence.py")


def binding():
    return {relative: {"sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                       "bytes": (ROOT / relative).stat().st_size} for relative in SOURCES}


def save(name, value):
    with (UNIT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    UNIT.mkdir(exist_ok=False)
    before = binding()
    for relative in SOURCES:
        target = UNIT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((ROOT / relative).read_bytes())
        assert hashlib.sha256(target.read_bytes()).hexdigest() == before[relative]["sha256"]
    command = [sys.executable, "-m", "pytest", "tests/test_identity_frame_sequence.py", "-q",
               "--junitxml=" + str(UNIT / "junit.xml"), "--basetemp=" + str(UNIT / "tmp")]
    save("preparation.json", {"atUtc": datetime.now(timezone.utc).isoformat(), "command": command,
                              "sourceBefore": before,
                              "scope": "Isolated observation codec/SQLite/raw-export tests; no inference or models."})
    started = time.perf_counter()
    with (UNIT / "output.log").open("xb") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False)
    after = binding()
    receipt = {"atUtc": datetime.now(timezone.utc).isoformat(), "exitCode": result.returncode,
               "elapsedSeconds": time.perf_counter() - started, "sourceBefore": before,
               "sourceAfter": after, "sourceUnchanged": before == after,
               "outputSha256": hashlib.sha256((UNIT / "output.log").read_bytes()).hexdigest()}
    if (UNIT / "junit.xml").exists():
        receipt["junitSha256"] = hashlib.sha256((UNIT / "junit.xml").read_bytes()).hexdigest()
    save("receipt.json", receipt)
    print((UNIT / "output.log").read_text(encoding="utf-8", errors="replace").encode("ascii", "backslashreplace").decode("ascii"))
    print(json.dumps(receipt, ensure_ascii=True))
    return result.returncode if before == after else 2


if __name__ == "__main__":
    raise SystemExit(main())
