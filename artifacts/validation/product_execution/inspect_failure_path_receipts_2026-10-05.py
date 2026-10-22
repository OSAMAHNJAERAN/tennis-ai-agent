"""Read-only summary of named Vitest receipts; no source or receipt changes."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[3]
for name in sys.argv[1:]:
    path = root / name
    result = json.loads(path.read_text(encoding="utf-8"))
    failures = [
        {"name": case["fullName"], "messages": [message[:500] for message in case.get("failureMessages", [])]}
        for suite in result.get("testResults", [])
        for case in suite.get("assertionResults", [])
        if case["status"] == "failed"
    ]
    print(json.dumps({"path": name, "passed": result.get("numPassedTests"),
                      "failed": result.get("numFailedTests"), "skipped": result.get("numPendingTests"),
                      "files": len(result.get("testResults", [])), "success": result.get("success"),
                      "failures": failures}, ensure_ascii=True))
