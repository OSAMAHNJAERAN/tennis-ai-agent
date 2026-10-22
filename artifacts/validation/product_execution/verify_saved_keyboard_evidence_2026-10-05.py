"""Read-only recheck of the saved final keyboard repair and restoration evidence."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


restore = Path("artifacts/validation/product_installation/frontend_source_restore_keyboard_2026-10-05")
seal = read(restore / "verification_seal.json")
receipt = read(restore / "receipt.json")
host = read("artifacts/validation/product_installation/keyboard_source_restoration_host_verification_2026-10-05.json")
for relative, expected in seal["evidenceBindings"].items():
    assert digest(relative) == expected, relative
for folder, bindings in (
    (Path("frontend"), receipt["sourceBindings"]),
    (Path("."), receipt["sharedBindings"]),
    (Path("frontend/dist"), seal["buildBindings"]),
):
    for relative, expected in bindings.items():
        assert digest(folder / relative) == expected, relative
assert host["checksPassed"] and len(host["checks"]) == 75
assert host["manifestSha256"] == digest("docs/deployment/PRODUCT_RESTORE_MANIFEST.json")
preview = read("artifacts/validation/product_execution/keyboard_final_preview_2026-10-05.json")
assert preview["entrySha256"] == digest(Path("frontend/dist") / preview["entry"].removeprefix("/"))
assert not host["inferenceRun"] and not host["productionQualified"]
print(json.dumps({"checkedAt": datetime.now(timezone.utc).isoformat(), "passed": True,
                  "boundEvidenceFiles": len(seal["evidenceBindings"]),
                  "currentInputs": len(receipt["sourceBindings"]) + len(receipt["sharedBindings"]),
                  "productionFiles": len(seal["buildBindings"]), "hostChecks": len(host["checks"]),
                  "manifestSha256": host["manifestSha256"], "qualificationEvidence": False}))
