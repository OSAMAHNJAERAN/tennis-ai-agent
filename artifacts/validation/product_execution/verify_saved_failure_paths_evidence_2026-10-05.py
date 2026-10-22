"""Recheck the explicit sealed input/build/evidence scope and current host receipt."""
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[3]
folder = root / "artifacts/validation/product_installation/frontend_source_restore_failure_paths_2026-10-05"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

seal = read(folder / "verification_seal.json")
receipt = read(folder / "receipt.json")
assert seal["passed"] and receipt["passed"]
for name, expected in seal["evidenceBindings"].items():
    assert digest(root / name) == expected, name
target = root / receipt["target"]
for selected, copied, bindings in (
    (root / "frontend", target, receipt["sourceBindings"]),
    (root, folder, receipt["sharedBindings"]),
):
    for name, expected in bindings.items():
        assert digest(selected / name) == digest(copied / name) == expected, name
verifier = root / "scripts/setup/verify_frontend_source_restore.py"
assert digest(verifier) == receipt["verifierSha256"]
spec = importlib.util.spec_from_file_location("restore_inputs", verifier)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for frontend in (root / "frontend", target):
    assert {path.relative_to(frontend).as_posix() for path in module.frontend_inputs(frontend)} == set(receipt["sourceBindings"])
    actual_build = {path.relative_to(frontend / "dist").as_posix(): digest(path)
                    for path in (frontend / "dist").rglob("*") if path.is_file()}
    assert actual_build == seal["buildBindings"]
host_path = root / "artifacts/validation/product_installation/failure_paths_source_restoration_host_verification_2026-10-05.json"
host = read(host_path)
manifest_path = root / "docs/deployment/PRODUCT_RESTORE_MANIFEST.json"
manifest = read(manifest_path)
assert host["checksPassed"] and len(host["checks"]) == 75 and all(check["passed"] for check in host["checks"])
assert host["warnings"] == [] and host["manifestSha256"] == digest(manifest_path)
assert host["apiSmoke"]["healthStatus"] == host["apiSmoke"]["analysesStatus"] == 200
assert host["apiSmoke"]["invalidUploadStatus"] == 415 and host["apiSmoke"]["emptyIsolatedLibrary"]
assert host["apiSmoke"]["workerDisabled"] and host["apiSmoke"]["modelModulesImported"] == []
assert not host["inferenceRun"] and not host["productionQualified"]
for item in manifest["validationEvidence"]:
    if item["path"].startswith(folder.relative_to(root).as_posix() + "/"):
        assert digest(root / item["path"]) == item["sha256"]
browser = read(root / "artifacts/validation/product_execution/failure_paths_browser_2026-10-05.json")
assert digest(root / "frontend/dist" / browser["entry"].removeprefix("/")) == seal["browserEntrySha256"]
original = root / "outputs/product/analyses/match_9de7cb2d09eb46a0af0e34c3b0e6b818/source.mp4"
assert digest(original) == "2143a4f12227ea0fd119610f124064968584676d43d0a847e5bf49b0cabb0bfb"
result = {
    "checkedAt": datetime.now(timezone.utc).isoformat(), "passed": True,
    "scope": "31 sealed evidence files, 122 selected/copied inputs, 37 identical selected/copied build files, refreshed 75-check host receipt and one original public source hash.",
    "hostVerifiedAt": host["verifiedAt"], "hostSha256": digest(host_path),
    "manifestSha256": digest(manifest_path), "sealSha256": digest(folder / "verification_seal.json"),
    "browserEntrySha256": seal["browserEntrySha256"], "originalSourceSha256": digest(original),
    "qualificationEvidence": False,
    "limitations": "No whole-workspace integrity claim, actual blocked muted autoplay or native GPU failure, independent model accuracy, valid reconstruction, public deployment rights or full-match capacity qualification.",
}
with (root / "artifacts/validation/product_execution/failure_paths_evidence_recheck_2026-10-05.json").open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2)
    output.write("\n")
print(json.dumps(result))
