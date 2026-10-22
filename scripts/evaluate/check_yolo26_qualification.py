"""Check retained promotion evidence; write an honest, hashed verification record."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

from src.detection.yolo26_bundle import REPO_ROOT, validate_yolo26_config
from src.evaluation.yolo26_qualification import CORE_TASKS, KNOWN_TASKS, sha256_file, validate_qualification_receipt


def check(config_path, *, root=REPO_ROOT, required_tasks=CORE_TASKS):
    root = Path(root).resolve()
    config_path = Path(config_path)
    config_path = config_path if config_path.is_absolute() else root / config_path
    record = {"schema_version": "tennis.yolo26.qualification-check.v1",
              "checked_at": datetime.now(timezone.utc).isoformat(),
              "verification_status": "UNQUALIFIED", "production_eligible": False,
              "vision_evaluation_executed": False, "required_tasks": sorted(required_tasks),
              "retained_hashes": {}}
    paths = {"configuration": config_path,
             "checker": root / "src/evaluation/yolo26_qualification.py",
             "bundle_gate": root / "src/detection/yolo26_bundle.py"}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        manifest_path = Path(config["model_system"]["artifact_manifest"])
        paths["artifact_manifest"] = manifest_path if manifest_path.is_absolute() else root / manifest_path
        manifest = validate_yolo26_config(config, check_assets=True, root=root)
        result = validate_qualification_receipt(manifest, config, root=root, required_tasks=required_tasks)
        record.update(verification_status="VERIFIED_EXTERNAL_RECEIPT", production_eligible=result["production_eligible"],
                      verified_receipt=result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        record["reason"] = str(exc)
    for key, path in paths.items():
        record["retained_hashes"][key] = {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/production/yolo26.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-task", action="append", choices=sorted(KNOWN_TASKS), default=[])
    args = parser.parse_args()
    record = check(args.config, required_tasks=CORE_TASKS | set(args.require_task))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        parser.error("Retain immutable check records; choose a new output path")
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    digest = sha256_file(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"verification_status": record["verification_status"], "record": str(args.output), "sha256": digest}))
    return 0 if record["production_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
