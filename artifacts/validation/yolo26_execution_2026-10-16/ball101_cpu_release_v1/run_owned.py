"""One root-reviewed CPU inspection; retained terminal and source observations."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.train import train_yolo26_author_ball_repair as t

directory = Path(__file__).resolve().parent
approval_path = directory / 'approval.json'
plan_path = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/ball101_trainer_preflight_v2/plan.json'
approval, plan = t.read(approval_path), t.read(plan_path)
assert t.digest(plan_path) == approval['plan_file_sha256']
assert t.exported.canonical_hash(plan) == approval['plan_sha256']
output = ROOT / approval['preview_output']
receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), approval=t.binding(approval_path),
    plan=t.binding(plan_path), output=str(output), status='CPU_ONLY_INSPECTION_STARTED',
    training_performed=False, gpu_training_approved=False, qualification_evidence=False)
clock = time.perf_counter()
code = 1
try:
    receipt['resources_before'] = t.resource_snapshot(output)
    result = t.cpu_preview(plan, output, approval)
    receipt['cpu_receipt'] = t.binding(output / 'receipt.json')
    receipt['status'] = result['status']
    code = 0
except BaseException as error:
    receipt.update(status='FAILED_CPU_INSPECTION_NO_AUTOMATIC_RETRY', failure_type=type(error).__name__,
        failure_reason=str(error), traceback=traceback.format_exc())
    if (output / 'receipt.json').exists():
        receipt['cpu_receipt'] = t.binding(output / 'receipt.json')
finally:
    receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
        exit_code=code, resources_after=t.resource_snapshot(output), runner=t.binding(Path(__file__)))
    t.exported.write_new_json(directory / 'receipt.json', receipt)
print(json.dumps(receipt, allow_nan=False))
raise SystemExit(code)
