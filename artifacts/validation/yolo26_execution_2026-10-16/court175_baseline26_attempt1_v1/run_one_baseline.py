from pathlib import Path
import json, sys
sys.path.insert(0, str(Path.cwd()))
from scripts.train import train_yolo26_court_repair as training
from scripts.evaluate import diagnose_yolo26_court_repair as diagnostic
root=Path.cwd()
artifact=root/"artifacts/validation/yolo26_execution_2026-10-16/court175_baseline26_attempt1_v1"
protocol=root/"artifacts/validation/yolo26_execution_2026-10-16/court175_trainer_preflight_v1/native26_protocol.json"
plan=diagnostic.prepare(protocol,"c805089d1d06ed42f686778bd8c080a5a069134415d2d9f05c7fb508630d5e73",training.INITIAL,training.INITIAL_SHA,phase="baseline")
training.write_json(artifact/"refreshed_frozen_plan.json",plan)
result=diagnostic.execute(plan,root/"outputs/yolo26_court175_baseline26_attempt1_2026-10-16")
print(json.dumps({"status":result["status"],"rows":len(result["rows"]),"summary":result.get("aggregates")}))
