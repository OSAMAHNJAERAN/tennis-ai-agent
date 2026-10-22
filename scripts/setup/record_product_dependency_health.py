"""Record pip dependency health for the interpreter running this check."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=30)
report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': result.returncode == 0, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'interpreter': sys.executable, 'modelsImported': False, 'inferenceRun': False}
destination = ROOT / 'artifacts/validation/product_installation/clean_dependency_health.json'
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=True))
raise SystemExit(result.returncode)
