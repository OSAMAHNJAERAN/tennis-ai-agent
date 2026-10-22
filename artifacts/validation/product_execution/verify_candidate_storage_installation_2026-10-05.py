"""One-shot, inference-free verification of the updated restoration manifest."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'artifacts/validation/product_installation/candidate_storage_host_verification_2026-10-05.json'
LOG = Path(__file__).with_suffix('.log')
BINDING = Path(__file__).with_suffix('.json')
MANIFEST = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
VERIFIER = ROOT / 'scripts/setup/verify_product_installation.py'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert not any(path.exists() for path in (REPORT, LOG, BINDING)), 'Preserve previous verification evidence'
before = {str(path.relative_to(ROOT).as_posix()): digest(path) for path in (MANIFEST, VERIFIER, CLEAN, Path(__file__))}
started = datetime.now(timezone.utc).isoformat()
with LOG.open('x', encoding='utf-8') as log:
    result = subprocess.run([str(CLEAN), str(VERIFIER), '--api-smoke', '--report', REPORT.relative_to(ROOT).as_posix()],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
after = {name: digest(ROOT / name) for name in before}
assert before == after, 'Verification inputs changed during the check'
report = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.is_file() else None
passed = bool(result.returncode == 0 and report and report['checksPassed'] and not report['inferenceRun']
              and not report['modelsImported'] and report['manifestSha256'] == before[MANIFEST.relative_to(ROOT).as_posix()])
receipt = {'startedAt': started, 'completedAt': datetime.now(timezone.utc).isoformat(), 'passed': passed,
           'exitCode': result.returncode, 'inputs': before, 'inputsUnchanged': before == after,
           'report': REPORT.relative_to(ROOT).as_posix(), 'reportSha256': digest(REPORT) if REPORT.exists() else None,
           'logSha256': digest(LOG), 'inferenceRun': False,
           'checkCount': len(report['checks']) if report else 0}
with BINDING.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'passed': passed, 'exitCode': result.returncode, 'checkCount': receipt['checkCount'],
                  'manifestSha256': before[MANIFEST.relative_to(ROOT).as_posix()], 'inferenceRun': False}))
raise SystemExit(0 if passed else 1)
