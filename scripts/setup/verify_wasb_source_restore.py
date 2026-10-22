"""Clone the pinned public source into a fresh check directory and verify runtime files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json').read_text(encoding='utf-8'))
source = manifest['sourceDependencies'][0]
output = ROOT / 'artifacts/validation/product_installation'
target = output / 'wasb_restore_check'
target.resolve().relative_to(ROOT.resolve())
if target.exists():
    raise ValueError('Restoration check target already exists; verify the retained report rather than replacing it.')
output.mkdir(parents=True, exist_ok=True)
log = []
for command in [
    ['git', 'clone', '--no-checkout', source['origin'], str(target)],
    ['git', '-C', str(target), '-c', 'core.autocrlf=true', 'checkout', '--detach', source['commit']],
]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    log.append({'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    if result.returncode:
        (output / 'wasb_source_restore_check.json').write_text(json.dumps({'passed': False, 'commands': log}), encoding='utf-8')
        raise RuntimeError('Pinned source restoration failed; retained command diagnostics identify the blocker.')
checks = []
for asset in manifest['assets']:
    prefix = source['path'] + '/'
    if not asset['path'].startswith(prefix):
        continue
    path = target / asset['path'][len(prefix):]
    with path.open('rb') as stream:
        observed = hashlib.file_digest(stream, 'sha256').hexdigest()
    checks.append({'path': asset['path'], 'expectedSha256': asset['sha256'], 'observedSha256': observed, 'passed': observed == asset['sha256'] and path.stat().st_size == asset['bytes']})
report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': all(row['passed'] for row in checks), 'checks': checks, 'commands': log, 'sourceCommit': source['commit'], 'lineEndingPolicy': 'core.autocrlf=true reproduces the observed Windows source checkout bytes', 'modelsImported': False, 'inferenceRun': False}
(output / 'wasb_source_restore_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'passed': report['passed'], 'checkCount': len(checks), 'sourceCommit': source['commit'], 'modelsImported': False}))
raise SystemExit(0 if report['passed'] else 1)
