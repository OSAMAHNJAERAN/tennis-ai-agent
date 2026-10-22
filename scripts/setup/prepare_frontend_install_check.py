"""Copy pinned frontend manifests and optionally verify an isolated frozen restore."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / 'artifacts/validation/product_installation/frontend_env'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--verify', action='store_true', help='Run pnpm frozen/offline restore and record unchanged manifest hashes.')
args = parser.parse_args()
target.mkdir(parents=True, exist_ok=True)
expected = {}
for name in ['package.json', 'pnpm-lock.yaml']:
    expected[name] = hashlib.sha256((ROOT / 'frontend' / name).read_bytes()).hexdigest()
    shutil.copy2(ROOT / 'frontend' / name, target / name)
if not args.verify:
    print(str(target))
else:
    pnpm = shutil.which('pnpm')
    node = shutil.which('node')
    if not pnpm or not node:
        raise RuntimeError('The recorded Node.js and pnpm tools must be on PATH.')
    command = [pnpm, '--dir', str(target), 'install', '--frozen-lockfile', '--offline']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    node_version = subprocess.run([node, '--version'], capture_output=True, text=True, timeout=10).stdout.strip()
    # pnpm can use packageManager to select a cached project-specific release.
    # Probe in the same directory as install; the launcher outside it may differ.
    launcher_version = subprocess.run([pnpm, '--version'], capture_output=True, text=True, timeout=10).stdout.strip()
    pnpm_version = subprocess.run([pnpm, '--version'], cwd=target, capture_output=True, text=True, timeout=10).stdout.strip()
    checks = [{'path': name, 'expectedSha256': digest, 'copiedSha256': hashlib.sha256((target / name).read_bytes()).hexdigest(), 'originalSha256': hashlib.sha256((ROOT / 'frontend' / name).read_bytes()).hexdigest()} for name, digest in expected.items()]
    for check in checks:
        check['passed'] = check['expectedSha256'] == check['copiedSha256'] == check['originalSha256']
    package_store = target / 'node_modules/.pnpm'
    package_count = sum(path.is_dir() and path.name != 'node_modules' for path in package_store.iterdir()) if package_store.is_dir() else 0
    report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': result.returncode == 0 and pnpm_version == '11.19.0' and all(check['passed'] for check in checks), 'command': command, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr, 'nodeVersion': node_version, 'pnpmVersion': pnpm_version, 'launcherPnpmVersion': launcher_version, 'versionProbeDirectory': str(target), 'packageStoreDirectories': package_count, 'checks': checks, 'frozenLockfile': True, 'offline': True, 'productionBuildRun': False, 'selectedFrontendMutated': False}
    (target.parent / 'frontend_restore_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['passed', 'nodeVersion', 'pnpmVersion', 'packageStoreDirectories', 'productionBuildRun', 'selectedFrontendMutated']}))
    raise SystemExit(0 if report['passed'] else 1)
