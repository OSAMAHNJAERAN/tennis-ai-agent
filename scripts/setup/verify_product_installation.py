"""Verify product restoration without model imports, inference, or state replacement."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
    parser.add_argument('--write-constraints', help='Write recorded transitive pins to a workspace path, then exit.')
    parser.add_argument('--report', help='Write verification JSON to a workspace path.')
    parser.add_argument('--api-smoke', action='store_true', help='Run an isolated API smoke check with its worker disabled; no inference.')
    args = parser.parse_args()
    manifest_path = inside(ROOT, args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if args.write_constraints:
        path = inside(ROOT, args.write_constraints)
        path.parent.mkdir(parents=True, exist_ok=True)
        distributions = manifest['python']['runtimeDistributions']
        path.write_text('\n'.join(name + '==' + row['version'] for name, row in sorted(distributions.items(), key=lambda item: item[0].lower())) + '\n', encoding='utf-8')
        print(json.dumps({'constraintsPath': str(path), 'distributions': len(distributions), 'modelsImported': False}))
        return

    checks, warnings = [], []
    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    requirements = inside(ROOT, manifest['python']['requirementsPath'])
    check('product_requirements_hash', requirements.is_file() and digest(requirements) == manifest['python']['requirementsSha256'])
    for name, row in manifest['python']['runtimeDistributions'].items():
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        check('package:' + name, version == row['version'], expected=row['version'], observed=version)
    providers = []
    for name in ['opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless']:
        try:
            providers.append(name + '==' + metadata.version(name))
        except metadata.PackageNotFoundError:
            pass
    if len(providers) != 1:
        warnings.append({'code': 'MULTIPLE_OR_MISSING_OPENCV_PROVIDERS', 'providers': providers})
    for asset in manifest['assets']:
        path = inside(ROOT, asset['path'])
        observed = digest(path) if path.is_file() else None
        check('asset:' + asset['path'], observed == asset['sha256'] and path.stat().st_size == asset['bytes'] if path.is_file() else False,
              expectedSha256=asset['sha256'], observedSha256=observed)
    for source in manifest['sourceDependencies']:
        path = inside(ROOT, source['path'])
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        revision = result.stdout.strip() if result.returncode == 0 else None
        status = subprocess.run(['git', '-C', str(path), 'status', '--short'], capture_output=True, text=True)
        check('source:' + source['path'], revision == source['commit'] and status.returncode == 0 and not status.stdout.strip(),
              expectedCommit=source['commit'], observedCommit=revision, clean=status.returncode == 0 and not status.stdout.strip())
    for tool in manifest['externalTools']:
        path = os.getenv(tool['mustBeOnPathOrSetEnvironment']) or shutil.which(tool['name'])
        if not path:
            check('tool:' + tool['name'], False, error='Missing executable')
            continue
        result = subprocess.run([path, '-version'], capture_output=True, text=True, timeout=20)
        version = result.stdout.splitlines()[0] if result.stdout else None
        check('tool:' + tool['name'], result.returncode == 0 and version == tool['versionLine'], expected=tool['versionLine'], observed=version)
        if tool['name'] == 'ffmpeg':
            help_result = subprocess.run([path, '-h', 'full'], capture_output=True, text=True, timeout=20)
            filters = subprocess.run([path, '-filters'], capture_output=True, text=True, timeout=20)
            codecs = subprocess.run([path, '-encoders'], capture_output=True, text=True, timeout=20)
            check('ffmpeg_source_timing_capabilities', '-stats_enc_pre' in help_result.stdout and all(name in filters.stdout for name in [' fps ', ' pad ', ' setpts ']) and 'libx264' in codecs.stdout)

    api_smoke = None
    if args.api_smoke:
        smoke_root = inside(ROOT, 'artifacts/validation/product_installation/api_smoke_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        smoke_root.mkdir(parents=True)
        os.environ['TENNIS_STORAGE_ROOT'] = str(smoke_root)
        sys.path.insert(0, str(ROOT))
        from src.api.app import create_app
        from src.api.jobs import Settings
        from fastapi.testclient import TestClient
        application = create_app(Settings(smoke_root, inside(ROOT, manifest['configuration'])), start_worker=False)
        with TestClient(application) as client:
            health = client.get('/api/health')
            analyses = client.get('/api/analyses')
            invalid = client.post('/api/analyses?filename=bad.mp4', content=b'not-a-video')
        imported = [name for name in ['torch', 'torchvision', 'ultralytics'] if name in sys.modules]
        api_smoke = {'healthStatus': health.status_code, 'health': health.json(), 'analysesStatus': analyses.status_code, 'emptyIsolatedLibrary': analyses.json() == [], 'invalidUploadStatus': invalid.status_code, 'workerDisabled': health.json().get('worker_available') is False, 'modelModulesImported': imported, 'storagePath': str(smoke_root), 'inferenceRun': False}
        check('isolated_api_smoke', health.status_code == 200 and analyses.status_code == 200 and analyses.json() == [] and invalid.status_code == 415 and not imported and health.json().get('worker_available') is False)

    report = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'manifestSha256': digest(manifest_path), 'pythonEnvironment': {'interpreter': sys.executable, 'version': platform.python_version(), 'platform': platform.platform(), 'isolated': sys.prefix != sys.base_prefix}, 'checksPassed': all(row['passed'] for row in checks), 'checks': checks, 'warnings': warnings, 'apiSmoke': api_smoke, 'modelsImported': [name for name in ['torch', 'torchvision', 'ultralytics'] if name in sys.modules], 'inferenceRun': False, 'productionQualified': False, 'publicDeploymentPermissionsVerified': False, 'remainingReleaseRequirements': manifest['unresolvedReleaseRequirements']}
    if args.report:
        path = inside(ROOT, args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'checksPassed': report['checksPassed'], 'checkCount': len(checks), 'failures': [row for row in checks if not row['passed']], 'warnings': warnings, 'apiSmoke': api_smoke, 'inferenceRun': False, 'productionQualified': False}, ensure_ascii=True))
    raise SystemExit(0 if report['checksPassed'] else 1)


if __name__ == '__main__':
    main()
