"""Capture the selected existing runtime, without downloading or importing models."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
snapshot_path = ROOT / 'artifacts/validation/product_installation/environment_snapshot.json'
snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
resolution_path = snapshot_path.parent / 'observed_dependency_resolution.json'
if not resolution_path.exists():
    resolution_path = snapshot_path.parent / 'dependency_resolution.json'
resolution = json.loads(resolution_path.read_text(encoding='utf-8'))
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
requirements = ROOT / 'requirements-product.txt'
direct = [line.strip() for line in requirements.read_text(encoding='utf-8').splitlines() if line.strip() and not line.startswith('#')]
distributions = {}
for row in resolution['install']:
    name = row['metadata']['name']
    current = metadata.distribution(name)
    distributions[name] = {'version': current.version, 'selection': 'OBSERVED_WORKING_ENVIRONMENT', 'requires': current.requires}
    if current.version == row['metadata']['version']:
        download = row.get('download_info', {})
        distributions[name]['resolvedWheelUrl'] = download.get('url')
        distributions[name]['resolvedWheelSha256'] = download.get('archive_info', {}).get('hashes', {}).get('sha256')
assets = []
for row in snapshot['assets']:
    if row['path'].startswith('C:'):
        continue
    row = dict(row)
    row.pop('exists', None)
    assets.append(row)
    if row['path'] == 'yolo11m.pt' or row['path'] == 'yolo11s-pose.pt':
        row.update(role='PERSON_RACKET_DETECTOR' if row['path'] == 'yolo11m.pt' else 'PLAYER_POSE_DETECTOR',
                   upstream='https://docs.ultralytics.com/models/yolo11/',
                   license='AGPL-3.0 or separately obtained Ultralytics Enterprise license',
                   licenseSource='https://www.ultralytics.com/license',
                   publicDeploymentPermission='NOT_VERIFIED_FOR_THIS_PROJECT',
                   restoration='Restore the retained exact bytes and verify this hash. Filename alone does not prove the publisher revision; do not silently substitute a newer asset.',
                   originalAcquisitionProvenance='NOT_RECORDED')
    elif row['path'] == 'models/keypoints_model.pth':
        row.update(role='FULL_COURT_KEYPOINT_STATE_DICT', upstream=None, license=None,
                   publicDeploymentPermission='UNKNOWN_SOURCE_AND_WEIGHT_LICENSE',
                   downloadUrl='https://github.com/OSAMAHNJAERAN/tennis-ai-agent/releases/download/project-snapshot-2026-10-22/keypoints_model.pth',
                   restoration='Download the exact retained asset from downloadUrl to models/keypoints_model.pth and verify the recorded SHA-256 and byte size. Original acquisition provenance and weight license remain unrecorded.',
                   originalAcquisitionProvenance='NOT_RECORDED')
    elif row['path'] == 'artifacts/models/ball/wasb_tennis_best.pth.tar':
        provenance = json.loads((ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.provenance.json').read_text())
        row.update(role='WASB_TENNIS_HRNET_CHECKPOINT', upstream=provenance['source_url'],
                   sourceModelZoo='https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md',
                   license=None, publicDeploymentPermission='WEIGHT_AND_TRAINING_FOOTAGE_TERMS_NOT_VERIFIED',
                   codeLicense='MIT applies to source code; a separate weight redistribution permission has not been established.',
                   restoration='Restore retained exact bytes, or acquire the publisher model-zoo asset after checking its terms and verify the recorded SHA-256/byte size.')
tooling = []
for name in ['ffmpeg', 'ffprobe']:
    found = shutil.which(name)
    if found:
        path = Path(found)
        result = subprocess.run([str(path), '-version'], capture_output=True, text=True, timeout=20)
        tooling.append({'name': name, 'observedPath': found, 'versionLine': result.stdout.splitlines()[0], 'bytes': path.stat().st_size, 'sha256': digest(path), 'mustBeOnPathOrSetEnvironment': 'TENNIS_' + name.upper(), 'restoration': 'Restore a matching trusted FFmpeg build with libx264, fps, pad, setpts, copyts and stats_enc_pre support; capability check is required.'})
manifest = {
    'schemaVersion': 1, 'capturedAt': datetime.now(timezone.utc).isoformat(),
    'scope': 'Current selected unqualified upload/report bundle; no accuracy certification or public deployment permission claim.',
    'productionQualified': False,
    'configuration': 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml',
    'python': {'observedVersion': snapshot['python']['version'], 'observedPlatform': snapshot['python']['platform'], 'validatedPlatform': 'Windows x86_64, CPython 3.13.5', 'requirementsPath': 'requirements-product.txt', 'requirementsSha256': digest(requirements), 'directRequirements': direct, 'runtimeDistributions': distributions, 'primaryIndex': 'https://pypi.org/simple', 'torchIndex': 'https://download.pytorch.org/whl/cu126', 'legacyRequirementsPreserved': 'requirements.txt', 'opencvProvider': 'opencv-python only; the working global environment contains multiple cv2 providers, which the isolated product environment avoids.'},
    'frontend': {'packageManifest': 'frontend/package.json', 'lockFile': 'frontend/pnpm-lock.yaml', 'packageManager': 'pnpm@11.19.0', 'nodeRequirement': '>=22', 'installCommand': 'pnpm --dir frontend install --frozen-lockfile', 'buildCommand': 'pnpm --dir frontend build'},
    'sourceDependencies': [{'path': 'artifacts/research/WASB-SBDT', 'origin': snapshot['source']['origin'], 'commit': snapshot['source']['commit'], 'workingTreeCleanAtCapture': snapshot['source']['status'] == '', 'license': 'MIT', 'licenseFile': 'artifacts/research/WASB-SBDT/LICENSE.md', 'restoration': 'Clone into an absent target, then git checkout --detach 923462cacdeb3353b84ddebdedb3f4b7a8553b0f. If the directory exists, verify it instead of replacing it.'}],
    'assets': assets, 'externalTools': tooling,
    'bootstrap': {'courtArchitectureWhenFullCheckpointIsPresent': 'ResNet50 with weights=None, strict full-state loading; no ImageNet cache required.', 'checkpointDeserialization': 'Existing local state dicts use weights_only=True. Only restore known trusted model assets.', 'modelImportsDuringManifestCapture': False},
    'persistentState': {'root': 'outputs/product', 'database': 'outputs/product/analyses.sqlite3', 'preserve': ['SQLite database and WAL/SHM files when present', 'uploads with their original video files', 'analyses with JSON, video, source timing and source credits'], 'backupPolicy': 'Take a coherent SQLite backup and retain upload/analysis directories. Do not replace this state during setup.'},
    'validation': {'directDependencyResolution': 'PASSED in isolated environment with no installed product packages; unconstrained transitive selection was inspected but not used as a working-environment lock.', 'observedTransitiveConstraintResolution': 'PENDING', 'cleanInstall': 'PENDING', 'apiSmokeWithoutWorker': 'PENDING', 'modelLoadAndInferenceInCleanEnvironment': 'NOT_RUN', 'courtBootstrapTests': '4 passed, including offline full checkpoint and strict missing/shape rejection', 'frontendFrozenInstall': 'NOT_RUN_BY_THIS_MANIFEST_CAPTURE'},
    'unresolvedReleaseRequirements': ['Court checkpoint source, authorship and weight license are unknown.', 'Existing YOLO asset acquisition revisions are not independently recorded; project public deployment licensing has not been verified.', 'WASB code is MIT, but separate checkpoint/training-footage redistribution terms are not certified.', 'A clean full model load, end-to-end inference and independent accuracy qualification remain required before claiming a reproducible qualified production release.'],
}
destination = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
destination.parent.mkdir(parents=True, exist_ok=True)
from refresh_product_installation_evidence import refresh
manifest = refresh(manifest)
destination.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
constraints = snapshot_path.parent / 'observed_constraints.txt'
constraints.write_text('\n'.join(name + '==' + row['version'] for name, row in sorted(distributions.items(), key=lambda item: item[0].lower())) + '\n', encoding='utf-8')
print(json.dumps({'manifest': destination.relative_to(ROOT).as_posix(), 'runtimeDistributions': len(distributions), 'constraints': constraints.relative_to(ROOT).as_posix(), 'assets': len(assets), 'modelsImported': False}))
