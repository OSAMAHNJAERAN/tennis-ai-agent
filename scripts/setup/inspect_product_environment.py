"""Inspect installed distribution metadata and runtime assets without importing models."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'artifacts/validation/product_installation'
out.mkdir(parents=True, exist_ok=True)
def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

pending = ['src.api.app', 'src.api.worker', 'src.pipeline.phase6_pipeline']
visited, external = {}, {}
while pending:
    module = pending.pop()
    path = ROOT / (module.replace('.', '/') + '.py')
    if not path.exists():
        path = ROOT / module.replace('.', '/') / '__init__.py'
    if not path.exists() or module in visited:
        continue
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    visited[module] = path.relative_to(ROOT).as_posix()
    for node in ast.walk(tree):
        modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        for name in modules:
            if name.startswith('src.'):
                pending.append(name)
            elif name.split('.')[0] not in sys.stdlib_module_names and name.split('.')[0] != '__future__':
                external.setdefault(name.split('.')[0], []).append(path.relative_to(ROOT).as_posix())
for rel in ['artifacts/research/WASB-SBDT/src/models/hrnet.py', 'artifacts/research/WASB-SBDT/src/utils/image.py']:
    path = ROOT / rel
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in ast.walk(tree):
        modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        for name in modules:
            if name.split('.')[0] not in sys.stdlib_module_names and name.split('.')[0] != '__future__':
                external.setdefault(name.split('.')[0], []).append(rel)
packages = ['aiofiles', 'fastapi', 'filelock', 'httpx', 'numpy', 'opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'pandas', 'pydantic', 'python-multipart', 'PyYAML', 'torch', 'torchvision', 'tqdm', 'ultralytics', 'uvicorn', 'lap', 'scipy', 'pillow', 'matplotlib', 'polars', 'ultralytics-thop', 'psutil', 'requests', 'pytest', 'starlette', 'pip']
distributions = {}
for name in packages:
    try:
        distribution = metadata.distribution(name)
        distributions[name] = {'version': distribution.version, 'requires': distribution.requires, 'license': distribution.metadata.get('License'), 'location': str(distribution.locate_file('')), 'directUrl': distribution.read_text('direct_url.json')}
    except metadata.PackageNotFoundError:
        distributions[name] = None
assets = []
for rel in ['configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml', 'yolo11m.pt', 'yolo11s-pose.pt', 'models/keypoints_model.pth', 'artifacts/models/ball/wasb_tennis_best.pth.tar', 'artifacts/models/ball/wasb_tennis_best.pth.provenance.json', 'artifacts/research/WASB-SBDT/LICENSE.md', 'artifacts/research/WASB-SBDT/src/models/hrnet.py', 'artifacts/research/WASB-SBDT/src/utils/image.py', 'artifacts/research/WASB-SBDT/src/configs/model/wasb.yaml', 'frontend/package.json', 'frontend/pnpm-lock.yaml']:
    path = ROOT / rel
    assets.append({'path': rel, 'exists': path.is_file(), 'bytes': path.stat().st_size if path.is_file() else None, 'sha256': digest(path) if path.is_file() else None})
cache = Path(os.getenv('TORCH_HOME', str(Path.home() / '.cache/torch'))) / 'hub/checkpoints/resnet50-11ad3fa6.pth'
assets.append({'path': str(cache), 'purpose': 'Implicit torchvision ResNet50_Weights.DEFAULT initialization cache', 'exists': cache.is_file(), 'bytes': cache.stat().st_size if cache.is_file() else None, 'sha256': digest(cache) if cache.is_file() else None})
repo = ROOT / 'artifacts/research/WASB-SBDT'
source = {'commit': subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(), 'status': subprocess.run(['git', '-C', str(repo), 'status', '--short'], capture_output=True, text=True).stdout.strip(), 'origin': subprocess.run(['git', '-C', str(repo), 'remote', 'get-url', 'origin'], capture_output=True, text=True).stdout.strip()}
report = {'capturedAt': datetime.now(timezone.utc).isoformat(), 'python': {'version': platform.python_version(), 'platform': platform.platform(), 'executable': sys.executable}, 'diskFreeBytes': shutil.disk_usage(ROOT).free, 'source': source, 'assets': assets, 'selectedRuntimeImportClosure': visited, 'externalImports': {k: sorted(set(v)) for k, v in external.items()}, 'distributions': distributions, 'modelsImported': False}
(out / 'environment_snapshot.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'python': report['python'], 'diskFreeBytes': report['diskFreeBytes'], 'source': source, 'externalImports': report['externalImports'], 'versions': {k: v['version'] if v else None for k, v in distributions.items()}, 'assets': assets}, ensure_ascii=True))
