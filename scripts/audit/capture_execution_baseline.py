"""Preserve the working source and git state without resetting or committing it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import zipfile


def capture(root: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    roots = ['src', 'frontend/src', 'configs', 'scripts', 'tests', 'docs']
    paths = set()
    for folder in roots:
        paths.update(p for p in (root / folder).rglob('*') if p.is_file()
                     and not any(s in {'__pycache__', 'node_modules', '.pytest_cache', '.git'} for s in p.parts)
                     and p.suffix.lower() in {'.py', '.ts', '.tsx', '.css', '.json', '.yaml', '.yml', '.md', '.toml'})
    paths.update(p for p in root.iterdir() if p.is_file() and p.suffix in {'.md', '.txt', '.json', '.css'})
    paths.update(root / p for p in ['frontend/package.json', 'frontend/pnpm-lock.yaml', 'data/data_registry.yaml'])
    files = []
    archive = destination / 'working-source.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(paths):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            out.writestr(relative, data)
            files.append({'path': relative, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    git = ['git', '-c', f'safe.directory={root.as_posix()}']
    def query(args):
        result = subprocess.run(git + args, cwd=root, capture_output=True, text=True)
        return {'returncode': result.returncode, 'output': result.stdout, 'error': result.stderr}
    receipt = {'captured_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
               'scope': 'Source snapshot at execution audit; existing weights, datasets and saved analyses remain in place.',
               'git_head': query(['rev-parse', 'HEAD']), 'git_branch': query(['branch', '--show-current']),
               'git_status': query(['status', '--short']), 'files': files,
               'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    (destination / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'destination': str(destination), 'files': len(files), 'archive_bytes': archive.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    capture(Path(__file__).resolve().parents[2], args.output.resolve())
