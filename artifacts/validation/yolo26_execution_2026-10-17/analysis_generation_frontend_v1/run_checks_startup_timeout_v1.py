"""Isolated frontend verification; reuse existing dependencies without install."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ARTIFACT = Path(__file__).resolve().parent
ROOT = ARTIFACT.parents[3]
FRONTEND = ROOT / 'frontend'
WORKSPACE = ARTIFACT / 'workspace'
NODE = Path('C:/Program Files/nodejs/node.exe')
DEPENDENCIES = FRONTEND / 'node_modules'
CHANGED = (
    'src/data/contracts.ts', 'src/data/publicDemos.ts', 'src/pages/SpatialAnalysis.tsx',
    'src/pages/ProductHome.tsx', 'src/pages/SpatialAnalysis.test.tsx',
    'src/pages/ProductHome.test.tsx', 'src/data/liveRepository.test.ts',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pins():
    paths = sorted((FRONTEND / 'src').rglob('*'))
    return {path.relative_to(FRONTEND).as_posix(): digest(path)
            for path in paths if path.is_file()}


def main():
    before = pins()
    copied = {name: digest(WORKSPACE / name) for name in before}
    if copied != before:
        raise AssertionError('Isolated frontend source differs before verification')
    (WORKSPACE / 'isolated.vitest.config.ts').write_text(
        "import base from './vite.config';\n"
        "import { fileURLToPath } from 'node:url';\n"
        "export default { ...base, cacheDir: fileURLToPath(new URL('./test-cache', import.meta.url)) };\n",
        encoding='utf-8')
    commands = [
        ('vitest', [str(NODE), str(DEPENDENCIES / 'vitest/vitest.mjs'), 'run',
                    'src/pages/SpatialAnalysis.test.tsx', 'src/pages/ProductHome.test.tsx',
                    'src/data/liveRepository.test.ts', 'src/data/repository.test.ts',
                    '--config', 'isolated.vitest.config.ts', '--reporter=json',
                    '--outputFile=' + str(ARTIFACT / 'vitest.json')]),
        ('typecheck_app', [str(NODE), str(DEPENDENCIES / 'typescript/bin/tsc'),
                           '-p', 'tsconfig.app.json', '--incremental',
                           '--tsBuildInfoFile', str(ARTIFACT / 'tsconfig.app.tsbuildinfo'),
                           '--pretty', 'false']),
        ('typecheck_config', [str(NODE), str(DEPENDENCIES / 'typescript/bin/tsc'),
                              '-p', 'tsconfig.node.json', '--incremental',
                              '--tsBuildInfoFile', str(ARTIFACT / 'tsconfig.node.tsbuildinfo'),
                              '--pretty', 'false']),
        ('eslint', [str(NODE), str(DEPENDENCIES / 'eslint/bin/eslint.js'),
                    *CHANGED, '--max-warnings=0']),
    ]
    results = []
    for name, args in commands:
        started = time.monotonic()
        completed = subprocess.run(args, cwd=WORKSPACE, capture_output=True, text=True,
            timeout=90, env={**os.environ, 'CI': '1'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (ARTIFACT / (name + '.stdout.txt')).write_text(completed.stdout, encoding='utf-8')
        (ARTIFACT / (name + '.stderr.txt')).write_text(completed.stderr, encoding='utf-8')
        row = {'name': name, 'args': args, 'exit_code': completed.returncode,
               'elapsed_seconds': time.monotonic() - started}
        results.append(row)
        print(json.dumps(row), flush=True)
    after = pins()
    report = json.loads((ARTIFACT / 'vitest.json').read_text(encoding='utf-8'))
    receipt = {'scope': 'FRONTEND_GENERATION_PIN_PRESENTATION_AND_LEGACY_COMPATIBILITY',
               'isolated_workspace': str(WORKSPACE), 'dependencies': str(DEPENDENCIES),
               'source_before': before, 'source_after': after,
               'source_unchanged': before == after, 'isolated_copy_matched_before': copied == before,
               'tests_total': report['numTotalTests'], 'tests_passed': report['numPassedTests'],
               'tests_failed': report['numFailedTests'], 'commands': results,
               'changed_source_files': list(CHANGED),
               'limitations': ['Component/schema verification; no browser or actual backend generation request.',
                               'No inference, training, accuracy qualification or correction UI.']}
    (ARTIFACT / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in
                     ('tests_total', 'tests_passed', 'tests_failed', 'source_unchanged')}), flush=True)
    return 0 if all(row['exit_code'] == 0 for row in results) and before == after else 1


if __name__ == '__main__':
    sys.exit(main())
