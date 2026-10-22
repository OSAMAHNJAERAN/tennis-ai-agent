"""Retry UI/data verification with full frontend type, lint and build checks."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FRONT = ROOT/'frontend'
NODE = r'C:\Program Files\nodejs\node.exe'


def pins():
    paths = {p for p in (FRONT/'src').rglob('*') if p.is_file()}
    paths.update(FRONT/name for name in ('package.json', 'pnpm-lock.yaml', 'vite.config.ts',
                 'tsconfig.app.json', 'tsconfig.node.json', 'tsconfig.json', 'eslint.config.js'))
    paths.add(Path(__file__).resolve())
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    if {p.name for p in OUT.iterdir()} != {'run_checks.py'}:
        raise RuntimeError('Preserve existing outputs; use a new folder')
    before = pins()
    for name in before:
        target = OUT/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    temp = OUT/'temp'
    temp.mkdir()
    env = dict(os.environ, TEMP=str(temp), TMP=str(temp), TMPDIR=str(temp))
    selected = ['src/data/retryIntent.test.ts', 'src/data/analysisApi.test.ts',
                'src/pages/Upload.test.tsx', 'src/pages/BatchUpload.test.tsx',
                'src/pages/ProductFlows.test.tsx', 'src/components/AccountAccess.test.tsx',
                'src/data/repository.test.ts', 'src/app/AnalysisContext.test.tsx']
    commands = [
        ('typecheck', [NODE, 'node_modules/typescript/bin/tsc', '-b', '--pretty', 'false']),
        ('lint', [NODE, 'node_modules/eslint/bin/eslint.js', '.', '--max-warnings=0']),
        ('tests', [NODE, 'node_modules/vitest/vitest.mjs', 'run', *selected,
             '--maxWorkers=1', '--reporter=dot', '--reporter=junit', '--outputFile.junit='+str(OUT/'junit.xml')]),
        ('build', [NODE, 'node_modules/vite/bin/vite.js', 'build']),
    ]
    (OUT/'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
        'source_before': before, 'commands': commands, 'scope': 'Software retry/account/refresh flow; no browser visual or model accuracy claim'}, indent=2), encoding='utf-8')
    results = []
    for name, command in commands:
        clock = time.perf_counter()
        with (OUT/(name+'.log')).open('x', encoding='utf-8') as log:
            done = subprocess.run(command, cwd=FRONT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        results.append({'name': name, 'command': command, 'exit_code': done.returncode,
                        'seconds': time.perf_counter()-clock})
        print(name, done.returncode, flush=True)
    after = pins()
    counts = None
    if (OUT/'junit.xml').is_file():
        suites = list(ET.parse(OUT/'junit.xml').getroot().iter('testsuite'))
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ('tests', 'errors', 'failures', 'skipped')}
    passed = (before == after and all(result['exit_code'] == 0 for result in results)
              and counts is not None and counts['tests'] > 0
              and not any(counts[key] for key in ('errors', 'failures', 'skipped')))
    receipt = {'status': 'PASS' if passed else 'FAIL', 'checks': results, 'counts': counts,
        'source_before': before, 'source_after': after, 'source_stable': before == after,
        'selected_tests': selected, 'actual_browser_visual_inspection': False,
        'host_ram_gpu_probes': False, 'model_training_or_inference': False}
    (OUT/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'counts': counts, 'stable_pins': len(before) if before == after else False}), flush=True)
    for result in results:
        if result['exit_code']:
            print((OUT/(result['name']+'.log')).read_text(encoding='utf-8')[-12000:])
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
