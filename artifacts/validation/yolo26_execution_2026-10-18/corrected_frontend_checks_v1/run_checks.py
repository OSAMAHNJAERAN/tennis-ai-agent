"""Retained frontend checks; no source mutations or external requests."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[4]
FRONT = ROOT / 'frontend'
OUT = Path(__file__).resolve().parent
NODE = r'C:\Program Files\nodejs\node.exe'
TEMP = OUT / 'temp'
TEMP.mkdir(exist_ok=False)
env = dict(os.environ, TEMP=str(TEMP), TMP=str(TEMP), TMPDIR=str(TEMP))
paths = sorted(p for p in (FRONT/'src').rglob('*') if p.is_file())
paths += [FRONT/name for name in ('package.json', 'pnpm-lock.yaml', 'vite.config.ts', 'tsconfig.app.json', 'tsconfig.node.json', 'tsconfig.json', 'eslint.config.js')]
paths += [Path(__file__).resolve()]
def pins():
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = pins()
(OUT/'pins_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
commands = [
    ('typecheck', [NODE, 'node_modules/typescript/bin/tsc', '-b', '--pretty', 'false']),
    ('lint', [NODE, 'node_modules/eslint/bin/eslint.js', '.', '--max-warnings=0']),
    ('tests', [NODE, 'node_modules/vitest/vitest.mjs', 'run', '--reporter=dot', '--reporter=junit', '--outputFile.junit='+str(OUT/'junit.xml')]),
    ('build', [NODE, 'node_modules/vite/bin/vite.js', 'build']),
]
results = []
for name, command in commands:
    start = time.monotonic()
    with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
        done = subprocess.run(command, cwd=FRONT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    results.append({'name': name, 'command': command, 'returncode': done.returncode, 'seconds': time.monotonic()-start,
                    'log_sha256': hashlib.sha256((OUT/(name+'.log')).read_bytes()).hexdigest()})
    print(name, done.returncode, flush=True)
after = pins()
(OUT/'pins_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
receipt = {'schema_version': 1, 'scope': 'frontend correction owner review and native forecast shared clock',
    'checks': results, 'source_pin_count': len(before), 'source_pins_stable': before == after,
    'changed_pins': [name for name, value in before.items() if after[name] != value],
    'tests_junit_sha256': hashlib.sha256((OUT/'junit.xml').read_bytes()).hexdigest() if (OUT/'junit.xml').is_file() else None,
    'claims': ['Synthetic frontend contract/workflow regression only; no independent accuracy qualification.']}
(OUT/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if before == after and all(result['returncode'] == 0 for result in results) else 1)
