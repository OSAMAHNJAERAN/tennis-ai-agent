"""Current focused closure plus full frontend type/lint/build and exact delta."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
FRONT = ROOT/'frontend'
OUT = Path(__file__).resolve().parent
BASE = OUT.parent/'corrected_frontend_checks_v2'
NODE = r'C:\Program Files\nodejs\node.exe'
TEMP = OUT/'temp'
TEMP.mkdir(exist_ok=False)
env = dict(os.environ, TEMP=str(TEMP), TMP=str(TEMP), TMPDIR=str(TEMP))
paths = sorted(p for p in (FRONT/'src').rglob('*') if p.is_file())
paths += [FRONT/name for name in ('package.json', 'pnpm-lock.yaml', 'vite.config.ts',
    'tsconfig.app.json', 'tsconfig.node.json', 'tsconfig.json', 'eslint.config.js')]
paths += [Path(__file__).resolve()]

def pins():
    return {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}

before = pins()
(OUT/'pins_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
baseline = json.loads((BASE/'pins_before.json').read_text(encoding='utf-8'))
changed = [name for name, value in baseline.items() if name.startswith('frontend/') and before.get(name) != value]
expected = {'frontend/src/components/AnalysisCorrection.tsx',
    'frontend/src/components/AnalysisCorrection.test.tsx',
    'frontend/src/components/SynchronizedTelemetryView.test.tsx'}
assert set(changed) == expected, changed
product = 'frontend/src/components/AnalysisCorrection.tsx'
new_text = (ROOT/product).read_text(encoding='utf-8')
new_line = "setCurrent(nextSource.is_current ? rows.find(row => activeStatus(row) && row.base_generation === generation) ?? null : null);"
old_line = "setCurrent(rows.find(row => activeStatus(row)) ?? null);"
assert new_text.count(new_line) == 1
old_text = new_text.replace(new_line, old_line)
assert hashlib.sha256(old_text.encode('utf-8')).hexdigest() == baseline[product]
(OUT/'baseline_AnalysisCorrection.tsx').write_text(old_text, encoding='utf-8')
(OUT/'product_delta.patch').write_text(''.join(difflib.unified_diff(old_text.splitlines(True),
    new_text.splitlines(True), fromfile='full-v2/'+product, tofile='current/'+product)), encoding='utf-8')
selected = ['src/components/AnalysisCorrection.test.tsx', 'src/components/NativeForecast.test.tsx',
    'src/components/SynchronizedTelemetryView.test.tsx', 'src/components/spatial/TelemetryDashboard.test.tsx',
    'src/data/correctionDraft.test.ts', 'src/data/correctionApi.test.ts',
    'src/data/nativeForecastRepository.test.ts', 'src/data/repository.test.ts',
    'src/app/AnalysisContext.test.tsx', 'src/pages/ProductFlows.test.tsx', 'src/pages/SpatialAnalysis.test.tsx']
commands = [
    ('typecheck', [NODE, 'node_modules/typescript/bin/tsc', '-b', '--pretty', 'false']),
    ('lint', [NODE, 'node_modules/eslint/bin/eslint.js', '.', '--max-warnings=0']),
    ('tests', [NODE, 'node_modules/vitest/vitest.mjs', 'run', *selected,
        '--reporter=dot', '--reporter=junit', '--outputFile.junit='+str(OUT/'junit.xml')]),
    ('build', [NODE, 'node_modules/vite/bin/vite.js', 'build']),
]
results = []
for name, command in commands:
    start = time.monotonic()
    with (OUT/(name+'.log')).open('w', encoding='utf-8') as log:
        done = subprocess.run(command, cwd=FRONT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    results.append({'name': name, 'command': command, 'returncode': done.returncode,
        'seconds': time.monotonic()-start, 'log_sha256': hashlib.sha256((OUT/(name+'.log')).read_bytes()).hexdigest()})
    print(name, done.returncode, flush=True)
after = pins()
(OUT/'pins_after.json').write_text(json.dumps(after, indent=2), encoding='utf-8')
suites = ET.parse(OUT/'junit.xml').getroot().findall('testsuite')
counts = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
receipt = {'schema_version': 1, 'scope': 'current correction review/history/source-frame/forecast closure',
    'checks': results, 'counts': counts, 'selected_test_files': selected,
    'source_pin_count': len(before), 'source_pins_stable': before == after,
    'changed_pins_during_run': [name for name, value in before.items() if after[name] != value],
    'full_v2_receipt_sha256': hashlib.sha256((BASE/'receipt.json').read_bytes()).hexdigest(),
    'delta_from_full_v2': changed, 'product_delta_exactly_reconstructed_against_baseline_hash': True,
    'claims': ['Full v2 has 264 passed and one incorrect decoder-expectation test; current focused closure '
        'includes that repaired actual decoder test and every changed source/test file. The sole product change '
        'is matching current-generation active restoration, as proven by exact baseline reconstruction. '
        'No browser visual inspection or independent prediction/CV accuracy claim.']}
(OUT/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if before == after and all(result['returncode'] == 0 for result in results) else 1)
