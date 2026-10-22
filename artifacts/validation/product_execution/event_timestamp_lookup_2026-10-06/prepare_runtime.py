"""Clone the exact verified read-only runtime gate into an exclusive new unit."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).parent
ORIGINAL = UNIT.parent / 'verify_result_storage_latency_runtime_2026-10-06.py'
TARGET = UNIT / 'verify_runtime.py'
EXPECTED = '140d462c754ff1a6a3656ee104a6b32a1674cf1928f1566fd8f2819525bedd51'
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == EXPECTED
text = ORIGINAL.read_text(encoding='utf-8')
replacements = {
    'ROOT = Path(__file__).resolve().parents[3]': 'ROOT = Path(__file__).resolve().parents[4]',
    "PREFLIGHT = BASE / 'result_storage_latency_runtime_preflight_2026-10-06.json'": "PREFLIGHT = BASE / 'runtime_preflight.json'",
    "RECEIPT = BASE / 'result_storage_latency_runtime_verification_2026-10-06.json'": "RECEIPT = BASE / 'runtime_verification.json'",
    "ORIGINAL = BASE / 'verify_result_storage_runtime_2026-10-06.py'": "ORIGINAL = BASE.parent / 'verify_result_storage_runtime_2026-10-06.py'",
    "names += sorted((ROOT / 'src/api').glob('*.py'))": "names += sorted((ROOT / 'src').rglob('*.py'))",
}
for before, after in replacements.items():
    assert text.count(before) == 1
    text = text.replace(before, after)
original_ast = ast.parse(ORIGINAL.read_text(encoding='utf-8'))
new_ast = ast.parse(text)
original_functions = {node.name: ast.dump(node, include_attributes=False)
                      for node in original_ast.body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: ast.dump(node, include_attributes=False)
                 for node in new_ast.body if isinstance(node, ast.FunctionDef)}
assert original_functions.keys() == new_functions.keys()
assert all(new_functions[name] == value for name, value in original_functions.items()
           if name != 'source_bindings')
with TARGET.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == EXPECTED
with (UNIT / 'runtime_preparation.json').open('x', encoding='utf-8') as stream:
    json.dump({'status': 'PREPARED_NOT_EXECUTED', 'originSha256': EXPECTED,
               'helperSha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
               'onlyChanges': replacements, 'unchangedSnapshotRequestAndExportFunctions': True,
               'productionOrHistoricalEvidenceChanged': False, 'inferenceRun': False}, stream, indent=2)
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'functionsUnchangedExceptSourceInventory': True}))
