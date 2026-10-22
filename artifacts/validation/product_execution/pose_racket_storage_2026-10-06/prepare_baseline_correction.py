"""Correct a transcribed preflight seal digest; failed helper remains intact."""
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
original = BASE / 'prepare_baseline.py'
assert not (BASE / 'before.json').exists() and not (BASE / 'before').exists()
actual = hashlib.sha256((BASE.parent / 'frontend_entry_recovery_2026-10-06/final_verification.json').read_bytes()).hexdigest()
assert actual == 'e81b4e627af107b0aba7d706790b390f21c6e40d9ccea8335a951fd7d4a3fae5'
source = original.read_text(encoding='utf-8')
assert source.count('e81b4e627af107b0aba7d706790b390f21c6e40d9ccaea8335a951fd7d4a3fae5') == 1
source = source.replace('e81b4e627af107b0aba7d706790b390f21c6e40d9ccaea8335a951fd7d4a3fae5', actual)
ast.parse(source)
target = BASE / 'prepare_baseline_v2.py'
with target.open('x', encoding='utf-8') as stream:
    stream.write(source)
with (BASE / 'preparation_correction.json').open('x', encoding='utf-8') as stream:
    json.dump({'reason': 'Transcribed seal digest had an extra letter; first preflight declined before any baseline copy or production change.',
               'originalHelperSha256': hashlib.sha256(original.read_bytes()).hexdigest(),
               'correctedHelperSha256': hashlib.sha256(target.read_bytes()).hexdigest(),
               'sealSha256': actual, 'productionChanged': False}, stream, indent=2)
print('Corrected helper prepared; original preflight preserved.')
