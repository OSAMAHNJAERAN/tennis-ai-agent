"""Make a fresh full-suite wrapper without replacing the failed run or source."""
import ast
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).parent
SOURCE = UNIT / 'verify_backend.py'
TARGET = UNIT / 'verify_backend_final.py'
before = SOURCE.read_bytes()
text = before.decode('utf-8')
assert text.count("UNIT = Path(__file__).parent / 'backend'") == 1
text = text.replace("UNIT = Path(__file__).parent / 'backend'", "UNIT = Path(__file__).parent / 'backend_final'")
assert text.count("'-o', 'faulthandler_timeout=30'") == 1
text = text.replace("'-o', 'faulthandler_timeout=30'", "'-o', 'faulthandler_timeout=30', '-o', 'junit_family=legacy'")
ast.parse(text)
with TARGET.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
assert SOURCE.read_bytes() == before
with (UNIT / 'backend_final_preparation.json').open('x', encoding='utf-8') as stream:
    json.dump({'status': 'PREPARED_NOT_EXECUTED',
               'originalWrapperSha256': hashlib.sha256(before).hexdigest(),
               'freshWrapperSha256': hashlib.sha256(TARGET.read_bytes()).hexdigest(),
               'preservesFailedRun': True,
               'changes': ['Exclusive backend_final directory', 'Legacy JUnit family retains recorded regression properties'],
               'testsRunByPreparation': False}, stream, indent=2)
print(json.dumps({'status': 'PREPARED_NOT_EXECUTED', 'failedRunPreserved': True}))
