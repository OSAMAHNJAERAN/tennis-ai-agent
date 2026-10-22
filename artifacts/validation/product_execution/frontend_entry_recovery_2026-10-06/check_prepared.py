"""Parse prepared one-shot helpers without dispatching them."""
import ast
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
files = {}
for name in ('publish_frontend.py', 'verify_runtime.py', 'verify_installation.py'):
    path = base / name
    ast.parse(path.read_text(encoding='utf-8'))
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
with (base / 'prepared_helpers.json').open('x', encoding='utf-8') as stream:
    json.dump({'parsed': True, 'helpersExecuted': False, 'sha256': files}, stream, indent=2)
print(json.dumps({'parsed': True, 'helpersExecuted': False, 'sha256': files}))
