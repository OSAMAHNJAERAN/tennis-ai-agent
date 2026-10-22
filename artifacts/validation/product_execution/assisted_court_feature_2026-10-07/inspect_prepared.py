"""Read prepared assisted sources and current host resources without CV imports."""
from pathlib import Path
import hashlib
import json
import psutil

unit = Path(__file__).resolve().parent
memory = psutil.virtual_memory()
print(json.dumps({'availableBytes': memory.available, 'totalBytes': memory.total,
                  'unitFiles': [{'name': p.name, 'bytes': p.stat().st_size,
                                 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                for p in unit.iterdir() if p.is_file()]}))
for path in unit.glob('*preparation*.json'):
    value = json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps({'file': path.name, 'keys': list(value),
                      'status': value.get('status'), 'testsRun': value.get('testsRun')}))
source = (unit / 'assisted_anchor.py').read_text(encoding='utf-8')
print('\n'.join(source.splitlines()[230:370]))
