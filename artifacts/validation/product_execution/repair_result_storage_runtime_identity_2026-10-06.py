"""Preserve header-fixed failed verifier, then match exact Python commands."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
SOURCE = BASE / 'verify_result_storage_runtime_2026-10-06.py'
COPY = BASE / 'verify_result_storage_runtime_2026-10-06_headers_fixed.py'
RECEIPT = Path(__file__).with_suffix('.json')
assert not COPY.exists() and not RECEIPT.exists()
original = SOURCE.read_bytes()
with COPY.open('xb') as stream:
    stream.write(original)
text = original.decode('utf-8')
old = """        if any('serve_product.py' in value or value == 'src.api.worker' for value in args):
            if Path(process.info['cwd'] or '').resolve() == ROOT:
                result.append(process.info)"""
new = """        if not args or Path(args[0]).resolve() != Path(sys.executable).resolve():
            continue
        server = (len(args) == 2 and (ROOT / args[1]).resolve() == ROOT / 'scripts/run/serve_product.py')
        worker = args[1:] == ['-u', '-m', 'src.api.worker']
        if (server or worker) and Path(process.info['cwd'] or '').resolve() == ROOT:
            result.append(process.info)"""
assert text.count(old) == 1
SOURCE.write_text(text.replace(old, new), encoding='utf-8')
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'initialVerificationCompleted': False,
               'initialExitCode': 1, 'initialError': 'AssertionError: len(processes) == 2',
               'cause': 'Substring command matching also included the cmd.exe launch wrapper.',
               'originalSourceSha256': hashlib.sha256(original).hexdigest(),
               'preservedSourceSha256': hashlib.sha256(COPY.read_bytes()).hexdigest(),
               'repairedSourceSha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'productCodeChanged': False, 'inferenceRun': False}, stream, indent=2)
print(json.dumps({'originalPreserved': COPY.read_bytes() == original, 'exactRuntimeCommandsRequired': True}))
