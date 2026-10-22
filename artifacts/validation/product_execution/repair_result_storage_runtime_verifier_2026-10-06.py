"""Preserve the failed verifier exactly, then fix its HTTP header case lookup."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
SOURCE = BASE / 'verify_result_storage_runtime_2026-10-06.py'
COPY = BASE / 'verify_result_storage_runtime_2026-10-06_initial.py'
RECEIPT = Path(__file__).with_suffix('.json')
assert not COPY.exists() and not RECEIPT.exists()
original = SOURCE.read_bytes()
with COPY.open('xb') as stream:
    stream.write(original)
text = original.decode('utf-8')
assert text.count('dict(response.headers)') == 1 and text.count("headers['Content-Disposition']") == 1
text = text.replace('dict(response.headers)', '{key.lower(): value for key, value in response.headers.items()}')
text = text.replace("headers['Content-Disposition']", "headers['content-disposition']")
SOURCE.write_text(text, encoding='utf-8')
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'initialVerificationCompleted': False,
               'initialExitCode': 1, 'initialError': "KeyError: 'Content-Disposition'",
               'cause': 'Verifier converted case-insensitive HTTP headers to a case-sensitive dictionary.',
               'originalSourceSha256': hashlib.sha256(original).hexdigest(),
               'preservedSourceSha256': hashlib.sha256(COPY.read_bytes()).hexdigest(),
               'repairedSourceSha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'productCodeChanged': False, 'inferenceRun': False}, stream, indent=2)
print(json.dumps({'originalPreserved': COPY.read_bytes() == original, 'verifierHeaderLookupRepaired': True}))
