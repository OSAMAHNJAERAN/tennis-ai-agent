"""Preserve the consumed host wrapper and prepare a fresh exclusive recheck."""
import ast
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
old = UNIT / 'verify_installation.py'
new = UNIT / 'verify_installation_final.py'
assert not new.exists()
assert hashlib.sha256(old.read_bytes()).hexdigest() == 'f0000e86c764b09582919c451ffbf5f399650cde8e6dd5b4263bb03a337715fe'
text = old.read_text(encoding='utf-8')
assert text.count('event_timestamp_lookup_host_verification_2026-10-06.json') == 1
text = text.replace('event_timestamp_lookup_host_verification_2026-10-06.json',
                    'event_timestamp_lookup_host_verification_final_2026-10-06.json')
ast.parse(text)
with new.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(text)
receipt = {'originalSha256': hashlib.sha256(old.read_bytes()).hexdigest(),
           'preparedSha256': hashlib.sha256(new.read_bytes()).hexdigest(),
           'scope': 'Same host verifier/guards, fresh wrapper/log/receipt and report paths only.',
           'hostVerificationRun': False}
with (UNIT / 'installation_final_preparation.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
