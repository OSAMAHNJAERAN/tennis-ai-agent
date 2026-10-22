"""Prepare fresh, unexecuted software compatibility checks for this frontend unit."""
import ast
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OLD = BASE.parent / 'event_timestamp_lookup_2026-10-06'

def create(name, source):
    ast.parse(source)
    with (BASE / name).open('x', encoding='utf-8', newline='') as stream:
        stream.write(source)
    print(name, hashlib.sha256((BASE / name).read_bytes()).hexdigest())

source = (OLD / 'verify_runtime.py').read_text(encoding='utf-8')
source = source.replace("    retired = previous_identities_retired(preflight['runtime'])", "    assert runtime_before == preflight['runtime'], 'Retain the exact current idle runtime without restart'\n    event_runtime = json.loads((BASE.parent / 'event_timestamp_lookup_2026-10-06/runtime_verification.json').read_text(encoding='utf-8'))\n    assert runtime_before == event_runtime['runtime'], 'The previously verified server/worker must remain unchanged'")
source = source.replace("'previousRuntimeIdentitiesRetired': retired,", "'runtimeRetainedWithoutRestart': True,")
source = source.replace("Existing public legacy-result idle-reload compatibility; no inference or scientific qualification.", "Existing public result compatibility after reversible frontend publication; retained runtime, no inference or scientific qualification.")
assert "retired = previous_identities_retired(preflight['runtime'])" not in source
create('verify_runtime.py', source)

source = (OLD / 'verify_installation_final.py').read_text(encoding='utf-8')
source = source.replace('event_timestamp_lookup_host_verification_final_2026-10-06.json', 'frontend_entry_recovery_host_verification_2026-10-06.json')
source = source.replace('documented event lookup revision', 'documented frontend entry recovery revision')
create('verify_installation.py', source)
