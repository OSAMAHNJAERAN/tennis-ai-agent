"""Append one current result summary, preserving the exact prior resume record."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
TARGET = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
before = binding(TARGET)
assert before['sha256'] == 'bd3e5e51a1133c878ae480641bca5037ec24d30bab19295958180b1a64495570'
assert binding(ROOT / 'docs/experiments/COURT_SURFACE_BOUNDARY_RESULTS_2026-10-07.md')['sha256'] == '22fe25614ac997f6e1be2e8b6af3b803244e6b3fd4ace57ec199521b1f6988e0'
original = TARGET.read_bytes()
with (UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md').open('xb') as stream:
    stream.write(original)
addition = '''

## Conditional surface-boundary development result — 7 October 2026

[Complete result and evidence](../experiments/COURT_SURFACE_BOUNDARY_RESULTS_2026-10-07.md). The sole frozen candidate remains research-only and **not promoted**: selection TP falls from 369 to 368 against the same boundary optimizer with original evidence; external-source TP remains 65. All64 observations (47 valid,17 initial abstentions),40 labeled cases and9 reused automatic views are retained. Five fits change; landmark2 of case27 is the sole controlled loss, with no gains. Dimitrov's image hypothesis visibly improves, but its13 exposed approximate marks and occluded point are not independent physical ground truth. Other automatic views reject or abstain; the known spectator fit remains rejected.

The first synthetic attempt retains34/35; a corrected saturation fixture and explicit inherited-whiteness limitation check pass36/36 without changing candidate/protocol. The real trial **failed finalization** after saving all73 records because a court-array variable shadowed the process object. No trial was rerun or failed receipt overwritten. Separate recovered-record audit passes2,240 native-error checks plus geometry/proposal/pixel/source/retirement/product checks; missing loaded-module/version inventories and actual historical replay arrays remain unverified. All15 panels and14 native originals were visually reviewed. These evidence checks do not convert the original run into a successful execution.

The prior website pair was absent; fresh documented restoration and compatibility checks pass3 demos,12 exports and7 index assets, with all10 jobs/319 saved files unchanged. The fresh pose/racket real-worker attempt is **NOT_STARTED**: availableRAM2,435,448,832 bytes below its unchanged2,500,000,000-byte preflight, zero child/model/samples and no retry. Current production sources/configuration remain bound to their retained 6 October software evidence; this development work supplies no new actual inference, automatic-participant qualification or full-match capacity claim. The [data recheck](../research/ACCURACY_DATA_RECHECK_2026-10-07.md) verifies no new suitable untouched qualification recordings. Continue the full prompt; independent sessions, speed/3D references and accuracy requirements remain open.
'''
assert '## Conditional surface-boundary development result — 7 October 2026'.encode('utf-8') not in original
assert TARGET.read_bytes() == original
with TARGET.open('ab') as stream:
    stream.write(addition.encode('utf-8'))
with (UNIT / 'progress_append_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'appendedAt': datetime.now(timezone.utc).isoformat(), 'before': before,
        'preservedBefore': binding(UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md'),
        'after': binding(TARGET), 'previousBytesPreservedExactly': TARGET.read_bytes().startswith(original),
        'scope': 'One appended resume entry; earlier evidence and seals remain historical.'}, stream, indent=2)
print('Current result appended; exact previous progress record preserved.')
