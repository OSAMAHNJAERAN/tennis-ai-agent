"""Actual rules/history retention probe; synthetic unknown events, never video accuracy."""
import builtins
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'pandas', 'PIL'}
original_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        raise RuntimeError('Pure scoring retention probe refuses ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded
from tests.test_scoring_service_phase import actual_classes

api = actual_classes()
paths = [ROOT / name for name in (
    'src/scoring/scoring_engine.py', 'src/scoring/event_history.py',
    'src/scoring/scoring_rules.py', 'src/scoring/match_state.py',
    'src/scoring/point_outcome_resolver.py', 'tests/test_scoring_service_phase.py')]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
rows = []
for count in (257, 1025, 4097):
    gc.collect()
    tracemalloc.start()
    engine = api.TennisScoringEngine()
    initial = copy.deepcopy(engine.state.to_dict())
    for index in range(count):
        engine.process_match_event(index + 1, 'UNRECOGNIZED_CONTACT', index / 50., confidence=.9)
    retained, peak = tracemalloc.get_traced_memory()
    row = dict(events=count, retained_history=len(engine.history.entries),
        retained_inputs=len(engine._inputs), retained_processed_ids=len(engine.history.processed_event_ids),
        retained_traced_bytes=retained, peak_traced_bytes=peak,
        score_unchanged=engine.state.to_dict() == initial)
    engine.correct_event(1, reason='Explicit synthetic confidence correction', expected_revision=0, confidence=.8)
    corrected_retained, corrected_peak = tracemalloc.get_traced_memory()
    row.update(correction_retained_traced_bytes=corrected_retained,
        correction_peak_traced_bytes=corrected_peak, audit_history_records=len(engine.corrections[0]['previous_history']),
        revision=engine.revision, score_unchanged_after_correction=engine.state.to_dict() == initial)
    rows.append(row)
    tracemalloc.stop()
    del engine
after = {str(path.relative_to(ROOT)): sha(path) for path in paths}
receipt = dict(status='RED_DURATION_GROWING_IN_MEMORY_RULE_JOURNAL',
    started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-clock,
    rows=rows, source_before=before, source_after=after, sources_unchanged=before == after,
    actual_rules_history_and_correction_definitions=True, input_scope='SYNTHETIC_UNKNOWN_EVENT_STREAM',
    forbidden_imports_present=sorted(FORBIDDEN.intersection(sys.modules)), models_used=False, media_used=False,
    gpu_used=False, qualification_evidence=False, runner_sha256=sha(Path(__file__)))
destination = Path(__file__).with_name('receipt.json')
with destination.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(receipt))
assert before == after and not receipt['forbidden_imports_present']
