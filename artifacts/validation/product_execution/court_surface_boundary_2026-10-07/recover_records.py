"""Recover flushed trial records without rerunning or replacing the failed trial."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import psutil

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RUN = UNIT / 'run'
OUTPUT = UNIT / 'recovery.json'
def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
def load(path):
    return json.loads(path.read_text(encoding='utf-8'))
assert not OUTPUT.exists()
prepared = load(UNIT / 'preparation.json')
resource = load(RUN / 'resource_receipt.json')
assert resource['status'] == 'FAILED' and resource['exitCode'] == 1 and not resource['passed']
assert not (RUN / 'report.json').exists()
assert resource['allOwnedIdentitiesRetired'] and resource['sourcesUnchanged'] and resource['savedProductUnchanged']
for name, expected in prepared['bindings'].items():
    path = Path(name) if Path(name).is_absolute() else ROOT / name
    assert binding(path) == expected, 'Frozen input changed: ' + name
for row in resource['ownedIdentities']:
    try:
        assert psutil.Process(row['pid']).create_time() != row['createTimeUnixSeconds'], 'Exact workload still live'
    except psutil.NoSuchProcess:
        pass
with sqlite3.connect((ROOT / 'outputs/product/analyses.sqlite3').as_uri() + '?mode=ro', uri=True) as connection:
    connection.row_factory = sqlite3.Row
    jobs = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
files = {path.relative_to(ROOT).as_posix(): binding(path) for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
assert {'jobs': jobs, 'files': files} == prepared['productBefore']
progress = [json.loads(line) for line in (RUN / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(progress) == 73
assert [row['kind'] for row in progress] == ['existing_seed'] * 64 + ['automatic'] * 9
cases = [row['entry'] for row in progress[:64]]
automatic = [row['entry'] for row in progress[64:]]
assert [row['index'] for row in cases] == list(range(64))
assert [row['index'] for row in automatic] == list(range(9))
assert sum(row['initialValid'] for row in cases) == 47
assert sum('scores' in row for row in cases) == 40
spectator = load(RUN / 'spectator_control.json')
for row in cases + automatic + [spectator]:
    assert binding(ROOT / row['source']) == row['sourceBinding']
    for key in ('panel', 'original'):
        if key in row:
            evidence = row[key]
            assert binding(ROOT / evidence['path']) == {key: evidence[key] for key in ('bytes', 'sha256')}
pooled = {}
for group in ('selection', 'external_source_check'):
    pooled[group] = {stage: {key: sum(row['scores'][stage][key] for row in cases if row['group'] == group)
                            for key in ('tp', 'fp', 'fn')}
                     for stage in ('projected', 'original', 'boundary_baseline', 'candidate')}
source_log = (RUN / 'stdout.log').read_text(encoding='utf-8')
assert "AttributeError: 'numpy.ndarray' object has no attribute 'memory_info'" in source_log
result = {'recoveredAt': datetime.now(timezone.utc).isoformat(), 'status': 'FLUSHED_RECORDS_RECOVERED_ORIGINAL_RUN_FAILED',
          'trialTerminalPassed': False, 'originalRunTerminalStatus': resource['status'],
          'experimentRecordsComplete': True, 'qualificationEvidence': False, 'productionPromoted': False,
          'modelInferenceRun': False, 'cases': cases, 'automatic': automatic,
          'spectatorHypothesisControl': spectator, 'pooled': pooled,
          'validInitializations': 47, 'initialAbstentions': 17, 'scoredCases': 40,
          'sourceBindingsVerified': len(prepared['bindings']), 'savedJobsUnchanged': len(jobs), 'savedFilesUnchanged': len(files),
          'changedCaseIndices': [row['index'] for row in cases if row.get('changed')],
          'recoveryScope': 'No candidate, proposal extraction, source decoding, optimizer, model or trial was rerun. Complete flushed records are copied and counted; an independent auditor must recompute their metric/geometry arithmetic.',
          'finalizationFailure': 'The local actual variable began as psutil.Process, then was reused for ndarray court points. The final actual.memory_info call raised before report.json could be saved.',
          'missingEvidence': ['Final report and successful child exit', 'Persisted ordinary loaded-library inventory before/after', 'Actual originalReplayReference arrays for independent recomputation of the reported historical replay maximum difference'],
          'noReconstructedMissingRuntimeMetadata': True,
          'evidenceBindings': {path.relative_to(UNIT).as_posix(): binding(path) for path in RUN.iterdir() if path.is_file()},
          'preparationBinding': binding(UNIT / 'preparation.json'), 'recoveryScriptBinding': binding(Path(__file__))}
with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(json.dumps({'status': result['status'], 'pooled': pooled, 'changedCases': result['changedCaseIndices'],
                  'automaticAccepted': [row['id'] for row in automatic if row['acceptedImageHypothesis']],
                  'spectatorAccepted': spectator['acceptedImageHypothesis']}))
