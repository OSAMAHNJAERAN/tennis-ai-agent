"""Read-only real JavaScript serialization -> native semantic validator probe."""
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
names = ['src/scoring/corrected_rule_rebuild.py', 'src/api/corrections.py',
         'frontend/src/data/correctionDraft.ts', 'frontend/src/data/correctionApi.ts']
def pins():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
before = pins()
source = json.loads((ART/'initial_review_source.json').read_text(encoding='utf-8'))
original = source['inputs'][0]['source_input']
node = r"""const fs=require('node:fs');
const page=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));
const row=structuredClone(page.inputs[0].source_input);
row.event_type='UNKNOWN_EVENT';
process.stdout.write(JSON.stringify(row));"""
result = subprocess.run([r'C:\Program Files\nodejs\node.exe', '-e', node,
    str(ART/'initial_review_source.json')], capture_output=True, text=True, timeout=30)
assert result.returncode == 0, result.stderr
replacement = json.loads(result.stdout)
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('No inference in JavaScript semantic roundtrip review')
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    from src.scoring.corrected_rule_rebuild import _replacement
    caught = None
    try:
        _replacement(original, {'replacement_source_input': replacement})
    except ValueError as error:
        caught = str(error)
finally:
    builtins.__import__ = native_import
immutable = ('frame_index', 'timestamp_s', 'ball_position_px', 'court_position_m',
             'trajectory_state', 'confidence', 'evidence')
same_values = all(original[name] == replacement[name] for name in immutable)
after = pins()
reproduced = same_values and caught is not None and 'changes source measurement' in caught
receipt = {'status': 'P2_REPRODUCED' if reproduced else 'NOT_REPRODUCED',
    'scope': 'ACTUAL_NODE_JSON_PARSE_STRUCTUREDCLONE_STRINGIFY_AND_PRODUCTION_NATIVE_REPLACEMENT; NO_API_MUTATION',
    'analysis_id': source['analysis_id'], 'event_id': original['event_id'],
    'original_event_type': original['event_type'], 'replacement_event_type': replacement['event_type'],
    'immutable_numeric_values_equal': same_values, 'native_rejection': caught,
    'original_timestamp_value': original['timestamp_s'], 'original_timestamp_python_type': type(original['timestamp_s']).__name__,
    'roundtripped_timestamp_value': replacement['timestamp_s'], 'roundtripped_timestamp_python_type': type(replacement['timestamp_s']).__name__,
    'source_before': before, 'source_after': after, 'source_stable': before == after,
    'inference_import_attempts': attempts, 'inference_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'browser_fixture_store_modified': False}
(ART/'js_numeric_roundtrip_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status', 'immutable_numeric_values_equal', 'native_rejection',
    'original_timestamp_python_type', 'roundtripped_timestamp_python_type', 'source_stable')}))
assert before == after and not attempts and not receipt['inference_loaded']
raise SystemExit(0)
