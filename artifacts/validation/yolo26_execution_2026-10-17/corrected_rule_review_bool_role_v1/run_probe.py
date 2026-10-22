"""Read-only production review: boolean semantic identity amendment admission."""
import builtins
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
TEMP = RUN / 'temp'
TEMP.mkdir(exist_ok=True)
os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
                  TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(TEMP / 'app_bootstrap'))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
FORBIDDEN = {'torch', 'ultralytics'}
attempts = []
original_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('Review must not load model runtimes: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded
SOURCES = ['src/scoring/corrected_rule_rebuild.py', 'src/scoring/correction_sources.py',
           'src/scoring/rule_replay_source.py', 'src/scoring/point_timeline.py',
           'tests/test_corrected_rule_rebuild.py', 'tests/test_correction_sources.py']

def pins():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}

before = pins()
begin = time.perf_counter()
started = datetime.now(timezone.utc).isoformat()
from tests.test_corrected_rule_rebuild import sealed_base, unchanged
from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline

fixture = sealed_base.__wrapped__(TEMP)
request = copy.deepcopy(fixture.request)
request['initial_state_amendment'] = None
original = fixture.document['inputs'][0]
replacement = copy.deepcopy(original['source_input'])
assert replacement['player_id'] == 1 and type(replacement['player_id']) is int
replacement['player_id'] = True
request['input_amendments'] = [{
    'event_id': original['source_input']['event_id'],
    'expected_input_fingerprint': original['input_fingerprint'],
    'replacement_source_input': replacement,
}]
with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
    accepted = rebuilt.owner.event(1)
    replay = rebuilt.owner.get('replay_inputs', accepted['uid'])
    observation = {
        'pending_revision_yielded': True,
        'original_source_player_id': original['source_input']['player_id'],
        'replacement_source_player_id': replay['source_input']['player_id'],
        'replacement_player_id_python_type': type(replay['source_input']['player_id']).__name__,
        'original_rule_player_id': original['scoring_input']['player_id'],
        'accepted_rule_player_id': replay['scoring_input']['player_id'],
        'original_rule_event_type': original['scoring_input']['event_type'],
        'accepted_rule_event_type': replay['scoring_input']['event_type'],
        'applied_event_ids': rebuilt.audit['applied_event_ids'],
        'published': rebuilt.audit['published'],
    }
unchanged(fixture)
after = pins()
receipt = {
    'status': 'REPRODUCED_MALFORMED_BOOL_ROLE_AMENDMENT' if
        observation['replacement_player_id_python_type'] == 'bool' else 'NOT_REPRODUCED',
    'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
    'seconds': time.perf_counter() - begin, 'observation': observation,
    'source_before': before, 'source_after': after, 'sources_unchanged': before == after,
    'forbidden_import_attempts': attempts,
    'forbidden_modules_present': sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN),
    'video_used': False, 'inference_used': False, 'independent_accuracy_validated': False,
    'scope': 'Synthetic committed native generation and real rule evaluator; no production data or files modified',
    'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(RUN / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status', 'seconds', 'observation',
    'sources_unchanged', 'forbidden_import_attempts', 'forbidden_modules_present')}))
