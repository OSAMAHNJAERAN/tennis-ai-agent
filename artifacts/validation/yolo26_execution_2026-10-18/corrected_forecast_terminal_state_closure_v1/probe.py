"""Recheck the exact independently reproduced contradictory-terminal-state case."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path.cwd()
output = root/'artifacts/validation/yolo26_execution_2026-10-18/corrected_forecast_terminal_state_closure_v1'
os.environ['TENNIS_STORAGE_ROOT'] = str(output/'app-storage')
os.environ['TEMP'] = os.environ['TMP'] = str(output/'temp')
(output/'temp').mkdir(exist_ok=True)
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
sys.path.insert(0,str(root))
paths = ['src/scoring/corrected_forecasts.py','src/prediction/tennis.py',
         'src/scoring/scoring_rules.py','src/scoring/corrected_rule_rebuild.py']
def pins():
    return {path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}
before = pins()
attempts = []
original_import = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.',1)[0] in {'torch','ultralytics'}:
        attempts.append(name)
        raise AssertionError('Forbidden model import '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__ = guarded
from tests.test_corrected_forecasts import fresh_shots
from tests.test_corrected_shot_rebuild import build_shot_base
from tests.test_corrected_rule_rebuild import unchanged
from src.scoring.corrected_forecasts import rebuild_corrected_forecasts
from src.scoring.match_state import CompletedSetScore
from src.scoring.match_state_codec import decode_match_state, encode_match_state
from src.scoring.scoring_rules import TennisScoringRules

fixture = build_shot_base(output)
initial = decode_match_state(fixture.request['initial_state_amendment']['replacement_initial_state'])
initial.sets_won_p1, initial.current_set = 2,3
initial.completed_sets = [CompletedSetScore(1,6,0,winner_id=1),CompletedSetScore(2,6,0,winner_id=1)]
assert initial.format.value == 'BEST_OF_3' and initial.match_complete is False
fixture.request['initial_state_amendment']['replacement_initial_state'] = encode_match_state(initial)
with fresh_shots(fixture) as (rebuilt,shots):
    with rebuild_corrected_forecasts(rebuilt,shots,fixture.scratch) as forecast:
        history = list(forecast.history)
        assert len(history) == 7
        for row in history:
            assert row['forecast']['status'] == 'unavailable'
            assert row['forecast']['probabilities'] is None
            assert row['forecast']['winner_id'] is None
            assert row['forecast']['sample_count'] == 0
            assert row['forecast']['withheld_reason_code'] == 'INVALID_TERMINAL_SCORE'
        assert forecast.audit['withheld_reason_code'] == 'INVALID_TERMINAL_SCORE'
        observed = {'status':'CLOSED','history_rows':len(history),
            'initial_forecast':history[0]['forecast'],'final_forecast':forecast.current,
            'independent_accuracy_validated':forecast.audit['independent_accuracy_validated'],
            'published':forecast.audit['published'],
            'shared_invariant_violations':TennisScoringRules.validate_invariants(initial)}
unchanged(fixture)
after = pins()
receipt = {'status':'PASS' if before==after and not attempts else 'SOURCE_CHANGED',
    'observed':observed,'source_pins_before':before,'source_pins_after':after,
    'pins_unchanged':before==after,'forbidden_import_attempts':attempts,
    'loaded_models':[name for name in ('torch','ultralytics') if name in sys.modules],
    'scope':'SYNTHETIC_NATIVE_PRIVATE_CORRECTION_ONLY; NO_PUBLICATION_OR_ACCURACY'}
(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
