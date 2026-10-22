"""Independent export consistency probes; synthetic native fixtures only."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(TEMP=str(ART / 'temp'), TMP=str(ART / 'temp'),
    TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'), PYTHONDONTWRITEBYTECODE='1')
(ART / 'temp').mkdir(exist_ok=False)
sys.dont_write_bytecode = True
paths = sorted([*ROOT.joinpath('src').rglob('*.py'),
                ROOT / 'tests/test_corrected_exports.py',
                ROOT / 'tests/test_corrected_landing_exports.py',
                ROOT / 'tests/test_corrected_forecasts.py',
                ROOT / 'tests/test_corrected_shot_rebuild.py',
                ROOT / 'tests/test_corrected_rule_rebuild.py'])
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}
before = pins()
attempts = []
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Review must not load inference: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
results = []
try:
    from tests.test_corrected_exports import fresh_forecast, export, read
    from tests.test_corrected_shot_rebuild import build_shot_base
    from tests.test_corrected_rule_rebuild import unchanged
    for damage in ('nonbounce_position', 'confidence', 'trajectory_state'):
        parent = ART / damage
        parent.mkdir(exist_ok=False)
        fixture = build_shot_base(parent)
        item = {'probe': damage}
        with fresh_forecast(fixture) as (rebuilt, shots, forecast):
            event = shots.events[0]
            original = dict(event.__dict__)
            if damage == 'nonbounce_position':
                event.court_position_m = (999., 999.)
            elif damage == 'confidence':
                event.confidence = .001
            else:
                event.trajectory_state = 'MUTABLE_UNSUPPORTED_STATE'
            shots.events[0] = event
            try:
                output = export(rebuilt, shots, forecast, fixture.scratch)
                doc = read(output.files['match_events.json'])
                replay = read(output.files['native_rule_replay.json'])
                item.update(export_accepted=True, export_directory=str(output.directory),
                    exported_physical_event=doc['events'][0],
                    lossless_source_event=replay['inputs'][0]['source_input'],
                    final_seal_present=output.files['corrected_export_manifest.json'].is_file())
            except (ValueError, TypeError, RuntimeError) as error:
                item.update(export_accepted=False, rejection=str(error))
            # Restore this caller-owned event snapshot for orderly context exit.
            event.__dict__.update(original)
            shots.events[0] = event
        unchanged(fixture)
        item['committed_base_unchanged'] = True
        results.append(item)
except BaseException:
    results.append({'unexpected_error': traceback.format_exc()})
finally:
    builtins.__import__ = original_import
receipt = {'results': results, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Private semantic export integrity; synthetic inputs; no vision accuracy or publication'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))
