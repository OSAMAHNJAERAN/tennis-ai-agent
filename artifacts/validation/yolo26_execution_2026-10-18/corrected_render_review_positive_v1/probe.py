"""Independent actual-codec render/source consistency probe."""
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
sys.dont_write_bytecode = True
(ART / 'temp').mkdir(exist_ok=False)
os.environ.update(TEMP=str(ART / 'temp'), TMP=str(ART / 'temp'),
    TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'), PYTHONDONTWRITEBYTECODE='1')
PATHS = ['src/scoring/corrected_rendering.py', 'src/scoring/corrected_exports.py',
    'src/scoring/corrected_landing_exports.py', 'src/scoring/corrected_rule_rebuild.py',
    'src/scoring/corrected_forecasts.py', 'src/scoring/corrected_shot_rebuild.py',
    'src/scoring/corrected_shot_sources.py', 'src/utils/event_analysis_sequence.py',
    'tests/test_corrected_rendering.py', 'tests/test_corrected_exports.py',
    'tests/test_corrected_forecasts.py', 'tests/test_corrected_shot_rebuild.py',
    'tests/test_corrected_rule_rebuild.py']
def pins():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PATHS}
before = pins()
attempts = []
original_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Render review may not load inference: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guard
result = {}
try:
    from tests.test_corrected_rendering import fresh_exports, render, read
    from tests.test_corrected_shot_rebuild import build_shot_base
    from tests.test_corrected_rule_rebuild import unchanged
    from src.events.event_detector import EventType
    from src.visualization.video_annotator import VideoAnnotator
    fixture_root = ART / 'fixture'
    fixture_root.mkdir(exist_ok=False)
    fixture = build_shot_base(fixture_root, playable_source=True)
    draws = []
    native = VideoAnnotator.draw_event_badge
    def observed(image, event_type, *args, **kwargs):
        draws.append(event_type)
        return native(image, event_type, *args, **kwargs)
    VideoAnnotator.draw_event_badge = staticmethod(observed)
    with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
        event = shots.events[1]
        original = event.event_type
        exported_type = read(exports.files['match_events.json'])['events'][1]['event_type']
        event.event_type = original
        shots.events[1] = event
        try:
            output = render(rebuilt, shots, forecast, exports, fixture.scratch)
            result.update(render_accepted=True, directory=str(output.directory),
                render_manifest_present=output.files['corrected_render_manifest.json'].is_file(),
                exported_event_type=exported_type, live_event_type=event.event_type.value,
                actual_drawn_badges=draws,
                original_bounce_badge_drawn='BOUNCE' in draws,
                substituted_unknown_badge_drawn='UNKNOWN_EVENT' in draws)
        except (ValueError, RuntimeError, TypeError) as error:
            result.update(render_accepted=False, rejection=str(error), actual_drawn_badges=draws)
        finally:
            event.event_type = original
            shots.events[1] = event
    unchanged(fixture)
    result['committed_base_unchanged'] = True
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = original_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Actual native/browser codecs on synthetic 12-frame source; software consistency only; no publication'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))
