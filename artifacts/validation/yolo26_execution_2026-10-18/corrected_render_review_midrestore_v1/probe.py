"""Actual render callback changes and restores a normal owned event write."""
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
    import cv2
    from tests.test_corrected_rendering import fresh_exports, render, digests
    from tests.test_corrected_shot_rebuild import build_shot_base
    from tests.test_corrected_rule_rebuild import unchanged
    from src.events.event_detector import EventType
    from src.visualization.video_annotator import VideoAnnotator
    fixture_root = ART / 'fixture'
    fixture_root.mkdir(exist_ok=False)
    fixture = build_shot_base(fixture_root, playable_source=True)
    native_ball = VideoAnnotator.draw_ball
    native_capture, native_writer = cv2.VideoCapture, cv2.VideoWriter
    handles = []
    def capture(*args, **kwargs):
        value = native_capture(*args, **kwargs)
        handles.append(value)
        return value
    def writer(*args, **kwargs):
        value = native_writer(*args, **kwargs)
        handles.append(value)
        return value
    cv2.VideoCapture, cv2.VideoWriter = capture, writer
    try:
        with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
            initial_digest = digests(exports.directory)
            original = shots.events[1]
            original_generation = shots.event_owner.events_generation
            calls = []
            def observed(image, position, state):
                if not calls:
                    event = shots.events[1]
                    event.event_type = EventType.UNKNOWN_EVENT
                    shots.events[1] = event
                    shots.events[1] = original
                    calls.append({'before': original_generation,
                        'after_restore': shots.event_owner.events_generation,
                        'event_restored': shots.events[1].event_type is original.event_type})
                return native_ball(image, position, state)
            VideoAnnotator.draw_ball = staticmethod(observed)
            try:
                output = render(rebuilt, shots, forecast, exports, fixture.scratch)
                result.update(render_accepted=True, directory=str(output.directory))
            except (ValueError, RuntimeError, TypeError) as error:
                result.update(render_accepted=False, rejection=str(error))
            finally:
                VideoAnnotator.draw_ball = staticmethod(native_ball)
            result.update(callback_mutation=calls, handles_closed=all(not value.isOpened() for value in handles),
                render_seals=[str(path) for path in fixture.scratch.rglob('corrected_render_manifest.json')],
                sealed_exports_unchanged=digests(exports.directory) == initial_digest)
        unchanged(fixture)
        result['committed_base_unchanged'] = True
    finally:
        cv2.VideoCapture, cv2.VideoWriter = native_capture, native_writer
        VideoAnnotator.draw_ball = staticmethod(native_ball)
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = original_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Actual source capture/writer on synthetic imagery; restore-equal event mutation guard; no publication'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))
