"""Read-only actual-codec HUD text bounds probe on the frozen final renderer."""
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
native_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Label review may not load inference: ' + name)
    return native_import(name, *args, **kwargs)
builtins.__import__ = guard
result = {}
try:
    import cv2
    from tests.test_corrected_rendering import fresh_exports, render, read, digests, decode
    from tests.test_corrected_shot_rebuild import build_shot_base
    from tests.test_corrected_rule_rebuild import unchanged
    from src.scoring.match_state import PointState
    labels = []
    native_text = cv2.putText
    def observed(image, text, position, font, scale, color, thickness, *args, **kwargs):
        if text.startswith('Serve ') and ' | ' in text:
            size, baseline = cv2.getTextSize(text, font, scale, thickness)
            mini_start = image.shape[1]-min(250, max(60, image.shape[1]//4))
            labels.append({'text': text, 'position': list(position), 'scale': scale,
                'pixel_width': size[0], 'right': position[0]+size[0],
                'mini_court_start_x': mini_start, 'clearance_pixels': mini_start-position[0]-size[0],
                'fits_before_mini': position[0]+size[0] <= mini_start-10})
        return native_text(image, text, position, font, scale, color, thickness, *args, **kwargs)
    fixture_root = ART / 'fixture'
    fixture_root.mkdir(exist_ok=False)
    fixture = build_shot_base(fixture_root, playable_source=True)
    cv2.putText = observed
    try:
        with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
            export_before = digests(exports.directory)
            output = render(rebuilt, shots, forecast, exports, fixture.scratch)
            native_meta, images = decode(output.files['annotated.mp4'], retain=(4,))
            browser_meta, _ = decode(output.files['annotated_web.mp4'])
            assert cv2.imwrite(str(ART / 'frame4.png'), images[4])
            result.update(render_accepted=True, native_metadata=native_meta,
                browser_metadata=browser_meta, labels=labels,
                all_actual_labels_fit=bool(labels) and all(row['fits_before_mini'] for row in labels),
                label_clearance_min=min(row['clearance_pixels'] for row in labels),
                sealed_exports_unchanged=digests(exports.directory) == export_before,
                render_manifest_sha256=hashlib.sha256(output.files['corrected_render_manifest.json'].read_bytes()).hexdigest(),
                frame4_path=str(ART / 'frame4.png'))
    finally:
        cv2.putText = native_text
    # Exhaust declared point-state labels under the exact 1280px layout rule.
    enum_bounds = []
    for state in PointState:
        label = f'Serve 2 (deuce) | {state.value.replace("_", " ").capitalize()}'
        left, right = 560, 1015
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)[0][0]
        scale = min(.55, .55*max(1, right-left)/max(1, width))
        actual_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        enum_bounds.append({'state': state.value, 'label': label, 'scale': scale,
            'right': left+actual_width, 'mini_start': 1030,
            'fits_before_mini': left+actual_width <= 1020})
    result.update(point_state_label_bounds=enum_bounds,
        all_point_state_labels_fit=all(row['fits_before_mini'] for row in enum_bounds))
    unchanged(fixture)
    result['committed_base_unchanged'] = True
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = native_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Actual text bounds and native/browser decode on 1280x960 synthetic source; all declared point-state labels; software presentation only'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))
