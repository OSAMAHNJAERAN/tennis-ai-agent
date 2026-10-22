"""One admitted synthetic display-frame H crosses the court horizon."""
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
    'src/scoring/corrected_landing_exports.py', 'src/scoring/correction_sources.py',
    'src/visualization/landing_zones.py', 'tests/test_corrected_rendering.py',
    'tests/test_corrected_exports.py', 'tests/test_corrected_shot_rebuild.py',
    'tests/test_corrected_rule_rebuild.py', 'tests/test_correction_sources.py']
def pins():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PATHS}
before = pins()
attempts = []
native_import = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Render review may not load inference: ' + name)
    return native_import(name, *args, **kwargs)
builtins.__import__ = guard
result = {}
try:
    import cv2
    import numpy as np
    import tests.test_correction_sources as assets
    from tests.test_corrected_rendering import fresh_exports, render
    from tests.test_corrected_shot_rebuild import build_shot_base
    from tests.test_corrected_rule_rebuild import unchanged
    from src.court.court_geometry import TennisCourtGeometry
    from src.scoring import corrected_rendering as renderer
    original_assets = assets.write_observation_assets
    matrix = [[.01, 0., 0.], [0., .01, 0.], [0., .01*(1.-1e-8)/float(TennisCourtGeometry.get_canonical_keypoints()[:, 1].max()), 1.]]
    def source_assets(*args, **kwargs):
        files = original_assets(*args, **kwargs)
        path = files['camera_registration.json']
        camera = json.loads(path.read_text(encoding='utf-8'))
        camera['frames'][9]['image_to_court'] = matrix
        path.write_text(json.dumps(camera, allow_nan=False), encoding='utf-8')
        path = files['court_geometry.json']
        court = json.loads(path.read_text(encoding='utf-8'))
        court['canonical_keypoints_m'] = TennisCourtGeometry.get_canonical_keypoints().tolist()
        path.write_text(json.dumps(court, allow_nan=False), encoding='utf-8')
        return files
    assets.write_observation_assets = source_assets
    fixture_root = ART / 'fixture'
    fixture_root.mkdir(exist_ok=False)
    try:
        fixture = build_shot_base(fixture_root, playable_source=True)
    finally:
        assets.write_observation_assets = original_assets
    completed, attempted = [], []
    native_draw = renderer._draw_frame
    def observed(image, frame, *args, **kwargs):
        attempted.append(frame)
        value = native_draw(image, frame, *args, **kwargs)
        completed.append(frame)
        return value
    renderer._draw_frame = observed
    try:
        with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
            result.update(semantic_exports_passed=True,
                current_camera_frame9_valid=rebuilt.sources._camera['frames'][9]['is_valid'],
                current_camera_frame9_matrix=matrix, determinant=float(np.linalg.det(matrix)))
            try:
                output = render(rebuilt, shots, forecast, exports, fixture.scratch)
                result.update(render_accepted=True, output_directory=str(output.directory))
            except (ValueError, TypeError, RuntimeError) as error:
                result.update(render_accepted=False, rejection=str(error))
            result.update(attempted_frames=attempted, completed_frames=completed,
                render_seals=[str(path) for path in fixture.scratch.rglob('corrected_render_manifest.json')])
        unchanged(fixture)
        result['committed_base_unchanged'] = True
    finally:
        renderer._draw_frame = native_draw
except BaseException:
    result['unexpected_error'] = traceback.format_exc()
finally:
    builtins.__import__ = native_import
receipt = {'result': result, 'source_before': before, 'source_after': pins(),
    'forbidden_import_attempts': attempts,
    'forbidden_imports_loaded': sorted({'torch', 'ultralytics'} & set(sys.modules)),
    'scope': 'Synthetic frozen source with one finite invertible but whole-court horizon-crossing display homography; no inference or accuracy claim'}
receipt['source_stable'] = receipt['source_before'] == receipt['source_after']
(ART / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in receipt.items()
    if key not in {'source_before', 'source_after'}}, indent=2))
