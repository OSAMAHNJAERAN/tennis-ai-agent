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
    from src.visualization.video_annotator import VideoAnnotator
    completed, attempted, court_draws, captions = [], [], [], []
    current_frame = None
    native_draw = renderer._draw_frame
    native_court_draw = VideoAnnotator.draw_court_keypoints
    native_text = renderer._text
    def court_draw(*args, **kwargs):
        court_draws.append(current_frame)
        return native_court_draw(*args, **kwargs)
    def text_draw(image, value, *args, **kwargs):
        if 'unavailable' in value.lower():
            captions.append([current_frame, value])
        return native_text(image, value, *args, **kwargs)
    def observed(image, frame, *args, **kwargs):
        global current_frame
        current_frame = frame
        attempted.append(frame)
        value = native_draw(image, frame, *args, **kwargs)
        completed.append(frame)
        return value
    renderer._draw_frame = observed
    VideoAnnotator.draw_court_keypoints = staticmethod(court_draw)
    renderer._text = text_draw
    try:
        with fresh_exports(fixture) as (rebuilt, shots, forecast, exports):
            result.update(semantic_exports_passed=True,
                current_camera_frame9_valid=rebuilt.sources._camera['frames'][9]['is_valid'],
                current_camera_frame9_matrix=matrix, determinant=float(np.linalg.det(matrix)))
            try:
                output = render(rebuilt, shots, forecast, exports, fixture.scratch)
                manifest = json.loads(output.files['corrected_render_manifest.json'].read_text(encoding='utf-8'))
                counts = {}
                for name, path in {'source.mp4': exports.files['source.mp4'],
                        'annotated.mp4': output.files['annotated.mp4'],
                        'annotated_web.mp4': output.files['annotated_web.mp4']}.items():
                    capture = cv2.VideoCapture(str(path))
                    count = 0
                    try:
                        assert capture.isOpened()
                        while True:
                            ok, image = capture.read()
                            if not ok:
                                break
                            assert image.shape == (960, 1280, 3)
                            count += 1
                    finally:
                        capture.release()
                    counts[name] = count
                result.update(render_accepted=True, output_directory=str(output.directory),
                    independent_decoded_frame_counts=counts,
                    geometry_withheld_frame_count=manifest['geometry_withheld_frame_count'],
                    geometry_withheld_reason_counts=manifest['geometry_withheld_reason_counts'],
                    court_drawn_frames=court_draws, unavailable_captions=captions,
                    camera_flag_after_render=rebuilt.sources._camera['frames'][9]['is_valid'])
            except (ValueError, TypeError, RuntimeError) as error:
                result.update(render_accepted=False, rejection=str(error))
            result.update(attempted_frames=attempted, completed_frames=completed,
                render_seals=[str(path) for path in fixture.scratch.rglob('corrected_render_manifest.json')])
        unchanged(fixture)
        result['committed_base_unchanged'] = True
    finally:
        renderer._draw_frame = native_draw
        VideoAnnotator.draw_court_keypoints = staticmethod(native_court_draw)
        renderer._text = native_text
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
