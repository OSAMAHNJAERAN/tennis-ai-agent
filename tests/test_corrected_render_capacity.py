"""Bounded corrected-frame rendering with actual CPU codecs and synthetic video.

This proves software frame-buffer ownership for 12/128 frames, not long-match
inference, completed-point-index growth, physical accuracy, or live throughput.
"""
import copy
import gc
import json
import time
import tracemalloc
from types import SimpleNamespace

import pytest

from tests.test_corrected_exports import export, fresh_forecast
from tests.test_corrected_rule_rebuild import unchanged
from tests.test_corrected_shot_rebuild import build_shot_base, no_model_imports


OBSERVATIONS = []


def _sources():
    matrix = [[.01, 0., 0.], [0., .01, 0.], [0., 0., 1.]]
    rows = []
    for frame in range(2):
        rows.append({0: {'analysis_allowed': True, 'measurement_transition_allowed': frame > 0,
            'segment_id': 'explicit-synthetic-camera', 'anchor_frame_index': 0,
            'local_reset': False, 'discontinuity_reasons': [],
            'source_pts': str(10**18+frame*100), 'source_time_base': '1/3000',
            'source_frame_index': frame, 'timestamp_seconds': frame/30.},
            1: {'analysis_id': 'fixture', 'epoch': 0, 'revision': 0, 'local_reset': False,
                'observations': [{'track_key': 'fixture-A', 'player_id': 'A', 'state': 'CONFIRMED'},
                                 {'track_key': 'fixture-B', 'player_id': 'B', 'state': 'CONFIRMED'}]}})
    return SimpleNamespace(source_frames=rows,
        _camera={'enabled': True, 'frames': [{'is_valid': True, 'image_to_court': matrix} for _ in rows]})


@pytest.mark.parametrize('damage', [
    'not_allowed', 'previous_not_allowed', 'measurement_cut', 'camera_segment',
    'camera_anchor', 'local_reset', 'source_discontinuity', 'identity_epoch',
    'identity_revision', 'identity_mapping', 'identity_reset',
    'current_registration_missing', 'previous_registration_missing',
])
def test_continuity_barrier_rejects_camera_cut_and_reviewed_identity_changes(damage):
    from src.scoring.corrected_rendering import _continuous
    sources = _sources()
    before, current = sources.source_frames
    assert _continuous(sources, (before[0], before[1]), 1)
    if damage == 'not_allowed':
        current[0]['analysis_allowed'] = False
    elif damage == 'previous_not_allowed':
        before[0]['analysis_allowed'] = False
    elif damage == 'measurement_cut':
        current[0]['measurement_transition_allowed'] = False
    elif damage == 'camera_segment':
        current[0]['segment_id'] = 'new-synthetic-camera'
    elif damage == 'camera_anchor':
        current[0]['anchor_frame_index'] = 1
    elif damage == 'local_reset':
        current[0]['local_reset'] = True
    elif damage == 'source_discontinuity':
        current[0]['discontinuity_reasons'] = ['SYNTHETIC_PTS_GAP']
    elif damage == 'identity_epoch':
        current[1]['epoch'] = 1
    elif damage == 'identity_revision':
        current[1]['revision'] = 1
    elif damage == 'identity_mapping':
        current[1]['observations'][0]['player_id'] = 'B'
    elif damage == 'identity_reset':
        current[1]['local_reset'] = True
    elif damage == 'current_registration_missing':
        sources._camera['frames'][1]['is_valid'] = False
    else:
        sources._camera['frames'][0]['is_valid'] = False
    assert not _continuous(sources, (before[0], before[1]), 1)


def test_display_contact_uses_current_frame_mapping_and_stops_at_barrier(tmp_path, monkeypatch):
    import numpy as np
    from src.scoring import corrected_rendering as renderer
    from src.utils.landing_zone_sequence import LandingZoneSequence
    from src.visualization import landing_zones
    from src.visualization.mini_court import MiniCourt
    # Actual fresh rule/shot/forecast owners; this isolated draw test supplies
    # explicit synthetic display homographies, not a newly qualified camera.
    fixture = build_shot_base(tmp_path)
    current_matrices = {4: np.diag([.01, .01, 1.]), 5: np.diag([.02, .02, 1.])}
    observed = []
    native_draw = landing_zones.draw_landing
    def drawing(image, record, **kwargs):
        if kwargs.get('transform') is None:
            observed.append(copy.deepcopy(record))
        return native_draw(image, record, **kwargs)
    monkeypatch.setattr(landing_zones, 'draw_landing', drawing)
    with fresh_forecast(fixture) as (rebuilt, shots, forecast):
        event = {'event_id': 2, 'event_type': 'BOUNCE', 'timestamp_s': 4/30.,
            'court_position_m': [3., 8.], 'evidence': {'active_play': True,
                'review_status': 'confirmed', 'position_type': 'ground_contact',
                'position_uncertainty_m': .1,
                'ground_contact_source': 'EXPLICIT_SYNTHETIC_CONTACT_NOT_PHYSICAL_GROUND_TRUTH'}}
        record = landing_zones.landing_record(event, current_matrices[4], analysis_id='fixture',
            revision=1, calibration_id='frame_4', partition='tactical_3x3')
        monkeypatch.setattr(renderer, '_camera', lambda sources, frame: current_matrices[frame])
        with LandingZoneSequence(fixture.scratch, 'fixture', 1) as landings:
            landings.append(record, {'analysis_id': 'fixture', 'analysis_revision': 1,
                'frame_index': 4, 'image_to_court': current_matrices[4].tolist()})
            landings.seal()
            mini = MiniCourt()
            initial = rebuilt.engine.initial_state.to_dict()
            prediction = next(row for row in reversed(forecast.history) if row['frame_index'] <= 4)
            for frame in (4, 5):
                image = renderer._draw_frame(np.zeros((960, 1280, 3), np.uint8), frame,
                    rebuilt, shots, landings, 'tactical_3x3', mini, [], 0,
                    prediction, None, initial)
                assert image.shape == (960, 1280, 3)
            assert len(observed) == 2
            assert observed[0]['contact_position_px'] == pytest.approx([300., 800.])
            assert observed[1]['contact_position_px'] == pytest.approx([150., 400.])
            for row in observed:
                assert row['contact_position_m'] == [3., 8.]
            np.testing.assert_allclose(observed[1]['zones'][0]['polygon_px'],
                                       np.asarray(observed[0]['zones'][0]['polygon_px'])/2)
            assert record['contact_position_px'] == pytest.approx([300., 800.])
            renderer._draw_frame(np.zeros((960, 1280, 3), np.uint8), 5, rebuilt, shots,
                landings, 'tactical_3x3', mini, [], 5, prediction, None, initial)
            assert len(observed) == 2  # Earlier segment's .8 second highlight is suppressed.
    unchanged(fixture)


def test_actual_private_render_keeps_python_frame_buffers_bounded_as_duration_grows(tmp_path, monkeypatch):
    from src.scoring import corrected_rendering as renderer
    native_decode = renderer.full_video_check
    measurements = []
    for frames in (12, 128):
        case = tmp_path / str(frames)
        case.mkdir()
        fixture = build_shot_base(case, frame_count=frames, playable_source=True, repeated_hits=True)
        with fresh_forecast(fixture) as (rebuilt, shots, forecast):
            exports = export(rebuilt, shots, forecast, fixture.scratch)
            decoded = []
            def checked(path, **metadata):
                native_decode(path, **metadata)
                decoded.append({'file': path.name, **metadata})
            monkeypatch.setattr(renderer, 'full_video_check', checked)
            # Fixture, source encoding, native rule/shot/forecast owners and
            # export preparation deliberately precede render tracing.
            counts = {'accepted_inputs': len(rebuilt.owner.rows('replay_inputs')),
                'physical_events': len(shots.events), 'shots': len(shots.shots),
                'forecast_history': len(forecast.history)}
            gc.collect()
            tracemalloc.start()
            started = time.perf_counter()
            try:
                result = renderer.render_corrected_video(rebuilt, shots, forecast, exports, fixture.scratch)
                gc.collect()
                retained, peak = tracemalloc.get_traced_memory()
                seconds = time.perf_counter()-started
            finally:
                tracemalloc.stop()
            seal = json.loads(result.files['corrected_render_manifest.json'].read_text(encoding='utf-8'))
            assert seal['frames'] == frames and seal['fps'] == 30.
            assert (seal['width'], seal['height']) == (1280, 960)
            assert set(seal['artifacts']) == {'annotated.mp4', 'annotated_web.mp4'}
            assert seal['published'] is False and seal['independent_accuracy_validated'] is False
            assert {row['file'] for row in decoded} == {'source.mp4', 'annotated.mp4', 'annotated_web.mp4'}
            assert len(decoded) == 3 and all(row['frames'] == frames for row in decoded)
            # render_corrected_video itself fully decoded source/native/browser
            # through ffprobe and bound those outputs before its immutable seal.
            measurements.append({'frames': frames, 'fps': seal['fps'],
                'width': seal['width'], 'height': seal['height'], **counts,
                'output_bytes': {name: record['bytes'] for name, record in seal['artifacts'].items()},
                'source_bytes': exports.files['source.mp4'].stat().st_size,
                'python_retained_bytes': retained, 'python_peak_bytes': peak, 'render_seconds': seconds,
                'full_decode_source_native_browser': True,
                'actual_full_decode_checks': decoded,
                'scope': 'FRAME_RENDER_BUFFERS_AND_VISUAL_OUTPUT_ONLY',
                'excludes': ['fixture_preparation', 'prior_native_owners', 'native_codec_allocations',
                             'native_sqlite_allocations', 'independent_physical_accuracy', 'live_inference']})
        unchanged(fixture)
    OBSERVATIONS.append({'actual_corrected_render_frame_capacity': measurements,
        'fixed_retained_growth_budget_bytes': 2*1024*1024,
        'fixed_peak_growth_budget_bytes': 12*1024*1024})
    small, large = measurements
    assert large['accepted_inputs'] > small['accepted_inputs']
    assert large['shots'] > small['shots'] and large['forecast_history'] > small['forecast_history']
    assert large['python_retained_bytes']-small['python_retained_bytes'] < 2*1024*1024
    assert large['python_peak_bytes']-small['python_peak_bytes'] < 12*1024*1024
