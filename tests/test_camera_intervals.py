"""Bounded scene/PTS/review fixtures, without replay or vision accuracy claims."""
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from src.api.media_timing import frame_mapping
from src.court.calibration import CourtCalibration, calibrate_court
from src.court.camera_intervals import CameraIntervals, CourtViewEvidence, IntervalReview, SUPPORTED_VIEW
from src.court.camera_registration import RegistrationResult
from src.court.court_geometry import TennisCourtGeometry

FRAME = np.full((180, 320, 3), 64, np.uint8)
CANONICAL = TennisCourtGeometry.get_canonical_keypoints()
POINTS = CANONICAL * np.array([20., 5.]) + np.array([40., 20.])
CALIBRATION = calibrate_court(POINTS, CANONICAL)


def timing(index, *, pts=None, source_index=None, valid=True):
    return {'frame_index': index, 'timestamp_seconds': index / 60.,
            'source_frame_index': index if source_index is None else source_index,
            'source_pts': str(index if pts is None else pts), 'source_time_base': '1/60',
            'validMeasurementTransitionFromPrevious': valid and index > 0}


def court(index, qualified=False):
    return CourtViewEvidence(index, POINTS, (True,) * 14, CALIBRATION,
                             SUPPORTED_VIEW if qualified else 'unknown', 'independent-view-validation' if qualified else None)


def registration(scale=1., valid=True):
    matrix = np.diag([scale, scale, 1.])
    calibration = CALIBRATION if valid else CourtCalibration(None, rejection_reason='TRACKING_LOST')
    return RegistrationResult(calibration, matrix if valid else None, {'is_valid': valid})


def review(label='match_play', view=SUPPORTED_VIEW, normal=True, start=0., end=100.):
    return IntervalReview('manual-interval-review', start, end, label, view, normal)


def supported(layer=None):
    layer = layer or CameraIntervals('session-one')
    result = layer.update(FRAME, timing(0), court_observation=court(0), registration=registration(), review=review())
    assert result['analysis_allowed'] and result['needs_new_registration']
    return layer


def test_histogram_similarity_does_not_confirm_tennis_or_play():
    layer = CameraIntervals('unknown-input')
    result = layer.update(FRAME, timing(0), registration=registration())
    assert not result['analysis_allowed']
    assert result['view_status'] == result['activity_status'] == 'unknown'
    assert 'fresh_court_calibration_required' in result['unavailable_reasons']
    assert result['classification_probability'] is None


def test_reviewed_play_has_independent_geometry_and_timing_gates():
    layer = CameraIntervals('reviewed-input')
    first = layer.update(FRAME, timing(0), review=review(), registration=registration())
    assert first['view_status'] == 'supported_reviewed' and not first['analysis_allowed']
    valid = layer.update(FRAME, timing(1), court_observation=court(1), registration=registration(), review=review())
    assert valid['analysis_allowed'] and not valid['measurement_transition_allowed']
    continuous = layer.update(FRAME, timing(2), registration=registration(), review=review())
    assert continuous['measurement_transition_allowed']


def test_qualified_view_geometry_does_not_prove_live_match_or_normal_speed():
    layer = CameraIntervals('qualified-view')
    first = layer.update(FRAME, timing(0), court_observation=court(0, qualified=True), registration=registration())
    assert first['view_status'] == 'supported_qualified_geometry'
    assert not first['analysis_allowed'] and first['activity_status'] == 'unknown'
    # The qualified view remains scoped to the continuous registered camera,
    # while a separate verified play review is still mandatory.
    second = layer.update(FRAME, timing(1), registration=registration(), review=review(view=None))
    assert second['view_status'] == 'supported_qualified_geometry' and second['analysis_allowed']


@pytest.mark.parametrize('label', ['replay', 'advertisement', 'close_up', 'slow_motion', 'warmup', 'between_points', 'unsupported_view'])
def test_explicit_exclusions_never_publish_measurements(label):
    layer = supported()
    result = layer.update(FRAME, timing(1), court_observation=court(1), registration=registration(), review=review(label))
    assert not result['analysis_allowed'] and not result['measurement_transition_allowed']
    assert 'reviewed_' + label in result['unavailable_reasons']
    assert result['activity_status'] == label


def test_cut_is_provisional_and_requires_new_observed_calibration():
    layer = supported()
    other = np.full_like(FRAME, 224)
    cut = layer.update(other, timing(1), registration=registration(), review=review())
    assert cut['local_reset'] and cut['discontinuity_status'] == 'provisional'
    assert cut['discontinuity_reasons'] == ['provisional_scene_cut']
    assert cut['epoch'] == 1 and not cut['analysis_allowed'] and cut['reanchor_required']
    # Existing/returning registration cannot relabel a new camera as anchored.
    held = layer.update(other, timing(2), registration=registration(), review=review())
    assert not held['analysis_allowed'] and held['reanchor_required']
    fresh = layer.update(other, timing(3), court_observation=court(3), registration=registration(), review=review())
    assert fresh['analysis_allowed'] and fresh['needs_new_registration']
    assert not fresh['measurement_transition_allowed']
    assert layer.update(other, timing(4), registration=registration(), review=review())['measurement_transition_allowed']


def test_registration_zoom_change_forces_current_court_reanchor():
    layer = supported()
    layer.update(FRAME, timing(1), registration=registration(), review=review())
    zoom = layer.update(FRAME, timing(2), registration=registration(1.2), review=review())
    assert zoom['local_reset'] and 'provisional_zoom_change' in zoom['discontinuity_reasons']
    assert zoom['relative_zoom_change'] == pytest.approx(.2)
    assert not zoom['analysis_allowed']
    fresh = layer.update(FRAME, timing(3), registration=registration(1.2), court_observation=court(3), review=review())
    assert fresh['analysis_allowed'] and fresh['anchor_frame_index'] == 3
    assert fresh['needs_new_registration']
    new_anchor = layer.update(FRAME, timing(4), registration=registration(), review=review())
    assert not new_anchor['local_reset'] and new_anchor['measurement_transition_allowed']
    assert new_anchor['anchor_frame_index'] == 3


def test_current_observed_geometry_reinitializes_lost_registration():
    layer = supported()
    current = layer.update(FRAME, timing(1), registration=registration(valid=False), court_observation=court(1), review=review())
    assert current['needs_new_registration'] and not current['reanchor_required']
    assert current['analysis_allowed']
    # The caller constructs from this current observation before gating.
    assert layer.gate_registration(current, registration()).calibration.is_valid


def test_reviewed_unsupported_view_overrides_previous_qualified_geometry():
    layer = CameraIntervals('review-conflict')
    layer.update(FRAME, timing(0), registration=registration(), court_observation=court(0, qualified=True), review=review())
    result = layer.update(FRAME, timing(1), registration=registration(), review=review(view='unsupported_view'))
    assert result['view_status'] == 'unknown' and not result['analysis_allowed']
    assert 'reviewed_unsupported_view' in result['unavailable_reasons']


def test_source_pts_gap_duplicates_and_frame_mapping_gaps_reset_scope():
    layer = supported()
    gap = layer.update(FRAME, timing(1, pts=60), registration=registration(), review=review())
    assert 'source_pts_gap_or_duplicate' in gap['discontinuity_reasons'] and not gap['analysis_allowed']
    layer = supported()
    duplicate = layer.update(FRAME, timing(1, pts=0, source_index=0, valid=False), registration=registration(), review=review())
    assert 'source_pts_gap_or_duplicate' in duplicate['discontinuity_reasons']
    assert 'source_frame_gap_or_duplicate' in duplicate['discontinuity_reasons']
    assert not duplicate['measurement_transition_allowed']
    layer = supported()
    skipped = layer.update(FRAME, timing(2), registration=registration(), review=review())
    assert 'frame_mapping_gap' in skipped['discontinuity_reasons'] and skipped['local_reset']


def test_missing_pts_never_falls_back_to_30fps_or_normalized_playback_time():
    layer = supported()
    missing = timing(1)
    missing.update(source_pts=None, source_time_base=None, source_frame_index=None)
    result = layer.update(FRAME, missing, registration=registration(), review=review())
    assert result['source_timestamp_seconds'] is None and not result['analysis_allowed']
    assert 'source_timing_unavailable' in result['unavailable_reasons']
    second = timing(2)
    second.update(source_pts=None, source_time_base=None, source_frame_index=None)
    held = layer.update(FRAME, second, registration=registration(), review=review())
    assert held['epoch'] == result['epoch'] and not held['local_reset']


def test_original_exact_pts_tokens_and_negative_capture_pts_are_preserved():
    layer = CameraIntervals('original-pts')
    original = timing(0)
    original.update(source_pts='-90000', source_time_base='1/90000', source_timestamp_seconds=999.)
    result = layer.update(FRAME, original, court_observation=court(0), registration=registration(), review=review(start=-2., end=0.))
    assert result['source_pts'] == '-90000' and result['source_time_base'] == '1/90000'
    assert result['source_timestamp_seconds'] == -1. and result['timestamp_seconds'] == 0.
    assert result['analysis_allowed']


def test_actual_worker_mapping_records_gate_duplicate_and_cadence_transitions(tmp_path):
    import json
    stats = tmp_path / 'stats.txt'
    mapping = tmp_path / 'source_frame_mapping.jsonl'
    stats.write_text('0 0 90000 1/90000 0 1/60\n1 1 91500 1/90000 1 1/60\n2 1 91500 1/90000 2 1/60\n')
    frame_mapping(stats, mapping, Fraction(60, 1), 2, 1)
    layer = CameraIntervals('worker-timing')
    results = [layer.update(FRAME, record, court_observation=court(record['frame_index']), registration=registration(), review=review())
               for record in (json.loads(line) for line in mapping.read_text().splitlines())]
    assert [record['source_pts'] for record in results] == ['90000', '91500', '91500']
    assert [record['measurement_transition_allowed'] for record in results] == [False, True, False]
    assert results[2]['local_reset']


def test_review_scope_and_missing_normal_speed_flag_withhold():
    layer = CameraIntervals('review-scope')
    result = layer.update(FRAME, timing(0), court_observation=court(0), registration=registration(), review=review(start=10., end=20.))
    assert result['activity_status'] == 'unknown' and not result['analysis_allowed']
    slow_unknown = layer.update(FRAME, timing(1), registration=registration(), review=review(normal=False))
    assert not slow_unknown['analysis_allowed']
    assert 'normal_speed_match_play_unconfirmed' in slow_unknown['unavailable_reasons']


def test_invalid_or_predicted_landmarks_cannot_reanchor():
    layer = supported()
    visibility = (False,) + (True,) * 13
    lost = layer.update(FRAME, timing(1), registration=registration(valid=False), review=review())
    assert lost['reanchor_required']
    partial = replace(court(2), observed=visibility)
    result = layer.update(FRAME, timing(2), registration=registration(), court_observation=partial, review=review())
    assert result['reanchor_required'] and not result['analysis_allowed']
    outside = replace(court(3), landmarks_px=POINTS + np.array([1000., 0.]))
    result = layer.update(FRAME, timing(3), registration=registration(), court_observation=outside, review=review())
    assert result['reanchor_required']
    with pytest.raises(ValueError, match='exact source-mapping frame'):
        layer.update(FRAME, timing(4), court_observation=court(3), registration=registration(), review=review())


def test_registration_gate_withholds_unknown_and_preserves_healthy_observation():
    layer = CameraIntervals('gate')
    existing = registration()
    unknown = layer.update(FRAME, timing(0), registration=existing)
    gated = layer.gate_registration(unknown, existing)
    assert not gated.calibration.is_valid and gated.anchor_to_frame_px is None
    assert gated.audit['image_to_court'] is None
    allowed = layer.update(FRAME, timing(1), registration=existing, court_observation=court(1), review=review())
    assert layer.gate_registration(allowed, existing) is existing


def test_independent_stream_ids_deduplicate_only_identical_evidence():
    first, second = supported(CameraIntervals('first')), supported(CameraIntervals('second'))
    a = first.update(FRAME, timing(1), registration=registration(), review=review())
    b = second.update(FRAME, timing(1), registration=registration(), review=review())
    assert a['segment_id'] != b['segment_id']
    assert first.update(FRAME, timing(1), registration=registration(), review=review()) == a
    with pytest.raises(ValueError, match='repeated with different'):
        first.update(FRAME, timing(1), registration=registration(), review=review('replay'))


def test_resolution_change_and_bounded_state_over_many_frames():
    layer = supported()
    changed = layer.update(np.full((240, 480, 3), 64, np.uint8), timing(1), registration=registration(), review=review())
    assert 'source_resolution_changed' in changed['discontinuity_reasons'] and not changed['analysis_allowed']
    layer = supported()
    for index in range(1, 1001):
        result = layer.update(FRAME, timing(index), registration=registration(), review=review())
    assert result['measurement_transition_allowed']
    assert layer.thumbnail.shape == (36, 64, 3) and layer.thumbnail.nbytes == 6912
    assert layer.histogram.shape == (512,) and layer.histogram.nbytes == 2048
    assert not any(isinstance(value, list) for value in layer.__dict__.values())
