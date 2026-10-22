"""Synthetic reference harness checks; they cannot certify real speed accuracy."""
from dataclasses import asdict
import copy
import json

import numpy as np
import pytest

from src.analytics.flight_evaluation import evaluate_flight_file, evaluate_flight_manifest
from src.analytics.flight_reconstruction import FlightBoundary
from tests.test_flight_reconstruction import known_camera, trajectory


def manifest():
    camera = asdict(known_camera())
    for field in ('intrinsic', 'world_to_camera_rotation', 'world_to_camera_translation_m'):
        camera[field] = camera[field].tolist()
    points, _ = trajectory()
    times = np.linspace(0, .7, 10001)
    speeds = np.linalg.norm(np.array([2,18,3]) + times[:, None]*[0,0,-9.80665], axis=1)
    average = float(np.trapezoid(speeds, times)/.7*3.6)
    return {'schema_version': '1.0', 'usage_rights': 'authorized_for_evaluation',
        'all_strokes_enumerated': True, 'data_kind': 'synthetic_software_test', 'strokes': [
            {'stroke_id': 'stroke-1', 'match_id': 'synthetic-match', 'stroke_type': 'forehand',
             'camera': camera, 'observations': [asdict(point) for point in points],
             'reference': {'definition': 'segment_average_physical_speed', 'speed_kmh': average,
                           'status': 'independently_measured', 'source': 'synthetic_fixture_only',
                           'synchronization_evidence_id': 'synthetic_clock', 'start_seconds': 120, 'end_seconds': 120.7}},
            {'stroke_id': 'stroke-unobserved', 'match_id': 'synthetic-match', 'stroke_type': 'forehand',
             'reference_unavailable_reason': 'synthetic_occlusion_fixture'}]}


def test_coverage_keeps_unmeasurable_strokes_and_proposed_error_gates_do_not_promote_qualification():
    report = evaluate_flight_manifest(manifest())
    assert report['data_kind'] == 'synthetic_software_test'
    summary = report['by_stroke_type']['forehand']
    assert summary['all_stroke_count'] == 2 and summary['unknown_count'] == 1
    assert summary['physical_estimate_coverage'] == .5 and summary['reference_unknown_count'] == 1
    measured = summary['definitions']['segment_average_physical_speed']
    assert measured['median_absolute_error_kmh'] < 1e-6 and measured['p95_absolute_error_kmh'] < 1e-6
    assert measured['proposed_error_gate_passed'] is True
    assert report['qualification_status'].startswith('unqualified')
    assert len(report['strokes']) == 2


def test_mismatched_average_definition_or_timing_is_rejected():
    data = manifest()
    data['strokes'][0]['reference']['start_seconds'] = 121
    with pytest.raises(ValueError, match='same source-time window'):
        evaluate_flight_manifest(data)
    data = manifest()
    data['strokes'][0]['reference']['definition'] = 'ground_plane_displacement'
    with pytest.raises(ValueError, match='distinguish'):
        evaluate_flight_manifest(data)
    data = manifest()
    data['strokes'][0]['reference']['synchronization_evidence_id'] = ''
    with pytest.raises(ValueError, match='synchronized timing'):
        evaluate_flight_manifest(data)


def test_impact_speed_reference_needs_evidenced_contact_and_correct_incoming_outgoing_side():
    data = manifest()
    row = data['strokes'][0]
    row['reference'].update(definition='racket_impact_speed', speed_kmh=float(np.linalg.norm([2,18,3])*3.6),
                            timestamp_seconds=120, impact_speed_side='outgoing')
    no_contact = evaluate_flight_manifest(data)['by_stroke_type']['forehand']['definitions']['racket_impact_speed']
    assert no_contact['evaluated_count'] == 0 and no_contact['unresolved_count'] == 1
    row['start_boundary'] = asdict(FlightBoundary('racket_contact', 120, 1.5, 'synthetic-contact'))
    with_contact = evaluate_flight_manifest(data)['by_stroke_type']['forehand']['definitions']['racket_impact_speed']
    assert with_contact['evaluated_count'] == 1
    row['reference']['impact_speed_side'] = 'incoming'
    with pytest.raises(ValueError, match='incoming/outgoing'):
        evaluate_flight_manifest(data)
    row['reference']['impact_speed_side'] = 'outgoing'
    row['reference']['timestamp_seconds'] = 120.1
    with pytest.raises(ValueError, match='same evidenced racket'):
        evaluate_flight_manifest(data)


def test_all_strokes_manifest_rights_and_duplicate_guards():
    data = manifest()
    data['all_strokes_enumerated'] = False
    with pytest.raises(ValueError, match='All strokes'):
        evaluate_flight_manifest(data)
    data = manifest()
    data['usage_rights'] = 'unknown'
    with pytest.raises(ValueError, match='rights'):
        evaluate_flight_manifest(data)
    data = manifest()
    data['strokes'][1]['stroke_id'] = 'stroke-1'
    with pytest.raises(ValueError, match='unique'):
        evaluate_flight_manifest(data)


def test_reference_report_binds_sources_and_has_no_residual_derived_ci(tmp_path):
    source, output = tmp_path/'synthetic_manifest.json', tmp_path/'synthetic_report.json'
    source.write_text(json.dumps(manifest()))
    report = evaluate_flight_file(source, output)
    assert len(report['manifest_sha256']) == 64 and len(report['implementation_sha256']) == 2
    assert json.loads(output.read_text()) == report
    assert report['strokes'][0]['estimate']['uncertainty_interval'] is None
