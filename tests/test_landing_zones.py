import numpy as np
import pytest

from src.visualization.landing_zones import draw_landing, landing_record, partition_zones, project_polygon


def event(position=(2., 20.), uncertainty=.1):
    return {'event_type': 'BOUNCE', 'event_id': 5, 'timestamp_s': 2., 'court_position_m': list(position),
            'evidence': {'review_status': 'confirmed', 'active_play': True, 'position_type': 'ground_contact',
                         'ground_contact_source': 'human_reviewed_test_fixture', 'position_uncertainty_m': uncertainty}}


def build(value, **kwargs):
    return landing_record(value, np.eye(3), analysis_id='test', revision=2, calibration_id='camera1', **kwargs)


def test_partitions_cover_singles_geometry_and_keep_types_distinct():
    assert len(partition_zones('service')) == 4
    assert len(partition_zones('tactical_3x3')) == 18
    assert len(partition_zones('tactical_4x3')) == 24
    for kind in ('tactical_3x3', 'tactical_4x3'):
        assert sum((z.bounds[2]-z.bounds[0])*(z.bounds[3]-z.bounds[1]) for z in partition_zones(kind)) == pytest.approx(8.23*23.77)


@pytest.mark.parametrize('change', [
    {'event_type': 'SERVE_CONTACT'}, {'evidence': {}},
    {'evidence': {**event()['evidence'], 'active_play': False}},
    {'evidence': {**event()['evidence'], 'position_type': 'airborne_projection'}},
    {'evidence': {**event()['evidence'], 'review_status': 'provisional'}},
    {'evidence': {**event()['evidence'], 'position_uncertainty_m': None}},
])
def test_no_zone_for_airborne_dead_or_unconfirmed_ball(change):
    assert build({**event(), **change}) is None


def test_boundary_uncertainty_highlights_both_candidate_zones():
    record = build(event((5.485, 8.), .05), partition='service')
    assert record['status'] == 'uncertain'
    assert len(record['zones']) == 2
    assert record['line_call'] is None
    assert build(event((2., 1.), .05), partition='service')['status'] == 'outside_partition'


def test_perspective_projection_roundtrips_every_zone_corner():
    matrix = np.array([[.04, .001, -3], [.003, .03, -1], [.0002, .0001, 1.]])
    polygon = partition_zones('service')[0].polygon
    image = project_polygon(polygon, matrix)
    back = np.c_[image, np.ones(4)] @ matrix.T
    np.testing.assert_allclose(back[:, :2] / back[:, 2, None], polygon, atol=1e-10)
    with pytest.raises(ValueError):
        project_polygon(polygon, np.zeros((3, 3)))


def test_complete_polygon_fill_is_source_time_bound_and_shared_with_mini_court():
    image_to_court = np.diag([.1, .1, 1.])
    record = landing_record(event((2., 20.)), image_to_court, analysis_id='test', revision=1, calibration_id='c1')
    frame = np.zeros((260, 140, 3), np.uint8)
    shown = draw_landing(frame, record, timestamp_seconds=2.4)
    assert shown[210, 25].any() and shown[220, 35].any()
    assert not frame.any()
    np.testing.assert_array_equal(draw_landing(frame, record, timestamp_seconds=1.99), frame)
    np.testing.assert_array_equal(draw_landing(frame, record, timestamp_seconds=2.8), frame)
    miniature = draw_landing(frame, record, timestamp_seconds=2.4, transform=lambda x,y: (x*10,y*10))
    # Both views share the full fill and contact marker; caption font scales
    # with the mini-court canvas independently of the geometric overlay.
    np.testing.assert_array_equal(shown[:225], miniature[:225])
