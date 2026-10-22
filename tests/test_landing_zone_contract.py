"""Contact-zone semantics/calibration guards and optional escaped report output."""
import copy

import numpy as np
import pytest

from src.api.landing_zone_contract import validate_landing_zones
from src.api.reporting import readable_report
from src.visualization.landing_zones import landing_record, partition_zones, project_polygon
from tests.test_bounded_reporting import report_result


def fixture(position=(5.485, 8.), radius=.05):
    matrix = np.diag([.1, .1, 1.])
    event = {'event_type': 'BOUNCE', 'event_id': 5, 'frame': 2, 'timestamp_s': 2.,
             'court_position_m': list(position),
             'evidence': {'review_status': 'confirmed', 'active_play': True, 'position_type': 'ground_contact',
                'ground_contact_source': 'independently_reviewed_fixture', 'position_uncertainty_m': radius,
                'analysis_id': 'test', 'analysis_revision': 3, 'calibration_id': 'frame_2',
                'contact_position_px': list(np.array(position)*10)}}
    # Normalize NumPy scalars to JSON numbers in fixture.
    event['evidence']['contact_position_px'] = [float(v) for v in event['evidence']['contact_position_px']]
    records = []
    for partition in ('service', 'tactical_3x3', 'tactical_4x3'):
        record = landing_record(event, matrix, analysis_id='test', revision=3, calibration_id='frame_2', partition=partition)
        record['contact_position_px'] = event['evidence']['contact_position_px']
        records.append(record)
    raw = {'schema_version': '1.0', 'analysis_id': 'test', 'analysis_revision': 3, 'records': records,
           'calibrations': {'frame_2': {'image_to_court': matrix.tolist(), 'analysis_id': 'test',
                                      'analysis_revision': 3, 'frame_index': 2}}}
    frames = [{'frame_index': i, 'isValid': True, 'imageToCourt': matrix.tolist()} for i in range(3)]
    return raw, [event], frames


def validate(raw, events, frames):
    return validate_landing_zones(raw, 'test', events, frame_calibrations=frames, expected_revision=3)


def test_multiple_partitions_same_event_valid_and_duplicate_partition_rejected():
    raw, events, frames = fixture()
    assert validate(raw, events, frames) == raw['records']
    raw['records'].append(copy.deepcopy(raw['records'][0]))
    with pytest.raises(ValueError, match='event or revision'):
        validate(raw, events, frames)


@pytest.mark.parametrize('field,value', [('status','inside'), ('label','Near right service box'),
                                       ('source','fake-source'), ('review_status','corrected'), ('line_call','in')])
def test_status_label_and_provenance_recomputed_from_actual_contact(field, value):
    raw, events, frames = fixture()
    raw['records'][0][field] = value
    with pytest.raises(ValueError, match='recomputed'):
        validate(raw, events, frames)


def test_canonical_but_wrong_zone_cannot_replace_actual_uncertain_cells():
    raw, events, frames = fixture()
    zone = partition_zones('service')[-1]
    raw['records'][0]['zones'] = [{'id': zone.id, 'label': zone.label, 'polygon_m': zone.polygon,
                                  'polygon_px': project_polygon(zone.polygon, frames[2]['imageToCourt'])}]
    with pytest.raises(ValueError, match='candidate cells'):
        validate(raw, events, frames)
    raw, events, frames = fixture((2.,8.))
    raw['records'][0]['zones'][0].update(id=zone.id, label=zone.label, polygon_m=zone.polygon,
                                       polygon_px=project_polygon(zone.polygon, frames[2]['imageToCourt']))
    with pytest.raises(ValueError, match='cell or perspective'):
        validate(raw, events, frames)


@pytest.mark.parametrize('change', [
    lambda raw: raw.update(analysis_revision=4),
    lambda raw: raw['records'][0].update(analysis_revision=4),
    lambda raw: raw['records'][0].update(analysis_revision=True),
    lambda raw: raw['records'][0].update(analysis_id='other'),
    lambda raw: raw['records'][0].update(calibration_id='frame_1'),
    lambda raw: raw['records'][0].update(timestamp_s=2.01),
    lambda raw: raw['records'][0].update(timestamp_s=float('nan')),
    lambda raw: raw['records'][0].update(duration_s=True),
    lambda raw: raw['records'][0].update(position_uncertainty_m=True),
    lambda raw: raw['records'][0].update(contact_position_m=[2.,20.]),
    lambda raw: raw['records'][0].update(contact_position_px=[2.,20.]),
    lambda raw: raw['calibrations']['frame_2'].update(analysis_revision=4),
    lambda raw: raw['calibrations']['frame_2'].update(frame_index=1),
    lambda raw: raw['calibrations']['frame_2'].update(image_to_court=np.eye(3).tolist()),
])
def test_tampered_revision_time_contact_or_registry_withheld(change):
    raw, events, frames = fixture()
    change(raw)
    with pytest.raises(ValueError):
        validate(raw, events, frames)


@pytest.mark.parametrize('field,value', [('active_play',False), ('review_status','provisional'),
    ('position_type','airborne_projection'), ('analysis_revision',4), ('calibration_id','frame_1'),
    ('position_uncertainty_m',None), ('position_uncertainty_m',-.1), ('contact_position_px',None),
    ('contact_position_px',[float('inf'),1])])
def test_unconfirmed_dead_airborne_or_missing_ground_contact_rejected(field, value):
    raw, events, frames = fixture()
    events[0]['evidence'][field] = value
    with pytest.raises(ValueError):
        validate(raw, events, frames)


@pytest.mark.parametrize('kind', ['wrong_image', 'invalid_frame', 'singular', 'nonfinite', 'boolean'])
def test_actual_frame_binding_and_finite_projected_polygon_required(kind):
    raw, events, frames = fixture()
    if kind == 'wrong_image':
        raw['records'][0]['zones'][0]['polygon_px'][0][0] += 20
    elif kind == 'invalid_frame':
        frames[2]['isValid'] = False
    else:
        matrix = np.zeros((3,3)).tolist() if kind == 'singular' else np.eye(3).tolist()
        if kind == 'nonfinite':
            matrix[0][0] = float('inf')
        if kind == 'boolean':
            matrix[0][0] = True
        frames[2]['imageToCourt'] = raw['calibrations']['frame_2']['image_to_court'] = matrix
    with pytest.raises(ValueError):
        validate(raw, events, frames)


def test_unbound_nonempty_document_duplicate_source_events_and_unlisted_event_rejected():
    raw, events, frames = fixture()
    with pytest.raises(ValueError, match='independent frame'):
        validate_landing_zones(raw, 'test', events)
    with pytest.raises(ValueError, match='unique'):
        validate(raw, events*2, frames)
    with pytest.raises(ValueError):
        validate(raw, [], frames)
    empty = {**raw, 'records': []}
    assert validate_landing_zones(empty, 'test', []) == []


def test_ground_contact_pixel_mapping_and_partition_semantics_are_consistent():
    raw, events, frames = fixture((2.,1.), .01)
    assert validate(raw, events, frames)[0]['status'] == 'outside_partition'
    assert raw['records'][0]['zones'] == []
    events[0]['evidence']['contact_position_px'] = [999.,999.]
    for row in raw['records']:
        row['contact_position_px'] = [999.,999.]
    with pytest.raises(ValueError, match='source calibration'):
        validate(raw, events, frames)


def test_report_zone_section_only_when_present_escaped_and_legal_call_separate():
    result = report_result(3)
    original = readable_report(result)
    assert 'Reviewed ground-contact zones' not in original
    result['landingZones'] = []
    assert readable_report(result) == original
    raw, events, frames = fixture()
    result['landingZones'] = validate(raw, events, frames)
    result['landingZones'][0]['source'] = '<script>bad</script>'
    rendered = readable_report(result)
    assert 'Reviewed ground-contact zones' in rendered and 'tactical_4x3' in rendered
    assert '&lt;script&gt;bad&lt;/script&gt;' in rendered and '<script>bad</script>' not in rendered
    assert 'do not establish a legal line or serve decision' in rendered
    assert 'Separate line evidence' in rendered and '0.050 m' in rendered
