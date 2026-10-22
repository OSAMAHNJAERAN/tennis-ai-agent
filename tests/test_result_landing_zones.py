"""Actual artifact-to-web integration; synthetic events never qualify vision."""
import json
from copy import deepcopy

import numpy as np

from src.api.results import build_result
from src.api.result_storage import ResultStorage, DiskSeries
from src.visualization.landing_zones import landing_record
from tests.test_product_api import write_artifacts


def contact_artifacts(directory):
    job = write_artifacts(directory)
    event = {'event_id': 5, 'event_type': 'BOUNCE', 'frame': 0, 'timestamp_s': 0.,
             'court_position_m': [2., 14.], 'confidence': 1., 'trajectory_state': 'DETECTED',
             'evidence': {'review_status': 'corrected', 'active_play': True, 'position_type': 'ground_contact',
                          'ground_contact_source': 'synthetic_software_fixture', 'position_uncertainty_m': .1,
                          'analysis_id': job['id'], 'analysis_revision': 2, 'calibration_id': 'frame_0',
                          'contact_position_px': [2., 14.]}}
    records = [landing_record(event, np.eye(3), analysis_id=job['id'], revision=2,
                              calibration_id='frame_0', partition=partition) for partition in ('service', 'tactical_3x3')]
    artifacts = {'metrics.json': {'authoritative_events_enabled': True},
                 'camera_registration.json': {'enabled': False, 'frames': []},
                 'match_events.json': {'analysis_id': job['id'], 'analysis_revision': 2, 'events': [event]},
                 'landing_zones.json': {'schema_version': '1.0', 'analysis_id': job['id'], 'analysis_revision': 2,
                    'calibrations': {'frame_0': {'analysis_id': job['id'], 'analysis_revision': 2, 'frame_index': 0,
                                               'image_to_court': np.eye(3).tolist()}}, 'records': records}}
    for name, value in artifacts.items():
        (directory / name).write_text(json.dumps(value), encoding='utf-8')
    return job


def test_static_camera_records_bind_both_zone_partitions_to_actual_geometry(tmp_path):
    job = contact_artifacts(tmp_path)
    result = build_result(tmp_path, job)
    assert result['analysisRevision'] == 2
    assert [row['partition'] for row in result['landingZones']] == ['service', 'tactical_3x3']
    assert result['cameraTracking']['frames'][0]['imageToCourt'] == np.eye(3).tolist()
    assert result['cameraTracking']['frames'][1]['imageToCourt'] is None  # original calibration gap retained
    assert result['landingZones'][0]['contact_position_px'] == [2., 14.]


def test_self_declared_zone_geometry_cannot_override_actual_frame_artifact(tmp_path):
    job = contact_artifacts(tmp_path)
    raw = json.loads((tmp_path / 'landing_zones.json').read_text())
    raw['calibrations']['frame_0']['image_to_court'][0][0] = 2.
    (tmp_path / 'landing_zones.json').write_text(json.dumps(raw))
    result = build_result(tmp_path, job)
    assert result['landingZones'] == []
    assert result['events'][0]['event_id'] == 5
    assert any('Landing-zone presentation unavailable' in value for value in result['warnings'])


def test_unqualified_events_withhold_zone_artifact_even_if_marked_confirmed(tmp_path):
    job = contact_artifacts(tmp_path)
    (tmp_path / 'metrics.json').write_text(json.dumps({'authoritative_events_enabled': False}))
    result = build_result(tmp_path, job)
    assert result['events'] == result['landingZones'] == []


def test_empty_presentation_does_not_require_or_export_a_missing_revision(tmp_path):
    job = write_artifacts(tmp_path)
    (tmp_path / 'landing_zones.json').write_text(json.dumps({'schema_version': '1.0', 'analysis_id': job['id'],
                                                           'analysis_revision': 0, 'records': []}))
    result = build_result(tmp_path, job)
    assert result['landingZones'] == []
    assert 'analysisRevision' not in result


def test_bounded_result_keeps_more_than_sixty_four_contact_rows(tmp_path):
    job = contact_artifacts(tmp_path)
    original = json.loads((tmp_path / 'match_events.json').read_text())['events'][0]
    events, records = [], []
    for ident in range(1, 66):
        event = deepcopy(original)
        event['event_id'] = ident
        events.append(event)
        records.extend(landing_record(event, np.eye(3), analysis_id=job['id'], revision=2,
            calibration_id='frame_0', partition=partition) for partition in ('service', 'tactical_3x3'))
    (tmp_path / 'match_events.json').write_text(json.dumps({
        'analysis_id': job['id'], 'analysis_revision': 2, 'events': events}))
    document = json.loads((tmp_path / 'landing_zones.json').read_text())
    document['records'] = records
    (tmp_path / 'landing_zones.json').write_text(json.dumps(document))
    with ResultStorage(tmp_path) as storage:
        result = build_result(tmp_path, job, _storage=storage)
        assert isinstance(result['landingZones'], DiskSeries)
        assert len(result['landingZones']) == 130
        assert result['landingZones'][129] == records[129]
        assert not any('Landing-zone presentation unavailable' in warning for warning in result['warnings'])
