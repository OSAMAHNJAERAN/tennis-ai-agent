"""Image person observations survive metric abstention without gaining identities."""

import json

import pytest

from src.api.results import build_result
from src.detection.image_people import image_people_artifact
from src.utils.bbox_utils import BBox
from tests.test_product_api import write_artifacts


def boxes():
    return [BBox(1, 1, 10, 20, confidence=.8, track_id=7),
            BBox(40, 3, 50, 30, confidence=.7, track_id=None),
            BBox(80, 4, 90, 35, confidence=.6, track_id=8)]


def test_raw_people_survive_calibration_abstention_without_top_two_promotion():
    proposals = boxes()
    raw = image_people_artifact([proposals, []], {1: [None, None], 2: [None, None]}, fps=30, width=100, height=100)
    assert len(raw['frames'][0]['people']) == 3
    assert [person['sourceTrackId'] for person in raw['frames'][0]['people']] == [7, None, 8]
    assert {person['participantStatus'] for person in raw['frames'][0]['people']} == {'UNKNOWN'}
    assert all(person['courtRole'] is None for person in raw['frames'][0]['people'])
    assert raw['frames'][1]['people'] == []
    assert raw['accuracyValidated'] is False


def test_candidate_role_attaches_only_to_exact_selected_observation():
    proposals = boxes()
    lookalike = BBox(1, 1, 10, 20, confidence=.8, track_id=7)
    raw = image_people_artifact([proposals], {1: [proposals[1]], 2: [lookalike]}, fps=30, width=100, height=100)
    assert [person['courtRole'] for person in raw['frames'][0]['people']] == [None, 1, None]
    assert raw['frames'][0]['people'][1]['participantStatus'] == 'COURT_ROLE_CANDIDATE'


def test_disk_selection_preserves_exact_proposal_role_with_duplicate_boxes_and_ids(tmp_path):
    import numpy as np
    from src.court.calibration import CourtCalibration
    from src.tracking.court_player_tracker import CourtPlayerTracker
    from src.tracking.player_identity import select_court_players_disk
    from src.utils.player_detection_sequence import PlayerDetectionSequence

    calibration = CourtCalibration(np.eye(3))
    proposals = [BBox(4, 10, 6, 20, confidence=.8, track_id=7),
                 BBox(4, 10, 6, 20, confidence=.8, track_id=7),
                 BBox(4, 1, 6, 4, confidence=.9, track_id=8)]
    with PlayerDetectionSequence(tmp_path) as observations:
        observations.append(proposals)
        with select_court_players_disk(CourtPlayerTracker(calibration), observations, 30, tmp_path) as selected:
            assert selected.index(1, 0) == 0
            assert selected[1][0] is not observations[0][0]
            raw = image_people_artifact(observations, selected, fps=30, width=100, height=100)
            assert [person['courtRole'] for person in raw['frames'][0]['people']] == [1, None, 2]


def test_invalid_box_is_withheld_but_valid_box_with_bad_tracker_id_remains():
    proposals = [BBox(1, 1, 10, 20, confidence=.8, track_id=float('nan')),
                 BBox(float('nan'), 1, 10, 20, confidence=.8),
                 BBox(1, 1, 10, 20, confidence=float('inf')),
                 BBox(99, 1, 105, 20, confidence=.8)]
    raw = image_people_artifact([proposals], {1: [None], 2: [None]}, fps=30, width=100, height=100)
    assert len(raw['frames'][0]['people']) == 1
    assert raw['frames'][0]['people'][0]['sourceTrackId'] is None
    json.dumps(raw, allow_nan=False)


def save_people(directory, *, selected=True):
    proposals = boxes()
    raw = image_people_artifact([proposals, [], proposals], {1: [proposals[0], None, proposals[0]] if selected else [None] * 3,
                                                          2: [None] * 3}, fps=30, width=100, height=100)
    (directory / 'image_people.json').write_text(json.dumps(raw), encoding='utf-8')
    path = directory / 'detections.json'
    detections = json.loads(path.read_text())
    for index in (0, 2):
        detections['frames'][index]['player_1']['source_track_id'] = 7
    path.write_text(json.dumps(detections), encoding='utf-8')
    return raw


def test_result_retains_raw_person_proposals_when_all_court_measurements_abstain(tmp_path):
    job = write_artifacts(tmp_path)
    save_people(tmp_path)
    (tmp_path / 'court_geometry.json').write_text('{"is_valid": false}', encoding='utf-8')
    result = build_result(tmp_path, job)
    assert result['court'] is None
    assert result['trajectories']['player1CourtPositions'] == [None] * 3
    assert result['playerMetrics'] is None and result['events'] == []
    assert len(result['imagePeople']['frames'][0]['people']) == 3
    assert result['imagePeople']['frames'][1]['people'] == []
    assert all(person['participantStatus'] == 'UNKNOWN' for person in result['imagePeople']['frames'][0]['people'])
    assert result['summary']['capabilities']['imagePeople'] is True
    assert result['imagePeople']['identityScope'] == 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID'
    json.dumps(result, allow_nan=False)


def test_result_candidate_role_requires_current_court_selected_box_and_source_id(tmp_path):
    job = write_artifacts(tmp_path)
    raw = save_people(tmp_path)
    raw['frames'][2]['people'][0]['sourceTrackId'] = 100
    (tmp_path / 'image_people.json').write_text(json.dumps(raw), encoding='utf-8')
    result = build_result(tmp_path, job)
    assert result['imagePeople']['frames'][0]['people'][0]['courtRole'] == 1
    assert result['imagePeople']['frames'][0]['people'][1]['courtRole'] is None
    assert result['imagePeople']['frames'][2]['people'][0]['courtRole'] is None
    assert result['imagePeople']['frames'][2]['people'][0]['sourceTrackId'] == 100


def test_result_isolates_malformed_person_and_time_sample_without_gap_fill(tmp_path):
    job = write_artifacts(tmp_path)
    raw = save_people(tmp_path)
    raw['frames'][0]['people'].extend([None, {'bbox': [1, 1, 101, 20], 'confidence': .8}])
    raw['frames'][0]['people'][1]['sourceTrackId'] = float('nan')
    raw['frames'][1]['timestamp_seconds'] = 10
    (tmp_path / 'image_people.json').write_text(json.dumps(raw), encoding='utf-8')
    result = build_result(tmp_path, job)
    assert len(result['imagePeople']['frames'][0]['people']) == 3
    assert result['imagePeople']['frames'][0]['people'][1]['sourceTrackId'] is None
    assert result['imagePeople']['frames'][1]['people'] is None
    assert len(result['imagePeople']['frames'][2]['people']) == 3
    assert result['detections']['frames'][0]['player_1'] is not None
    assert result['trajectories']['ballTrajectory'][0]['state'] == 'DETECTED'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('malformed', [None, '{broken', '[]'])
def test_missing_or_malformed_raw_people_never_synthesizes_from_selected_players(tmp_path, malformed):
    job = write_artifacts(tmp_path)
    if malformed is not None:
        (tmp_path / 'image_people.json').write_text(malformed, encoding='utf-8')
    result = build_result(tmp_path, job)
    assert result['imagePeople'] is None
    assert result['summary']['capabilities']['imagePeople'] is False
    assert result['detections']['frames'][0]['player_1'] is not None
