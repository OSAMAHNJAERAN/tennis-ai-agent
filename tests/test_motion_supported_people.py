import pytest

from src.tracking.motion_supported_people import select_motion_supported_people


def person(identity, x):
    return {'sourceTrackId': identity, 'bbox': [x, 10, x + 10, 30], 'confidence': .8,
            'courtRole': None, 'participantStatus': 'UNKNOWN'}


def camera(index, dx=0):
    return {'frame_index': index, 'timestamp_seconds': index / 10, 'isValid': True,
            'frameToAnchor': [[1, 0, -dx], [0, 1, 0], [0, 0, 1]]}


def test_camera_drift_does_not_promote_stationary_official_and_missing_boxes_stay_missing():
    rows = [[person(1, 30 + i * 5), person(2, 70 + i * 3), person(3, 100 + i)] for i in range(10)]
    rows[5] = [rows[5][1], rows[5][2]]
    selected, audit = select_motion_supported_people(rows, 10, [camera(i, i) for i in range(10)])
    assert [person['sourceTrackId'] for person in selected[0]] == [1, 2]
    assert [person['sourceTrackId'] for person in selected[5]] == [2]
    assert audit['statistics'][3]['eligibleMovingPersonCandidate'] is False
    assert audit['gapFilling'] is False


def test_three_moving_people_abstain_instead_of_selecting_top_two():
    rows = [[person(1, i * 3), person(2, 100 + i * 3), person(3, 200 + i * 3)] for i in range(10)]
    selected, audit = select_motion_supported_people(rows, 10, [camera(i) for i in range(10)])
    assert selected == [[]] * 10
    assert audit['ambiguousFrames'] == list(range(10))


def test_unavailable_camera_and_nullable_source_id_never_gain_active_status():
    rows = [[person(1, i * 5), person(None, 100 + i * 5)] for i in range(10)]
    selected, audit = select_motion_supported_people(rows, 10, [None] * 10)
    assert selected == [[]] * 10
    assert audit['statistics'][1]['cameraSupportedSamples'] == 0


def test_source_fragments_are_never_linked_and_no_geometry_or_name_is_created():
    rows = [[person(1 if i < 6 else 8, i * 5)] for i in range(12)]
    selected, audit = select_motion_supported_people(rows, 10, [camera(i) for i in range(12)])
    assert selected[0][0] == rows[0][0] and selected[-1][0] == rows[-1][0]
    assert selected[0][0]['courtRole'] is None
    assert selected[0][0]['participantStatus'] == 'UNKNOWN'
    assert set(audit['statistics']) == {1, 8}
    assert audit['identityMerging'] is False
    with pytest.raises(ValueError):
        select_motion_supported_people(rows, 0, [None] * 12)


def test_bad_observation_with_eligible_id_is_never_promoted_and_duplicates_abstain():
    rows = [[person(1, i * 5)] for i in range(12)]
    rows[5].append({**person(1, 20), 'bbox': [20, 10, 20, 30]})
    rows[6].append({**person(True, 20), 'confidence': .9})
    rows[7].append(None)
    selected, audit = select_motion_supported_people(rows, 10, [camera(i) for i in range(12)])
    assert selected[5] == [] and audit['ambiguousFrames'] == [5]
    assert selected[6] == [rows[6][0]]
    assert selected[7] == [rows[7][0]]
