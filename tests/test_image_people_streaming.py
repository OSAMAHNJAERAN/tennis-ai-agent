"""Lazy raw-person export preserves the published observation and identity contract."""

import json
import math

import pytest

from src.detection.image_people import (
    image_people_artifact,
    image_people_metadata,
    image_people_records,
)
from src.utils.bbox_utils import BBox


EXPECTED_METADATA = {
    'schema_version': '1.0',
    'coordinateSystem': 'normalized_video_pixels',
    'identityScope': 'DETECTOR_TRACK_ID_NOT_VERIFIED_PERSON_ID',
    'participantSemantics': (
        'Person detector proposals may include spectators, officials and false positives. '
        'Court-role candidates are not independently verified active participants.'
    ),
    'accuracyValidated': False,
}


def test_exact_values_and_serialized_compatibility_for_independent_observation_fixture():
    edge = BBox(0, -0.0, 100, 80, confidence='1', track_id='0')
    player = BBox('1', '2', '10', '20', confidence='.8', track_id=7.0)
    unknown = BBox(20, 5, 30, 15, confidence=.4, track_id=True)
    later = BBox(12, 20, 18, 35, confidence=.6, track_id=-1)
    invalid = BBox(90, 0, 101, 20, confidence=.9, track_id=8)
    detections = [[edge, invalid, player, unknown], [], [later]]
    selected = {1: [player, None, None], 2: [edge, None, later]}
    expected_frames = [
        {'frame_index': 0, 'timestamp_seconds': 0.0, 'people': [
            {'bbox': [0.0, -0.0, 100.0, 80.0], 'confidence': 1.0, 'sourceTrackId': 0,
             'courtRole': 2, 'participantStatus': 'COURT_ROLE_CANDIDATE'},
            {'bbox': [1.0, 2.0, 10.0, 20.0], 'confidence': .8, 'sourceTrackId': 7,
             'courtRole': 1, 'participantStatus': 'COURT_ROLE_CANDIDATE'},
            {'bbox': [20.0, 5.0, 30.0, 15.0], 'confidence': .4, 'sourceTrackId': None,
             'courtRole': None, 'participantStatus': 'UNKNOWN'},
        ]},
        {'frame_index': 1, 'timestamp_seconds': 1 / 29.97, 'people': []},
        {'frame_index': 2, 'timestamp_seconds': 2 / 29.97, 'people': [
            {'bbox': [12.0, 20.0, 18.0, 35.0], 'confidence': .6, 'sourceTrackId': None,
             'courtRole': 2, 'participantStatus': 'COURT_ROLE_CANDIDATE'},
        ]},
    ]
    expected = {**EXPECTED_METADATA, 'frames': expected_frames}
    records = list(image_people_records(detections, selected, fps=29.97, width=100, height=80))
    artifact = image_people_artifact(detections, selected, fps=29.97, width=100, height=80)
    assert image_people_metadata() == EXPECTED_METADATA
    assert records == expected_frames
    assert artifact == expected
    assert json.dumps(artifact, allow_nan=False) == json.dumps(expected, allow_nan=False)
    assert math.copysign(1, records[0]['people'][0]['bbox'][1]) == -1


def test_metadata_is_fresh_and_never_contains_materialized_records():
    first = image_people_metadata()
    first['accuracyValidated'] = True
    first['frames'] = ['not an observation']
    assert image_people_metadata() == EXPECTED_METADATA


class ReadBoundedFrames:
    """A configured-scale sequence that refuses reads beyond requested frames."""

    def __init__(self, count, proposals, allowed):
        self.count = count
        self.proposals = proposals
        self.allowed = allowed
        self.reads = []

    def __len__(self):
        return self.count

    def __iter__(self):
        for index in range(self.count):
            assert index < self.allowed, 'Exporter eagerly read an unrequested frame'
            self.reads.append(index)
            yield self.proposals


class ReadBoundedSelection:
    def __init__(self, count, box, allowed):
        self.count = count
        self.box = box
        self.allowed = allowed
        self.reads = []

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        assert index < self.allowed, 'Exporter eagerly inspected an unrequested role'
        self.reads.append(index)
        return self.box


def test_two_requested_frames_do_not_read_the_remaining_432000_frame_clip():
    box = BBox(1, 2, 10, 20, confidence=.75, track_id=7)
    detections = ReadBoundedFrames(432_000, [box], allowed=2)
    role_1 = ReadBoundedSelection(432_000, box, allowed=2)
    role_2 = ReadBoundedSelection(432_000, None, allowed=2)
    records = image_people_records(detections, {1: role_1, 2: role_2}, fps=60, width=100, height=100)
    assert iter(records) is records
    assert detections.reads == role_1.reads == role_2.reads == []
    first = next(records)
    assert detections.reads == role_1.reads == [0]
    assert role_2.reads == []
    second = next(records)
    assert detections.reads == role_1.reads == [0, 1]
    assert role_2.reads == []
    assert [first['frame_index'], second['frame_index']] == [0, 1]
    assert [first['timestamp_seconds'], second['timestamp_seconds']] == [0.0, 1 / 60]
    assert first['people'][0]['courtRole'] == second['people'][0]['courtRole'] == 1
    records.close()
    assert detections.reads == [0, 1]


def test_same_value_distinct_objects_do_not_inherit_selected_identity():
    first = BBox(1, 2, 10, 20, confidence=.8, track_id=7)
    second = BBox(1, 2, 10, 20, confidence=.8, track_id=7)
    unobserved = BBox(1, 2, 10, 20, confidence=.8, track_id=7)
    assert first == second == unobserved
    assert first is not second and first is not unobserved
    frame = next(image_people_records([[first, second]], {1: [second], 2: [unobserved]},
                                      fps=30, width=100, height=100))
    assert [person['sourceTrackId'] for person in frame['people']] == [7, 7]
    assert [person['courtRole'] for person in frame['people']] == [None, 1]
    assert [person['participantStatus'] for person in frame['people']] == ['UNKNOWN', 'COURT_ROLE_CANDIDATE']


def test_same_observed_object_selected_for_both_roles_keeps_first_role_precedence():
    box = BBox(1, 2, 10, 20)
    frame = next(image_people_records([[box, box]], {1: [box], 2: [box]},
                                      fps=30, width=100, height=100))
    assert [person['courtRole'] for person in frame['people']] == [1, 1]


@pytest.mark.parametrize('invalid', [
    BBox(float('nan'), 1, 10, 20),
    BBox(1, float('inf'), 10, 20),
    BBox(1, 1, float('-inf'), 20),
    BBox(1, 1, 10, None),
    BBox(True, 1, 10, 20),
    BBox('bad', 1, 10, 20),
    BBox(-1, 1, 10, 20),
    BBox(1, -1, 10, 20),
    BBox(10, 1, 10, 20),
    BBox(1, 20, 10, 20),
    BBox(1, 1, 101, 20),
    BBox(1, 1, 10, 101),
    BBox(1, 1, 10, 20, confidence=None),
    BBox(1, 1, 10, 20, confidence=True),
    BBox(1, 1, 10, 20, confidence=0),
    BBox(1, 1, 10, 20, confidence=-.5),
    BBox(1, 1, 10, 20, confidence=1.01),
    BBox(1, 1, 10, 20, confidence=float('nan')),
    BBox(1, 1, 10, 20, confidence=float('inf')),
])
def test_invalid_selected_box_is_withheld_without_losing_frame_or_other_person(invalid):
    valid = BBox(20, 30, 40, 50, confidence=.5, track_id=3)
    frame = next(image_people_records([[invalid, valid]], {1: [invalid], 2: [None]},
                                      fps=30, width=100, height=100))
    assert frame == {'frame_index': 0, 'timestamp_seconds': 0.0, 'people': [
        {'bbox': [20.0, 30.0, 40.0, 50.0], 'confidence': .5, 'sourceTrackId': 3,
         'courtRole': None, 'participantStatus': 'UNKNOWN'},
    ]}
    json.dumps(frame, allow_nan=False)


@pytest.mark.parametrize(('track_id', 'expected'), [
    (None, None), (True, None), (False, None), (float('nan'), None),
    (float('inf'), None), (-1, None), (2.5, None), ('bad', None),
    (0, 0), (-0.0, 0), (7.0, 7), ('8', 8),
])
def test_invalid_tracker_id_abstains_without_withholding_valid_observation(track_id, expected):
    box = BBox(1, 2, 10, 20, confidence=.8, track_id=track_id)
    frame = next(image_people_records([[box]], {1: [None], 2: [None]},
                                      fps=30, width=100, height=100))
    assert len(frame['people']) == 1
    assert frame['people'][0]['sourceTrackId'] == expected
    assert frame['people'][0]['courtRole'] is None
    json.dumps(frame, allow_nan=False)


def test_empty_input_and_observed_empty_frames_are_preserved():
    assert list(image_people_records([], {}, fps=30, width=100, height=100)) == []
    assert image_people_artifact([], {}, fps=30, width=100, height=100) == {**EXPECTED_METADATA, 'frames': []}
    assert list(image_people_records([[], []], {1: [None, None], 2: [None, None]},
                                     fps=30, width=100, height=100)) == [
        {'frame_index': 0, 'timestamp_seconds': 0.0, 'people': []},
        {'frame_index': 1, 'timestamp_seconds': 1 / 30, 'people': []},
    ]


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), float('-inf')])
def test_invalid_fps_is_rejected_before_iteration(fps):
    for export in (image_people_records, image_people_artifact):
        with pytest.raises(ValueError, match='^FPS must be positive and finite$'):
            export([], {}, fps=fps, width=100, height=100)


@pytest.mark.parametrize(('width', 'height'), [(0, 100), (-1, 100), (100, 0), (100, -1)])
def test_invalid_dimensions_are_rejected_before_iteration(width, height):
    for export in (image_people_records, image_people_artifact):
        with pytest.raises(ValueError, match='^Video dimensions must be positive$'):
            export([], {}, fps=30, width=width, height=height)


@pytest.mark.parametrize('selected', [{}, {1: [None], 2: []}, {1: [], 2: [None]},
                                      {1: [None, None], 2: [None]}])
def test_misaligned_roles_are_rejected_before_iteration(selected):
    for export in (image_people_records, image_people_artifact):
        with pytest.raises(ValueError, match='^Court-role observations must align with person detection frames$'):
            export([[]], selected, fps=30, width=100, height=100)


@pytest.mark.parametrize(('fps', 'selected'), [('30', {}), (None, {}), (30, {1: None, 2: []})])
def test_malformed_configuration_keeps_existing_type_error(fps, selected):
    for export in (image_people_records, image_people_artifact):
        with pytest.raises(TypeError):
            export([], selected, fps=fps, width=100, height=100)


@pytest.mark.parametrize(('malformed', 'error'), [(None, TypeError), ([None], AttributeError)])
def test_later_malformed_frame_errors_only_when_that_frame_is_consumed(malformed, error):
    detections = [[], malformed]
    selected = {1: [None, None], 2: [None, None]}
    records = image_people_records(detections, selected, fps=30, width=100, height=100)
    assert next(records) == {'frame_index': 0, 'timestamp_seconds': 0.0, 'people': []}
    with pytest.raises(error):
        next(records)
    with pytest.raises(error):
        image_people_artifact(detections, selected, fps=30, width=100, height=100)
