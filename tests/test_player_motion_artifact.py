import json

import pytest

from src.utils.identity_frame_sequence import IdentityFrameSequence
from src.utils.player_motion_artifact import (FrameTimes, UnknownGroundPositions,
                                             accumulate_motion_frames, write_player_motion_json)


def test_owned_frame_samples_export_two_players_without_materializing(tmp_path):
    positions = UnknownGroundPositions(300)
    times = FrameTimes(300, 25)
    intervals = [{0: dict(source_timestamp_seconds=100 + index / 25, source_pts=1000 + index,
                         source_time_base='1/25', segment_id='s1', measurement_transition_allowed=index != 100)}
                 for index in range(300)]
    with IdentityFrameSequence(tmp_path) as store:
        summaries = accumulate_motion_frames(positions, positions, times, store, source_frames=intervals)
        assert len(store) == 300
        assert summaries[1]['observed_distance_m'] is None
        assert not isinstance(summaries[1]['samples'], list)
        output = tmp_path / 'motion.json'
        write_player_motion_json(output, dict(schema_version='2.0', player_1=summaries[1], player_2=summaries[2]), [], lambda value: value)
        parsed = json.loads(output.read_text())
        assert len(parsed['player_1']['samples']) == len(parsed['player_2']['samples']) == 300
        assert parsed['player_1']['samples'][100]['source_pts'] == 1100
        assert all(item['speed_kmh'] is None for item in parsed['player_1']['samples'])
        assert parsed['pose_frames'] == []
    assert not list(tmp_path.glob('*.sqlite3'))


def test_unknowns_cannot_upgrade_source_motion_measurements(tmp_path):
    with IdentityFrameSequence(tmp_path) as store:
        with pytest.raises(ValueError, match='qualified ground-contact integration'):
            accumulate_motion_frames([(0, 0)], [(0, 0)], [0], store,
                                     source_frames=[{0: {'measurement_transition_allowed': True}}])


def test_atomic_export_failure_keeps_previous_version_and_removes_owned_temp(tmp_path):
    output = tmp_path / 'motion.json'
    output.write_text('previous validated artifact')
    metadata = {'player_1': {'samples': [{'speed_kmh': float('nan')}]}, 'player_2': {'samples': []}}
    with pytest.raises(ValueError):
        write_player_motion_json(output, metadata, [], lambda value: value)
    assert output.read_text() == 'previous validated artifact'
    assert not list(tmp_path.glob('.motion.json.*.tmp'))


def test_normalized_time_sequence_and_unknown_position_indexing():
    times = FrameTimes(10, 25)
    assert times[-1] == pytest.approx(.36)
    assert times[2:4] == pytest.approx([.08, .12])
    assert UnknownGroundPositions(10)[-1] is None
    with pytest.raises(IndexError):
        times[10]
    with pytest.raises(ValueError):
        FrameTimes(10, 0)
