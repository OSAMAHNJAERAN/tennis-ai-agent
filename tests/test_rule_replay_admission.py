"""Authority admission for replay exports; synthetic diagnostic observations."""
import pytest

from tests.test_rule_replay import api, accepted_analysis
from tests.test_point_timeline import shot
from src.scoring.point_timeline import PointTimeline
from src.scoring.rule_replay import write_rule_replay


def test_authority_off_diagnostic_orphan_shot_is_not_published(api, tmp_path):
    with accepted_analysis(api, tmp_path, empty=True) as (engine, owner, source, *_):
        owner.sealed = False
        timeline = PointTimeline(owner, engine.initial_state.to_dict(), source_frames=source)
        diagnostic = shot(1, 'PLAYER_1_HIT', 1)
        timeline.attach_shots([diagnostic])
        timeline.finish()
        assert len(owner.rows('replay_inputs')) == len(owner.rows('transitions')) == 0
        assert len(owner.rows('replay_shots')) == len(owner.member_rows()) == 1
        assert owner.member_rows()[0]['play_scope'] == 'UNKNOWN'
        destination = tmp_path / 'replay.json'
        previous = b'previous complete authority-off replay'
        destination.write_bytes(previous)
        with pytest.raises(ValueError, match='authority-off'):
            write_rule_replay(destination, engine, owner, source,
                              authoritative_events_enabled=False)
        assert destination.read_bytes() == previous
        assert not owner.readers
        assert not list(tmp_path.glob('.replay.json.*.tmp'))
