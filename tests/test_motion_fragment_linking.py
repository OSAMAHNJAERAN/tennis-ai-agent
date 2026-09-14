from src.tracking.motion_fragment_linking import link_player_fragments
from src.tracking.player_fragment_linking import link_player_fragments as static_link, remap_observations


def person(i, x):
    return {'id': i, 'box': [x, 0, x+20, 60], 'confidence': .8}


def test_bidirectional_motion_recovers_moving_player_without_gap_fill():
    frames = [[person(4, i*4)] for i in range(6)]+[[], []]+[[person(230, i*4)] for i in range(8, 14)]
    assert static_link(frames, 30)[1] == []
    mapping, links = link_player_fragments(frames, 30)
    assert mapping == {4: 4, 230: 4}
    assert links[0]['forward_iou'] > .99 and links[0]['backward_iou'] > .99
    changed, _ = remap_observations(frames, [], mapping)
    assert changed[6:8] == [[], []]
    assert changed[-1][0]['box'] == frames[-1][0]['box']


def test_opposite_motion_rejects_forward_only_coincidence():
    frames = [[person(4, i*4)] for i in range(6)]+[[], []]+[[person(230, 32-(i-8)*4)] for i in range(8, 14)]
    assert link_player_fragments(frames, 30)[1] == []


def test_insufficient_history_and_temporal_overlap_stay_separate():
    assert link_player_fragments([[person(1, 0)], [person(2, 0)]], 30)[1] == []
    assert link_player_fragments([[person(1, 0), person(2, 0)]]*5, 30)[1] == []


def test_ambiguous_successors_stay_separate():
    frames = [[person(1, i*4)] for i in range(6)]+[[], []]+[[person(2, i*4), person(3, i*4)] for i in range(8, 14)]
    assert link_player_fragments(frames, 30)[1] == []
